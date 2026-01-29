"""Generic test classes for TTSDB model packages.

Each model package can subclass these and set a few class attributes to get a
consistent unit/integration test suite without duplicating code.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from ttsdb_core.config import ModelConfig


def _import_object(dotted_path: str):
    """Import `pkg.mod.Symbol` and return Symbol."""

    module_name, _, attr = dotted_path.rpartition(".")
    if not module_name:
        raise ValueError(f"Expected dotted path like 'pkg.mod.Symbol', got: {dotted_path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


class BaseModelConfigTests:
    """Common tests for `config.yaml` loading."""

    PACKAGE_NAME: str
    EXPECTED_MODEL_NAME: str
    EXPECTED_SAMPLE_RATE: int | None = None
    EXPECTED_CODE_ROOT: str | None = None

    def test_config_loads_from_package(self):
        config = ModelConfig.from_package(self.PACKAGE_NAME)
        assert config is not None

    def test_config_has_metadata(self):
        config = ModelConfig.from_package(self.PACKAGE_NAME)
        assert "metadata" in config
        assert config.metadata.name == self.EXPECTED_MODEL_NAME

    def test_config_has_sample_rate(self):
        config = ModelConfig.from_package(self.PACKAGE_NAME)
        if self.EXPECTED_SAMPLE_RATE is None:
            assert config.metadata.sample_rate > 0
        else:
            assert config.metadata.sample_rate == self.EXPECTED_SAMPLE_RATE

    def test_config_has_code_root_if_expected(self):
        if self.EXPECTED_CODE_ROOT is None:
            pytest.skip("No EXPECTED_CODE_ROOT specified for this model")
        config = ModelConfig.from_package(self.PACKAGE_NAME)
        assert config.code.root == self.EXPECTED_CODE_ROOT


class BaseModelImportTests:
    """Common tests for importing a model wrapper class."""

    PACKAGE_NAME: str
    MODEL_CLASS_PATH: str  # e.g. "ttsdb_f5_tts.F5TTS"
    EXPECTED_SAMPLE_RATE_CONST: int | None = None

    def test_model_class_importable(self):
        cls = _import_object(self.MODEL_CLASS_PATH)
        assert cls is not None

    def test_model_has_package_name(self):
        cls = _import_object(self.MODEL_CLASS_PATH)
        assert getattr(cls, "_package_name", None) == self.PACKAGE_NAME

    def test_model_has_sample_rate_constant_if_expected(self):
        if self.EXPECTED_SAMPLE_RATE_CONST is None:
            pytest.skip("No EXPECTED_SAMPLE_RATE_CONST specified for this model")
        cls = _import_object(self.MODEL_CLASS_PATH)
        assert getattr(cls, "SAMPLE_RATE", None) == self.EXPECTED_SAMPLE_RATE_CONST


class BaseModelIntegrationTests:
    """Common integration tests for synthesis.

    Requires the model package's `conftest.py` to provide:
    - `weights_path`
    - `test_data`
    - `reference_audio`
    - `audio_examples_dir`
    """

    PACKAGE_NAME: str
    MODEL_CLASS_PATH: str
    WEIGHTS_PREPARE_HINT: str | None = None
    EXPECTED_SAMPLE_RATE: int | None = None

    @pytest.fixture(scope="class")
    def model(self, weights_path):
        cls = _import_object(self.MODEL_CLASS_PATH)
        if not weights_path.exists():
            hint = self.WEIGHTS_PREPARE_HINT or self.PACKAGE_NAME
            pytest.skip(f"Weights not found at {weights_path}. Run: just hf-weights-prepare {hint}")
        return cls(model_path=str(weights_path))

    @pytest.fixture(scope="class")
    def synthesis_results(self, model, reference_audio, test_data):
        all_sentences: dict[str, list[dict[str, Any]]] = (test_data or {}).get("test_sentences", {}) or {}
        results: dict[tuple[str, int], tuple[Any, int]] = {}

        # If the model exposes supported languages in config, respect it.
        supported = getattr(getattr(getattr(model, "model_config", None), "metadata", None), "languages", None)

        for lang, sentences in all_sentences.items():
            if supported is not None and lang not in supported:
                continue
            ref = (reference_audio or {}).get(lang)
            if not ref:
                continue
            for i, sentence in enumerate(sentences or []):
                text = sentence["text"]
                audio, sr = model.synthesize(
                    text=text,
                    reference_audio=ref["path"],
                    text_reference=ref.get("text", ""),
                    language=lang,
                )
                results[(lang, i)] = (audio, sr)

        return results

    def test_model_loads(self, model):
        assert model is not None
        assert getattr(model, "model", None) is not None

    def test_synthesize_returns_audio(self, synthesis_results):
        if not synthesis_results:
            pytest.skip("No synthesis results (missing ref/sentences or languages)")
        (audio, sr) = next(iter(synthesis_results.values()))
        assert audio is not None
        assert len(audio) > 0
        if self.EXPECTED_SAMPLE_RATE is None:
            assert sr > 0
        else:
            assert sr == self.EXPECTED_SAMPLE_RATE

    def test_generate_audio_examples(self, synthesis_results, audio_examples_dir):
        import soundfile as sf

        for (lang, i), (audio, sr) in synthesis_results.items():
            output_path = audio_examples_dir / f"{lang}_test_{i+1:03d}.wav"
            sf.write(str(output_path), audio, sr)
            assert output_path.exists()
            assert output_path.stat().st_size > 0

