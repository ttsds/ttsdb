"""Generic test classes for TTSDB model packages.

Each model package can subclass these and set a few class attributes to get a
consistent unit/integration test suite without duplicating code.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
import yaml

from ttsdb_core.config import ModelConfig

try:
    from importlib.resources import as_file, files
except ImportError:  # pragma: no cover - Python <3.9 fallback
    from importlib_resources import as_file, files


def _import_object(dotted_path: str):
    """Import `pkg.mod.Symbol` and return Symbol."""

    module_name, _, attr = dotted_path.rpartition(".")
    if not module_name:
        raise ValueError(f"Expected dotted path like 'pkg.mod.Symbol', got: {dotted_path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _load_raw_config(package_name: str, filename: str = "config.yaml") -> dict[str, Any]:
    """Load raw config.yaml from a package without applying variants."""
    package_files = files(package_name)
    config_file = package_files.joinpath(filename)

    if not config_file.exists():
        # try looking for local config.yaml, (in case of editable install)
        config_file = package_files.parent.parent / filename
        if not config_file.exists():
            raise FileNotFoundError(
                f"Config file {config_file} not found in package {package_name}"
            )

    with as_file(config_file) as config_path:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}


class BaseModelConfigTests:
    """Common tests for `config.yaml` loading."""

    PACKAGE_NAME: str
    EXPECTED_MODEL_NAME: str
    EXPECTED_SAMPLE_RATE: int | None = None
    EXPECTED_CODE_ROOT: str | None = None
    # Optional: expected variants for models that support them
    EXPECTED_VARIANTS: list[str] | None = None
    EXPECTED_DEFAULT_VARIANT: str | None = None

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

    def test_config_has_expected_variants(self):
        if self.EXPECTED_VARIANTS is None:
            pytest.skip("No EXPECTED_VARIANTS specified for this model")
        config = ModelConfig.from_package(self.PACKAGE_NAME)
        assert config.available_variants == self.EXPECTED_VARIANTS

    def test_config_has_default_variant(self):
        if self.EXPECTED_DEFAULT_VARIANT is None:
            pytest.skip("No EXPECTED_DEFAULT_VARIANT specified for this model")
        config = ModelConfig.from_package(self.PACKAGE_NAME)
        assert config.default_variant == self.EXPECTED_DEFAULT_VARIANT

    def test_config_loads_each_variant(self):
        if self.EXPECTED_VARIANTS is None:
            pytest.skip("No EXPECTED_VARIANTS specified for this model")
        for variant in self.EXPECTED_VARIANTS:
            config = ModelConfig.from_package(self.PACKAGE_NAME, variant=variant)
            assert config is not None
            assert config.variant == variant

    def test_config_variant_raises_for_unknown(self):
        if self.EXPECTED_VARIANTS is None:
            pytest.skip("No EXPECTED_VARIANTS specified for this model")
        with pytest.raises(ValueError, match="Unknown variant"):
            ModelConfig.from_package(self.PACKAGE_NAME, variant="nonexistent_variant_xyz")

    def test_variant_metadata_overrides(self):
        raw = _load_raw_config(self.PACKAGE_NAME)
        variants = raw.get("variants", {}) or {}
        metadata_overrides = {
            name: (data or {}).get("metadata", {}) or {}
            for name, data in variants.items()
            if name != "default"
        }

        if not any(metadata_overrides.values()):
            pytest.skip("No variant metadata overrides specified")

        for variant, overrides in metadata_overrides.items():
            if not overrides:
                continue
            config = ModelConfig.from_package(self.PACKAGE_NAME, variant=variant)
            for key, value in overrides.items():
                assert getattr(config.metadata, key) == value


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

    Supports `--variant` CLI option to run tests for specific variants only:
        pytest tests/ -m integration --variant=v3
        pytest tests/ -m integration --variant=v3,v4
    """

    PACKAGE_NAME: str
    MODEL_CLASS_PATH: str
    WEIGHTS_PREPARE_HINT: str | None = None
    # Sample rate: can be int (all variants) or dict[variant, int] (per-variant)
    EXPECTED_SAMPLE_RATE: int | dict[str | None, int] | None = None

    @pytest.fixture(scope="class")
    def variants(self, request):
        from ttsdb_core.testing import get_selected_variants

        config = ModelConfig.from_package(self.PACKAGE_NAME)
        all_variants = config.available_variants or [None]

        # Filter by --variant CLI option if provided
        selected = get_selected_variants(request.config)
        if selected is not None:
            # Filter to only selected variants that exist
            filtered = [v for v in all_variants if v in selected]
            if not filtered:
                available_str = ", ".join(str(v) for v in all_variants)
                pytest.skip(
                    f"Requested variant(s) {selected} not in available variants: [{available_str}]"
                )
            return filtered

        return all_variants

    @pytest.fixture(scope="class")
    def models(self, weights_path, variants):
        cls = _import_object(self.MODEL_CLASS_PATH)
        if not weights_path.exists():
            hint = self.WEIGHTS_PREPARE_HINT or self.PACKAGE_NAME
            pytest.skip(f"Weights not found at {weights_path}. Run: just hf-weights-prepare {hint}")

        missing = []
        for variant in variants:
            if variant is None:
                continue
            variant_dir = weights_path / variant
            if not variant_dir.exists():
                missing.append(variant)

        if missing:
            hint = self.WEIGHTS_PREPARE_HINT or self.PACKAGE_NAME
            pytest.skip(
                "Missing weights for variants: "
                f"{missing}. Run: just hf-weights-prepare {hint} --all"
            )

        models: dict[str | None, Any] = {}
        for variant in variants:
            models[variant] = cls(model_path=str(weights_path), variant=variant)
        return models

    @pytest.fixture(scope="class")
    def synthesis_results(self, models, reference_audio, test_data):
        all_sentences: dict[str, list[dict[str, Any]]] = (test_data or {}).get(
            "test_sentences", {}
        ) or {}
        results: dict[tuple[str | None, str, int], tuple[Any, int]] = {}

        for variant, model in models.items():
            # If the model exposes supported languages in config, respect it.
            supported = getattr(
                getattr(getattr(model, "model_config", None), "metadata", None), "languages", None
            )

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
                    results[(variant, lang, i)] = (audio, sr)

        return results

    def test_model_loads(self, models):
        assert models is not None
        for model in models.values():
            assert model is not None
            assert getattr(model, "model", None) is not None

    def test_synthesize_returns_audio(self, synthesis_results):
        if not synthesis_results:
            pytest.skip("No synthesis results (missing ref/sentences or languages)")

        # Check each synthesis result
        for (variant, lang, i), (audio, sr) in synthesis_results.items():
            assert audio is not None, f"Audio is None for variant={variant}, lang={lang}, i={i}"
            assert len(audio) > 0, f"Audio is empty for variant={variant}, lang={lang}, i={i}"

            # Check sample rate
            if self.EXPECTED_SAMPLE_RATE is None:
                assert sr > 0, f"Sample rate must be positive for variant={variant}"
            elif isinstance(self.EXPECTED_SAMPLE_RATE, dict):
                # Per-variant sample rate
                expected_sr = self.EXPECTED_SAMPLE_RATE.get(variant)
                if expected_sr is not None:
                    assert (
                        sr == expected_sr
                    ), f"Expected {expected_sr}Hz for variant={variant}, got {sr}Hz"
                else:
                    assert sr > 0, f"Sample rate must be positive for variant={variant}"
            else:
                # Single sample rate for all variants
                assert (
                    sr == self.EXPECTED_SAMPLE_RATE
                ), f"Expected {self.EXPECTED_SAMPLE_RATE}Hz, got {sr}Hz for variant={variant}"

    def test_generate_audio_examples(self, synthesis_results, audio_examples_dir):
        import soundfile as sf

        for (variant, lang, i), (audio, sr) in synthesis_results.items():
            variant_name = variant or "default"
            output_dir = audio_examples_dir / variant_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{lang}_test_{i + 1:03d}.wav"
            sf.write(str(output_path), audio, sr)
            assert output_path.exists()
            assert output_path.stat().st_size > 0
