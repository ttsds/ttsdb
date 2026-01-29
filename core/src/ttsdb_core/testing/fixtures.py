"""Reusable pytest fixtures for model packages.

These are implemented as *fixture factories* so each model package can bind its
own project root (where `test_data.yaml`, `audio_examples/`, etc. live).
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any


def _resolve_weights_path(model_root: Path, *, prefer_weights_dir: bool) -> Path:
    env_path = os.environ.get("TTSDB_WEIGHTS_PATH")
    if env_path:
        return Path(env_path)

    if prefer_weights_dir:
        weights_dir = model_root / "weights"
        if weights_dir.exists():
            return weights_dir

    return model_root / "weights"


def make_weights_path_fixture(model_root: Path, *, prefer_weights_dir: bool = True, scope: str = "class"):
    import pytest

    @pytest.fixture(scope=scope)
    def weights_path() -> Path:
        """Path to local model weights directory."""

        return _resolve_weights_path(model_root, prefer_weights_dir=prefer_weights_dir)

    return weights_path


def make_test_data_fixture(model_root: Path, *, scope: str = "class"):
    import pytest
    import yaml

    @pytest.fixture(scope=scope)
    def test_data() -> dict[str, Any]:
        """Loaded `test_data.yaml`."""

        test_data_path = model_root / "test_data.yaml"
        if not test_data_path.exists():
            pytest.skip(f"Test data not found at {test_data_path}")
        with open(test_data_path) as f:
            data = yaml.safe_load(f) or {}
        return data

    return test_data


def make_reference_audio_fixture(*, scope: str = "class"):
    import pytest

    @pytest.fixture(scope=scope)
    def reference_audio(test_data: dict[str, Any], tmp_path_factory) -> dict[str, dict[str, str]]:
        """Download reference audio clips described by `test_data.yaml`."""

        ref_data = (test_data or {}).get("reference_audio", {}) or {}
        base = tmp_path_factory.mktemp("reference_audio")

        result: dict[str, dict[str, str]] = {}
        for lang, ref_info in ref_data.items():
            url = (ref_info or {}).get("url")
            if not url:
                continue

            ext = url.split(".")[-1] if "." in url else "wav"
            audio_path = base / f"reference_{lang}.{ext}"
            urllib.request.urlretrieve(url, audio_path)

            result[lang] = {
                "path": str(audio_path),
                "text": (ref_info or {}).get("text", "") or "",
                "language": str(lang),
            }

        if not result:
            pytest.skip("No reference audio found in test_data.yaml")

        return result

    return reference_audio


def make_audio_examples_dir_fixture(model_root: Path, *, scope: str = "function"):
    import pytest

    @pytest.fixture(scope=scope)
    def audio_examples_dir() -> Path:
        """Directory where tests may write example WAV files."""

        examples_dir = model_root / "audio_examples"
        examples_dir.mkdir(exist_ok=True)
        return examples_dir

    return audio_examples_dir

