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


def make_weights_path_fixture(
    model_root: Path, *, prefer_weights_dir: bool = True, scope: str = "class"
):
    import pytest

    @pytest.fixture(scope=scope)
    def weights_path() -> Path:
        """Path to local model weights directory."""

        return _resolve_weights_path(model_root, prefer_weights_dir=prefer_weights_dir)

    return weights_path


def make_test_data_fixture(model_root: Path, *, scope: str = "class"):
    import pytest
    import yaml

    def _find_test_data_path() -> Path | None:
        # Back-compat: allow per-model `test_data.yaml`.
        local = model_root / "test_data.yaml"
        if local.exists():
            return local

        # Preferred: shared repo asset at `assets/test_data.yaml`.
        for parent in (model_root, *model_root.parents):
            candidate = parent / "assets" / "test_data.yaml"
            if candidate.exists():
                return candidate

        return None

    @pytest.fixture(scope=scope)
    def test_data() -> dict[str, Any]:
        """Loaded `test_data.yaml`."""

        test_data_path = _find_test_data_path()
        if not test_data_path:
            pytest.skip(
                "Test data not found. Expected either model_root/test_data.yaml "
                "or repo assets/test_data.yaml"
            )
        with open(test_data_path) as f:
            data = yaml.safe_load(f) or {}
        return data

    return test_data


def make_reference_audio_fixture(*, scope: str = "class"):
    import pytest

    @pytest.fixture(scope=scope)
    def reference_audio(
        test_data: dict[str, Any], tmp_path_factory, request
    ) -> dict[str, dict[str, str]]:
        """Resolve reference audio clips described by `test_data.yaml`.

        Supports either:
        - `url`: remote URL downloaded into a temp directory
        - `path`: local path (absolute or relative to pytest rootdir)
        """

        ref_data = (test_data or {}).get("reference_audio", {}) or {}
        base = tmp_path_factory.mktemp("reference_audio")

        result: dict[str, dict[str, str]] = {}
        for lang, ref_info in ref_data.items():
            info = ref_info or {}

            local_path = info.get("path")
            if local_path:
                p = Path(str(local_path))
                if not p.is_absolute():
                    # Prefer resolving relative paths from the pytest rootdir,
                    # since model tests often run with cwd set to the model dir.
                    root = Path(str(request.config.rootpath))
                    candidate = root / p
                    if candidate.exists():
                        p = candidate
                    else:
                        # If tests are running from a nested root (e.g. models/<pkg>),
                        # walk up to find the repo-level assets path.
                        for parent in (root, *root.parents):
                            candidate = parent / p
                            if candidate.exists():
                                p = candidate
                                break
                if not p.exists():
                    raise FileNotFoundError(
                        f"Reference audio path for {lang!r} not found: {p} "
                        f"(from test_data.yaml path={local_path!r})"
                    )
                result[str(lang)] = {
                    "path": str(p),
                    "text": info.get("text", "") or "",
                    "language": str(lang),
                }
                continue

            url = info.get("url")
            if not url:
                continue

            ext = url.split(".")[-1] if "." in url else "wav"
            audio_path = base / f"reference_{lang}.{ext}"
            urllib.request.urlretrieve(url, audio_path)

            result[str(lang)] = {
                "path": str(audio_path),
                "text": info.get("text", "") or "",
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
