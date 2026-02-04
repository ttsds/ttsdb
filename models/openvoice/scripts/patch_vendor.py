#!/usr/bin/env python3
"""Patch vendored OpenVoice source for OpenVoice v2 integration."""

from __future__ import annotations

import argparse
from pathlib import Path


def _replace_text(path: Path, needle: str, replacement: str) -> None:
    if not path.exists():
        return
    text = path.read_text()
    if needle not in text:
        return
    path.write_text(text.replace(needle, replacement))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", required=True, help="Path to vendored source root")
    parser.add_argument("--config", required=False, help="Path to config.yaml (unused)")
    args = parser.parse_args()

    vendor_root = Path(args.vendor_root).resolve()

    se_extractor = vendor_root / "openvoice" / "se_extractor.py"
    _replace_text(
        se_extractor,
        "assert num_splits > 0, 'input audio is too short'",
        "num_splits = max(num_splits, 1)",
    )
    _replace_text(
        se_extractor,
        "from faster_whisper import WhisperModel\n",
        "",
    )
    _replace_text(
        se_extractor,
        'if model is None:\n        model = WhisperModel(model_size, device="cuda", compute_type="float16")',
        'if model is None:\n        from faster_whisper import WhisperModel\n        model = WhisperModel(model_size, device="cuda", compute_type="float16")',
    )

    api_path = vendor_root / "openvoice" / "api.py"
    _replace_text(
        api_path,
        "if kwargs.get('enable_watermark', True):",
        "if False:",
    )


if __name__ == "__main__":
    main()
