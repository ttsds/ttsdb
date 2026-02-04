#!/usr/bin/env python3
"""Prepare weights for HierSpeech.

Downloads weights from HuggingFace and prepares them for local use.
This script should be customized per model to reflect upstream layout.

Standard directory structure after download:
    weights/
    ├── shared/           # Common files (vocab, deps) - always downloaded
    │   └── <dependency>/ # External dependencies (e.g., vocoder)
    └── <variant>/        # Variant checkpoint directory (if variants are used)
"""

import argparse
import sys
from pathlib import Path

import yaml

# Ensure repo root is on sys.path so we can import builder helpers.
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# =============================================================================
# Model-specific definitions (static knowledge about this model's structure)
# =============================================================================

VARIANT_FILES: dict[str, dict[str, tuple[str, str]]] = {
    "v1": {
        "hierspeechpp_eng_kor/hierspeechpp_v1_ckpt.pth": (
            "https://drive.google.com/file/d/1_rYQZ7YEIxJbXEpJ3Vf4NXXRxLbcfys9/view",
            "hierspeechpp_eng_kor/hierspeechpp_v1_ckpt.pth",
        ),
        "hierspeechpp_eng_kor/config.json": (
            "https://drive.google.com/file/d/1qp4rmTdecnui_DGJbBkrahqp5JCsQ1KZ/view",
            "hierspeechpp_eng_kor/config.json",
        ),
    },
    "v1.1": {
        "hierspeechpp_eng_kor/hierspeechpp_v1_ckpt.pth": (
            "https://drive.google.com/file/d/1xMfhg4qeehGO0RN-zxq-hAnW-omXmpdq/view",
            "hierspeechpp_eng_kor/hierspeechpp_v1_ckpt.pth",
        ),
        "hierspeechpp_eng_kor/config.json": (
            "https://drive.google.com/file/d/1qp4rmTdecnui_DGJbBkrahqp5JCsQ1KZ/view",
            "hierspeechpp_eng_kor/config.json",
        ),
    },
    "lt460": {
        "hierspeechpp_libritts460/hierspeechpp_lt460_ckpt.pth": (
            "https://drive.google.com/file/d/1JxjU40OZfkICqjP7gD2Qn40EiVEzubDo/view",
            "hierspeechpp_libritts460/hierspeechpp_lt460_ckpt.pth",
        ),
        "hierspeechpp_libritts460/config.json": (
            "https://drive.google.com/file/d/1xcbruEoaOiDLm4fgyVb7CSf3oV-SmNlN/view",
            "hierspeechpp_libritts460/config.json",
        ),
    },
    "lt960": {
        "hierspeechpp_libritts960/hierspeechpp_lt960_ckpt.pth": (
            "https://drive.google.com/file/d/1pNDRafZ7DU1WALkGIkVyEJFIlcxp4DnE/view",
            "hierspeechpp_libritts960/hierspeechpp_lt960_ckpt.pth",
        ),
        "hierspeechpp_libritts960/config.json": (
            "https://drive.google.com/file/d/1AArVxxMSIr8fbZyq2DoXwv76YKBrs3Hd/view",
            "hierspeechpp_libritts960/config.json",
        ),
    },
}

DEFAULT_VARIANT = "v1"

# Patterns to always download (in addition to shared/* and variant subdir).
SHARED_PATTERNS: list[str] = []

# Optional: model architecture config written into weights dir.
MODEL_CONFIG: dict[str, object] = {}
MODEL_CONFIG_FILENAME = "model_config.json"

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
CONFIG_PATH = MODEL_DIR / "config.yaml"
WEIGHTS_DIR = MODEL_DIR / "weights"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _download_gdrive(url: str, dest: Path) -> None:
    from gdown import download

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    download(url, str(dest), fuzzy=True)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare HierSpeech weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--variant",
        "-v",
        type=str,
        default=None,
        help="Variant to download (e.g., 'base', 'v1'). Default: download all variants.",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Download all available variants.",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available variants and exit.",
    )

    args = parser.parse_args()
    load_config()

    from builder.prepare_weights import write_model_config

    variants = list(VARIANT_FILES.keys())

    if args.list:
        print(f"Default variant: {DEFAULT_VARIANT}")
        print(f"Available variants: {variants}")
        return

    if args.variant:
        variants = [args.variant]
    elif not args.all:
        variants = list(VARIANT_FILES.keys())

    for variant in variants:
        files = VARIANT_FILES.get(variant)
        if not files:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(VARIANT_FILES.keys())}"
            )

        variant_dir = WEIGHTS_DIR / variant
        for _, (url, rel_path) in files.items():
            _download_gdrive(url, variant_dir / rel_path)

    # Shared Text2W2V checkpoint
    shared_dir = WEIGHTS_DIR / "shared" / "ttv_libritts_v1"
    _download_gdrive(
        "https://drive.google.com/file/d/1JTi3OOhIFFElj1X1u5jBeNa3CPbVS_gk/view",
        shared_dir / "ttv_lt960_ckpt.pth",
    )
    _download_gdrive(
        "https://drive.google.com/file/d/1JMYEGHtljxaTodek4e6cRASQEQ4KVTE6/view",
        shared_dir / "config.json",
    )

    if MODEL_CONFIG:
        write_model_config(MODEL_CONFIG, WEIGHTS_DIR / MODEL_CONFIG_FILENAME)

    print(f"✓ HierSpeech weights ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
