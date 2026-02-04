#!/usr/bin/env python3
"""Prepare weights for Vevo.

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

VARIANT_REPOS: dict[str, str] = {
    # "vevo": "amphion/Vevo",
    "vevo1.5": "amphion/Vevo1.5",
}

DEFAULT_VARIANT = "vevo1.5"

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


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Vevo weights",
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
    config = load_config()

    import urllib.request

    from builder.prepare_weights import (
        download_model_weights,
        get_weights_config,
        write_model_config,
    )

    _, commit = get_weights_config(config)

    if args.list:
        print(f"Default variant: {DEFAULT_VARIANT}")
        print(f"Available variants: {list(VARIANT_REPOS.keys())}")
        return

    variants = list(VARIANT_REPOS.keys())
    if args.variant:
        variants = [args.variant]
    elif not args.all:
        variants = list(VARIANT_REPOS.keys())

    for variant in variants:
        repo_id = VARIANT_REPOS.get(variant)
        if not repo_id:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(VARIANT_REPOS.keys())}"
            )

        download_model_weights(
            repo_id=repo_id,
            weights_dir=WEIGHTS_DIR,
            commit=commit,
            variant=variant,
            shared_patterns=SHARED_PATTERNS if SHARED_PATTERNS else None,
        )

    # Download HuBERT checkpoint expected by Vevo pipeline.
    hubert_dir = WEIGHTS_DIR / "shared" / "hubert"
    hubert_dir.mkdir(parents=True, exist_ok=True)
    hubert_path = hubert_dir / "hubert_fairseq_large_ll60k.pth"
    if not hubert_path.exists():
        urllib.request.urlretrieve(
            "https://download.pytorch.org/torchaudio/models/hubert_fairseq_large_ll60k.pth",
            hubert_path,
        )

    if MODEL_CONFIG:
        write_model_config(MODEL_CONFIG, WEIGHTS_DIR / MODEL_CONFIG_FILENAME)

    print(f"✓ Vevo weights ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
