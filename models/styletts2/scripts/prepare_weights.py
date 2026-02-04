#!/usr/bin/env python3
"""Prepare weights for StyleTTS2.

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

# StyleTTS2 uses a single checkpoint set.
DEFAULT_VARIANT = "base"

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
        description="Prepare StyleTTS2 weights",
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

    from builder.prepare_weights import (
        download_model_weights,
        get_weights_config,
        write_model_config,
    )

    repo_id, commit = get_weights_config(config)

    if args.list:
        print("No variants defined.")
        return
    if args.all or args.variant:
        print("Warning: variants not configured; downloading full weights.")

    download_model_weights(
        repo_id=repo_id,
        weights_dir=WEIGHTS_DIR,
        commit=commit,
        variant=None,
        shared_patterns=SHARED_PATTERNS if SHARED_PATTERNS else None,
    )

    if MODEL_CONFIG:
        write_model_config(MODEL_CONFIG, WEIGHTS_DIR / MODEL_CONFIG_FILENAME)

    print(f"✓ StyleTTS2 weights ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
