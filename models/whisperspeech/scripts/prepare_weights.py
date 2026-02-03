#!/usr/bin/env python3
"""Prepare weights for WhisperSpeech.

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

# WhisperSpeech uses a single supported variant.
DEFAULT_VARIANT = "small"

# Model files required for the supported pipeline variant.
MODEL_FILES = [
    "t2s-v1.95-small-8lang.model",
    "s2a-v1.95-medium-7lang.model",
]

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
        description="Prepare WhisperSpeech weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available variants and exit.",
    )

    args = parser.parse_args()
    config = load_config()

    from builder.prepare_weights import download_dependency, get_weights_config, write_model_config

    repo_id, commit = get_weights_config(config)

    if args.list:
        print(f"Default variant: {DEFAULT_VARIANT}")
        print(f"Available variants: ['{DEFAULT_VARIANT}']")
        return

    variant_dir = WEIGHTS_DIR / DEFAULT_VARIANT
    variant_dir.mkdir(parents=True, exist_ok=True)

    download_dependency(
        repo_id=repo_id,
        local_dir=variant_dir,
        allow_patterns=MODEL_FILES,
        commit=commit,
    )

    if MODEL_CONFIG:
        write_model_config(MODEL_CONFIG, WEIGHTS_DIR / MODEL_CONFIG_FILENAME)

    print(f"✓ WhisperSpeech weights ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
