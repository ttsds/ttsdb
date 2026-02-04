#!/usr/bin/env python3
"""Prepare weights for XTTS.

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
    "v2": "coqui/XTTS-v2",
}

DEFAULT_VARIANT = "v2"

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
        description="Prepare XTTS weights",
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
    load_config()

    from huggingface_hub import snapshot_download

    from builder.prepare_weights import write_model_config

    variants = list(VARIANT_REPOS.keys())

    if args.list:
        print(f"Default variant: {DEFAULT_VARIANT}")
        print(f"Available variants: {variants}")
        return

    if not variants:
        raise ValueError("No XTTS variants configured.")

    for variant in variants:
        repo_id = VARIANT_REPOS.get(variant)
        if not repo_id:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(VARIANT_REPOS.keys())}"
            )

        variant_dir = WEIGHTS_DIR / variant
        variant_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=repo_id,
            local_dir=variant_dir,
            local_dir_use_symlinks=False,
            allow_patterns=SHARED_PATTERNS or None,
        )

    if MODEL_CONFIG:
        write_model_config(MODEL_CONFIG, WEIGHTS_DIR / MODEL_CONFIG_FILENAME)

    print(f"✓ XTTS weights ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
