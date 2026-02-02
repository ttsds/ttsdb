#!/usr/bin/env python3
"""Prepare weights for F5-TTS model.

Downloads weights from HuggingFace, optionally for a specific variant.
When a variant is specified, uses allow_patterns to only download that
variant's checkpoint files (avoiding unnecessary downloads).

Standard directory structure after download:
    weights/
    ├── shared/           # Common files (vocab, deps) - always downloaded
    │   └── vocos-mel-24khz/  # Vocoder dependency
    ├── base/             # Default variant checkpoint
    └── v1/               # v1 variant checkpoint
        └── f5_model_config.json  # Variant-specific config

Usage:
    python prepare_weights.py              # Download all variants
    python prepare_weights.py --variant v1 # Download v1 variant only
    python prepare_weights.py --all        # Download all variants
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# =============================================================================
# Model-specific definitions (static knowledge about this model's structure)
# =============================================================================

# Mapping from variant name to HuggingFace repo subdirectory.
# This is static model knowledge - the HF repo structure is fixed.
VARIANT_SUBDIRS: dict[str, str] = {
    "base": "F5TTS_Base",
    "v1": "F5TTS_v1_Base",
}

# Default variant when none specified
DEFAULT_VARIANT = "base"

# Patterns to always download (in addition to shared/* and variant subdir)
SHARED_PATTERNS: list[str] = ["*.md"]

# Model architecture configs (written to variant dirs for runtime)
MODEL_CONFIGS: dict[str, dict[str, object]] = {
    # Base variant config
    "base": {
        "dim": 1024,
        "depth": 22,
        "heads": 16,
        "ff_mult": 2,
        "text_dim": 512,
        "text_mask_padding": False,
        "conv_layers": 4,
        "pe_attn_head": 1,
        "mel_spec_type": "vocos",
    },
    # v1 uses the original config
    "v1": {
        "dim": 1024,
        "depth": 22,
        "heads": 16,
        "ff_mult": 2,
        "text_dim": 512,
        "conv_layers": 4,
        "mel_spec_type": "vocos",
    },
}
MODEL_CONFIG_FILENAME = "f5_model_config.json"

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
CONFIG_PATH = MODEL_DIR / "config.yaml"
WEIGHTS_DIR = MODEL_DIR / "weights"


# =============================================================================
# Main logic
# =============================================================================


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare F5-TTS weights",
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
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if weights exist.",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available variants and exit.",
    )

    args = parser.parse_args()
    config = load_config()

    if args.list:
        print(f"Default variant: {DEFAULT_VARIANT}")
        print(f"Available variants: {list(VARIANT_SUBDIRS.keys())}")
        return

    # Import core weight utilities
    from builder.prepare_weights import (
        download_all_variants,
        download_dependencies_from_config,
        download_model_weights,
        get_weights_config,
        write_model_config,
    )

    # Get repo info from config
    repo_id, commit = get_weights_config(config)

    if args.variant:
        variant = args.variant
        if variant not in VARIANT_SUBDIRS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(VARIANT_SUBDIRS.keys())}"
            )
        download_model_weights(
            repo_id=repo_id,
            weights_dir=WEIGHTS_DIR,
            commit=commit,
            variant=variant,
            variant_subdirs=VARIANT_SUBDIRS,
            shared_patterns=SHARED_PATTERNS,
        )
        print(f"✓ Variant '{variant}' ready in {WEIGHTS_DIR}")
    else:
        # Default: download all variants
        print(f"Downloading all variants: {list(VARIANT_SUBDIRS.keys())}")
        download_all_variants(
            repo_id=repo_id,
            weights_dir=WEIGHTS_DIR,
            variant_subdirs=VARIANT_SUBDIRS,
            commit=commit,
            shared_patterns=SHARED_PATTERNS,
        )

    # Download dependencies (vocoder, etc.)
    download_dependencies_from_config(config, WEIGHTS_DIR)

    # Write model architecture config into each variant directory
    for variant in VARIANT_SUBDIRS:
        variant_dir = WEIGHTS_DIR / variant
        if not variant_dir.exists():
            continue
        model_cfg = MODEL_CONFIGS.get(variant)
        if model_cfg:
            write_model_config(model_cfg, variant_dir / MODEL_CONFIG_FILENAME)

    print(f"✓ F5-TTS weights ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
