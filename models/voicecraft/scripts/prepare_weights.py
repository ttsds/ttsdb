#!/usr/bin/env python3
"""Prepare weights for VoiceCraft.

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

# Mapping from variant name to checkpoint filename in the main HF repo.
VARIANT_CHECKPOINTS: dict[str, str] = {
    "giga830m_tts_enhanced": "830M_TTSEnhanced.pth",
}

DEFAULT_VARIANT = "giga830m_tts_enhanced"

# HuggingFace repo containing the checkpoints.
WEIGHTS_REPO = "pyp1/VoiceCraft"

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
        description="Prepare VoiceCraft weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--variant",
        "-v",
        type=str,
        default=None,
        help="Variant to download. Default: download the default variant only.",
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
        download_dependencies_from_config,
        download_dependency,
        write_model_config,
    )

    variants = list(VARIANT_CHECKPOINTS.keys())

    if args.list:
        print(f"Default variant: {DEFAULT_VARIANT}")
        print(f"Available variants: {variants}")
        return

    if args.variant:
        variants = [args.variant]
    elif not args.all:
        variants = [DEFAULT_VARIANT]

    for variant in variants:
        checkpoint_name = VARIANT_CHECKPOINTS.get(variant)
        if not checkpoint_name:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(VARIANT_CHECKPOINTS.keys())}"
            )

        variant_dir = WEIGHTS_DIR / variant
        variant_dir.mkdir(parents=True, exist_ok=True)

        download_dependency(
            repo_id=WEIGHTS_REPO,
            local_dir=variant_dir,
            allow_patterns=[checkpoint_name],
        )

    # Shared EnCodec tokenizer used by VoiceCraft.
    shared_dir = WEIGHTS_DIR / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    download_dependency(
        repo_id=WEIGHTS_REPO,
        local_dir=shared_dir / "encodec",
        allow_patterns=["encodec_4cb2048_giga.th"],
    )

    download_dependencies_from_config(config, WEIGHTS_DIR)

    if MODEL_CONFIG:
        write_model_config(MODEL_CONFIG, WEIGHTS_DIR / MODEL_CONFIG_FILENAME)

    print(f"✓ VoiceCraft weights ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
