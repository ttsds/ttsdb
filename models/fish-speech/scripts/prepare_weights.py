#!/usr/bin/env python3
"""Prepare weights for Fish Speech.

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

# Mapping from variant name to HuggingFace repo IDs.
VARIANT_REPOS: dict[str, str] = {
    # "1.0": "fishaudio/fish-speech-1",
    # "1.0-large": "fishaudio/fish-speech-1",
    # "1.1": "fishaudio/fish-speech-1",
    # "1.1-large": "fishaudio/fish-speech-1",
    # "1.2": "fishaudio/fish-speech-1.2",
    # "1.2-sft": "fishaudio/fish-speech-1.2-sft",
    # "1.4": "fishaudio/fish-speech-1.4",
    # "1.5": "fishaudio/fish-speech-1.5",
    "s1-mini": "fishaudio/openaudio-s1-mini",
}

DEFAULT_VARIANT = "s1-mini"

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


# def _cleanup_variant_files(variant: str, variant_dir: Path) -> None:
#     """Remove files not needed for a specific Fish Speech variant."""
#     if variant == "1.0":
#         for name in [
#             "fish-speech-v1.1.zip",
#             "text2semantic-sft-large-v1.1-4k.pth",
#             "text2semantic-sft-medium-v1.1-4k.pth",
#             "vits_decoder_v1.1.ckpt",
#         ]:
#             (variant_dir / name).unlink(missing_ok=True)
#     elif variant == "1.0-large":
#         for name in [
#             "fish-speech-v1.1.zip",
#             "fish-speech-v1.0.zip",
#             "text2semantic-sft-large-v1.1-4k.pth",
#             "text2semantic-sft-medium-v1.1-4k.pth",
#             "text2semantic-sft-medium-v1-4k.pth",
#             "vits_decoder_v1.1.ckpt",
#         ]:
#             (variant_dir / name).unlink(missing_ok=True)
#     elif variant == "1.1":
#         for name in [
#             "fish-speech-v1.1.zip",
#             "text2semantic-sft-medium-v1-4k.pth",
#             "vits_decoder_v1.1.ckpt",
#         ]:
#             (variant_dir / name).unlink(missing_ok=True)
#     elif variant == "1.1-large":
#         for name in [
#             "fish-speech-v1.1.zip",
#             "fish-speech-v1.0.zip",
#             "text2semantic-sft-medium-v1.1-4k.pth",
#             "text2semantic-sft-large-v1-4k.pth",
#             "text2semantic-sft-medium-v1-4k.pth",
#             "vits_decoder_v1.1.ckpt",
#         ]:
#             (variant_dir / name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Fish Speech weights",
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

    from huggingface_hub import snapshot_download

    from builder.prepare_weights import download_dependencies_from_config, write_model_config

    variants = list(VARIANT_REPOS.keys())

    if args.list:
        print(f"Default variant: {DEFAULT_VARIANT}")
        print(f"Available variants: {variants}")
        return

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

        variant_dir = WEIGHTS_DIR / variant
        variant_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=repo_id,
            local_dir=variant_dir,
            local_dir_use_symlinks=False,
            allow_patterns=SHARED_PATTERNS or None,
        )

        # _cleanup_variant_files(variant, variant_dir)

    download_dependencies_from_config(config, WEIGHTS_DIR)

    if MODEL_CONFIG:
        write_model_config(MODEL_CONFIG, WEIGHTS_DIR / MODEL_CONFIG_FILENAME)

    print(f"✓ Fish Speech weights ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
