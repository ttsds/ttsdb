#!/usr/bin/env python3
"""Prepare weights for Pheme.

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
    "base": "PolyAI/pheme",
    "small": "PolyAI/pheme_small",
}

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


def _prepare_variant_layout(variant_dir: Path) -> None:
    """Normalize checkpoint layout to match the original demo scripts."""
    t2s_bin = variant_dir / "t2s.bin"
    s2a_ckpt = variant_dir / "s2a.ckpt"
    t2s_cfg = variant_dir / "config_t2s.json"
    s2a_cfg = variant_dir / "config_s2a.json"

    t2s_dir = variant_dir / "t2s"
    s2a_dir = variant_dir / "s2a"
    t2s_dir.mkdir(parents=True, exist_ok=True)
    s2a_dir.mkdir(parents=True, exist_ok=True)

    if t2s_bin.exists():
        (t2s_dir / "pytorch_model.bin").write_bytes(t2s_bin.read_bytes())
    if t2s_cfg.exists():
        (t2s_dir / "config.json").write_bytes(t2s_cfg.read_bytes())
    if s2a_ckpt.exists():
        (s2a_dir / "s2a.ckpt").write_bytes(s2a_ckpt.read_bytes())
    if s2a_cfg.exists():
        (s2a_dir / "config.json").write_bytes(s2a_cfg.read_bytes())


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Pheme weights",
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

        _prepare_variant_layout(variant_dir)

    download_dependencies_from_config(config, WEIGHTS_DIR)

    if MODEL_CONFIG:
        write_model_config(MODEL_CONFIG, WEIGHTS_DIR / MODEL_CONFIG_FILENAME)

    print(f"✓ Pheme weights ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
