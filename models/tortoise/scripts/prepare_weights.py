#!/usr/bin/env python3
"""Prepare weights for Tortoise.

Downloads the official weights repo and auxiliary HF models used by
alignment/tokenization into `weights/checkpoints/`.

Standard directory structure after download:
    weights/
    ├── shared/           # Common files (if any) - always downloaded
    ├── .models/          # Tortoise model checkpoints
    └── checkpoints/      # Auxiliary models (wav2vec2, etc.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# =============================================================================
# Model-specific definitions (static knowledge about this model's structure)
# =============================================================================

# Patterns to always download (in addition to shared/*)
# Tortoise doesn't have specific shared patterns beyond the default
SHARED_PATTERNS: list[str] = []

# Auxiliary HF repos needed for Tortoise (mirrors Cog container patching).
# These are optional but allow fully offline inference if vendor code is patched.
AUXILIARY_REPOS: dict[str, str] = {
    "wav2vec2-large-robust-ft-libritts-voxpopuli": "jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli",
    "wav2vec2-large-960h": "facebook/wav2vec2-large-960h",
    "tacotron_symbols": "jbetker/tacotron-symbols",
}

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
        return yaml.safe_load(f) or {}


def download_auxiliary_repos(weights_dir: Path) -> None:
    """Download auxiliary HF repos needed for offline inference."""
    from builder.prepare_weights import download_dependency

    ckpt_root = weights_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    for folder, repo_id in AUXILIARY_REPOS.items():
        out = ckpt_root / folder
        if out.exists() and any(out.iterdir()):
            print(f"Aux checkpoint already present: {out}")
            continue
        print(f"Downloading aux checkpoint {repo_id} -> {out} ...")
        download_dependency(repo_id=repo_id, local_dir=out)


def main() -> None:
    # Import core weight utilities
    from builder.prepare_weights import (
        download_dependencies_from_config,
        download_model_weights,
        get_weights_config,
    )

    config = load_config()

    # Get repo info from config
    repo_id, commit = get_weights_config(config)

    # Download main Tortoise weights (no variants)
    print(f"Downloading Tortoise weights from {repo_id}...")
    download_model_weights(
        repo_id=repo_id,
        weights_dir=WEIGHTS_DIR,
        commit=commit,
        variant=None,  # No variants - download all
        shared_patterns=SHARED_PATTERNS if SHARED_PATTERNS else None,
    )

    # Download dependencies from config (if any)
    download_dependencies_from_config(config, WEIGHTS_DIR)

    # Download auxiliary repos for offline inference
    download_auxiliary_repos(WEIGHTS_DIR)

    print(f"✓ Tortoise weights ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
