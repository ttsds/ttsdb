#!/usr/bin/env python3
"""Prepare weights for MaskGCT model.

Downloads the model weights from HuggingFace and prepares them
for use with the model.

Standard directory structure after download:
    weights/
    ├── shared/           # Common files (if any) - always downloaded
    └── ...               # Model checkpoints
"""

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
# MaskGCT doesn't have specific shared patterns beyond the default
SHARED_PATTERNS: list[str] = []

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
    # Import core weight utilities
    from builder.prepare_weights import (
        download_dependencies_from_config,
        download_model_weights,
        get_weights_config,
    )

    config = load_config()

    # Get repo info from config
    repo_id, commit = get_weights_config(config)

    # Download main weights (no variants for MaskGCT)
    download_model_weights(
        repo_id=repo_id,
        weights_dir=WEIGHTS_DIR,
        commit=commit,
        variant=None,  # No variants - download all
        shared_patterns=SHARED_PATTERNS if SHARED_PATTERNS else None,
    )

    # Download dependencies (if any)
    download_dependencies_from_config(config, WEIGHTS_DIR)

    print(f"✓ MaskGCT weights ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
