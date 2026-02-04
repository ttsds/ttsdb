#!/usr/bin/env python3
"""Prepare weights for E2-TTS model.

Downloads weights from HuggingFace into the `weights/` directory and writes a
small model config file so the runtime can discover model hyperparameters.

Standard directory structure after download:
    weights/
    ├── shared/           # Common files (vocab, deps) - always downloaded
    │   └── vocos-mel-24khz/  # Vocoder dependency
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
SHARED_PATTERNS: list[str] = ["*.md"]

# Model architecture config (written to weights dir for runtime)
MODEL_CONFIG = {
    "dim": 1024,
    "depth": 24,
    "heads": 16,
    "ff_mult": 4,
}
MODEL_CONFIG_FILENAME = "e2_model_config.json"

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
        write_model_config,
    )

    config = load_config()

    # Get repo info from config
    repo_id, commit = get_weights_config(config)

    # Download main weights (no variants for E2-TTS)
    # variant=None means download everything, but shared/* is always included
    download_model_weights(
        repo_id=repo_id,
        weights_dir=WEIGHTS_DIR,
        commit=commit,
        variant=None,  # No variants - download all
        shared_patterns=SHARED_PATTERNS,
    )

    # Download dependencies (vocoder)
    download_dependencies_from_config(config, WEIGHTS_DIR)

    # Write model architecture config
    write_model_config(MODEL_CONFIG, WEIGHTS_DIR / MODEL_CONFIG_FILENAME)

    print(f"✓ E2-TTS weights ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
