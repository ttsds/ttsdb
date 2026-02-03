#!/usr/bin/env python3
"""Prepare weights for VibeVoice.

Downloads weights from HuggingFace and prepares them for local use.
This script also downloads the required Qwen2.5-1.5B tokenizer.
"""

import argparse
import sys
from pathlib import Path

import yaml

# Ensure repo root is on sys.path so we can import builder helpers.
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
CONFIG_PATH = MODEL_DIR / "config.yaml"
WEIGHTS_DIR = MODEL_DIR / "weights"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare VibeVoice weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.parse_args()

    config = load_config()

    from builder.prepare_weights import (
        download_dependencies_from_config,
        download_model_weights,
        get_weights_config,
    )

    repo_id, commit = get_weights_config(config)

    download_model_weights(
        repo_id=repo_id,
        weights_dir=WEIGHTS_DIR,
        commit=commit,
    )

    download_dependencies_from_config(config, WEIGHTS_DIR)

    print(f"✓ VibeVoice weights ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
