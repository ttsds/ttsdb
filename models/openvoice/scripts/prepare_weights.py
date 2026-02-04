#!/usr/bin/env python3
"""Prepare weights for OpenVoice.

Downloads weights from HuggingFace and prepares them for local use.
This script should be customized per model to reflect upstream layout.

Standard directory structure after download:
    weights/
    ├── openvoice/         # OpenVoice converter + speaker embeddings
    ├── openvoice_en/      # MeloTTS English
    ├── openvoice_zh/      # MeloTTS Chinese
    ├── openvoice_es/      # MeloTTS Spanish
    └── openvoice_fr/      # MeloTTS French
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

REPO_MAP: dict[str, str] = {
    "openvoice": "myshell-ai/OpenVoiceV2",
    "openvoice_en": "myshell-ai/MeloTTS-English",
    "openvoice_zh": "myshell-ai/MeloTTS-Chinese",
    "openvoice_es": "myshell-ai/MeloTTS-Spanish",
    "openvoice_fr": "myshell-ai/MeloTTS-French",
}

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
        description="Prepare OpenVoice weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available repos and exit.",
    )

    args = parser.parse_args()
    load_config()

    from huggingface_hub import snapshot_download

    from builder.prepare_weights import write_model_config

    if args.list:
        for name, repo_id in REPO_MAP.items():
            print(f"{name}: {repo_id}")
        return

    for name, repo_id in REPO_MAP.items():
        target_dir = WEIGHTS_DIR / name
        target_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )

    if MODEL_CONFIG:
        write_model_config(MODEL_CONFIG, WEIGHTS_DIR / MODEL_CONFIG_FILENAME)

    print(f"✓ OpenVoice weights ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
