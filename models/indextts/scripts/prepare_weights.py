#!/usr/bin/env python3
"""Prepare weights for IndexTTS.

Downloads weights from HuggingFace and prepares them for local use.
"""

import argparse
import sys
from pathlib import Path

import yaml

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare IndexTTS weights",
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

    config = load_config()

    from huggingface_hub import snapshot_download

    from builder.prepare_weights import (
        download_dependency,
        get_weights_config,
        write_model_config,
    )

    if args.list:
        print("No variants defined.")
        return

    repo_id, commit = get_weights_config(config)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        revision=commit,
        local_dir=WEIGHTS_DIR,
        local_dir_use_symlinks=False,
    )

    vendor_assets_dir = WEIGHTS_DIR / "vendor_assets"
    download_dependency(
        repo_id="Plachta/JDCnet",
        local_dir=vendor_assets_dir
        / "indextts"
        / "utils"
        / "maskgct"
        / "models"
        / "codec"
        / "facodec"
        / "modules"
        / "JDC",
        allow_patterns=["bst.t7"],
    )

    write_model_config({}, WEIGHTS_DIR / "model_config.json")

    print(f"✓ IndexTTS weights ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
