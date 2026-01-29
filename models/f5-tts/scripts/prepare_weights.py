#!/usr/bin/env python3
"""Prepare weights for F5-TTS model.

Downloads weights from HuggingFace, writes model config (from config.yaml
model.f5) into the weights directory as f5_model_config.json so the runtime
loads it from there. The weights dir is then uploaded to our HF repo.
"""

import json
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

# Paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
CONFIG_PATH = MODEL_DIR / "config.yaml"
# Same convention as builder get_huggingface_dir
WEIGHTS_DIR = MODEL_DIR / "weights"
F5_CONFIG_FILENAME = "f5_model_config.json"
VOCOS_REPO_ID = "charactr/vocos-mel-24khz"
VOCOS_DIRNAME = "vocos-mel-24khz"


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    weights = config.get("weights", {})
    url = weights.get("url")
    commit = weights.get("commit")
    if not url or "huggingface.co/" not in url:
        raise ValueError("config.yaml weights.url must be a HuggingFace URL")

    repo_id = url.rstrip("/").split("huggingface.co/")[-1].split("/tree/")[0].strip("/")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading weights from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        revision=commit,
        local_dir=WEIGHTS_DIR,
        local_dir_use_symlinks=False,
    )

    # Download vocoder for offline inference (mirrors Space usage of Vocos.from_pretrained).
    # This allows running without network by pointing TTSDB_VOCOS_PATH at this folder or
    # relying on the default `<weights>/vocos-mel-24khz/` fallback.
    vocos_dir = WEIGHTS_DIR / VOCOS_DIRNAME
    vocos_config = vocos_dir / "config.yaml"
    vocos_bin = vocos_dir / "pytorch_model.bin"
    if not (vocos_config.exists() and vocos_bin.exists()):
        print(f"Downloading vocoder from {VOCOS_REPO_ID}...")
        snapshot_download(
            repo_id=VOCOS_REPO_ID,
            local_dir=vocos_dir,
            local_dir_use_symlinks=False,
        )
        print(f"✓ Vocos ready in {vocos_dir}")
    else:
        print(f"Vocos already present at {vocos_dir}")

    # Write model config into weights dir so runtime and HF upload both have it
    model_config = dict(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4,
    )
    config_path = WEIGHTS_DIR / F5_CONFIG_FILENAME
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Wrote {F5_CONFIG_FILENAME} -> {config_path}")

    print(f"✓ Weights and config ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
