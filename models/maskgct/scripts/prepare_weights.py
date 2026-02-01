#!/usr/bin/env python3
"""Prepare weights for MaskGCT model.

This script downloads the model weights from HuggingFace and prepares them
for upload to our repository.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

# Configuration
SOURCE_REPO = "amphion/MaskGCT"
SOURCE_COMMIT = "e3c5700"

# Paths (relative to model directory)
MODEL_DIR = Path(__file__).parent.parent
HF_DIR = MODEL_DIR / "weights"


def main():
    print(f"Downloading weights from {SOURCE_REPO}...")

    HF_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=SOURCE_REPO,
        revision=SOURCE_COMMIT,
        local_dir=HF_DIR,
        local_dir_use_symlinks=False,
    )

    print(f"✓ Weights downloaded to {HF_DIR}")


if __name__ == "__main__":
    main()
