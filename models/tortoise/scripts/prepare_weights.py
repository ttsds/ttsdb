#!/usr/bin/env python3
"""Prepare weights for Tortoise.

Downloads the official weights repo (contains `.models/`) and also downloads
auxiliary HF models used by alignment/tokenization into `weights/checkpoints/`.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
CONFIG_PATH = MODEL_DIR / "config.yaml"
WEIGHTS_DIR = MODEL_DIR / "weights"


def _repo_id_from_hf_url(url: str) -> str:
    if "huggingface.co/" not in url:
        raise ValueError("weights.url must be a HuggingFace URL")
    return url.rstrip("/").split("huggingface.co/")[-1].split("/tree/")[0].strip("/")


def main() -> None:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}

    weights = config.get("weights", {}) or {}
    url = weights.get("url") or ""
    commit = weights.get("commit") or None
    if not url:
        raise ValueError("config.yaml weights.url is required")

    repo_id = _repo_id_from_hf_url(url)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Tortoise weights from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        revision=commit,
        local_dir=WEIGHTS_DIR,
        local_dir_use_symlinks=False,
    )

    # Auxiliary repos (mirrors Cog container patching of wav2vec_alignment.py).
    # These are optional but allow fully offline inference if vendor code is patched to use them.
    aux = {
        "wav2vec2-large-robust-ft-libritts-voxpopuli": "jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli",
        "wav2vec2-large-960h": "facebook/wav2vec2-large-960h",
        "tacotron_symbols": "jbetker/tacotron-symbols",
    }

    ckpt_root = WEIGHTS_DIR / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    for folder, rid in aux.items():
        out = ckpt_root / folder
        if out.exists() and any(out.iterdir()):
            print(f"Aux checkpoint already present: {out}")
            continue
        print(f"Downloading aux checkpoint {rid} -> {out} ...")
        snapshot_download(
            repo_id=rid,
            local_dir=out,
            local_dir_use_symlinks=False,
        )

    print(f"✓ Weights ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
