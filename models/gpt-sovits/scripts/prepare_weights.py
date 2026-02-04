#!/usr/bin/env python3
"""Prepare weights for GPT-SoVITS model.

Downloads weights from HuggingFace for the specified variant(s).

GPT-SoVITS has 4 major versions with different HF repo structures:
- v1: Original release (commit 021ac208...) - pretrained_models/
- v2: Extended release - gsv-v2final-pretrained/
- v3: DiT-based (commit 3725636c...) - s1v3.ckpt, s2Gv3.pth, bigvgan_v2_24khz_100band_256x/
- v4: Custom vocoder (commit 336b2ec4...) - gsv-v4-pretrained/

Standard directory structure after download:
    weights/
    ├── shared/                 # Common dependencies (cnhubert, bert)
    │   ├── chinese-hubert-base/
    │   └── chinese-roberta-wwm-ext-large/
    ├── v1/                     # v1 pretrained models
    │   ├── s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt  (GPT)
    │   └── s2G488k.pth         (SoVITS)
    ├── v2/                     # v2 pretrained models
    │   ├── s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
    │   ├── s2G2333k.pth
    │   └── G2PWModel/          # G2PW for Chinese TTS
    ├── v3/                     # v3 pretrained models
    │   ├── s1v3.ckpt
    │   ├── s2Gv3.pth
    │   └── bigvgan_v2_24khz_100band_256x/
    └── v4/                     # v4 pretrained models
        ├── s1v3.ckpt           # Same GPT as v3
        ├── s2v4.pth
        └── vocoder.pth

Usage:
    python prepare_weights.py              # Download default (v1)
    python prepare_weights.py --variant v2 # Download v2 variant only
    python prepare_weights.py --all        # Download all variants
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# =============================================================================
# Model-specific definitions
# =============================================================================

# HuggingFace repo ID
REPO_ID = "lj1995/GPT-SoVITS"

# Commits for each variant
# v1 has different structure, others use main which has all files
VARIANT_COMMITS: dict[str, str] = {
    "v1": "021ac208db367e69e1982d66d793d0a2af53bfb8",
    "v2": "336b2ec4e8d4ac74740798dd40af44e74659ecaf",
    "v3": "336b2ec4e8d4ac74740798dd40af44e74659ecaf",
    "v4": "336b2ec4e8d4ac74740798dd40af44e74659ecaf",
}

# Files to download for each variant (from HF repo structure)
# Note: v1 commit has files in root, v2+ are in subdirs on main
VARIANT_PATTERNS: dict[str, list[str]] = {
    "v1": [
        "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        "s2G488k.pth",
    ],
    "v2": [
        "gsv-v2final-pretrained/*",
    ],
    "v3": [
        "s1v3.ckpt",
        "s2Gv3.pth",
        "models--nvidia--bigvgan_v2_24khz_100band_256x/*",
    ],
    "v4": [
        "s1v3.ckpt",  # v4 reuses v3 GPT model
        "gsv-v4-pretrained/*",
    ],
}

# Rename mapping: source -> destination within variant dir
VARIANT_RENAMES: dict[str, dict[str, str]] = {
    "v1": {
        "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt": "gpt.ckpt",
        "s2G488k.pth": "sovits.pth",
    },
    "v2": {
        "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt": "gpt.ckpt",
        "gsv-v2final-pretrained/s2G2333k.pth": "sovits.pth",
    },
    "v3": {
        "s1v3.ckpt": "gpt.ckpt",
        "s2Gv3.pth": "sovits.pth",
        "models--nvidia--bigvgan_v2_24khz_100band_256x": "bigvgan",
    },
    "v4": {
        "s1v3.ckpt": "gpt.ckpt",
        "gsv-v4-pretrained/s2Gv4.pth": "sovits.pth",
        "gsv-v4-pretrained/vocoder.pth": "vocoder.pth",
    },
}

DEFAULT_VARIANT = "v1"

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


def download_variant(variant: str, weights_dir: Path, force: bool = False) -> None:
    """Download weights for a specific variant."""
    from huggingface_hub import snapshot_download

    variant_dir = weights_dir / variant
    if not force and variant_dir.exists() and any(variant_dir.iterdir()):
        print(f"Variant '{variant}' already exists. Use --force to re-download.")
        return

    variant_dir.mkdir(parents=True, exist_ok=True)

    commit = VARIANT_COMMITS.get(variant)
    patterns = VARIANT_PATTERNS.get(variant, [])

    print(f"Downloading variant '{variant}' (commit: {commit})...")

    # Download to a temp directory first
    temp_dir = weights_dir / f".tmp_{variant}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=REPO_ID,
            revision=commit,
            local_dir=temp_dir,
            local_dir_use_symlinks=False,
            allow_patterns=patterns,
        )

        # Move and rename files according to VARIANT_RENAMES
        renames = VARIANT_RENAMES.get(variant, {})
        for src_rel, dst_name in renames.items():
            src_path = temp_dir / src_rel
            dst_path = variant_dir / dst_name
            if src_path.exists():
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                if dst_path.exists():
                    if dst_path.is_dir():
                        shutil.rmtree(dst_path)
                    else:
                        dst_path.unlink()
                shutil.move(str(src_path), str(dst_path))
                print(f"  {src_rel} -> {variant}/{dst_name}")

    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    print(f"✓ Variant '{variant}' ready in {variant_dir}")


def download_dependencies(config: dict, weights_dir: Path) -> None:
    """Download shared dependencies (cnhubert, bert)."""
    from builder.prepare_weights import download_dependencies_from_config

    download_dependencies_from_config(config, weights_dir)


def download_g2pw(weights_dir: Path, force: bool = False) -> None:
    """Download G2PWModel for Chinese TTS (v2+).

    G2PWModel is not in the main GPT-SoVITS HF repo, so we download it separately.
    It's stored in shared/ since it's used by all v2+ variants.
    """
    import urllib.request
    import zipfile

    g2pw_dir = weights_dir / "shared" / "G2PWModel"
    if not force and g2pw_dir.exists() and any(g2pw_dir.iterdir()):
        print(f"G2PWModel already exists at {g2pw_dir}")
        return

    print("Downloading G2PWModel for Chinese TTS...")
    g2pw_url = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
    zip_path = weights_dir / "G2PWModel.zip"

    try:
        urllib.request.urlretrieve(g2pw_url, zip_path)
        print(f"  Downloaded {zip_path}")

        # Extract
        g2pw_dir.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(weights_dir / "shared")
        print(f"  Extracted to {g2pw_dir}")
    finally:
        if zip_path.exists():
            zip_path.unlink()

    print("✓ G2PWModel ready")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare GPT-SoVITS weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--variant",
        "-v",
        type=str,
        default=None,
        help=f"Variant to download ({', '.join(VARIANT_COMMITS.keys())}). Default: download all variants.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if weights exist.",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available variants and exit.",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip downloading shared dependencies.",
    )

    args = parser.parse_args()
    config = load_config()

    if args.list:
        print(f"Default variant: {DEFAULT_VARIANT}")
        print(f"Available variants: {list(VARIANT_COMMITS.keys())}")
        for v, c in VARIANT_COMMITS.items():
            print(f"  {v}: {c}")
        return

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Download shared dependencies first
    if not args.skip_deps:
        download_dependencies(config, WEIGHTS_DIR)
        # G2PWModel is needed for v2+ Chinese TTS
        download_g2pw(WEIGHTS_DIR, args.force)

    # Download variant(s)
    # Default: download all variants (needed for HuggingFace upload)
    if args.variant:
        variant = args.variant
        if variant not in VARIANT_COMMITS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(VARIANT_COMMITS.keys())}"
            )
        download_variant(variant, WEIGHTS_DIR, args.force)
    else:
        # Download all variants by default
        print(f"Downloading all variants: {list(VARIANT_COMMITS.keys())}")
        for variant in VARIANT_COMMITS:
            download_variant(variant, WEIGHTS_DIR, args.force)

    print(f"\n✓ GPT-SoVITS weights ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
