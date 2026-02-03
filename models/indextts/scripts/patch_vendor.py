#!/usr/bin/env python3
"""Patch vendored IndexTTS code to remove large assets."""

from __future__ import annotations

import argparse
from pathlib import Path

BST_REL_PATH = "indextts/utils/maskgct/models/codec/facodec/modules/JDC/bst.t7"


def _remove(path: Path) -> None:
    if path.is_file():
        path.unlink()
        print(f"Removed {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch vendored IndexTTS assets")
    parser.add_argument("--vendor-root", required=True)
    parser.add_argument("--config", required=False)
    args = parser.parse_args()

    vendor_root = Path(args.vendor_root)

    commons_path = (
        vendor_root
        / "indextts"
        / "utils"
        / "maskgct"
        / "models"
        / "codec"
        / "facodec"
        / "modules"
        / "commons.py"
    )

    if commons_path.exists():
        text = commons_path.read_text(encoding="utf-8")
        if "_ttsdb_resolve_bst_t7" not in text:
            helper = f"""

def _ttsdb_resolve_bst_t7(path):
    if os.path.exists(path):
        return path
    assets_root = os.environ.get("TTSDB_VENDOR_ASSETS_DIR")
    if assets_root:
        alt = os.path.join(assets_root, "{BST_REL_PATH}")
        if os.path.exists(alt):
            return alt
    return path
"""
            insert_marker = "from huggingface_hub import hf_hub_download"
            if insert_marker in text:
                text = text.replace(insert_marker, insert_marker + helper)

            text = text.replace(
                'if not os.path.exists(path):\n        path = hf_hub_download(repo_id="Plachta/JDCnet", filename="bst.t7")',
                'path = _ttsdb_resolve_bst_t7(path)\n    if not os.path.exists(path):\n        path = hf_hub_download(repo_id="Plachta/JDCnet", filename="bst.t7")',
            )

            commons_path.write_text(text, encoding="utf-8")

    targets = [
        vendor_root / BST_REL_PATH,
        vendor_root / "assets" / "IndexTTS2.mp4",
    ]

    for target in targets:
        _remove(target)


if __name__ == "__main__":
    main()
