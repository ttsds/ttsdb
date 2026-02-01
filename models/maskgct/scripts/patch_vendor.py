#!/usr/bin/env python3
"""Patch vendored upstream code for MaskGCT.

This runs as part of `builder/vendor.py` after cloning Amphion into:
  src/ttsdb_maskgct/_vendor/source/

Goals:
- Strip large binary assets from vendored code (keep wheels small)
- Patch upstream code to load those assets from the HF *weights* repo instead

This script is intentionally model-local so other models can implement their own
patch logic without hardcoding anything into `builder/vendor.py`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


def patch_poly_bert_onnx_fallback(
    *, vendor_root: Path, py_file: str, env_var: str, asset_rel: str
) -> None:
    """Patch chinese_model_g2p.py to resolve poly_bert_model.onnx via an env var."""
    file_path = vendor_root / py_file
    if not file_path.exists():
        raise FileNotFoundError(f"Patch target not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    if "_ttsdb_resolve_poly_bert_onnx" in text:
        return

    marker = "from onnxruntime import InferenceSession, GraphOptimizationLevel, SessionOptions"
    if marker not in text:
        raise RuntimeError(f"Patch marker not found in {file_path}")

    helper = f"""

def _ttsdb_resolve_poly_bert_onnx(bert_model: str) -> str:
    \"\"\"Resolve poly_bert_model.onnx path.

    If the upstream ONNX isn't present in the vendored tree (we strip it to keep wheels small),
    fall back to an extracted copy shipped alongside model weights.
    \"\"\"
    local = os.path.join(bert_model, "poly_bert_model.onnx")
    if os.path.exists(local):
        return local

    assets_root = os.environ.get("{env_var}")
    if assets_root:
        alt = os.path.join(assets_root, "{asset_rel}")
        if os.path.exists(alt):
            return alt

    return local
"""

    text = text.replace(
        'os.path.join(bert_model, "poly_bert_model.onnx")',
        "_ttsdb_resolve_poly_bert_onnx(bert_model)",
    )
    text = text.replace(marker, marker + helper)
    file_path.write_text(text, encoding="utf-8")


def strip_path(p: Path) -> None:
    if not p.exists():
        return
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor-root", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    vendor_root: Path = args.vendor_root
    config_path: Path = args.config

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    code = (config.get("code") or {}) if isinstance(config, dict) else {}
    vendor_assets = code.get("vendor_assets", []) or []

    for entry in vendor_assets:
        source_rel = entry.get("source")
        if not source_rel:
            continue

        # 1) Strip from vendored source tree
        strip_path(vendor_root / source_rel)

        # 2) Apply patch if requested
        patch = entry.get("patch") or {}
        patch_file = patch.get("file")
        env_var = patch.get("env", "TTSDB_VENDOR_ASSETS_DIR")
        dest_rel = entry.get("dest") or f"vendor_assets/{source_rel}"

        if patch_file and source_rel.endswith("poly_bert_model.onnx"):
            patch_poly_bert_onnx_fallback(
                vendor_root=vendor_root,
                py_file=patch_file,
                env_var=env_var,
                asset_rel=dest_rel,
            )


if __name__ == "__main__":
    main()
