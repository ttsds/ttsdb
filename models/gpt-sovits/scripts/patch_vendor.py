#!/usr/bin/env python3
"""Patch vendored upstream code for GPT-SoVITS.

This runs as part of `builder/vendor.py` after cloning GPT-SoVITS into:
  src/ttsdb_gpt_sovits/_vendor/source/

Goals:
- Remove reference audio length restrictions (allow any length)
- Add any necessary runtime patches for TTSDB integration

This script is intentionally model-local so other models can implement their own
patch logic without hardcoding anything into `builder/vendor.py`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def patch_wav16k_length_check(vendor_root: Path) -> None:
    """Remove the wav16k length restriction from inference_webui.py.

    The original check rejects reference audio outside 3-10 seconds.
    We patch it to always pass (if True:) to allow any reference length.
    """
    target_file = vendor_root / "GPT_SoVITS" / "inference_webui.py"
    if not target_file.exists():
        print(f"  Skipping wav16k patch: {target_file} not found")
        return

    text = target_file.read_text(encoding="utf-8")

    # Pattern: if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
    old_pattern = r"if\s*\(wav16k\.shape\[0\]\s*>\s*160000\s+or\s+wav16k\.shape\[0\]\s*<\s*48000\):"
    new_pattern = "if True:  # patched by ttsdb - removed length restriction"

    if re.search(old_pattern, text):
        text = re.sub(old_pattern, new_pattern, text)
        target_file.write_text(text, encoding="utf-8")
        print(f"  Patched wav16k length check in {target_file.name}")
    elif "# patched by ttsdb" in text:
        print(f"  wav16k patch already applied in {target_file.name}")
    else:
        print(f"  Warning: wav16k length check pattern not found in {target_file.name}")


def patch_g2pw_model_path(vendor_root: Path) -> None:
    """Patch G2PW model path resolution for TTSDB.

    Allows G2PW model to be loaded from weights directory via environment variable.
    """
    target_file = vendor_root / "GPT_SoVITS" / "text" / "chinese.py"
    if not target_file.exists():
        target_file = vendor_root / "GPT_SoVITS" / "text" / "g2pw" / "__init__.py"
        if not target_file.exists():
            print("  Skipping G2PW patch: target files not found")
            return

    text = target_file.read_text(encoding="utf-8")

    if "_ttsdb_resolve_g2pw" in text:
        print(f"  G2PW patch already applied in {target_file.name}")
        return

    # Add helper function at the top after imports
    helper = '''
import os as _ttsdb_os

def _ttsdb_resolve_g2pw_path(default_path: str) -> str:
    """Resolve G2PW model path, preferring TTSDB weights directory."""
    if _ttsdb_os.path.exists(default_path):
        return default_path
    # Check TTSDB weights directory
    weights_dir = _ttsdb_os.environ.get("TTSDB_WEIGHTS_DIR", "")
    if weights_dir:
        alt = _ttsdb_os.path.join(weights_dir, "G2PWModel")
        if _ttsdb_os.path.exists(alt):
            return alt
    return default_path

'''

    # Insert helper after imports
    lines = text.split("\n")
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_idx = i + 1

    lines.insert(insert_idx, helper)
    text = "\n".join(lines)

    target_file.write_text(text, encoding="utf-8")
    print(f"  Added G2PW path resolver to {target_file.name}")


def patch_hubert_path(vendor_root: Path) -> None:
    """Patch cnhubert path resolution for TTSDB.

    Allows chinese-hubert-base to be loaded from weights/shared directory.
    """
    target_file = vendor_root / "GPT_SoVITS" / "feature_extractor" / "cnhubert.py"
    if not target_file.exists():
        print(f"  Skipping cnhubert patch: {target_file} not found")
        return

    text = target_file.read_text(encoding="utf-8")

    if "_ttsdb_resolve_cnhubert" in text:
        print(f"  cnhubert patch already applied in {target_file.name}")
        return

    # Add helper function
    helper = '''
import os as _ttsdb_os

def _ttsdb_resolve_cnhubert_path(default_path: str) -> str:
    """Resolve cnhubert model path, preferring TTSDB weights directory."""
    if _ttsdb_os.path.exists(default_path):
        return default_path
    weights_dir = _ttsdb_os.environ.get("TTSDB_WEIGHTS_DIR", "")
    if weights_dir:
        alt = _ttsdb_os.path.join(weights_dir, "shared", "chinese-hubert-base")
        if _ttsdb_os.path.exists(alt):
            return alt
    return default_path

'''

    # Insert after imports, before class/function definitions
    lines = text.split("\n")
    insert_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            insert_idx = i + 1
        elif stripped.startswith("class ") or stripped.startswith("def "):
            break

    lines.insert(insert_idx, helper)
    text = "\n".join(lines)

    target_file.write_text(text, encoding="utf-8")
    print(f"  Added cnhubert path resolver to {target_file.name}")


def patch_g2pw_onnx_bert_source(vendor_root: Path) -> None:
    """Patch G2PW ONNX converter to use local BERT tokenizer path.

    Upstream config may set model_source to a path like
    GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large, which is not
    a valid HuggingFace repo_id. Prefer TTSDB_WEIGHTS_DIR/shared/chinese-roberta-wwm-ext-large
    when available.
    """
    target_file = vendor_root / "GPT_SoVITS" / "text" / "g2pw" / "onnx_api.py"
    if not target_file.exists():
        print(f"  Skipping G2PW ONNX patch: {target_file} not found")
        return

    text = target_file.read_text(encoding="utf-8")

    if "TTSDB patch: prefer local packaged weights" in text:
        print(f"  G2PW ONNX bert source patch already applied in {target_file.name}")
        return

    # Insert TTSDB block after "self.enable_opencc = ..." and before
    # "self.tokenizer = AutoTokenizer.from_pretrained(self.model_source)"
    old_line = "self.tokenizer = AutoTokenizer.from_pretrained(self.model_source)"
    insert_block = """        # TTSDB patch: prefer local packaged weights if available.
        # Some upstream configs set model_source to a path like
        # `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large`, which is
        # not a valid HuggingFace repo_id and may not exist locally.
        weights_dir = os.environ.get("TTSDB_WEIGHTS_DIR", "")
        if weights_dir:
            local_bert = os.path.join(weights_dir, "shared", "chinese-roberta-wwm-ext-large")
            if os.path.exists(local_bert):
                self.model_source = local_bert

        """
    if old_line not in text:
        print(f"  Warning: G2PW ONNX patch target line not found in {target_file.name}")
        return

    text = text.replace(
        old_line,
        insert_block + old_line,
        1,
    )
    target_file.write_text(text, encoding="utf-8")
    print(f"  Patched G2PW ONNX bert source in {target_file.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch GPT-SoVITS vendored code")
    ap.add_argument("--vendor-root", required=True, type=Path)
    ap.add_argument("--config", required=False, type=Path)
    args = ap.parse_args()

    vendor_root: Path = args.vendor_root

    if not vendor_root.exists():
        raise FileNotFoundError(f"Vendor root not found: {vendor_root}")

    print(f"Patching GPT-SoVITS in {vendor_root}...")

    # Apply patches
    patch_wav16k_length_check(vendor_root)
    patch_g2pw_model_path(vendor_root)
    patch_hubert_path(vendor_root)
    patch_g2pw_onnx_bert_source(vendor_root)

    print("✓ Patches applied successfully")


if __name__ == "__main__":
    main()
