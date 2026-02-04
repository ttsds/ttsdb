#!/usr/bin/env python3
"""Patch vendored tortoise code to support offline checkpoints.

This mirrors the intent of `cog_tts/models/tortoise/cog.yaml`, but uses env vars
so the wheel can work both online (default HF ids) and offline (local paths).

Called automatically by `python builder/vendor.py models/tortoise` if present.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _patch_file(path: Path) -> None:
    s = path.read_text()

    # Ensure `import os` exists for env var usage
    if "import os" not in s:
        s = "import os\n" + s

    repl = {
        '"jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli"': 'os.environ.get("TTSDB_TORTOISE_W2V_ROBUST", "jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli")',
        'f"facebook/wav2vec2-large-960h"': 'os.environ.get("TTSDB_TORTOISE_W2V_960H", "facebook/wav2vec2-large-960h")',
        "'jbetker/tacotron-symbols'": 'os.environ.get("TTSDB_TORTOISE_TACOTRON_SYMBOLS", "jbetker/tacotron-symbols")',
    }

    changed = False
    for needle, replacement in repl.items():
        if needle in s:
            s = s.replace(needle, replacement)
            changed = True

    if changed:
        path.write_text(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor-root", required=True, type=Path)
    ap.add_argument("--config", required=False, type=Path)
    args = ap.parse_args()

    vendor_root: Path = args.vendor_root

    # Vendored repo root should contain the `tortoise/` package.
    target = vendor_root / "tortoise" / "utils" / "wav2vec_alignment.py"
    if not target.exists():
        # Some forks use `tortoise_tts/` or different layouts; fall back to a search.
        matches = list(vendor_root.rglob("wav2vec_alignment.py"))
        if matches:
            target = matches[0]
        else:
            raise FileNotFoundError("Could not find wav2vec_alignment.py in vendored tortoise repo")

    _patch_file(target)
    print(f"Patched {target}")


if __name__ == "__main__":
    main()
