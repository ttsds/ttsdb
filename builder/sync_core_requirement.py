#!/usr/bin/env python3
"""Sync model pyproject dependency with local ttsdb-core version."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def _read_core_version(core_pyproject: Path) -> str:
    text = core_pyproject.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', text, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find core version in {core_pyproject}")
    return match.group(1)


def _sync_model_pyproject(model_pyproject: Path, core_version: str) -> bool:
    text = model_pyproject.read_text()
    pattern = r'"ttsdb-core>=[^"]+"'
    replacement = f'"ttsdb-core>={core_version}"'
    updated, count = re.subn(pattern, replacement, text)
    if count == 0:
        raise ValueError(f"Could not find ttsdb-core dependency in {model_pyproject}")
    if updated != text:
        model_pyproject.write_text(updated)
        return True
    return False


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: sync_core_requirement.py <model_dir>", file=sys.stderr)
        raise SystemExit(2)

    model_dir = Path(sys.argv[1]).resolve()
    repo_root = Path(__file__).resolve().parent.parent
    core_pyproject = repo_root / "core" / "pyproject.toml"
    model_pyproject = model_dir / "pyproject.toml"

    if not model_pyproject.exists():
        raise FileNotFoundError(f"Missing pyproject.toml at {model_pyproject}")

    core_version = _read_core_version(core_pyproject)
    changed = _sync_model_pyproject(model_pyproject, core_version)
    status = "updated" if changed else "already up-to-date"
    print(f"{model_pyproject}: {status} (ttsdb-core>={core_version})")


if __name__ == "__main__":
    main()
