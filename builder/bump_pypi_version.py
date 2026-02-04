#!/usr/bin/env python3
"""Bump local pyproject version if it matches PyPI."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def _read_project_name_version(pyproject_path: Path) -> tuple[str, str]:
    data = pyproject_path.read_text()
    try:
        import tomllib  # py3.11+
    except Exception:  # pragma: no cover
        import tomli as tomllib  # type: ignore

    parsed = tomllib.loads(data)
    project = parsed.get("project", {})
    name = project.get("name")
    version = project.get("version")
    if not name or not version:
        raise ValueError(f"Missing project.name or project.version in {pyproject_path}")
    return name, version


def _fetch_pypi_version(package_name: str) -> str | None:
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        with urlopen(url, timeout=10) as resp:
            payload = json.load(resp)
            return payload.get("info", {}).get("version")
    except HTTPError as e:
        if e.code == 404:
            return None
        raise
    except URLError:
        raise


def _bump_patch(version: str) -> str:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
    if not match:
        raise ValueError(f"Unsupported version format: {version}")
    major, minor, patch = match.groups()
    return f"{major}.{minor}.{int(patch) + 1}"


def _replace_version(pyproject_path: Path, new_version: str) -> None:
    text = pyproject_path.read_text()
    updated, count = re.subn(
        r'^(version\s*=\s*")([^"]+)(")\s*$',
        lambda m: f"{m.group(1)}{new_version}{m.group(3)}",
        text,
        flags=re.MULTILINE,
    )
    if count == 0:
        raise ValueError(f"Could not update version in {pyproject_path}")
    pyproject_path.write_text(updated)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: bump_pypi_version.py <package_dir>", file=sys.stderr)
        raise SystemExit(2)

    package_dir = Path(sys.argv[1]).resolve()
    pyproject_path = package_dir / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError(f"Missing pyproject.toml at {pyproject_path}")

    name, local_version = _read_project_name_version(pyproject_path)
    remote_version = _fetch_pypi_version(name)

    if remote_version is None:
        print(f"{name}: not on PyPI yet; keeping version {local_version}")
        return

    if remote_version == local_version:
        new_version = _bump_patch(local_version)
        _replace_version(pyproject_path, new_version)
        print(f"{name}: bumped {local_version} -> {new_version} to avoid PyPI clash")
        return

    print(f"{name}: local {local_version}, PyPI {remote_version} (no bump)")


if __name__ == "__main__":
    main()
