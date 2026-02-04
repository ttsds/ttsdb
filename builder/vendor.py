#!/usr/bin/env python3
"""Vendoring utilities for fetching external research code."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def get_import_name(model_dir: Path) -> str:
    """Get the import name from pyproject.toml."""
    pyproject_path = model_dir / "pyproject.toml"
    if pyproject_path.exists():
        import tomllib

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        return data.get("project", {}).get("name", "").replace("-", "_")

    # Fallback: derive from directory name
    return "ttsdb_" + model_dir.name.replace("-", "_")


def _run_model_vendor_patch(model_dir: Path, vendor_root: Path) -> None:
    """Run an optional model-specific vendoring patch script.

    We don't commit vendored code; any patching/asset stripping must therefore happen
    at vendoring time. Models can provide a script at:
      scripts/patch_vendor.py
    """
    script = model_dir / "scripts" / "patch_vendor.py"
    if not script.exists():
        return

    config_path = model_dir / "config.yaml"
    print(f"Running vendor patch script: {script}")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--vendor-root",
            str(vendor_root),
            "--config",
            str(config_path),
        ],
        cwd=model_dir,
        check=True,
    )


def fetch_source(
    model_dir: str | Path,
    vendor_dirname: str = "_vendor",
    source_dirname: str = "source",
    clean: bool = True,
) -> Path | None:
    """Fetch external source code defined in config.yaml.

    Clones the repository at the specified commit, removes .git metadata,
    and places it in src/<package>/_vendor/ so it's included in the wheel.

    Args:
        model_dir: Path to the model directory containing config.yaml.
        vendor_dirname: Name of the vendor directory (default: "_vendor").
        source_dirname: Name of the source subdirectory (default: "source").
        clean: If True, removes existing vendor directory before cloning.

    Returns:
        Path to the vendored source directory, or None if no external code defined.
    """
    model_dir = Path(model_dir).resolve()
    config_path = model_dir / "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # PyPI-only packages do not vendor upstream code
    pypi_config = config.get("package", {}).get("pypi")
    if pypi_config:
        pypi_name = pypi_config if isinstance(pypi_config, str) else pypi_config.get("name", "")
        print(f"Package is PyPI-only (package.pypi.name={pypi_name}); skipping vendor.")
        return None

    # Check for code section with URL
    code = config.get("code", {})
    url = code.get("url")

    if not url:
        print(f"No code.url defined in {config_path}")
        return None

    commit = code.get("commit")

    if not commit:
        print(f"Warning: No commit pinned for {url}, using HEAD")

    # Get import name to find package directory
    import_name = get_import_name(model_dir)

    # Target: src/<import_name>/_vendor/source/
    package_dir = model_dir / "src" / import_name
    vendor_dir = package_dir / vendor_dirname
    target_dir = vendor_dir / source_dirname

    # Clean slate if requested
    if clean and vendor_dir.exists():
        shutil.rmtree(vendor_dir)

    vendor_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cloning {url}" + (f" @ {commit}" if commit else "") + "...")

    env = dict(**os.environ)
    # Skip Git LFS smudge to avoid failing on large binary assets during vendoring.
    env.setdefault("GIT_LFS_SKIP_SMUDGE", "1")

    # Clone the repository
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(target_dir)],
        check=True,
        env=env,
    )

    # Checkout specific commit if provided
    if commit:
        # Need to fetch the specific commit first (shallow clone doesn't have it)
        subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", commit],
            cwd=target_dir,
            check=True,
            env=env,
        )
        subprocess.run(
            ["git", "checkout", commit],
            cwd=target_dir,
            check=True,
            env=env,
        )

    # Remove .git folder to save space and prevent confusion
    git_dir = target_dir / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)

    # Model-specific patching/asset stripping (optional)
    _run_model_vendor_patch(model_dir, target_dir)

    # Create __init__.py so Python treats _vendor as a package
    (vendor_dir / "__init__.py").touch()

    print(f"✓ Vendored to {target_dir}")
    return target_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch external source code for a model.")
    parser.add_argument("model_dir", help="Path to model directory")
    parser.add_argument("--no-clean", action="store_true", help="Don't remove existing vendor dir")

    args = parser.parse_args()
    fetch_source(args.model_dir, clean=not args.no_clean)


if __name__ == "__main__":
    main()
