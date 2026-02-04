"""Prepare weights utilities for TTSDB models.

These helpers are only used by `scripts/prepare_weights.py` to download and
restructure upstream weights into the standardized TTSDB layout.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def extract_repo_id(url: str) -> str:
    """Extract HuggingFace repo ID from URL."""
    if not url or "huggingface.co/" not in url:
        raise ValueError(f"Not a valid HuggingFace URL: {url}")
    return url.rstrip("/").split("huggingface.co/")[-1].split("/tree/")[0].strip("/")


def get_weights_config(config: dict) -> tuple[str, str | None]:
    """Extract repo_id and commit from config."""
    weights_cfg = config.get("weights", {}) or {}
    url = weights_cfg.get("url", "")
    commit = weights_cfg.get("commit")

    repo_id = extract_repo_id(url)
    return repo_id, commit


def download_model_weights(
    repo_id: str,
    weights_dir: Path,
    *,
    commit: str | None = None,
    variant: str | None = None,
    variant_subdirs: dict[str, str] | None = None,
    shared_patterns: list[str] | None = None,
) -> Path:
    """Download model weights from HuggingFace with variant support."""
    from huggingface_hub import snapshot_download

    weights_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = _build_allow_patterns(variant, variant_subdirs, shared_patterns)

    print(f"Downloading from {repo_id}...")
    if allow_patterns:
        print(f"  allow_patterns: {allow_patterns}")

    snapshot_download(
        repo_id=repo_id,
        revision=commit,
        local_dir=weights_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )

    if variant and variant_subdirs:
        source_subdir = variant_subdirs.get(variant)
        if source_subdir and source_subdir != variant:
            _rename_subdir(weights_dir, source_subdir, variant)

    return weights_dir


def download_all_variants(
    repo_id: str,
    weights_dir: Path,
    variant_subdirs: dict[str, str],
    *,
    commit: str | None = None,
    shared_patterns: list[str] | None = None,
) -> Path:
    """Download all variants from a HuggingFace repo."""
    for variant in variant_subdirs:
        download_model_weights(
            repo_id=repo_id,
            weights_dir=weights_dir,
            commit=commit,
            variant=variant,
            variant_subdirs=variant_subdirs,
            shared_patterns=shared_patterns,
        )
    return weights_dir


def download_dependency(
    repo_id: str,
    local_dir: Path,
    *,
    allow_patterns: list[str] | None = None,
    commit: str | None = None,
    force: bool = False,
) -> Path:
    """Download an external dependency from HuggingFace."""
    from huggingface_hub import snapshot_download

    if not force and local_dir.exists() and any(local_dir.iterdir()):
        print(f"Dependency already present: {local_dir}")
        return local_dir

    print(f"Downloading dependency from {repo_id}...")

    snapshot_download(
        repo_id=repo_id,
        revision=commit,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )

    return local_dir


def download_dependencies_from_config(config: dict, weights_dir: Path) -> None:
    """Download all dependencies listed in config into shared/."""
    weights_cfg = config.get("weights", {}) or {}
    dependencies = weights_cfg.get("dependencies", []) or []

    shared_dir = weights_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    for dep in dependencies:
        repo_id = dep.get("repo_id")
        local_dir = dep.get("local_dir")
        dep_patterns = dep.get("allow_patterns")
        dep_commit = dep.get("commit")

        if not repo_id or not local_dir:
            print(f"Skipping invalid dependency: {dep}")
            continue

        download_dependency(
            repo_id=repo_id,
            local_dir=shared_dir / local_dir,
            allow_patterns=dep_patterns,
            commit=dep_commit,
        )
        print(f"✓ {local_dir} ready")


def write_model_config(config_data: dict, path: Path) -> None:
    """Write model architecture config as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"Wrote {path.name} -> {path}")


def _build_allow_patterns(
    variant: str | None,
    variant_subdirs: dict[str, str] | None,
    shared_patterns: list[str] | None,
) -> list[str] | None:
    if variant is None:
        return None

    patterns = ["shared/*"]

    if variant_subdirs and variant in variant_subdirs:
        source_subdir = variant_subdirs[variant]
        patterns.append(f"{source_subdir}/*")
    else:
        patterns.append(f"{variant}/*")

    if shared_patterns:
        patterns.extend(shared_patterns)

    return patterns


def _rename_subdir(weights_dir: Path, source: str, dest: str) -> None:
    src_path = weights_dir / source
    dst_path = weights_dir / dest

    if not src_path.exists():
        return

    if dst_path.exists():
        print(f"  Removing existing {dest}/ before rename")
        shutil.rmtree(dst_path)

    print(f"  Renaming {source}/ -> {dest}/")
    shutil.move(str(src_path), str(dst_path))
