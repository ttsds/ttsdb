"""Runtime weight loading utilities for TTSDB models."""

from __future__ import annotations

from pathlib import Path

from .config import ModelConfig


def resolve_weights_path(load_path: str) -> Path:
    """Resolve the weights path, downloading from HuggingFace if needed."""
    path = Path(load_path)
    if path.exists():
        return path

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=load_path))


def get_variant_checkpoint_dir(
    weights_base: Path,
    config: ModelConfig | None = None,
    variant: str | None = None,
) -> Path:
    """Get the checkpoint directory for a variant."""
    if variant is None and config is not None:
        variant = config.variant

    if variant:
        variant_dir = weights_base / variant
        if variant_dir.exists():
            return variant_dir

    return weights_base


def find_checkpoint(
    checkpoint_dir: Path,
    pattern: str = "model_*.pt",
    prefer_safetensors: bool = True,
) -> Path | None:
    """Find a checkpoint file in the directory."""
    if not checkpoint_dir.exists():
        return None

    pt_files = list(checkpoint_dir.glob(pattern))
    st_files = list(checkpoint_dir.glob(pattern.replace(".pt", ".safetensors")))

    if prefer_safetensors and st_files:
        candidates = st_files
    elif pt_files:
        candidates = pt_files
    elif st_files:
        candidates = st_files
    else:
        return None

    def get_step(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return 0

    candidates.sort(key=get_step, reverse=True)
    return candidates[0] if candidates else None
