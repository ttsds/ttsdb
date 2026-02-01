"""Naming helpers for model/package generation."""

from __future__ import annotations

import re


def normalize_name(name: str) -> dict[str, str]:
    """Generate common name variants from a model name.

    Args:
        name: The full model name (e.g., "MaskGCT", "XTTS_v2", "F5-TTS")

    Returns:
        Dictionary with name variants:
        - model_name: Original name for display (e.g., "MaskGCT", "F5-TTS")
        - folder_name: Lowercase with hyphens (e.g., "maskgct", "f5-tts")
        - package_name: Same as folder_name (pip/uv style)
        - import_name: Lowercase with underscores, prefixed (e.g., "ttsdb_maskgct")
        - class_name: Original with hyphens/spaces removed (e.g., "MaskGCT", "F5TTS")
    """
    model_name = name

    # Folder name: lowercase, underscores/spaces to hyphens
    folder_name = name.lower().replace("_", "-").replace(" ", "-")
    folder_name = re.sub(r"-+", "-", folder_name)

    package_name = folder_name
    import_name = "ttsdb_" + folder_name.replace("-", "_")

    # Class name: preserve original casing, remove invalid identifier chars
    class_name = re.sub(r"[-\s]+", "", name)

    return {
        "model_name": model_name,
        "folder_name": folder_name,
        "package_name": package_name,
        "import_name": import_name,
        "class_name": class_name,
    }
