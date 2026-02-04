"""Configuration loading utilities for TTSDB models."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import yaml

if sys.version_info >= (3, 9):
    from importlib.resources import as_file, files
else:
    from importlib_resources import as_file, files


class ModelConfig(dict):
    """Model configuration loaded from config.yaml.

    A dict subclass with attribute access and helper methods.
    Supports model variants through the `variants` section.

    Example:
        >>> config = ModelConfig.from_package("ttsdb_maskgct")
        >>> config["metadata"]["name"]
        'MaskGCT'
        >>> config.metadata.name  # Also works
        'MaskGCT'
        >>> config.metadata.sample_rate
        24000

    Variants example:
        >>> config = ModelConfig.from_package("ttsdb_f5_tts", variant="v1")
        >>> config.variant  # resolved variant name
        'v1'
        >>> config.metadata.name  # base metadata preserved
        'F5-TTS'
    """

    _variant: str | None = None

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
            if isinstance(value, dict):
                return ModelConfig(value)
            return value
        except KeyError as e:
            raise AttributeError(f"Config has no attribute '{name}'") from e

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    @property
    def variant(self) -> str | None:
        """Currently loaded variant name, or None if no variant."""
        return self._variant

    @property
    def available_variants(self) -> list[str]:
        """List of available variant names."""
        variants = self.get("variants", {})
        if not isinstance(variants, dict):
            return []
        return [k for k in variants.keys() if k != "default"]

    @property
    def default_variant(self) -> str | None:
        """Default variant name, or None if no variants defined."""
        variants = self.get("variants", {})
        if not isinstance(variants, dict):
            return None
        return variants.get("default")

    @classmethod
    def from_yaml(cls, path: str | Path, variant: str | None = None) -> ModelConfig:
        """Load config from a YAML file.

        Args:
            path: Path to YAML config file.
            variant: Optional variant name. If provided, merges variant-specific
                    config over the base config.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._apply_variant(data, variant)

    @classmethod
    def from_package(
        cls, package_name: str, filename: str = "config.yaml", variant: str | None = None
    ) -> ModelConfig:
        """Load config from a package's bundled config.yaml.

        Args:
            package_name: Name of the Python package.
            filename: Config filename (default: config.yaml).
            variant: Optional variant name. If provided, merges variant-specific
                    config over the base config.
        """
        package_files = files(package_name)
        config_file = package_files.joinpath(filename)

        if not config_file.exists():
            # try looking for local config.yaml, (in case of editable install)
            config_file = package_files.parent.parent / filename
            if not config_file.exists():
                raise FileNotFoundError(
                    f"Config file {config_file} not found in package {package_name}"
                )

        with as_file(config_file) as config_path:
            return cls.from_yaml(config_path, variant=variant)

    @classmethod
    def _apply_variant(cls, data: dict, variant: str | None) -> ModelConfig:
        """Apply variant-specific overrides to config data.

        Args:
            data: Raw config dict.
            variant: Variant name, or None for default.

        Returns:
            ModelConfig with variant overrides applied.
        """
        config = cls(copy.deepcopy(data))
        variants = data.get("variants", {})

        if not isinstance(variants, dict) or not variants:
            # No variants defined - return base config
            config._variant = None
            return config

        # Resolve variant name
        if variant is None:
            variant = variants.get("default")

        if variant is None:
            # No default variant specified, return base config
            config._variant = None
            return config

        variant_data = variants.get(variant)
        if variant_data is None:
            available = [k for k in variants.keys() if k != "default"]
            raise ValueError(f"Unknown variant '{variant}'. Available variants: {available}")

        # Deep merge variant overrides into config
        config = cls(_deep_merge(copy.deepcopy(data), variant_data))
        config._variant = variant
        return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override dict into base dict.

    Args:
        base: Base dictionary (modified in place).
        override: Override dictionary.

    Returns:
        Merged dictionary.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
