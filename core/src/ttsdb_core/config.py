"""Configuration loading utilities for TTSDB models."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Union

import yaml

if sys.version_info >= (3, 9):
    from importlib.resources import files, as_file
else:
    from importlib_resources import files, as_file


class ModelConfig(dict):
    """Model configuration loaded from config.yaml.
    
    A dict subclass with attribute access and helper methods.
    
    Example:
        >>> config = ModelConfig.from_package("ttsdb_maskgct")
        >>> config["metadata"]["name"]
        'MaskGCT'
        >>> config.metadata.name  # Also works
        'MaskGCT'
        >>> config.metadata.sample_rate
        24000
    """
    
    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
            if isinstance(value, dict):
                return ModelConfig(value)
            return value
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ModelConfig":
        """Load config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(data)
    
    @classmethod
    def from_package(cls, package_name: str, filename: str = "config.yaml") -> "ModelConfig":
        """Load config from a package's bundled config.yaml."""
        package_files = files(package_name)
        config_file = package_files.joinpath(filename)

        if not config_file.exists():
            # try looking for local config.yaml, (in case of editable install)
            config_file = package_files.parent.parent / filename
            if not config_file.exists():
                raise FileNotFoundError(f"Config file {config_file} not found in package {package_name}")
        
        with as_file(config_file) as config_path:
            return cls.from_yaml(config_path)
