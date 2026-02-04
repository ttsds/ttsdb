"""Runtime vendor path utilities for external research code."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .config import ModelConfig


def _get_code_root(config: ModelConfig) -> str | None:
    """Get code.root from config, handling nested access."""
    try:
        code = config.get("code")
        if code and isinstance(code, dict):
            return code.get("root")
    except (KeyError, TypeError):
        pass
    return None


def _get_extra_paths(config: ModelConfig) -> list[str]:
    """Get code.extra_paths from config, handling nested access.

    Extra paths are additional subdirectories within the vendor source
    that should be added to sys.path. This is useful when vendored code
    has internal imports that expect certain directories to be on the path.

    Example config.yaml:
        code:
          url: https://github.com/example/repo
          root: .
          extra_paths:
            - subdir  # Adds _vendor/source/subdir to sys.path
    """
    try:
        code = config.get("code")
        if code and isinstance(code, dict):
            extra = code.get("extra_paths")
            if extra and isinstance(extra, list):
                return [str(p) for p in extra]
    except (KeyError, TypeError):
        pass
    return []


def setup_vendor_path(
    package_name: str,
    vendor_dirname: str = "_vendor",
    source_dirname: str = "source",
) -> Path:
    """Add vendored source to sys.path for imports.

    Reads `code.root` from the package's config.yaml to determine the
    importable root within the vendored source. Also reads `code.extra_paths`
    for additional subdirectories to add to sys.path.

    The vendor directory is expected to be inside the package directory:
    <package>/_vendor/source/

    Call this at the top of your model's __init__.py before importing
    from the vendored code.

    Args:
        package_name: The package name (e.g., "ttsdb_maskgct").
        vendor_dirname: Name of the vendor directory (default: "_vendor").
        source_dirname: Name of the source subdirectory (default: "source").

    Returns:
        Path to the main vendor directory added to sys.path.

    Example:
        >>> # In models/bark/src/ttsdb_bark/__init__.py
        >>> from ttsdb_core import setup_vendor_path
        >>> setup_vendor_path("ttsdb_bark")
        >>>
        >>> # Now you can import from the vendored bark code
        >>> from bark.generation import generate_text_semantic

    Config example with extra_paths:
        code:
          url: https://github.com/example/repo
          root: .
          extra_paths:
            - subpackage  # Also adds _vendor/source/subpackage to sys.path
    """
    # Load config from package
    config = ModelConfig.from_package(package_name)
    code_root = _get_code_root(config)
    extra_paths = _get_extra_paths(config)

    # Find package location - vendor is inside the package directory
    import importlib

    package = importlib.import_module(package_name)
    package_dir = Path(package.__file__).parent

    # Base vendor path: <package_dir>/_vendor/source/
    base_vendor_path = package_dir / vendor_dirname / source_dirname

    # Main vendor path (with code.root applied)
    vendor_path = base_vendor_path
    if code_root and code_root != ".":
        vendor_path = vendor_path / code_root

    vendor_path_str = str(vendor_path)

    if vendor_path_str not in sys.path:
        sys.path.insert(0, vendor_path_str)

    # Add extra paths (relative to base vendor path, not code.root)
    for extra in extra_paths:
        extra_full = base_vendor_path / extra
        extra_str = str(extra_full)
        if extra_full.exists() and extra_str not in sys.path:
            sys.path.insert(0, extra_str)

    return vendor_path


def get_vendor_path(
    package_name: str,
    vendor_dirname: str = "_vendor",
    source_dirname: str = "source",
) -> Path:
    """Get the vendor path for a package without modifying sys.path.

    Args:
        package_name: The package name (e.g., "ttsdb_maskgct").
        vendor_dirname: Name of the vendor directory (default: "_vendor").
        source_dirname: Name of the source subdirectory (default: "source").

    Returns:
        Path to the vendor directory.
    """
    config = ModelConfig.from_package(package_name)
    code_root = _get_code_root(config)

    import importlib

    package = importlib.import_module(package_name)
    package_dir = Path(package.__file__).parent

    vendor_path = package_dir / vendor_dirname / source_dirname

    if code_root and code_root != ".":
        vendor_path = vendor_path / code_root

    return vendor_path


@contextmanager
def vendor_context(
    package_name: str,
    cwd: bool = False,
    env: dict[str, str] | None = None,
) -> Iterator[Path]:
    """Context manager for running code that requires specific environment setup.

    Some research code expects to be run from a specific working directory
    or requires certain environment variables. This context manager handles
    setting up and tearing down that environment.

    Args:
        package_name: The package name (e.g., "ttsdb_maskgct").
        cwd: If True, change working directory to vendor path during context.
        env: Dictionary of environment variables to set. Values can use
             "{vendor_path}" as a template variable.

    Yields:
        Path to the vendor directory.

    Example:
        >>> from ttsdb_core import vendor_context
        >>>
        >>> with vendor_context("ttsdb_maskgct", cwd=True, env={"WORK_DIR": "{vendor_path}"}):
        ...     # Working directory is now the vendor path
        ...     # WORK_DIR env var is set
        ...     from models.tts.maskgct.maskgct_utils import build_semantic_model
        ...     model = build_semantic_model(device)
        >>> # Original working directory and env vars are restored
    """
    vendor_path = get_vendor_path(package_name)
    vendor_path_str = str(vendor_path)

    # Save original state
    original_cwd = os.getcwd()
    original_env = {}

    try:
        # Set up environment variables
        if env:
            for key, value in env.items():
                original_env[key] = os.environ.get(key)
                # Replace template variables
                resolved_value = value.replace("{vendor_path}", vendor_path_str)
                os.environ[key] = resolved_value

        # Change working directory if requested
        if cwd:
            if not vendor_path.exists():
                raise FileNotFoundError(
                    f"Vendored research code not found at {vendor_path}\n"
                    f"- If you are developing locally: run `just fetch {package_name.replace('ttsdb_', '')}` "
                    f"or `just setup <model>` to vendor the upstream repo.\n"
                    f"- If you installed from PyPI/HuggingFace Space: the wheel must include "
                    f"`{package_name}/_vendor/...` (publish a new release after vendoring)."
                )
            os.chdir(vendor_path_str)

        yield vendor_path

    finally:
        # Restore working directory
        if cwd:
            os.chdir(original_cwd)

        # Restore environment variables
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
