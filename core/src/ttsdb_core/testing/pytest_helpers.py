"""Pytest hook helpers shared across model packages."""

from __future__ import annotations


def configure_integration_marker(config, *, extra: str = "") -> None:
    """Register the shared `integration` marker."""

    suffix = f" {extra}".rstrip()
    config.addinivalue_line(
        "markers",
        f"integration: mark test as integration test (requires model weights){suffix}",
    )


def skip_integration_by_default(config, items) -> None:
    """Skip `@pytest.mark.integration` unless `-m integration` is used."""

    # Avoid importing pytest at module import time (keeps this "test-only" helper
    # lightweight and side-effect free outside test runs).
    import pytest

    marker_expr = config.getoption("-m")
    if marker_expr != "integration":
        skip_integration = pytest.mark.skip(reason="integration tests require -m integration")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

