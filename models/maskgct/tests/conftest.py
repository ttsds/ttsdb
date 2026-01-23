"""Pytest configuration for MaskGCT tests."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires model weights and Amphion)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests by default unless -m integration is specified."""
    if config.getoption("-m") != "integration":
        skip_integration = pytest.mark.skip(reason="integration tests require -m integration")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
