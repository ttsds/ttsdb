"""Shared testing helpers for TTSDB model packages.

This subpackage is intended to be imported by *model* test suites, so they can
share common pytest fixtures and generic test classes.
"""

from .fixtures import (
    make_audio_examples_dir_fixture,
    make_reference_audio_fixture,
    make_test_data_fixture,
    make_weights_path_fixture,
)
from .pytest_helpers import configure_integration_marker, skip_integration_by_default
from .testkit import BaseModelConfigTests, BaseModelImportTests, BaseModelIntegrationTests

__all__ = [
    "configure_integration_marker",
    "skip_integration_by_default",
    "make_weights_path_fixture",
    "make_test_data_fixture",
    "make_reference_audio_fixture",
    "make_audio_examples_dir_fixture",
    "BaseModelConfigTests",
    "BaseModelImportTests",
    "BaseModelIntegrationTests",
]
