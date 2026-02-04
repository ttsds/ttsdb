"""Tests for OpenVoice.

Generated at 2026-02-03T00:00:00Z from templates/init/test_model.py.j2.
"""

import pytest
from ttsdb_core.testing import BaseModelConfigTests, BaseModelImportTests, BaseModelIntegrationTests


class TestConfig(BaseModelConfigTests):
    PACKAGE_NAME = "ttsdb_openvoice"
    EXPECTED_MODEL_NAME = "OpenVoice"


class TestModelImport(BaseModelImportTests):
    PACKAGE_NAME = "ttsdb_openvoice"
    MODEL_CLASS_PATH = "ttsdb_openvoice.OpenVoice"


# Integration tests - require model weights to be downloaded
@pytest.mark.integration
class TestModelIntegration(BaseModelIntegrationTests):
    PACKAGE_NAME = "ttsdb_openvoice"
    MODEL_CLASS_PATH = "ttsdb_openvoice.OpenVoice"
    WEIGHTS_PREPARE_HINT = "openvoice"
