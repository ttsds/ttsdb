"""Tests for Pheme.

Generated at 2026-02-02T09:52:12Z from templates/init/test_model.py.j2.
"""

import pytest
from ttsdb_core.testing import BaseModelConfigTests, BaseModelImportTests, BaseModelIntegrationTests


class TestConfig(BaseModelConfigTests):
    PACKAGE_NAME = "ttsdb_pheme"
    EXPECTED_MODEL_NAME = "Pheme"
    EXPECTED_SAMPLE_RATE = 16000


class TestModelImport(BaseModelImportTests):
    PACKAGE_NAME = "ttsdb_pheme"
    MODEL_CLASS_PATH = "ttsdb_pheme.Pheme"
    EXPECTED_SAMPLE_RATE_CONST = 16000


# Integration tests - require model weights to be downloaded
@pytest.mark.integration
class TestModelIntegration(BaseModelIntegrationTests):
    PACKAGE_NAME = "ttsdb_pheme"
    MODEL_CLASS_PATH = "ttsdb_pheme.Pheme"
    WEIGHTS_PREPARE_HINT = "pheme"
    EXPECTED_SAMPLE_RATE = 16000
