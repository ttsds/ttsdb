"""Tests for TorToise."""

import pytest
from ttsdb_core.testing import BaseModelConfigTests, BaseModelImportTests, BaseModelIntegrationTests


class TestConfig(BaseModelConfigTests):
    PACKAGE_NAME = "ttsdb_tortoise"
    EXPECTED_MODEL_NAME = "TorToise"
    EXPECTED_SAMPLE_RATE = 24000
    EXPECTED_CODE_ROOT = "."


class TestModelImport(BaseModelImportTests):
    PACKAGE_NAME = "ttsdb_tortoise"
    MODEL_CLASS_PATH = "ttsdb_tortoise.TorToise"
    EXPECTED_SAMPLE_RATE_CONST = 24000


# Integration tests - require model weights to be downloaded
@pytest.mark.integration
class TestModelIntegration(BaseModelIntegrationTests):
    PACKAGE_NAME = "ttsdb_tortoise"
    MODEL_CLASS_PATH = "ttsdb_tortoise.TorToise"
    WEIGHTS_PREPARE_HINT = "tortoise"
    EXPECTED_SAMPLE_RATE = 24000
