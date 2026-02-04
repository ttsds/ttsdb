"""Tests for MaskGCT."""

import pytest
from ttsdb_core.testing import BaseModelConfigTests, BaseModelImportTests, BaseModelIntegrationTests


class TestConfig(BaseModelConfigTests):
    PACKAGE_NAME = "ttsdb_maskgct"
    EXPECTED_MODEL_NAME = "MaskGCT"
    EXPECTED_SAMPLE_RATE = 24000
    EXPECTED_CODE_ROOT = "."


class TestModelImport(BaseModelImportTests):
    PACKAGE_NAME = "ttsdb_maskgct"
    MODEL_CLASS_PATH = "ttsdb_maskgct.MaskGCT"
    EXPECTED_SAMPLE_RATE_CONST = 24000


# Integration tests - require model weights to be downloaded
@pytest.mark.integration
class TestModelIntegration(BaseModelIntegrationTests):
    PACKAGE_NAME = "ttsdb_maskgct"
    MODEL_CLASS_PATH = "ttsdb_maskgct.MaskGCT"
    WEIGHTS_PREPARE_HINT = "maskgct"
    EXPECTED_SAMPLE_RATE = 24000
