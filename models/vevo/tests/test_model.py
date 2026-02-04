"""Tests for Vevo.

Generated at 2026-02-02T09:52:12Z from templates/init/test_model.py.j2.
"""

import pytest
from ttsdb_core.testing import BaseModelConfigTests, BaseModelImportTests, BaseModelIntegrationTests


class TestConfig(BaseModelConfigTests):
    PACKAGE_NAME = "ttsdb_vevo"
    EXPECTED_MODEL_NAME = "Vevo"
    EXPECTED_SAMPLE_RATE = 24000


class TestModelImport(BaseModelImportTests):
    PACKAGE_NAME = "ttsdb_vevo"
    MODEL_CLASS_PATH = "ttsdb_vevo.Vevo"
    EXPECTED_SAMPLE_RATE_CONST = 24000


# Integration tests - require model weights to be downloaded
@pytest.mark.integration
class TestModelIntegration(BaseModelIntegrationTests):
    PACKAGE_NAME = "ttsdb_vevo"
    MODEL_CLASS_PATH = "ttsdb_vevo.Vevo"
    WEIGHTS_PREPARE_HINT = "vevo"
    EXPECTED_SAMPLE_RATE = 24000
