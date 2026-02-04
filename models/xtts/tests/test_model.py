"""Tests for XTTS.

Generated at 2026-02-02T10:50:14Z from templates/init/test_model.py.j2.
"""

import pytest
from ttsdb_core.testing import BaseModelConfigTests, BaseModelImportTests, BaseModelIntegrationTests


class TestConfig(BaseModelConfigTests):
    PACKAGE_NAME = "ttsdb_xtts"
    EXPECTED_MODEL_NAME = "XTTS"
    EXPECTED_SAMPLE_RATE = 24000


class TestModelImport(BaseModelImportTests):
    PACKAGE_NAME = "ttsdb_xtts"
    MODEL_CLASS_PATH = "ttsdb_xtts.XTTS"
    EXPECTED_SAMPLE_RATE_CONST = 24000


# Integration tests - require model weights to be downloaded
@pytest.mark.integration
class TestModelIntegration(BaseModelIntegrationTests):
    PACKAGE_NAME = "ttsdb_xtts"
    MODEL_CLASS_PATH = "ttsdb_xtts.XTTS"
    WEIGHTS_PREPARE_HINT = "xtts"
    EXPECTED_SAMPLE_RATE = 24000
