"""Tests for VibeVoice.

Generated at 2026-02-03T00:00:00Z from templates/init/test_model.py.j2.
"""

import pytest
from ttsdb_core.testing import BaseModelConfigTests, BaseModelImportTests, BaseModelIntegrationTests


class TestConfig(BaseModelConfigTests):
    PACKAGE_NAME = "ttsdb_vibevoice"
    EXPECTED_MODEL_NAME = "VibeVoice"
    EXPECTED_SAMPLE_RATE = 24000


class TestModelImport(BaseModelImportTests):
    PACKAGE_NAME = "ttsdb_vibevoice"
    MODEL_CLASS_PATH = "ttsdb_vibevoice.VibeVoice"
    EXPECTED_SAMPLE_RATE_CONST = 24000


# Integration tests - require model weights to be downloaded
@pytest.mark.integration
class TestModelIntegration(BaseModelIntegrationTests):
    PACKAGE_NAME = "ttsdb_vibevoice"
    MODEL_CLASS_PATH = "ttsdb_vibevoice.VibeVoice"
    WEIGHTS_PREPARE_HINT = "vibevoice"
    EXPECTED_SAMPLE_RATE = 24000
