"""Tests for E2 TTS."""

import pytest
from ttsdb_core.testing import BaseModelConfigTests, BaseModelImportTests, BaseModelIntegrationTests


class TestConfig(BaseModelConfigTests):
    PACKAGE_NAME = "ttsdb_e2_tts"
    EXPECTED_MODEL_NAME = "E2 TTS"
    EXPECTED_SAMPLE_RATE = 24000
    EXPECTED_CODE_ROOT = "src"


class TestModelImport(BaseModelImportTests):
    PACKAGE_NAME = "ttsdb_e2_tts"
    MODEL_CLASS_PATH = "ttsdb_e2_tts.E2TTS"
    EXPECTED_SAMPLE_RATE_CONST = 24000


# Integration tests - require model weights to be downloaded
@pytest.mark.integration
class TestModelIntegration(BaseModelIntegrationTests):
    PACKAGE_NAME = "ttsdb_e2_tts"
    MODEL_CLASS_PATH = "ttsdb_e2_tts.E2TTS"
    WEIGHTS_PREPARE_HINT = "e2-tts"
    EXPECTED_SAMPLE_RATE = 24000
