"""Tests for F5-TTS."""

import pytest
from ttsdb_core.testing import BaseModelConfigTests, BaseModelImportTests, BaseModelIntegrationTests


class TestConfig(BaseModelConfigTests):
    PACKAGE_NAME = "ttsdb_f5_tts"
    EXPECTED_MODEL_NAME = "F5-TTS"
    EXPECTED_SAMPLE_RATE = 24000
    EXPECTED_CODE_ROOT = "src"
    EXPECTED_VARIANTS = ["base", "v1"]
    EXPECTED_DEFAULT_VARIANT = "base"


class TestModelImport(BaseModelImportTests):
    PACKAGE_NAME = "ttsdb_f5_tts"
    MODEL_CLASS_PATH = "ttsdb_f5_tts.F5TTS"
    EXPECTED_SAMPLE_RATE_CONST = 24000


# Integration tests - require model weights to be downloaded
@pytest.mark.integration
class TestModelIntegration(BaseModelIntegrationTests):
    PACKAGE_NAME = "ttsdb_f5_tts"
    MODEL_CLASS_PATH = "ttsdb_f5_tts.F5TTS"
    WEIGHTS_PREPARE_HINT = "f5-tts"
    EXPECTED_SAMPLE_RATE = 24000
