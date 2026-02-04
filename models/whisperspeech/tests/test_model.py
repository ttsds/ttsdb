"""Tests for WhisperSpeech.

Generated at 2026-02-02T10:50:14Z from templates/init/test_model.py.j2.
"""

import pytest
from ttsdb_core.testing import BaseModelConfigTests, BaseModelImportTests, BaseModelIntegrationTests


class TestConfig(BaseModelConfigTests):
    PACKAGE_NAME = "ttsdb_whisperspeech"
    EXPECTED_MODEL_NAME = "WhisperSpeech"
    EXPECTED_SAMPLE_RATE = 24000


class TestModelImport(BaseModelImportTests):
    PACKAGE_NAME = "ttsdb_whisperspeech"
    MODEL_CLASS_PATH = "ttsdb_whisperspeech.WhisperSpeech"
    EXPECTED_SAMPLE_RATE_CONST = 24000


# Integration tests - require model weights to be downloaded
@pytest.mark.integration
class TestModelIntegration(BaseModelIntegrationTests):
    PACKAGE_NAME = "ttsdb_whisperspeech"
    MODEL_CLASS_PATH = "ttsdb_whisperspeech.WhisperSpeech"
    WEIGHTS_PREPARE_HINT = "whisperspeech"
    EXPECTED_SAMPLE_RATE = 24000
