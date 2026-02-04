"""Tests for Fish Speech.

Generated at 2026-02-02T09:52:12Z from templates/init/test_model.py.j2.
"""

import pytest
from ttsdb_core.testing import BaseModelConfigTests, BaseModelImportTests, BaseModelIntegrationTests


class TestConfig(BaseModelConfigTests):
    PACKAGE_NAME = "ttsdb_fish_speech"
    EXPECTED_MODEL_NAME = "Fish Speech"
    EXPECTED_SAMPLE_RATE = 44100


class TestModelImport(BaseModelImportTests):
    PACKAGE_NAME = "ttsdb_fish_speech"
    MODEL_CLASS_PATH = "ttsdb_fish_speech.FishSpeech"
    EXPECTED_SAMPLE_RATE_CONST = 44100


# Integration tests - require model weights to be downloaded
@pytest.mark.integration
class TestModelIntegration(BaseModelIntegrationTests):
    PACKAGE_NAME = "ttsdb_fish_speech"
    MODEL_CLASS_PATH = "ttsdb_fish_speech.FishSpeech"
    WEIGHTS_PREPARE_HINT = "fish-speech"
