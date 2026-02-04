"""Tests for HierSpeech.

Generated at 2026-02-02T10:50:14Z from templates/init/test_model.py.j2.
"""

import pytest
from ttsdb_core.testing import BaseModelConfigTests, BaseModelImportTests, BaseModelIntegrationTests


class TestConfig(BaseModelConfigTests):
    PACKAGE_NAME = "ttsdb_hierspeech"
    EXPECTED_MODEL_NAME = "HierSpeech"
    EXPECTED_SAMPLE_RATE = 16000


class TestModelImport(BaseModelImportTests):
    PACKAGE_NAME = "ttsdb_hierspeech"
    MODEL_CLASS_PATH = "ttsdb_hierspeech.HierSpeech"
    EXPECTED_SAMPLE_RATE_CONST = 16000


# Integration tests - require model weights to be downloaded
@pytest.mark.integration
class TestModelIntegration(BaseModelIntegrationTests):
    PACKAGE_NAME = "ttsdb_hierspeech"
    MODEL_CLASS_PATH = "ttsdb_hierspeech.HierSpeech"
    WEIGHTS_PREPARE_HINT = "hierspeech"
    EXPECTED_SAMPLE_RATE = 16000
