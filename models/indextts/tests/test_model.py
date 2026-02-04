"""Tests for IndexTTS.

Generated at 2026-02-03T00:00:00Z from templates/init/test_model.py.j2.
"""

import pytest
from ttsdb_core.testing import BaseModelConfigTests, BaseModelImportTests, BaseModelIntegrationTests


class TestConfig(BaseModelConfigTests):
    PACKAGE_NAME = "ttsdb_indextts"
    EXPECTED_MODEL_NAME = "IndexTTS"
    EXPECTED_SAMPLE_RATE = 22050


class TestModelImport(BaseModelImportTests):
    PACKAGE_NAME = "ttsdb_indextts"
    MODEL_CLASS_PATH = "ttsdb_indextts.IndexTTS"
    EXPECTED_SAMPLE_RATE_CONST = 22050


# Integration tests - require model weights to be downloaded
@pytest.mark.integration
class TestModelIntegration(BaseModelIntegrationTests):
    PACKAGE_NAME = "ttsdb_indextts"
    MODEL_CLASS_PATH = "ttsdb_indextts.IndexTTS"
    WEIGHTS_PREPARE_HINT = "indextts"
    EXPECTED_SAMPLE_RATE = 22050
