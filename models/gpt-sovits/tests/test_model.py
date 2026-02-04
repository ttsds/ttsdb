"""Tests for GPT-SoVITS.

Generated at 2026-02-01T19:12:21Z from templates/init/test_model.py.j2.
"""

import pytest
from ttsdb_core.testing import BaseModelConfigTests, BaseModelImportTests, BaseModelIntegrationTests


class TestConfig(BaseModelConfigTests):
    PACKAGE_NAME = "ttsdb_gpt_sovits"
    EXPECTED_MODEL_NAME = "GPT-SoVITS"
    EXPECTED_SAMPLE_RATE = 48000  # default v4; v1/v2=32000, v3=24000


class TestModelImport(BaseModelImportTests):
    PACKAGE_NAME = "ttsdb_gpt_sovits"
    MODEL_CLASS_PATH = "ttsdb_gpt_sovits.GPTSoVITS"
    EXPECTED_SAMPLE_RATE_CONST = None  # GPT-SoVITS doesn't expose a SAMPLE_RATE constant


# Integration tests - require model weights to be downloaded
@pytest.mark.integration
class TestModelIntegration(BaseModelIntegrationTests):
    PACKAGE_NAME = "ttsdb_gpt_sovits"
    MODEL_CLASS_PATH = "ttsdb_gpt_sovits.GPTSoVITS"
    WEIGHTS_PREPARE_HINT = "gpt-sovits"
    # Per-variant sample rates
    EXPECTED_SAMPLE_RATE = {
        "v1": 32000,
        "v2": 32000,
        "v3": 24000,
        "v4": 48000,
    }
