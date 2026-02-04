"""Pytest configuration for model tests (shared via ttsdb_core).

Generated at 2026-02-02T09:52:12Z from templates/init/conftest.py.j2.
"""

import sys
from pathlib import Path

from ttsdb_core.testing import (
    add_variant_option,
    configure_integration_marker,
    make_audio_examples_dir_fixture,
    make_reference_audio_fixture,
    make_test_data_fixture,
    make_weights_path_fixture,
    skip_integration_by_default,
    write_integration_result,
)

MODEL_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = MODEL_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def pytest_addoption(parser):
    add_variant_option(parser)


def pytest_configure(config):
    configure_integration_marker(config)


def pytest_collection_modifyitems(config, items):
    skip_integration_by_default(config, items)


def pytest_sessionfinish(session, exitstatus):
    write_integration_result(MODEL_ROOT, session, exitstatus)


# Fixtures (bound to this model's root directory)
weights_path = make_weights_path_fixture(MODEL_ROOT, prefer_weights_dir=True)
test_data = make_test_data_fixture(MODEL_ROOT)
reference_audio = make_reference_audio_fixture()
audio_examples_dir = make_audio_examples_dir_fixture(MODEL_ROOT)
