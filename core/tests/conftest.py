"""Pytest configuration for ttsdb_core tests."""

import sys
from pathlib import Path

# Add src to path for editable install compatibility
CORE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CORE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
