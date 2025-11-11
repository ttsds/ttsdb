import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

TTSDB_CACHE = os.getenv("TTSDB_CACHE", "~/.cache/ttsdb")

VENV_PATH = Path(TTSDB_CACHE) / "venv"
VENV_PATH.mkdir(parents=True, exist_ok=True)

HF_SPACE_PATH = Path(TTSDB_CACHE) / "hf_spaces"
HF_SPACE_PATH.mkdir(parents=True, exist_ok=True)