import os

from TTS.api import TTS

os.environ["COQUI_TOS_AGREED"] = "1"

TTS("tts_models/multilingual/multi-dataset/xtts_v2")