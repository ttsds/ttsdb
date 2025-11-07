import os
import sys

os.chdir("voicecraft")
sys.path.append(".")

import getpass

os.environ["USER"] = getpass.getuser()

from models import voicecraft
from data.tokenizer import AudioTokenizer, TextTokenizer


def load_voicecraft_model(model_name):
    model = voicecraft.VoiceCraft.from_pretrained(f"pyp1/VoiceCraft_{model_name}")
    return model

def load_audio_tokenizer(encodec_fn="./pretrained_models/encodec_4cb2048_giga.th"):
    if not os.path.exists(encodec_fn):
        os.system(f"wget -q https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th -O {encodec_fn}")
    print(f"Loaded audio tokenizer from {encodec_fn}")
    audio_tokenizer = AudioTokenizer(signature=encodec_fn)
    return audio_tokenizer

def load_text_tokenizer():
    text_tokenizer = TextTokenizer(backend="espeak")
    return text_tokenizer

if __name__ == "__main__":
    load_voicecraft_model("giga330M")
    load_voicecraft_model("giga830M")
    load_voicecraft_model("330M_TTSEnhanced")
    load_voicecraft_model("830M_TTSEnhanced")
    load_audio_tokenizer("encodec_4cb2048_giga.th")
    load_text_tokenizer()