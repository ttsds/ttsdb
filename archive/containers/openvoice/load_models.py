from melo.api import TTS
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
import torch
import nltk

nltk.download('averaged_perceptron_tagger_eng')

device = ("cuda" if torch.cuda.is_available() else "cpu")

ckpt_base_v1 = 'openvoice_v1/checkpoints/base_speakers/EN'
ckpt_converter_v1 = 'openvoice_v1/checkpoints/converter'

base_speaker_tts_v1 = BaseSpeakerTTS(f'{ckpt_base_v1}/config.json', device=device)
base_speaker_tts_v1.load_ckpt(f'{ckpt_base_v1}/checkpoint.pth')

tone_color_converter_v1 = ToneColorConverter(f'{ckpt_converter_v1}/config.json', device=device)
tone_color_converter_v1.load_ckpt(f'{ckpt_converter_v1}/checkpoint.pth')

source_se_v1 = torch.load(f'{ckpt_base_v1}/en_default_se.pth', map_location=device)

# Initialize OpenVoice v2 models
ckpt_converter_v2 = 'openvoice_v2/checkpoints_v2/converter'
tone_color_converter_v2 = ToneColorConverter(f'{ckpt_converter_v2}/config.json', device=device)
tone_color_converter_v2.load_ckpt(f'{ckpt_converter_v2}/checkpoint.pth')

language_v2 = "EN_NEWEST"
source_se_v2 = torch.load(f'openvoice_v2/checkpoints_v2/base_speakers/ses/en-newest.pth', map_location=device)

model_v2 = TTS(language=language_v2, device=device)
speaker_id_v2 = 0