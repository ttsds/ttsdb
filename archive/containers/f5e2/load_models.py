from huggingface_hub import hf_hub_download

from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model
)

# Load vocoder
vocos = load_vocoder()

# Load models
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_model_path = hf_hub_download(repo_id="SWivid/F5-TTS", filename="F5TTS_Base/model_1200000.safetensors")
F5TTS_ema_model = load_model(DiT, F5TTS_model_cfg, F5TTS_model_path)

E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
E2TTS_model_path = hf_hub_download(repo_id="SWivid/E2-TTS", filename="E2TTS_Base/model_1200000.safetensors")
E2TTS_ema_model = load_model(UNetT, E2TTS_model_cfg, E2TTS_model_path)