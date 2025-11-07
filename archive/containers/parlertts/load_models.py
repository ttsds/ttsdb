
import torch
from transformers import AutoFeatureExtractor, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

def setup(model_id, device="cpu"):
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    SAMPLING_RATE = model.config.sampling_rate
    return model, tokenizer, feature_extractor, SAMPLING_RATE

setup("parler-tts/parler-tts-mini-v1")
setup("parler-tts/parler-tts-large-v1")