from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)
from speechbrain.pretrained import EncoderClassifier

SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    run_opts={"device": device},
    savedir="/tmp/speechbrain_speaker_embedding",
)