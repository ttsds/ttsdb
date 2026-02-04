<!-- Generated at 2026-02-03T00:00:00Z from templates/init/README.md.j2 -->

# OpenVoice

The model description is sourced from config.yaml (metadata.description).

OpenVoice v2 voice cloning model built on MeloTTS and a tone color converter
for zero-shot speaker adaptation.

## Installation

pip install ttsdb_openvoice

## Usage

from ttsdb_openvoice import OpenVoice

model = OpenVoice(model_id="myshell-ai/OpenVoiceV2")
audio, sr = model.synthesize(
    text="Hello, world!",
    reference_audio="path/to/reference.wav"
)
