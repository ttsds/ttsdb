<!-- Generated at 2026-02-03T00:00:00Z from templates/init/README.md.j2 -->

# VibeVoice

The model description is sourced from config.yaml (metadata.description).

VibeVoice is a large-scale text-to-speech model that generates expressive,
long-form speech. This integration focuses on single-speaker voice cloning
with a reference audio prompt.

## Installation

pip install ttsdb_vibevoice

## Usage

from ttsdb_vibevoice import VibeVoice

model = VibeVoice(model_id="microsoft/VibeVoice-1.5B")
audio, sr = model.synthesize(
    text="Hello, world!",
    reference_audio="path/to/reference.wav"
)
