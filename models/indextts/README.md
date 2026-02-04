<!-- Generated at 2026-02-03T00:00:00Z from templates/init/README.md.j2 -->

# IndexTTS

The model description is sourced from config.yaml (metadata.description).

IndexTTS2 is an emotionally expressive, duration-controlled zero-shot TTS model.

## Installation

pip install ttsdb_indextts

## Usage

from ttsdb_indextts import IndexTTS

model = IndexTTS(model_id="IndexTeam/IndexTTS-2")
audio, sr = model.synthesize(
    text="Hello, world!",
    reference_audio="path/to/reference.wav"
)
