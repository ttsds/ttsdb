# F5-TTS

TODO: Add model description.

## Installation

```bash
cd models/f5-tts
uv sync
```

## Usage

```python
from ttsdb_f5_tts import F5TTS

model = F5TTS(model_id="TODO: huggingface/model-id")
audio, sr = model.synthesize(
    text="Hello, world!",
    reference_audio="path/to/reference.wav"
)
```
