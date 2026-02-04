<!-- Generated at 2026-01-29T19:21:50Z from templates/init/README.md.j2 -->

# TorToise (Tortoise TTS)

The model description is sourced from `config.yaml` (`metadata.description`).

## Installation

```bash
just setup tortoise
```

## Usage

```python
from ttsdb_tortoise import Tortoise

model = Tortoise(model_id="ttsds/tortoise")
audio, sr = model.synthesize(
    text="Hello, world!",
    reference_audio="path/to/reference.wav"
)
```

## Audio Examples

Generated audio samples from integration tests:

| Language | Text | Audio |
|----------|------|-------|
| English | "With tenure, Suzie'd have all the more leisure for yachting, but her publications are no good." | [audio_examples/en_test_001.wav](audio_examples/en_test_001.wav) |
| Chinese | "視野無限廣，窗外有藍天" | [audio_examples/zh_test_001.wav](audio_examples/zh_test_001.wav) |

To regenerate examples, run:
```bash
just test-integration tortoise
```
