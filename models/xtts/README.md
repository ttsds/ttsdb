<!-- Generated at 2026-02-02T10:50:14Z from templates/init/README.md.j2 -->

# XTTS

The model description is sourced from `config.yaml` (`metadata.description`).


TODO: Add a short markdown description of the model.


## Installation

```bash
pip install ttsdb_xtts
```

## Usage

```python
from ttsdb_xtts import XTTS

model = XTTS(model_id="ttsds/xtts")
audio, sr = model.synthesize(
    text="Hello, world!",
    reference_audio="path/to/reference.wav"
)
```

## Audio Examples

Generated audio samples from integration tests:

| Language | Text | Audio |
|----------|------|-------|
| English | "With tenure, Suzie'd have all the more leisure for yachting, but her publications are no good." | [audio_examples/eng_test_001.wav](https://github.com/ttsds/ttsdb/raw/refs/heads/v2/models/xtts/audio_examples/eng_test_001.wav) |

To regenerate examples, run:
```bash
just test-integration xtts
```
