# GPT-SoVITS

GPT-SoVITS is a powerful few-shot voice conversion and text-to-speech system by [RVC-Boss](https://github.com/RVC-Boss/GPT-SoVITS). It achieves high-quality voice cloning with just 1 minute of training data, supporting zero-shot and few-shot TTS with cross-lingual synthesis capabilities.

## Features

- **Zero-shot TTS**: Input a 5-second vocal sample and experience instant text-to-speech conversion
- **Few-shot TTS**: Fine-tune the model with just 1 minute of training data for improved voice similarity
- **Cross-lingual Support**: Synthesize in languages different from the training/reference audio
- **Multiple Versions**: Support for v1, v2, v3, and v4 with different capabilities

## Version Comparison

| Version | Languages | Training Hours | Parameters | Output Sample Rate |
|---------|-----------|---------------|------------|-------------------|
| v1 | English, Chinese, Japanese | 2k | 167M | 32kHz |
| v2 | +Korean, Cantonese | 5k | 167M | 32kHz |
| v3 | Same as v2 | 7k | 407M | 24kHz |
| v4 | Same as v2 | 7k | 407M | 48kHz |

## Installation

```bash
# Setup the model environment
just setup gpt-sovits

# Download all variants + shared dependencies (for HuggingFace upload)
just hf-weights-prepare gpt-sovits

# Download a specific variant only:
cd models/gpt-sovits
source .venv/bin/activate
python scripts/prepare_weights.py --variant v1  # v1 only
python scripts/prepare_weights.py --variant v2  # v2 only
python scripts/prepare_weights.py --variant v3  # v3 only
python scripts/prepare_weights.py --variant v4  # v4 only
```

## Usage

```python
from ttsdb_gpt_sovits import GPTSoVITS

# Load model (default: v1)
model = GPTSoVITS(model_id="ttsds/gpt-sovits")

# Load specific variant
model = GPTSoVITS(model_id="ttsds/gpt-sovits", variant="v2")

# Synthesize speech
audio, sr = model.synthesize(
    text="Hello, this is a test of the GPT-SoVITS system.",
    reference_audio="reference_speaker.wav",
    text_reference="This is the speaker's reference text.",
    language="en",
)

# Additional parameters
audio, sr = model.synthesize(
    text="你好，这是一个测试。",
    reference_audio="reference_speaker.wav",
    text_reference="这是说话人的参考音频。",
    language="zh",
    top_k=15,
    top_p=1.0,
    temperature=1.0,
    speed=1.0,
)
```

## Supported Languages

- **v1**: English (`en`), Chinese (`zh`), Japanese (`ja`)
- **v2+**: English (`en`), Chinese (`zh`), Japanese (`ja`), Korean (`ko`), Cantonese (`yue`)

## License

MIT License - See [LICENSE](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE)

## Citation

```bibtex
@misc{RVCBoss2024,
  author = {RVC-Boss},
  title = {GPT-SoVITS: 1 min voice data can also be used to train a good TTS model},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/RVC-Boss/GPT-SoVITS}},
}
```
