# MaskGCT

MaskGCT is a zero-shot text-to-speech model using masked generative codec transformer from Amphion.

## Installation

```bash
cd models/maskgct
just setup maskgct
```

## Usage

```python
from ttsdb_maskgct import MaskGCT

model = MaskGCT(model_id="ttsds/maskgct")
audio, sr = model.synthesize(
    text="Hello, world!",
    reference_audio="path/to/reference.wav",
    text_reference="Text spoken in the reference audio.",
    language="en"
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
just hf-weights-prepare maskgct  # Download weights first
just test-integration maskgct
```

## Supported Languages

- English (en)
- Chinese (zh)
- Korean (ko)
- Japanese (ja)
- French (fr)
- German (de)

## Citation

```bibtex
@article{wang2024maskgct,
  title={MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer},
  author={Wang, Yuancheng and Zhan, Haoyue and Liu, Liwei and Zeng, Ruihong and Guo, Haotian and Zheng, Jiachen and Zhang, Qiang and Zhang, Xueyao and Zhang, Shunsi and Wu, Zhizheng},
  journal={arXiv preprint arXiv:2409.00750},
  year={2024}
}
```
