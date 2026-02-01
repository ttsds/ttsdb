# TTSDB

TTSDB is a monorepo of small, installable Python packages for text-to-speech (TTS) models.

## Requirements

[just](https://github.com/casey/just) is the only dependency.

## Quickstart

```bash
# Install uv, pre-commit, and git hooks (one-time setup)
just bootstrap

# List available models
just models

# Set up a model for development
just setup maskgct cpu

# Run linters
just lint
```

## TTS systems

| **System** | **Training Data** | **Multilingual** | **Languages** | **Training (k hours)** | **Num. Parameters (M)** | **Target Repr.** | **NAR** | **AR** | **Diffusion** |
|---|---|---|---|---|---|---|---|---|---|
| [**E2 TTS**](models/e2-tts) | Emilia Dataset | ✅ | eng, zho | 100 | 335 | Mel | ✅ | ❌ | ✅ |
| [**F5-TTS**](models/f5-tts) | Emilia Dataset | ✅ | eng, zho | 100 | 335 | Mel | ✅ | ❌ | ✅ |
| [**MaskGCT**](models/maskgct) | Emilia Dataset | ✅ | eng, zho, kor, jpn, fra, deu | 100 | 1010 | Codec | ✅ | ❌ | ❌ |
| [**TorToise**](models/tortoise) | LibriTTS, HifiTTS | ❌ | eng | Unknown | 960 | Mel | ❌ | ✅ | ✅ |


## Disclaimers

- Respect each upstream model's license and terms before use.
- Don't clone anyone's voice without permission.
