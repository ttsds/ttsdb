# TTSDB

TTSDB is a collection of python packages for text-to-speech (TTS) models.

## Requirements

[just](https://github.com/casey/just) is the only dependency that needs to be installed before getting started.

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

## Dataset synthesis (CLI)

```bash
# Synthesize across all models (default variant per model)
just synth-dataset run --all-models --gpus 0,1

# Run with a custom dataset YAML (pairs or test_data format)
just synth-dataset run --dataset assets/example_dataset.yaml --models vibevoice xtts --gpus 0

# Folder-based pairs (suffix -01 as reference, -02 as target)
just synth-dataset run --dataset-folder ../sap_readable --language eng --gpus 0,1

# Check progress
just synth-dataset status
```

## TTS systems

| **System** | **Training Data** | **Multilingual** | **Languages** | **Training (k hours)** | **Num. Parameters (M)** | **Target Repr.** | **NAR** | **AR** | **Diffusion** |
|---|---|---|---|---|---|---|---|---|---|
| [**E2 TTS**](models/e2-tts) | Emilia Dataset | ✅ | eng, zho | 100 | 335 | Mel | ✅ | ❌ | ✅ |
| [**F5-TTS**](models/f5-tts) | Emilia Dataset | ✅ | eng, zho | 100 | 335 | Mel | ✅ | ❌ | ✅ |
| [**Fish Speech**](models/fish-speech) | Unknown | ✅ | eng, zho, deu, jpn, fra, spa, kor, ara, rus, nld, ita, pol, por | Unknown | 500 | Codec, Quantized Mel Tokens | ❌ | ✅ | ❌ |
| [**GPT-SoVITS**](models/gpt-sovits) | Internal Dataset | ✅ | eng, zho, jpn | 2 | 167 | Codec | ✅ | ❌ | ❌ |
| [**HierSpeech**](models/hierspeech) | LibriTTS, LibriLight, Expresso, MSSS, NIKL | ✅ | eng, kor | Unknown | 204 | Waveform | ✅ | ❌ | ❌ |
| [**IndexTTS**](models/indextts) | Unknown | ✅ | eng, zho | Unknown | Unknown | Mel spectrogram | ❌ | ✅ | ❌ |
| [**MaskGCT**](models/maskgct) | Emilia Dataset | ✅ | eng, zho, kor, jpn, fra, deu | 100 | 1010 | Codec | ✅ | ❌ | ❌ |
| [**Metavoice**](models/metavoice) | Unknown | ❌ | eng | Unknown | 1200 | Codec | ✅ | ❌ | ❌ |
| [**OpenVoice**](models/openvoice) | Unknown | ✅ | eng, zho, spa, fra | Unknown | Unknown | Mel spectrogram | ✅ | ❌ | ❌ |
| [**Pheme**](models/pheme) | GigaSpeech, MLS, LibriTTS | ❌ | eng | Unknown | 300 | Codec | ✅ | ❌ | ❌ |
| [**StyleTTS2**](models/styletts2) | LibriTTS | ❌ | eng | 0.24 | 191 | Mel | ✅ | ❌ | ✅ |
| [**TorToise**](models/tortoise) | LibriTTS, HifiTTS | ❌ | eng | Unknown | 960 | Mel | ❌ | ✅ | ✅ |
| [**Vevo**](models/vevo) | Emilia Dataset | ✅ | eng, zho, deu, fra, jpn, kor | 101 | 900 | Codec, RepCodec | ✅ | ❌ | ✅ |
| [**VibeVoice**](models/vibevoice) | Unknown | ✅ | eng, zho | Unknown | 1500 | Codec | ❌ | ❌ | ✅ |
| [**VoiceCraft**](models/voicecraft) | GigaSpeech | ❌ | eng | Unknown | 830 | Codec | ❌ | ✅ | ❌ |
| [**WhisperSpeech**](models/whisperspeech) | MLS, LibriLight | ✅ | eng, pol, deu, fra, ita, nld, spa, por | Unknown | 1300 | Codec, EnCodec | ✅ | ❌ | ❌ |
| [**XTTS**](models/xtts) | LibriTTS, CommonVoice | ✅ | eng, spa, fra, deu, ita, por, pol, tur, rus, nld, ces, ara, zho, hun, hin | Unknown | 466 | VQ-VAE | ❌ | ✅ | ❌ |


## Disclaimers

- Respect each upstream model's license and terms before use.
- Don't clone anyone's voice without permission.
