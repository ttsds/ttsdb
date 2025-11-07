# TTSDS Datasets
This repository allows generation of synthetic speech datasets using 18 state-of-the-art open-source TTS models. Since many of these systems require different system dependencies and/or python versions, they are organised as docker containers.

These containers are used to generate the datasets for our TTSDS benchmark @ https://huggingface.co/ttsds

**NOTE**: The v2_evaluation dataset is not public at the moment, so you will have to edit ``run.sh`` with the path(s) to your own datasets.

## TTS Systems

| **System**       | **Training Data**                                  | 🌐 **Multilingual** | 📚 **Training Amount (k hours)** | 🧠 **Num. Parameters (M)** | 🎯 **Target Repr.**        | 📖 **LibriVox Only** | 🔄 **NAR** | 🔁 **AR** | 🔡 **G2P** | 🧩 **Language Model** | 🎵 **Prosody Prediction** | 🌊 **Diffusion** | ⏱️ **Delay Pattern** |
|-------------------|---------------------------------------------------|---------------------|-----------------------------------|----------------------------|----------------------------|----------------------|------------|-----------|------------|-----------------------|--------------------------|------------------|---------------------|
| [**Bark**](https://github.com/suno-ai/bark)          | Unknown                                           | ✅                  | Unknown                           | 240                        | Audio Codec Code          | ❌                   | ✅          | ✅         | ❌          | ✅                     | ❌                        | ❌                | ❌                   |
| [**F5-TTS**](https://github.com/SWivid/F5-TTS)        | Emilia                                           | ✅                  | 95                                | 330                        | Mel Spectrogram           | ❌                   | ✅          | ❌         | ✅          | ❌                     | ❌                        | ✅                | ❌                   |
| [**Fish (1.4)**](https://github.com/fishaudio/fish-speech)    | LibriLight, PlayerFM, StarRail...?               | ✅                  | 700                               | 500                        | Audio Codec Code          | ❌                   | ❌          | ✅         | ❌          | ✅                     | ❌                        | ❌                | ❌                   |
| [**GPT-SoVITS**](https://github.com/RVC-Boss/GPT-SoVITS)    | Chinese, English, Japanese (1000+700+300 hrs)     | ✅                  | 2                                 | 200                        | Audio Codec Code          | ❌                   | ✅          | ✅         | ✅          | ✅                     | ❌                        | ❌                | ❌                   |
| [**Hierspeech++**](https://github.com/sh-lee-prml/HierSpeechpp)  | LibriTTS, LibriLight, Expresso, MSSS, NIKL        | ✅                  | 2.7                               | 97                         | Waveform                  | ❌                   | ✅          | ❌         | ✅          | ❌                     | ✅                        | ❌                | ❌                   |
| [**MetaVoice**](https://github.com/metavoiceio/metavoice-src)     | Unknown                                           | ❌                  | 100                               | 1000                       | Audio Codec Code          | ❌                   | ✅          | ✅         | ❌          | ✅                     | ❌                        | ❌                | ✅                   |
| [**NaturalSpeech2**](https://github.com/open-mmlab/Amphion)| LibriTTS                                         | ❌                  | 0.58                              | 380                        | Audio Codec Code          | ✅                   | ✅          | ❌         | ✅          | ❌                     | ✅                        | ✅                | ❌                   |
| [**OpenVoice**](https://github.com/myshell-ai/OpenVoice)     | Unknown                                           | ❌                  | 0.6                               | 73                         | Mel Spectrogram           | ❌                   | ✅          | ❌         | ✅          | ❌                     | ✅                        | ❌                | ❌                   |
| [**ParlerTTS**](https://github.com/huggingface/parler-tts)     | MLS, LibriTTS                                     | ❌                  | 23                                | 2200                       | Audio Codec Code          | ✅                   | ❌          | ✅         | ❌          | ✅                     | ❌                        | ❌                | ✅                   |
| [**Pheme**](https://github.com/PolyAI-LDN/pheme)         | GigaSpeech, MLS, LibriTTS                        | ❌                  | 11                                | 300                        | Audio Codec Code          | ❌                   | ✅          | ✅         | ✅          | ✅                     | ❌                        | ❌                | ❌                   |
| [**SpeechT5**](https://github.com/microsoft/SpeechT5)      | LibriTTS, LibriSpeech                            | ❌                  | 1.4                               | 144                        | Mel Spectrogram           | ✅                   | ❌          | ✅         | ❌          | ❌                     | ❌                        | ❌                | ❌                   |
| [**StyleTTS2**](https://github.com/yl4579/StyleTTS2)     | LibriTTS                                         | ❌                  | 0.24                              | 191                        | Mel Spectrogram           | ✅                   | ✅          | ❌         | ✅          | ❌                     | ✅                        | ✅                | ❌                   |
| [**TorToiSe**](https://github.com/neonbjb/tortoise-tts)      | LibriTTS, HifiTTS, Podcasts + Audiobooks         | ❌                  | 49.5                              | 960                        | Mel Spectrogram           | ❌                   | ✅          | ✅         | ❌          | ✅                     | ❌                        | ✅                | ❌                   |
| [**VallEv1**](https://github.com/open-mmlab/Amphion)       | LibriLight                                       | ❌                  | 6                                 | 370                        | Audio Codec Code          | ✅                   | ❌          | ✅         | ✅          | ✅                     | ❌                        | ❌                | ❌                   |
| [**VoiceCraft**](https://github.com/jasonppy/VoiceCraft)    | GigaSpeech                                       | ❌                  | 9                                 | 830                        | Audio Codec Code          | ❌                   | ❌          | ✅         | ✅          | ✅                     | ❌                        | ❌                | ✅                   |
| [**WhisperSpeech**](https://github.com/collabora/WhisperSpeech) | MLS                                              | ✅                  | 80                                | 2054                       | Audio Codec Code          | ✅                   | ✅          | ✅         | ❌          | ✅                     | ❌                        | ❌                | ❌                   |
| [**XTTSv2**](https://github.com/idiap/coqui-ai-TTS)        | LibriTTS, Common Voice, Internal                 | ✅                  | 27                                | 456                        | Audio Codec Code (Own)    | ❌                   | ❌          | ✅         | ❌          | ✅                     | ❌                        | ❌                | ❌                   |
| [**E2-TTS**](https://github.com/SWivid/F5-TTS)        | Emilia                                           | ✅                  | 95                                | 330                        | Mel Spectrogram           | ❌                   | ✅          | ❌         | ✅          | ❌                     | ✅                        | ❌                | ❌                   |

## Legend

- 🌐 Multilingual
  - The ISO codes of languages the model is capable off. ❌ if English only.
- 📚 Training Amount (k hours)
  - The number of hours the model was trained on
- 🧠 Num. Parameters (M)
  - How many parameters the model has, excluding vocoder and text-only components
- 🎯 Target Repr.
  - Which output representations the model uses, for example audio codecs or mel spectrograms
- 📖 LibriVox Only
  - If the model was trained on librivox-like (audiobook) data alone
- 🔄 NAR
  - If the model has a significant non-autoregressive component
- 🔁 AR
  - If the model has a significant autoregressive component
- 🔡 G2P
  - If the model uses G2P (phone inputs)
- 🧩 Language Model
  - If an LM-like approach is used (next token prediction)
- 🎵 Prosody Prediction
  - If prosodic correlates such as pitch or energy are predicted
- 🌊 Diffusion
  - If diffusion is used (outside vocoder)
- ⏱️ Delay Pattern
  - If a delay pattern is used for audio codes (see [Lyth & King, 2024](https://arxiv.org/abs/2402.01912))

## Requirements
 - Python 3.10
 - huggingface-cli

## Disclaimers
 - You need to respect the TOS and license(s) of the TTS systems in this repository before using them using this tool.
 - We intend for this repository to be used for academic and educational purposes only.
 - Do not clone anyone's voice without their permission.
