# GENERIC SETUP
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y tzdata apt-transport-https
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y git git-lfs libsox-dev ffmpeg gcc build-essential g++-12 espeak-ng

RUN --mount=type=cache,target=/root/.cache/pip pip install torch torchvision torchaudio

WORKDIR /app

# PHEME INSTALLATION

RUN apt-get install -y wget
RUN git clone https://github.com/PolyAI-LDN/pheme.git
WORKDIR pheme
RUN git checkout a0b73f9379fd55a069842ff2c3c48ec0fd0bdc6a
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --no-deps
RUN mkdir -p "ckpt/speechtokenizer/"
WORKDIR ckpt/speechtokenizer/
RUN wget "https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/SpeechTokenizer.pt"
RUN wget "https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/config.json"
WORKDIR ..
RUN wget "https://huggingface.co/fnlp/USLM/resolve/main/USLM_libritts/unique_text_tokens.k2symbols"
RUN git clone https://huggingface.co/PolyAI/pheme ckpt/pheme
RUN mkdir -p "ckpt/t2s"
RUN mkdir -p "ckpt/s2a"
RUN mv ckpt/pheme/config_t2s.json ckpt/t2s/config.json
RUN mv ckpt/pheme/generation_config.json ckpt/t2s/generation_config.json
RUN mv ckpt/pheme/t2s.bin ckpt/t2s/pytorch_model.bin
RUN mv ckpt/pheme/config_s2a.json ckpt/s2a/config.json
RUN mv ckpt/pheme/s2a.ckpt ckpt/s2a/s2a.ckpt

# FASTAPI AND UVICORN

RUN --mount=type=cache,target=/root/.cache/pip pip install fastapi[standard] uvicorn

ENV port=8000

ADD . /app

WORKDIR /app

CMD uvicorn pheme_service:app --host 0.0.0.0 --port $port --timeout-keep-alive 300