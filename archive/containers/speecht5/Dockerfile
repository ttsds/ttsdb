# GENERIC SETUP
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y tzdata apt-transport-https
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y git git-lfs libsox-dev ffmpeg gcc build-essential g++-12 espeak-ng

RUN --mount=type=cache,target=/root/.cache/pip pip install torch torchvision torchaudio

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip pip install transformers soundfile numpy sentencepiece speechbrain

RUN --mount=type=cache,target=/root/.cache/pip pip install fastapi[standard] uvicorn

ENV port=8000

ADD . /app

WORKDIR /app

CMD uvicorn speecht5_service:app --host 0.0.0.0 --port $port --timeout-keep-alive 300
