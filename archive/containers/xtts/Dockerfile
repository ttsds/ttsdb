FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y tzdata apt-transport-https
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y git git-lfs libsox-dev ffmpeg gcc build-essential g++-12 espeak-ng

RUN --mount=type=cache,target=/root/.cache/pip pip3 install --upgrade pip

RUN --mount=type=cache,target=/root/.cache/pip pip3 install torch torchvision torchaudio transformers==4.42.4

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip pip3 install coqui-tts

ADD . /app

RUN python3 load_models.py

RUN --mount=type=cache,target=/root/.cache/pip pip3 install fastapi[standard] uvicorn

ENV port=8000

CMD uvicorn xtts_service:app --host 0.0.0.0 --port $port --timeout-keep-alive 300