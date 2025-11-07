FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y git git-lfs libsox-dev ffmpeg gcc build-essential g++-12 -y

RUN --mount=type=cache,target=/root/.cache/pip pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

RUN --mount=type=cache,target=/root/.cache/pip pip install git+https://github.com/SWivid/F5-TTS.git@700039b55422db5aaa80a1b07fbf671e039ba66b

WORKDIR /app

COPY load_models.py /app/load_models.py
RUN python load_models.py

RUN --mount=type=cache,target=/root/.cache/pip pip install fastapi[standard] uvicorn

COPY . /app

ENV port=8000

WORKDIR /app

CMD uvicorn f5e2_service:app --host 0.0.0.0 --port $port --timeout-keep-alive 300