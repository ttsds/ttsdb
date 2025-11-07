# GENERIC SETUP
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y tzdata apt-transport-https
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y git git-lfs libsox-dev ffmpeg gcc build-essential g++-12 espeak-ng python3 python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN --mount=type=cache,target=/root/.cache/pip pip install torch torchvision torchaudio

WORKDIR /app

# Tortoise installation
ADD ./requirements.txt /app

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

ADD load_models.py /app/load_models.py

RUN python load_models.py

RUN --mount=type=cache,target=/root/.cache/pip pip install fastapi[standard] uvicorn

ENV port=8000

ADD . /app

CMD uvicorn tortoise_service:app --host 0.0.0.0 --port $port --timeout-keep-alive 300