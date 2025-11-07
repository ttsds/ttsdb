FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y tzdata apt-transport-https
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y git git-lfs libsox-dev ffmpeg gcc build-essential g++-12 espeak-ng

RUN --mount=type=cache,target=/root/.cache/pip pip install -U torch torchvision torchaudio

RUN --mount=type=cache,target=/root/.cache/pip pip install git+https://github.com/collabora/WhisperSpeech.git@8567548dcc9d0501d1488884c7bcf194d546c0aa

RUN --mount=type=cache,target=/root/.cache/pip pip install webdataset

ADD load_models.py /app/load_models.py

WORKDIR /app

RUN python load_models.py

RUN --mount=type=cache,target=/root/.cache/pip pip install fastapi[standard] uvicorn

ENV port=8000

ADD . /app

CMD uvicorn whisperspeech_service:app --host 0.0.0.0 --port $port --timeout-keep-alive 300