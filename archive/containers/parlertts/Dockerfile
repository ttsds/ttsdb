# GENERIC SETUP
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y tzdata apt-transport-https
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y git git-lfs libsox-dev ffmpeg gcc build-essential g++-12 espeak-ng

RUN --mount=type=cache,target=/root/.cache/pip pip install -U torch torchvision torchaudio

WORKDIR /app
# PARLER TTS INSTALLATION

RUN git clone https://github.com/apresence/parler-tts parler-tts
WORKDIR parler-tts
RUN git checkout 91063f38c4079e051d2c5db32db2f7d04bafafc4

RUN --mount=type=cache,target=/root/.cache/pip pip install -e .

# FASTAPI AND UVICORN

RUN --mount=type=cache,target=/root/.cache/pip pip install fastapi[standard] uvicorn

ENV port=8000

WORKDIR /app

ADD load_models.py /app/load_models.py

RUN python load_models.py

RUN --mount=type=cache,target=/root/.cache/pip pip install git+https://github.com/wenet-e2e/wespeaker.git umap hdbscan datasets

ADD parlertts_descriptions.py /app/parlertts_descriptions.py

RUN python parlertts_descriptions.py

ADD . /app

CMD uvicorn parlertts_service:app --host 0.0.0.0 --port $port --timeout-keep-alive 300
