FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y tzdata apt-transport-https
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y git git-lfs libsox-dev ffmpeg gcc build-essential g++-12 espeak-ng

RUN --mount=type=cache,target=/root/.cache/pip pip install -U torch torchvision torchaudio

WORKDIR /app

RUN git clone https://github.com/yl4579/StyleTTS2.git

WORKDIR /app/StyleTTS2

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

RUN git checkout 5cedc71c333f8d8b8551ca59378bdcc7af4c9529
RUN git clone https://huggingface.co/yl4579/StyleTTS2-LibriTTS

RUN mv StyleTTS2-LibriTTS/Models .

RUN --mount=type=cache,target=/root/.cache/pip pip install phonemizer

RUN --mount=type=cache,target=/root/.cache/pip pip install fastapi[standard] uvicorn

ENV port=8000

ADD styletts2_service.py /app

WORKDIR /app

CMD uvicorn styletts2_service:app --host 0.0.0.0 --port $port --timeout-keep-alive 300