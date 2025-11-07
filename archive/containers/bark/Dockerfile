FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y git git-lfs

WORKDIR /app
RUN git clone https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer.git bark-vc
WORKDIR bark-vc
RUN git checkout 4f42e44
RUN apt-get update && apt-get install gcc build-essential g++-12 -y
RUN --mount=type=cache,target=/root/.cache/pip pip install fairseq
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

RUN pip install git+https://github.com/suno-ai/bark.git@f4f32d4cd480dfec1c245d258174bc9bde3c2148

WORKDIR /app

COPY load_models.py /app/load_models.py
RUN python load_models.py

COPY . /app
RUN --mount=type=cache,target=/root/.cache/pip pip install fastapi[standard] uvicorn
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /app/requirements.txt

ENV port=8000

CMD uvicorn bark_service:app --host 0.0.0.0 --port $port --timeout-keep-alive 300