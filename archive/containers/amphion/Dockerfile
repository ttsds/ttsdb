ARG gpus=all
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y git

WORKDIR /app
RUN git clone https://github.com/open-mmlab/Amphion.git
WORKDIR Amphion
RUN git checkout f7cb4b4

RUN mkdir -p ckpts/tts
RUN ln -s  /downloads/valle_v1_small  ckpts/tts/valle_v1_small
RUN ln -s  /downloads/valle_v1_medium ckpts/tts/valle_v1_medium
RUN ln -s  /downloads/valle_v2 ckpts/tts/valle_v2
RUN ln -s  /downloads/naturalspeech2_v1 ckpts/tts/naturalspeech2_v1
RUN ln -s  /downloads/maskgct_v1 ckpts/tts/maskgct_v1

# copy requirements.txt from current local directory to /app
RUN apt-get update && apt-get install gcc build-essential g++-12 -y
RUN --mount=type=cache,target=/root/.cache/pip pip install fairseq

RUN --mount=type=cache,target=/root/.cache/pip pip install fastapi[standard] uvicorn

COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /app/requirements.txt

COPY download_encodec_nltk.py /app/download_encodec_nltk.py
RUN --mount=type=cache,target=/root/.cache/huggingface \
python /app/download_encodec_nltk.py

# espeak (for VALL-E)
RUN apt-get install -y espeak-ng

# amphion_service.py is the entrypoint
COPY amphion_service.py /app/amphion_service.py

# insert the following at the beginning of /app/Amphion/bins/tts/inference.py
# import sys\nsys.path.append('.')\n
RUN sed -i '1s/^/import sys\nsys.path.append(".")\n/' /app/Amphion/bins/tts/inference.py
RUN sed -i 's/self.args.text.replace(" ", "_", 100)/"output"/' /app/Amphion/models/tts/naturalspeech2/ns2_inference.py

WORKDIR /app

ENV port=8000

CMD uvicorn amphion_service:app --host 0.0.0.0 --port $port --timeout-keep-alive 300