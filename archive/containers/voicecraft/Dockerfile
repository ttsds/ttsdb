# GENERIC SETUP
FROM mmcauliffe/montreal-forced-aligner:v2.2.17

USER root

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y tzdata apt-transport-https
RUN apt-get install -y git git-lfs libsox-dev ffmpeg gcc build-essential g++-12 espeak-ng

USER mfauser

WORKDIR /app

# VOICECRAFT INSTALLATION

RUN --mount=type=cache,target=/root/.cache/pip pip install torchaudio==2.0.2 torch==2.0.1 xformers==0.0.22

RUN pip install -e git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft

RUN pip install -U hydra-core omegaconf

RUN mfa model download dictionary english_us_arpa
RUN mfa model download acoustic english_us_arpa

RUN pip install tensorboard==2.16.2 phonemizer==3.2.1 datasets==2.16.0 torchmetrics==0.11.1 numpy=="1.*" huggingface_hub

RUN git clone https://github.com/jasonppy/VoiceCraft.git voicecraft

WORKDIR voicecraft

RUN git checkout 4873249

WORKDIR /app

ADD load_models.py /app/load_models.py

RUN python load_models.py

# FASTAPI AND UVICORN

RUN --mount=type=cache,target=/root/.cache/pip pip install fastapi[standard] uvicorn

ENV port=8000

ADD voicecraft_service.py /app/voicecraft_service.py

CMD uvicorn voicecraft_service:app --host 0.0.0.0 --port $port --timeout-keep-alive 300