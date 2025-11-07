# GENERIC SETUP
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update && apt-get install -y tzdata apt-transport-https
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y git git-lfs libsox-dev ffmpeg gcc build-essential g++ espeak-ng

RUN --mount=type=cache,target=/root/.cache/pip pip install torch=="1.13.1+cu116" torchvision=="0.14.1+cu116" torchaudio=="0.13.1" --extra-index-url https://download.pytorch.org/whl/cu116

WORKDIR /app
# OPENVOICE INSTALLATION

# V1

RUN apt-get install -y wget unzip

RUN wget https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip

RUN git clone https://github.com/myshell-ai/OpenVoice.git openvoice_v1

RUN unzip checkpoints_1226.zip -d openvoice_v1

WORKDIR openvoice_v1

RUN git checkout 34874b77e8ed4fd63ebafcfefafbe31002748486

RUN sed -i "s/assert num_splits > 0, 'input audio is too short'/num_splits=max(num_splits,1)/g" openvoice/se_extractor.py

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# remove watermark
RUN sed -i 's/if kwargs.get('enable_watermark', True):/if False:/' openvoice/api.py

RUN --mount=type=cache,target=/root/.cache/pip pip install -e .

# V2

WORKDIR /app

RUN wget https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip

RUN mkdir openvoice_v2 && unzip checkpoints_v2_0417.zip -d openvoice_v2

RUN pip install git+https://github.com/myshell-ai/MeloTTS.git@5b538481e24e0d578955be32a95d88fcbde26dc8

RUN python -m unidic download

ADD load_models.py /app/load_models.py

RUN python load_models.py

# FASTAPI AND UVICORN

RUN --mount=type=cache,target=/root/.cache/pip pip install fastapi[standard] uvicorn

ENV port=8000

ADD . /app

CMD uvicorn openvoice_service:app --host 0.0.0.0 --port $port --timeout-keep-alive 300
