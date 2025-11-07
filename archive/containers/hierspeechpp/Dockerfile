# GENERIC SETUP
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update && apt-get install -y tzdata apt-transport-https
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y git git-lfs libsox-dev ffmpeg gcc build-essential espeak-ng

RUN --mount=type=cache,target=/root/.cache/pip pip install torch torchvision torchaudio

WORKDIR /app

# HierSpeech++ installation

RUN git clone https://github.com/sh-lee-prml/HierSpeechpp.git

WORKDIR HierSpeechpp

RUN git checkout 318c633

RUN sed -i 's/torch==1.13.1+cu117//g' requirements.txt
RUN sed -i 's/torchaudio==0.13.1+cu117//g' requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip pip install gdown

ADD models /app/models

RUN gdown --fuzzy https://drive.google.com/file/d/1xMfhg4qeehGO0RN-zxq-hAnW-omXmpdq/view?usp=drive_link
RUN gdown --fuzzy https://drive.google.com/file/d/1JTi3OOhIFFElj1X1u5jBeNa3CPbVS_gk/view?usp=drive_link

RUN mv hierspeechpp_v1.1_ckpt.pth models/main/hierspeechpp_v1.1_ckpt.pth
RUN mv ttv_lt960_ckpt.pth models/ttv/ttv_lt960_ckpt.pth

# FASTAPI AND UVICORN

RUN --mount=type=cache,target=/root/.cache/pip pip install fastapi[standard] uvicorn

ADD . /app


RUN sed -i 's/.cuda()//g' HierSpeechpp/inference.py
RUN sed -i 's/torch.load(a.ckpt)/torch.load(a.ckpt, map_location=torch.device("cpu"))/g' HierSpeechpp/inference.py
RUN sed -i 's/torch.load(a.ckpt_text2w2v)/torch.load(a.ckpt_text2w2v, map_location=torch.device("cpu"))/g' HierSpeechpp/inference.py
RUN sed -i 's/.cuda()//g' HierSpeechpp/hierspeechpp_speechsynthesizer.py


# APPEND FUNCTION to inference.py

RUN echo "def set_globals(_device,_hps,_hps_t2w2v,_h_sr,_h_sr48,_hps_denoiser):" >> HierSpeechpp/inference.py
RUN echo "    global device, hps, hps_t2w2v, h_sr, h_sr48, hps_denoiser" >> HierSpeechpp/inference.py
RUN echo "    device = _device" >> HierSpeechpp/inference.py
RUN echo "    hps = _hps" >> HierSpeechpp/inference.py
RUN echo "    hps_t2w2v = _hps_t2w2v" >> HierSpeechpp/inference.py
RUN echo "    h_sr = _h_sr" >> HierSpeechpp/inference.py
RUN echo "    h_sr48 = _h_sr48" >> HierSpeechpp/inference.py
RUN echo "    hps_denoiser = _hps_denoiser" >> HierSpeechpp/inference.py

ENV port=8000

CMD uvicorn hierspeechpp_service:app --host 0.0.0.0 --port $port --timeout-keep-alive 300

