# GENERIC SETUP
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y tzdata apt-transport-https
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y git git-lfs libsox-dev ffmpeg gcc build-essential g++-12 espeak-ng

RUN --mount=type=cache,target=/root/.cache/pip pip install -U torch torchvision torchaudio

WORKDIR /app

# GPT-SOVITS INSTALLATION

RUN git clone https://github.com/RVC-Boss/GPT-SoVITS.git gpt-sovits
WORKDIR gpt-sovits
RUN git checkout eee607b
RUN sed -i 's/if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):/if True:/g' GPT_SoVITS/inference_webui.py

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

RUN mkdir -p pretrained_models
RUN mkdir -p tools/damo_asr/models
RUN mkdir -p tools/uvr5

WORKDIR pretrained_models

RUN git clone https://huggingface.co/lj1995/GPT-SoVITS hf_gpt_sovits

WORKDIR tools/damo_asr/models

RUN git clone https://www.modelscope.cn/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git
RUN git clone https://www.modelscope.cn/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch.git
RUN git clone https://www.modelscope.cn/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch.git

WORKDIR tools/uvr5

RUN git clone https://huggingface.co/Delik/uvr5_weights

RUN git config core.sparseCheckout true
RUN mv hf_gpt_sovits/pretrained_models/GPT-SoVITS/* /app/gpt-sovits/pretrained_models/

WORKDIR /app

# FASTAPI AND UVICORN

ADD gptsovits_service.py /app/gptsovits_service.py

RUN --mount=type=cache,target=/root/.cache/pip pip install fastapi[standard] uvicorn

ENV port=8000

CMD uvicorn gptsovits_service:app --host 0.0.0.0 --port $port --timeout-keep-alive 300