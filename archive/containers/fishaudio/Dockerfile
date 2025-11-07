FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y git git-lfs libsox-dev ffmpeg gcc build-essential g++-12 -y

RUN --mount=type=cache,target=/root/.cache/pip pip install -U torch torchvision torchaudio

WORKDIR /app
RUN git clone https://github.com/fishaudio/fish-speech.git
WORKDIR /app/fish-speech
RUN git fetch --all --tags
# checkout tag v1.4.1
RUN git checkout tags/v1.4.1

# we remove @click decorators from the scripts so that we can run them as modules
RUN sed -i '596,624d' tools/llama/generate.py
# print number of parameters in the model
RUN sed -i 's/logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")/print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")/g' tools/llama/generate.py
RUN sed -i '48,68d' tools/vqgan/inference.py
RUN sed -i 's/fake_audios\[0, 0\].float().cpu().numpy()/fake_audios\[0, 0\].detach().float().cpu().numpy()/g' tools/vqgan/inference.py

RUN --mount=type=cache,target=/root/.cache/pip pip install -e .[stable]

RUN --mount=type=cache,target=/root/.cache/pip pip install -U "huggingface_hub[cli]"

RUN --mount=type=cache,target=/root/.cache/huggingface huggingface-cli download fishaudio/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4

RUN --mount=type=cache,target=/root/.cache/pip pip install fastapi[standard] uvicorn

COPY . /app

ENV port=8000

WORKDIR /app

CMD uvicorn fish_service:app --host 0.0.0.0 --port $port --timeout-keep-alive 300