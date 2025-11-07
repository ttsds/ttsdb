# GENERIC SETUP
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y tzdata apt-transport-https
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y git git-lfs libsox-dev ffmpeg gcc build-essential g++-12 espeak-ng

RUN --mount=type=cache,target=/root/.cache/pip pip install torch==2.2.1 torchaudio==2.2.1 torchvision

WORKDIR /app

# METAVOICE INSTALLATION

RUN git clone https://github.com/metavoiceio/metavoice-src.git
WORKDIR metavoice-src
RUN git checkout de3fa21
RUN sed -i 's/threshold_s=30/threshold_s=1/g' fam/llm/utils.py
RUN sed -i '51s/.*/#torch._inductor.config.fx_graph_cache = (/' fam/llm/fast_inference_utils.py
RUN sed -i '52s/.*/#    True/' fam/llm/fast_inference_utils.py
RUN sed -i '53s/.*/#)/' fam/llm/fast_inference_utils.py
RUN sed -i 's/compile=True/compile=False/g' fam/llm/fast_inference.py
RUN sed -i 's/self.precision = {"float16": torch.float16, "bfloat16": torch.bfloat16}\[self._dtype\]/self.precision = torch.float32/g' fam/llm/fast_inference.py
RUN sed -i 's/self.dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}\[get_default_dtype()\]/self.dtype = torch.float32/g' fam/llm/fast_model.py
RUN sed -i 's/return dtype/return "float32"/g' fam/llm/utils.py
RUN sed -i 's/model = model.to(device=device, dtype=torch.bfloat16)/model = model.to(device=device, dtype=precision)/g' fam/llm/fast_inference_utils.py
RUN sed -i 's/"cuda",/("cuda" if torch.cuda.is_available() else "cpu"),/g' fam/llm/fast_inference_utils.py
RUN sed -i 's/"cuda")/("cuda" if torch.cuda.is_available() else "cpu"))/g' fam/llm/fast_inference_utils.py
RUN sed -i 's/mbd = MultiBandDiffusion.get_mbd_24khz(bw=6)/mbd = MultiBandDiffusion.get_mbd_24khz(bw=6, device=("cuda" if torch.cuda.is_available() else "cpu"))/g' fam/llm/decoders.py
RUN sed -i 's/.to("cuda")/.to("cuda" if torch.cuda.is_available() else "cpu")/g' fam/llm/decoders.py
RUN sed -i 's/device="cuda"/device=("cuda" if torch.cuda.is_available() else "cpu")/g' fam/llm/decoders.py
RUN sed -i 's/with torch.amp.autocast(device_type="cuda", dtype=torch.float32):/with open("test.txt", "w") as f:/g' fam/llm/decoders.py
RUN sed -i 's/"gpu": torch.cuda.get_device_name(0)/"gpu": ("cuda" if torch.cuda.is_available() else "cpu")/g' fam/llm/fast_inference.py

WORKDIR /app

# remove --hash lines from requirements.txt
RUN sed -i '/--hash/d' metavoice-src/requirements.txt
# remove \ from the end of the lines
RUN sed -i 's/\\//g' metavoice-src/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip pip install -r metavoice-src/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip pip install fastapi[standard] uvicorn

RUN --mount=type=cache,target=/root/.cache/pip pip install python-dotenv posthog

ADD load_models.py /app/load_models.py

RUN python load_models.py

ADD . /app

ENV port=8000

CMD uvicorn metavoice_service:app --host 0.0.0.0 --port $port --timeout-keep-alive 300