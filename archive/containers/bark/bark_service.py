import tempfile
import os
import sys
from pathlib import Path
import subprocess
import shutil
import argparse

sys.path.append("bark-vc")

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import torchaudio
from tqdm import tqdm
from encodec import EncodecModel
from encodec.utils import convert_audio
from bark_hubert_quantizer.hubert_manager import HuBERTManager
from bark_hubert_quantizer.pre_kmeans_hubert import CustomHubert
from bark_hubert_quantizer.customtokenizer import CustomTokenizer
from bark import SAMPLE_RATE, generate_audio, preload_models
import soundfile as sf

app = FastAPI()

def create_speaker_npz(audio_path, output_path):
    wav_file = audio_path
    out_file = output_path

    wav, sr = torchaudio.load(wav_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ("quantifier_V1_hubert_base_ls960_23.pth", "tokenizer_large.pth")
    print("Loading HuBERT...")
    hubert_model = CustomHubert(HuBERTManager.make_sure_hubert_installed(), device=device)
    print("Loading Quantizer...")
    quant_model = CustomTokenizer.load_from_checkpoint(
        HuBERTManager.make_sure_tokenizer_installed(model=model[0], local_file=model[1]),
        device,
    )
    print("Loading Encodec...")
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.set_target_bandwidth(6.0)
    encodec_model.to(device)

    wav_hubert = wav.to(device)

    if wav_hubert.shape[0] == 2:  # Stereo to mono if needed
        wav_hubert = wav_hubert.mean(0, keepdim=True)

    print("Extracting semantics...")
    semantic_vectors = hubert_model.forward(wav_hubert, input_sample_hz=sr)
    print("Tokenizing semantics...")
    semantic_tokens = quant_model.get_token(semantic_vectors)
    print("Creating coarse and fine prompts...")
    wav = convert_audio(wav, sr, encodec_model.sample_rate, 1).unsqueeze(0)

    wav = wav.to(device)

    with torch.no_grad():
        encoded_frames = encodec_model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()

    codes = codes.cpu()
    semantic_tokens = semantic_tokens.cpu()

    np.savez(
        out_file,
        semantic_prompt=semantic_tokens,
        fine_prompt=codes,
        coarse_prompt=codes[:2, :],
    )

@app.post("/synthesize", response_class=FileResponse)
def synthesize(
    text: str = Form(...),
    version: str = Form(...),
    speaker_wav: UploadFile = File(...),
    speaker_txt: str = Form(...),
):
    if version not in ["Bark"]:
        return {"error": "Invalid version"}
    shutil.rmtree("/results_bark", ignore_errors=True)
    Path("/results_bark").mkdir()
    if version == "Bark":
        audio_prompt = process_speaker_reference(speaker_wav.file.read())
        create_speaker_npz(audio_prompt, "bark.npz")
        audio_array = generate_audio(
            text, history_prompt="bark.npz"
        )
        output_path = '/results_bark/output.wav'
        sf.write(output_path, audio_array, SAMPLE_RATE)
        return FileResponse(output_path)
        
def process_speaker_reference(speaker_wav: bytes):
    with open("/app/speaker.wav", "wb") as f:
        f.write(speaker_wav)
    return "/app/speaker.wav"

@app.get("/info")
def info():
    return {
         "versions": ["Bark"],
         "requires_text": [False],
    }

@app.get("/ready")
def ready():
    return "ready"