import tempfile
import os
import sys
from pathlib import Path
import subprocess
import shutil
import argparse

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import torchaudio
from tqdm import tqdm
from tools.vqgan.inference import main as encode_audio
from tools.llama.generate import main as generate_semantic_tokens
import soundfile as sf

app = FastAPI()

def codes_to_wav(audio_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encode_audio(
        input_path=Path(audio_path),
        output_path=Path(output_path),
        config_name="firefly_gan_vq",
        checkpoint_path="/app/fish-speech/checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        device=device,
    )


@app.post("/synthesize", response_class=FileResponse)
def synthesize(
    text: str = Form(...),
    version: str = Form(...),
    speaker_wav: UploadFile = File(...),
    speaker_txt: str = Form(...),
):
    if version not in ["Fish"]:
        return {"error": "Invalid version"}
    shutil.rmtree("/results_fish", ignore_errors=True)
    Path("/results_fish").mkdir()
    if version == "Fish":
        audio_prompt = process_speaker_reference(speaker_wav.file.read())
        codes_to_wav(audio_prompt, "/results_fish/fish_fake.wav")
        ref_text = text
        prompt_text = speaker_txt
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generate_semantic_tokens(
            text=ref_text,
            prompt_text=[prompt_text],
            prompt_tokens=[Path(f"/results_fish/fish_fake.npy")],
            num_samples=1,
            max_new_tokens=1024,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            checkpoint_path="/app/fish-speech/checkpoints/fish-speech-1.4",
            device=device,
            compile=False,
            seed=42,
            half=False,
            iterative_prompt=True,
            chunk_length=100,
        )
        codes_to_wav("codes_0.npy", "/results_fish/output.wav")
        return FileResponse("/results_fish/output.wav")
        
def process_speaker_reference(speaker_wav: bytes):
    with open("/app/speaker.wav", "wb") as f:
        f.write(speaker_wav)
    return "/app/speaker.wav"

@app.get("/info")
def info():
    return {
         "versions": ["Fish"],
         "requires_text": [True],
    }

@app.get("/ready")
def ready():
    return "ready"