import os
import uuid
import tempfile

import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import soundfile as sf

from tortoise import api
from tortoise import utils

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

tts = None

@app.post("/synthesize", response_class=FileResponse)
def synthesize(
    text: str = Form(...),
    version: str = Form(...),
    speaker_wav: UploadFile = File(...),
):
    global tts
    if version != "tortoise":
        raise ValueError("Invalid version")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if tts is None:
        try:
            tts = api.TextToSpeech(kv_cache=True)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                device = "cpu"
                tts = api.TextToSpeech(kv_cache=True)
            else:
                raise e

    # Create a directory to store results if it doesn't exist
    output_dir = "/results_tortoise"
    os.makedirs(output_dir, exist_ok=True)

    # Save the uploaded speaker reference audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(speaker_wav.file.read())
        init_audio_file = tmp.name

    # Load and preprocess the initial audio
    reference_clips = [utils.audio.load_audio(init_audio_file, 22050)]
    pcm_audio = tts.tts_with_preset(text, voice_samples=reference_clips, preset='fast')

    # Save the output audio to a file
    output_path = "/results_tortoise/output.wav"
    sf.write(output_path, pcm_audio[0][0], 24000)

    return FileResponse(output_path)

@app.get("/info")
def info():
    return {
        "versions": ["tortoise"],
        "requires_text": [False],
    }

@app.get("/ready")
def ready():
    return "ready"
