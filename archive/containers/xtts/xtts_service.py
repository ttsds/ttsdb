import sys
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from TTS.api import TTS
import torch
import shutil
import soundfile as sf

os.environ["COQUI_TOS_AGREED"] = "1"

app = FastAPI()

# Load the XTTS v2 model once when the app starts
gpu_available = torch.cuda.is_available()
tts_xtts_v2 = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=gpu_available)

@app.post("/synthesize", response_class=FileResponse)
def synthesize(
    text: str = Form(...),
    version: str = Form(...),
    speaker_wav: UploadFile = File(...),
):
    if version not in ["v2"]:
        return {"error": "Invalid version"}
    output_dir = Path("/results_xttsv2")
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=True)

    # Process speaker_wav
    speaker_wav_path = process_speaker_reference(speaker_wav.file.read())
    output_path = output_dir / "output.wav"

    if version == "v2":
        tts_xtts_v2.tts_to_file(
            text=text,
            file_path=str(output_path),
            speaker_wav=str(speaker_wav_path),
            language="en"
        )
        return FileResponse(str(output_path))

def process_speaker_reference(speaker_wav: bytes):
    speaker_wav_path = Path("/app/speaker.wav")
    with open(speaker_wav_path, "wb") as f:
        f.write(speaker_wav)
    return speaker_wav_path

@app.get("/info")
def info():
    return {
        "versions": ["v2"],
        "requires_text": [False],
    }

@app.get("/ready")
def ready():
    return "ready"