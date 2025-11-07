import sys
import os
import shutil
from pathlib import Path
import soundfile as sf
import tempfile
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse

# Ensure GPT-SoVITS is in the Python path
os.chdir("GPT-SoVITS")
sys.path.append(".")

# Import the get_tts_wav function from GPT-SoVITS
from inference_webui import get_tts_wav

app = FastAPI()

def process_speaker_reference(speaker_wav_bytes: bytes):
    # Write the uploaded speaker WAV bytes to a temporary file
    temp_file = tempfile.mktemp(suffix=".wav")
    with open(temp_file, "wb") as f:
        f.write(speaker_wav_bytes)
    return temp_file

def create_wav(speaker, text):
    # Load the speaker audio
    y, sr = librosa.load(speaker)
    
    # Ensure the audio is at least 3 seconds long
    if len(y) < sr * 3:
        y = np.pad(y, (sr * 3 - len(y), 0), mode="constant")
    # If longer than 8 seconds, take a random 8-second segment
    elif len(y) > sr * 8:
        start = np.random.randint(0, len(y) - sr * 8)
        y = y[start:start + sr * 8]

    # Write the processed speaker audio to a temporary file
    temp_speaker_file = tempfile.mktemp(suffix=".wav")
    sf.write(temp_speaker_file, y, sr)

    # Generate the TTS audio using GPT-SoVITS
    wav_generator = get_tts_wav(str(temp_speaker_file), None, "en", text, "en")

    wavs = []
    for chunk in wav_generator:
        sr, chunk_wav = chunk
        wavs.append(chunk_wav)

    # Clean up the temporary speaker file
    os.remove(temp_speaker_file)

    return sr, np.concatenate(wavs)

@app.post("/synthesize", response_class=FileResponse)
def synthesize(
    text: str = Form(...),
    version: str = Form(...),
    speaker_wav: UploadFile = File(...),
    speaker_txt: str = Form(...),
):
    if version not in ["GPT-SoVITS"]:
        return {"error": "Invalid version"}

    # Process the uploaded speaker reference
    speaker_path = process_speaker_reference(speaker_wav.file.read())

    # Generate the TTS audio
    sr, wav = create_wav(speaker_path, text)

    # Clean up the temporary speaker file
    os.remove(speaker_path)

    # Save the output to a temporary file
    temp_output_file = tempfile.mktemp(suffix=".wav")
    sf.write(temp_output_file, wav, sr)

    # Return the generated audio file
    return FileResponse(temp_output_file, media_type='audio/wav')

@app.get("/info")
def info():
    return {
        "versions": ["GPT-SoVITS"],
        "requires_text": [False],
    }

@app.get("/ready")
def ready():
    return "ready"
