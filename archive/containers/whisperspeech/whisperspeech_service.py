from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
from whisperspeech.pipeline import Pipeline
import soundfile as sf
import torch
import numpy as np

app = FastAPI()

def process_speaker_reference(speaker_wav_bytes: bytes):
    speaker_audio_path = "/app/speaker.wav"
    with open(speaker_audio_path, "wb") as f:
        f.write(speaker_wav_bytes)
    return speaker_audio_path

def step_callback(*step):
    print(f"Step {step}")

pipe = None

@app.post("/synthesize", response_class=FileResponse)
def synthesize(
    text: str = Form(...),
    version: str = Form(...),
    speaker_wav: UploadFile = File(...),
):
    global pipe
    print(f"Received request for {version}")
    if version not in ["Tiny", "Base", "Small", "Medium"]:
        return {"error": "Invalid version"}
    version = version.lower()
    if pipe is None:
        if version != "medium":
            pipe = Pipeline(
                optimize=False,
                torch_compile=False,
                s2a_ref=f"collabora/whisperspeech:s2a-q4-{version}-en+pl.model",
                t2s_ref=f"collabora/whisperspeech:t2s-{version}-en+pl.model"
            )
        else:
            pipe = Pipeline(
                optimize=False,
                torch_compile=False,
                s2a_ref="collabora/whisperspeech:s2a-v1.95-medium-7lang.model",
                t2s_ref="collabora/whisperspeech:t2s-v1.95-medium-7lang.model"
            )
        pipe.t2s.optimize(max_batch_size=1, dtype=torch.float32, torch_compile=False)
        pipe.s2a.optimize(max_batch_size=1, dtype=torch.float32, torch_compile=False)
    # Clean up previous results
    shutil.rmtree("/results_whisperspeech", ignore_errors=True)
    Path("/results_whisperspeech").mkdir(parents=True, exist_ok=True)
    # Process speaker reference
    print("Processing speaker reference")
    speaker_audio_path = process_speaker_reference(speaker_wav.file.read())
    output_path = "/results_whisperspeech/output.wav"
    # Generate speech using WhisperSpeech
    print("Generating speech")
    # hack to fix faulty inference code
    pipe.s2a.dtype = torch.float32
    pipe.generate_to_file(
        output_path,
        text=text,
        lang='en',
        step_callback=step_callback,
        speaker=speaker_audio_path
    )
    return FileResponse(output_path)

@app.get("/info")
def info():
    return {
        "versions": ["Tiny", "Base", "Small", "Medium"],
        "requires_text": [False, False, False, False],
    }

@app.get("/ready")
def ready():
    return "ready"