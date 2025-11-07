import tempfile
import os
import sys
from pathlib import Path
import shutil

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
import torchaudio
import soundfile as sf
from huggingface_hub import hf_hub_download
import tqdm

# Import F5 TTS modules
from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
)

app = FastAPI()

# Load vocoder
vocos = load_vocoder()

# Load models
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_model_path = hf_hub_download(repo_id="SWivid/F5-TTS", filename="F5TTS_Base/model_1200000.safetensors")
F5TTS_ema_model = load_model(DiT, F5TTS_model_cfg, F5TTS_model_path)

E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
E2TTS_model_path = hf_hub_download(repo_id="SWivid/E2-TTS", filename="E2TTS_Base/model_1200000.safetensors")
E2TTS_ema_model = load_model(UNetT, E2TTS_model_cfg, E2TTS_model_path)


@app.post("/synthesize", response_class=FileResponse)
def synthesize(
    text: str = Form(...),
    version: str = Form(...),
    speaker_wav: UploadFile = File(...),
    speaker_txt: str = Form(...),
):
    # Save the uploaded reference audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref_audio:
        tmp_ref_audio.write(speaker_wav.file.read())
        ref_audio_path = tmp_ref_audio.name

    # Process the reference audio and text
    ref_audio_waveform, ref_text_processed = preprocess_ref_audio_text(
        ref_audio_path, speaker_txt, show_info=print
    )

    # Select the model
    if version == "F5-TTS":
        ema_model = F5TTS_ema_model
    elif version == "E2-TTS":
        ema_model = E2TTS_ema_model
    else:
        return {"error": "Invalid model selection"}

    # Perform inference
    final_wave, final_sample_rate, _ = infer_process(
        ref_audio_waveform,
        ref_text_processed,
        text,
        ema_model,
        show_info=print,
        progress=tqdm,
    )

    # Save the output audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_output_audio:
        sf.write(tmp_output_audio.name, final_wave, final_sample_rate)
        output_audio_path = tmp_output_audio.name

    # Return the output audio file
    return FileResponse(output_audio_path, media_type='audio/wav', filename='output.wav')

@app.get("/info")
def info():
    return {
        "versions": ["F5-TTS", "E2-TTS"],
        "requires_text": [True, True],
    }

@app.get("/ready")
def ready():
    return "ready"
