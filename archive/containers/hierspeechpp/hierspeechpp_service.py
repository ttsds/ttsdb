import sys
from pathlib import Path
from argparse import Namespace
import os

sys.path.append('/app/HierSpeechpp')
os.chdir('/app/HierSpeechpp')

import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import inference
import utils

app = FastAPI()

def process_speaker_reference(speaker_wav: bytes):
    with open("/app/speaker.wav", "wb") as f:
        f.write(speaker_wav)
    return "/app/speaker.wav"

def process_text(text: str):
    with open("/app/text.txt", "w") as f:
        f.write(text)
    return "/app/text.txt"

@app.post("/synthesize", response_class=FileResponse)
def synthesize(
    text: str = Form(...),
    version: str = Form(...),
    speaker_wav: UploadFile = File(...),
):
    global device, hps, hps_t2w2v,h_sr,h_sr48, hps_denoiser
    if version not in ["v1.1"]:
        raise ValueError("Invalid version")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a directory to store results if it doesn't exist
    output_dir = "/results_hierspeechpp"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_prompt = process_speaker_reference(speaker_wav.file.read())
    input_txt = process_text(text)

    a = {
        'input_prompt': input_prompt,
        'input_txt': input_txt,
        'output_dir': output_dir,
        'ckpt': '../models/main/hierspeechpp_v1.1_ckpt.pth',
        'ckpt_text2w2v': '../models/ttv/ttv_lt960_ckpt.pth',
        'ckpt_sr': './speechsr24k/G_340000.pth',
        'ckpt_sr48': './speechsr48k/G_100000.pth',
        'denoiser_ckpt': 'denoiser/g_best',
        'scale_norm': 'max',
        'output_sr': 48000,
        'noise_scale_ttv': 0.333,
        'noise_scale_vc': 0.333,
        'denoise_ratio': 0.8
    }

    a = Namespace(**a)

    hps = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt)[0], 'config.json'))
    hps_t2w2v = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt_text2w2v)[0], 'config.json'))
    h_sr = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt_sr)[0], 'config.json') )
    h_sr48 = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt_sr48)[0], 'config.json') )
    hps_denoiser = utils.get_hparams_from_file(os.path.join(os.path.split(a.denoiser_ckpt)[0], 'config.json'))

    # make sure global variables are defined in inference
    inference.set_globals(device, hps, hps_t2w2v, h_sr, h_sr48, hps_denoiser)

    inference.inference(a)

    output_path = Path(output_dir).rglob("*.wav").__next__()

    return FileResponse(output_path)

@app.get("/info")
def info():
    return {
        "versions": ["v1.1"],
        "requires_text": [False],
    }

@app.get("/ready")
def ready():
    return "ready"