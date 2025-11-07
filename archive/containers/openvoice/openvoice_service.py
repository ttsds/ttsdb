import os
import sys
import shutil
import torch
from pathlib import Path
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from melo.api import TTS

app = FastAPI()

def process_speaker_reference(speaker_wav: UploadFile, tmpdir):
    speaker_path = os.path.join(tmpdir, 'speaker.wav')
    with open(speaker_path, 'wb') as f:
        f.write(speaker_wav.file.read())
    return speaker_path

@app.post("/synthesize", response_class=FileResponse)
def synthesize(
    text: str = Form(...),
    version: str = Form(...),
    speaker_wav: UploadFile = File(...),
):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    if Path("/results_openvoice").exists():
        shutil.rmtree("/results_openvoice")
    Path("/results_openvoice").mkdir(parents=True, exist_ok=True)
    speaker_path = process_speaker_reference(speaker_wav, "/results_openvoice")
    if version == "OpenVoice_v1":
        ckpt_base_v1 = 'openvoice_v1/checkpoints/base_speakers/EN'
        ckpt_converter_v1 = 'openvoice_v1/checkpoints/converter'

        base_speaker_tts_v1 = BaseSpeakerTTS(f'{ckpt_base_v1}/config.json', device=device)
        base_speaker_tts_v1.load_ckpt(f'{ckpt_base_v1}/checkpoint.pth')

        tone_color_converter_v1 = ToneColorConverter(f'{ckpt_converter_v1}/config.json', device=device)
        tone_color_converter_v1.load_ckpt(f'{ckpt_converter_v1}/checkpoint.pth')

        source_se_v1 = torch.load(f'{ckpt_base_v1}/en_default_se.pth', map_location=device)
        target_se, audio_name = se_extractor.get_se(
            speaker_path,
            tone_color_converter_v1,
            target_dir="/results_openvoice",
            vad=True
        )
        # Generate base speech
        src_path = os.path.join("/results_openvoice", 'tmp.wav')
        base_speaker_tts_v1.tts(text, src_path, speaker='default', language='English', speed=1.0)
        # Run tone color converter
        output_path = os.path.join("/results_openvoice", 'output.wav')
        encode_message = "@MyShell"
        tone_color_converter_v1.convert(
            audio_src_path=src_path, 
            src_se=source_se_v1, 
            tgt_se=target_se, 
            output_path=output_path,
            message=encode_message)
        return FileResponse(output_path)
    elif version == "OpenVoice_v2":
        ckpt_converter_v2 = 'openvoice_v2/checkpoints_v2/converter'
        tone_color_converter_v2 = ToneColorConverter(f'{ckpt_converter_v2}/config.json', device=device)
        tone_color_converter_v2.load_ckpt(f'{ckpt_converter_v2}/checkpoint.pth')

        language_v2 = "EN_NEWEST"
        source_se_v2 = torch.load(f'openvoice_v2/checkpoints_v2/base_speakers/ses/en-newest.pth', map_location=device)

        model_v2 = TTS(language=language_v2, device=device)
        speaker_id_v2 = 0
        target_se, audio_name = se_extractor.get_se(
            speaker_path,
            tone_color_converter_v2,
            vad=True
        )
        # Generate base speech
        src_path = os.path.join("/results_openvoice", 'tmp.wav')
        model_v2.tts_to_file(text, speaker_id_v2, src_path, speed=1.0)
        # Run tone color converter
        output_path = os.path.join("/results_openvoice", 'output.wav')
        encode_message = "@MyShell"
        tone_color_converter_v2.convert(
            audio_src_path=src_path, 
            src_se=source_se_v2, 
            tgt_se=target_se, 
            output_path=output_path,
            message=encode_message
        )
        return FileResponse(output_path)
    else:
        return {"error": "Invalid version"}

@app.get("/info")
def info():
    return {
        "versions": ["OpenVoice_v1", "OpenVoice_v2"],
        "requires_text": [False, False],
    }

@app.get("/ready")
def ready():
    return "ready"