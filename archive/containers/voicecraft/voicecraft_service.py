import os
import sys
import shutil
from pathlib import Path
import tempfile
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import argparse
import random
import numpy as np
import librosa
import soundfile as sf

# Set the working directory and system path
os.chdir("voicecraft")
sys.path.append(".")

import getpass

os.environ["USER"] = getpass.getuser()

from inference_tts_scale import inference_one_sample
from models import voicecraft
from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

@app.post("/synthesize", response_class=FileResponse)
def synthesize(
    text: str = Form(...),
    version: str = Form(...),
    speaker_wav: UploadFile = File(...),
    speaker_txt: str = Form(...),
):

    if version not in ["330M", "830M", "330M_TTSEnhanced", "830M_TTSEnhanced"]:
        return {"error": "Invalid version"}

    # Save the speaker_wav to a temporary file
    speaker_audio_path = "/app/speaker.wav"
    with open(speaker_audio_path, "wb") as f:
        f.write(speaker_wav.file.read())

    # Save the speaker_txt to a variable
    speaker_transcript = speaker_txt

    # Create an output directory
    output_dir = "./results_voicecraft"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique name for the output file
    name = "output.wav"

    # Call the synthesise function
    synthesise(speaker_audio_path, speaker_transcript, text, output_dir, name, version)

    # Return the output audio as a FileResponse
    output_path = os.path.join(output_dir, name)
    return FileResponse(output_path)

def synthesise(orig_audio, orig_transcript, target_transcript, output_dir, name, version="830M_TTSEnhanced"):

    # Set default parameters
    silence_tokens = [1388, 1898, 131]
    codec_audio_sr = 16000
    codec_sr = 50
    top_k = 0
    top_p = 0.8
    temperature = 1
    kvcache = 1
    stop_repetition = -1
    sample_batch_size = 3
    seed = 1
    beam_size = 50
    retry_beam_size = 200
    cut_off_sec = 3.6
    margin = 0.04
    cutoff_tolerance = 1

    voicecraft_name = version

    # Load the model
    if voicecraft_name == "330M":
        voicecraft_name = "giga330M"
    elif voicecraft_name == "830M":
        voicecraft_name = "giga830M"
    elif voicecraft_name == "330M_TTSEnhanced":
        voicecraft_name = "330M_TTSEnhanced"
    elif voicecraft_name == "830M_TTSEnhanced":
        voicecraft_name = "830M_TTSEnhanced"
    model = voicecraft.VoiceCraft.from_pretrained(
        f"pyp1/VoiceCraft_{voicecraft_name.replace('.pth', '')}")
    phn2num = model.args.phn2num
    config = vars(model.args)
    model.to(device)

    encodec_fn = "./pretrained_models/encodec_4cb2048_giga.th"
    if not os.path.exists(encodec_fn):
        os.system(
        f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th -O ./pretrained_models/encodec_4cb2048_giga.th")
    audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=device)

    text_tokenizer = TextTokenizer(backend="espeak")
    
    # Move the audio and transcript to temp folder
    temp_folder = "./demo/temp"
    os.makedirs(temp_folder, exist_ok=True)
    # Copy the orig_audio to temp_folder
    filename = os.path.splitext(os.path.basename(orig_audio))[0]
    temp_audio_path = os.path.join(temp_folder, f"{filename}.wav")
    shutil.copyfile(orig_audio, temp_audio_path)

    # Save the orig_transcript to temp_folder
    with open(f"{temp_folder}/{filename}.txt", "w") as f:
        f.write(orig_transcript)

    # Run MFA to get the alignment
    align_temp = f"{temp_folder}/mfa_alignments"
    alignments = f"{temp_folder}/mfa_alignments/{filename}.csv"
    if not os.path.isfile(alignments):
        os.system(f"mfa align -v --clean -j 1 --output_format csv {temp_folder} \
                english_us_arpa english_us_arpa {align_temp} --beam {beam_size} --retry_beam {retry_beam_size}")

    # Find cutoff point
    cut_off_sec, cut_off_word_idx = find_closest_word_boundary(alignments, cut_off_sec, margin, cutoff_tolerance)

    # Read the original transcript
    orig_split = orig_transcript.split(" ")

    # Make sure cut_off_word_idx is valid
    cut_off_word_idx = min(cut_off_word_idx, len(orig_split) - 1)

    # Create the target transcript
    target_transcript = " ".join(orig_split[:cut_off_word_idx+1]) + " " + target_transcript

    # Get the duration of the audio
    info = torchaudio.info(temp_audio_path)
    audio_dur = info.num_frames / info.sample_rate

    cut_off_sec = min(cut_off_sec, audio_dur)
    prompt_end_frame = int(cut_off_sec * info.sample_rate)

    # Seed everything
    seed_everything(seed)

    # Inference
    decode_config = {'top_k': top_k, 'top_p': top_p, 'temperature': temperature, 'stop_repetition': stop_repetition, 'kvcache': kvcache,
                    "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr, "silence_tokens": silence_tokens, "sample_batch_size": sample_batch_size}

    concated_audio, gen_audio = inference_one_sample(model, argparse.Namespace(
        **config), phn2num, text_tokenizer, audio_tokenizer, temp_audio_path, target_transcript, device, decode_config, prompt_end_frame)

    # Save the generated audio
    concated_audio, gen_audio = concated_audio[0].cpu(), gen_audio[0].cpu()

    # Save the audio
    seg_save_fn_gen = f"{output_dir}/{name}"
    torchaudio.save(seg_save_fn_gen, gen_audio, codec_audio_sr)

def find_closest_word_boundary(alignments, cut_off_sec, margin, cutoff_tolerance=1):
    with open(alignments, 'r') as file:
        # skip header
        next(file)
        cutoff_time = None
        cutoff_index = None
        cutoff_time_best = None
        cutoff_index_best = None
        lines = [l for l in file.readlines() if "words" in l]
        try:
            for i, line in enumerate(lines):
                end = float(line.strip().split(',')[1])
                if end >= cut_off_sec and cutoff_time == None:
                    cutoff_time = end
                    cutoff_index = i
                if end >= cut_off_sec and end < cut_off_sec + cutoff_tolerance and float(lines[i+1].strip().split(',')[0]) - end >= margin:
                        cutoff_time_best = end + margin * 2 / 3
                        cutoff_index_best = i
                        break
            if cutoff_time_best != None:
                cutoff_time = cutoff_time_best
                cutoff_index = cutoff_index_best
        except:
            pass
        if cutoff_time == None:
            cutoff_time = float(lines[-1].strip().split(',')[1])
            cutoff_index = len(lines) - 1
        return cutoff_time, cutoff_index

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

@app.get("/info")
def info():
    return {
         "versions": ["330M", "830M", "330M_TTSEnhanced", "830M_TTSEnhanced"],
         "requires_text": [True, True, True, True],
    }

@app.get("/ready")
def ready():
    return "ready"