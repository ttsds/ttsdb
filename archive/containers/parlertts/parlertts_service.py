import os
import uuid
import tempfile
from pathlib import Path

import torch
import torchaudio
from transformers import set_seed
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import soundfile as sf
from transformers.generation.configuration_utils import GenerationConfig
import numpy as np
import wespeaker

from load_models import setup

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_closest_description(speaker_file):
    wespeaker_model = wespeaker.load_model('english')
    emb_list = sorted(list(Path("/tmp/embeddings").rglob("*.npy")))
    emb_list = [np.load(str(emb)) for emb in emb_list]
    dists = []
    speaker_emb = wespeaker_model.extract_embedding(speaker_file)
    if isinstance(speaker_emb, torch.Tensor):
        speaker_emb = speaker_emb.cpu().numpy()
    for emb in emb_list:
        dist = np.linalg.norm(emb - speaker_emb)
        dists.append(dist)
    descriptions = sorted(list(Path("/tmp/embeddings").rglob("*.txt")))
    descriptions = [open(str(desc)).read() for desc in descriptions]
    desc = descriptions[np.argmin(dists)]
    if "Or:" in desc:
        desc = desc.split("Or:")
        # randomly choose one of the descriptions
        desc = desc[np.random.randint(0, len(desc))]
    desc = desc.strip()
    return desc

model, tokenizer, feature_extractor, SAMPLING_RATE = None, None, None, None

@app.post("/synthesize", response_class=FileResponse)
def synthesize(
    text: str = Form(...),
    version: str = Form(...),
    speaker_wav: UploadFile = File(...),
    speaker_txt: str = Form(...),
):
    global model, tokenizer, feature_extractor, SAMPLING_RATE
    if version not in ["Mini-v1", "Large-v1"]:
        raise ValueError("Invalid version")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None:
        try:
            if version == "Mini-v1":
                model, tokenizer, feature_extractor, SAMPLING_RATE = setup("parler-tts/parler-tts-mini-v1", device)
            elif version == "Large-v1":
                model, tokenizer, feature_extractor, SAMPLING_RATE = setup("parler-tts/parler-tts-large-v1", device)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                device = "cpu"
                if version == "Mini-v1":
                    model, tokenizer, feature_extractor, SAMPLING_RATE = setup("parler-tts/parler-tts-mini-v1", device)
                elif version == "Large-v1":
                    model, tokenizer, feature_extractor, SAMPLING_RATE = setup("parler-tts/parler-tts-large-v1", device)
            else:
                raise e

    # Create a directory to store results if it doesn't exist
    output_dir = "/results_parler"
    os.makedirs(output_dir, exist_ok=True)

    # Save the uploaded speaker reference audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(speaker_wav.file.read())
        init_audio_file = tmp.name

    description = get_closest_description(init_audio_file)
    print(description)

    init_prompt = speaker_txt
    prompt = text

    # Load and preprocess the initial audio
    init_audio, init_sr = torchaudio.load(init_audio_file)
    init_audio = torchaudio.functional.resample(init_audio, init_sr, 16_000)
    init_audio = torchaudio.functional.resample(init_audio, 16_000, SAMPLING_RATE)
    init_audio = init_audio.mean(0)  # Convert to mono if necessary

    # Encode the initial audio using the feature extractor
    input_values = feature_extractor(
        init_audio, sampling_rate=SAMPLING_RATE, return_tensors="pt"
    )
    padding_mask = input_values.padding_mask.to(device)
    input_values = input_values.input_values.to(device)

    # Tokenize the description and prompts
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    print(init_prompt + " " + prompt)
    prompt_input_ids = tokenizer(
        init_prompt + " " + prompt, return_tensors="pt"
    ).input_ids.to(device)

    set_seed(2)  # For reproducibility

    config = model.generation_config

    # Generate the audio using the Parler-TTS model
    generation = model.generate(
        input_ids=input_ids,
        prompt_input_ids=prompt_input_ids,
        input_values=input_values,
        padding_mask=padding_mask,
        generation_config=config,
    )

    generation = generation[0, input_values.shape[2]:]

    # Save the generated audio to a unique file
    audio_arr = generation.cpu().numpy().squeeze()
    output_filename = f"output_{uuid.uuid4().hex}.wav"
    output_path = os.path.join(output_dir, output_filename)
    sf.write(output_path, audio_arr, SAMPLING_RATE)

    # Clean up the temporary file
    os.remove(init_audio_file)

    return FileResponse(output_path)

@app.get("/info")
def info():
    return {
        "versions": ["Mini-v1", "Large-v1"],
        "requires_text": [True, True],
    }

@app.get("/ready")
def ready():
    return "ready"
