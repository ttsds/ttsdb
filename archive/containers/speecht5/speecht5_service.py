import os
from pathlib import Path
import shutil
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import soundfile as sf
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)
from speechbrain.pretrained import EncoderClassifier
import torch.nn.functional as F
import tempfile

app = FastAPI()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the TTS models
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

# Load SpeechBrain x-vector speaker embedding model
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    run_opts={"device": device},
    savedir="/tmp/speechbrain_speaker_embedding",
)

def process_speaker_reference(speaker_wav: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(speaker_wav)
        tmp_path = tmp.name
    return tmp_path

@app.post("/synthesize", response_class=FileResponse)
def synthesize(
    text: str = Form(...),
    speaker_wav: UploadFile = File(...),
):
    speaker_wav_bytes = speaker_wav.file.read()
    speaker_wav_path = process_speaker_reference(speaker_wav_bytes)

    # Load and prepare reference audio
    speech_array, sampling_rate = torchaudio.load(speaker_wav_path)
    speech_array = speech_array.squeeze(0).to(device)

    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000).to(device)
        speech_array = resampler(speech_array)
        sampling_rate = 16000

    # Extract speaker embedding
    with torch.no_grad():
        embeddings = classifier.encode_batch(speech_array.unsqueeze(0))
        embeddings = F.normalize(embeddings, dim=2)
        speaker_embedding = embeddings.squeeze(0)  # Shape: (512,)

    # Prepare input text
    inputs = processor(text=text, return_tensors="pt").to(device)

    # Generate speech
    with torch.no_grad():
        speech = tts_model.generate_speech(
            inputs["input_ids"], speaker_embedding, vocoder=vocoder
        )

    # Save output speech
    speech = speech.cpu()
    output_dir = "/results_speecht5"
    shutil.rmtree(output_dir, ignore_errors=True)
    Path(output_dir).mkdir()
    output_path = os.path.join(output_dir, "output.wav")
    sf.write(output_path, speech.numpy(), samplerate=16000)

    # Clean up temporary files
    os.remove(speaker_wav_path)

    return FileResponse(output_path)

@app.get("/info")
def info():
    return {
        "versions": ["SpeechT5"],
        "requires_text": [True],
    }

@app.get("/ready")
def ready():
    return "ready"
