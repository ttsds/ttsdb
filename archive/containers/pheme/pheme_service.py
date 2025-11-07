import tempfile
import os
import sys
from pathlib import Path
import shutil
import argparse

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse

import numpy as np
import soundfile as sf
import torch
from einops import rearrange
from librosa.util import normalize
from pyannote.audio import Inference
from transformers import GenerationConfig, T5ForConditionalGeneration

os.chdir("pheme")
sys.path.append(".")

import constants as c
from data.collation import get_text_semantic_token_collater
from data.semantic_dataset import TextTokenizer
from modules.s2a_model import Pheme
from modules.vocoder import VocoderType
from modules.speech_tokenizer import SpeechTokenizer

app = FastAPI()

# Initialize arguments
args = argparse.Namespace()
args.manifest_path = "demo/manifest.json"
args.outputdir = "demo/"
args.featuredir = "demo/"
args.text_tokens_file = "ckpt/unique_text_tokens.k2symbols"
args.t2s_path = "ckpt/ckpt/t2s/"
args.s2a_path = "ckpt/ckpt/s2a/s2a.ckpt"
args.target_sample_rate = 16000
args.temperature = 0.7
args.top_k = 210
args.voice = "male_voice"

# Initialize PhemeClient
class PhemeClient():
    def __init__(self, args):
        self.args = args
        self.outputdir = args.outputdir
        self.target_sample_rate = args.target_sample_rate
        self.featuredir = Path(args.featuredir).expanduser()
        self.collater = get_text_semantic_token_collater(args.text_tokens_file)
        self.phonemizer = TextTokenizer()
    
        # T2S model
        self.t2s = T5ForConditionalGeneration.from_pretrained(args.t2s_path)
        self.t2s.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.t2s.eval()

        # S2A model
        self.s2a = Pheme.load_from_checkpoint(args.s2a_path)
        self.s2a.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.s2a.eval()

        # Vocoder
        vocoder = VocoderType["SPEECHTOKENIZER"].get_vocoder(None, None)
        self.vocoder = vocoder.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocoder.eval()

        self.spkr_embedding = Inference(
            "pyannote/embedding",
            window="whole",
            use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN", None),
        )

        self.speech_tokenizer = SpeechTokenizer(
            config_path="ckpt/speechtokenizer/config.json",
            ckpt_path="ckpt/speechtokenizer/SpeechTokenizer.pt",
            device=('cuda' if torch.cuda.is_available() else 'cpu')
        )

    def infer(
        self, text, prompt_file_path, initial_text="", temperature=0.7,
        top_k=210, max_new_tokens=750,
    ):
        sampling_config = GenerationConfig.from_pretrained(
            self.args.t2s_path,
            top_k=top_k,
            num_beams=1,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True
        )

        text = initial_text + " " + text

        audio_array = self.generate_audio(
            text, sampling_config, prompt_file_path)

        return audio_array

    def generate_audio(self, text, sampling_config, prompt_file_path):
        output_semantic = self.infer_text(
            text, sampling_config, prompt_file_path
        )

        codes = self.infer_acoustic(output_semantic, prompt_file_path)

        audio_array = self.vocoder.decode(codes)
        audio_array = rearrange(audio_array, "1 1 T -> T").cpu().numpy()

        return audio_array

    def infer_text(self, text, sampling_config, prompt_file_path):
        # copy prompt file to input folder
        Path("/results_pheme/input").mkdir(parents=True, exist_ok=True)
        if Path("/results_pheme/output").exists():
            shutil.rmtree("/results_pheme/output")
        shutil.copy(prompt_file_path, "/results_pheme/input")
        self.speech_tokenizer.encode_file(
            folder_path="/results_pheme/input",
            destination_folder="/results_pheme/output",
            filename=Path(prompt_file_path).name
        )
        print(
            [
                f for f in Path("/results_pheme/").rglob("*")
            ]
        )
        semantic_prompt = np.load(f"/results_pheme/output/semantic/{Path(prompt_file_path).stem}.npy")
        phones_seq = self.phonemizer(text)[0]
        input_ids = self.collater([phones_seq])
        input_ids = input_ids.type(torch.IntTensor).to('cuda' if torch.cuda.is_available() else 'cpu')

        labels = [str(lbl) for lbl in semantic_prompt]
        labels = self.collater([labels])[:, :-1]
        decoder_input_ids = labels.to('cuda' if torch.cuda.is_available() else 'cpu').long()

        counts = 1E10
        while (counts > 100):  # MAX_TOKEN_COUNT = 100
            output_ids = self.t2s.generate(
                input_ids, decoder_input_ids=decoder_input_ids,
                generation_config=sampling_config).sequences
            
            # check repetitiveness
            _, counts = torch.unique_consecutive(output_ids, return_counts=True)
            counts = max(counts).item()

        output_semantic = self.lazy_decode(
            output_ids[0], self.collater.idx2token)

        # remove the prompt
        return output_semantic[len(semantic_prompt):].reshape(1, -1)

    def lazy_decode(self, decoder_output, symbol_table):
        semantic_tokens = map(lambda x: symbol_table[x], decoder_output)
        semantic_tokens = [int(x) for x in semantic_tokens if x.isdigit()]

        return np.array(semantic_tokens)

    def infer_acoustic(self, output_semantic, prompt_file_path):
        semantic_tokens = output_semantic.reshape(1, -1)
        acoustic_tokens = np.full(
            [semantic_tokens.shape[1], 7], fill_value=c.PAD)

        element_id_prompt = Path(prompt_file_path).stem
        acoustic_prompt = np.load(f"/results_pheme/output/acoustic/{element_id_prompt}.npy").squeeze().T
        semantic_prompt = np.load(f"/results_pheme/output/semantic/{element_id_prompt}.npy")[None]

        # Prepend prompt
        acoustic_tokens = np.concatenate(
            [acoustic_prompt, acoustic_tokens], axis=0)
        semantic_tokens = np.concatenate([
            semantic_prompt, semantic_tokens], axis=1)

        # Add speaker
        acoustic_tokens = np.pad(
            acoustic_tokens, [[1, 0], [0, 0]], constant_values=c.SPKR_1)
        semantic_tokens = np.pad(
            semantic_tokens, [[0,0], [1, 0]], constant_values=c.SPKR_1)

        speaker_emb = None
        if self.s2a.hp.use_spkr_emb:
            speaker_emb = self._load_speaker_emb(prompt_file_path)
            speaker_emb = np.repeat(
                speaker_emb, semantic_tokens.shape[1], axis=0)
            speaker_emb = torch.from_numpy(speaker_emb).to('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            speaker_emb = None

        acoustic_tokens = torch.from_numpy(
            acoustic_tokens).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu').long()
        semantic_tokens = torch.from_numpy(semantic_tokens).to('cuda' if torch.cuda.is_available() else 'cpu').long()
        start_t = torch.tensor(
            [acoustic_prompt.shape[0]], dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu')
        length = torch.tensor([
            semantic_tokens.shape[1]], dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu')

        codes = self.s2a.model.inference(
            acoustic_tokens,
            semantic_tokens,
            start_t=start_t,
            length=length,
            maskgit_inference=True,
            speaker_emb=speaker_emb
        )

        # Remove the prompt
        synth_codes = codes[:, :, start_t:]
        synth_codes = rearrange(synth_codes, "b c t -> c b t")

        return synth_codes

    def _load_speaker_emb(self, prompt_file_path):
        wav, _ = sf.read(prompt_file_path)
        audio = normalize(wav) * 0.95
        speaker_emb = self.spkr_embedding(
            {
                "waveform": torch.FloatTensor(audio).unsqueeze(0),
                "sample_rate": self.target_sample_rate
            }
        ).reshape(1, -1)

        return speaker_emb

pheme_client = PhemeClient(args)

@app.post("/synthesize", response_class=FileResponse)
def synthesize(
    text: str = Form(...),
    version: str = Form(...),
    speaker_wav: UploadFile = File(...),
    speaker_txt: str = Form(...),
):
    if version not in ["Pheme"]:
        return {"error": "Invalid version"}
    
    # Process speaker_wav
    if Path("/results_pheme").exists():
        shutil.rmtree("/results_pheme")
    Path("/results_pheme").mkdir(parents=True, exist_ok=True)
    temp_wav_file = Path("/results_pheme") / "speaker.wav"
    speaker_wav_contents = speaker_wav.file.read()
    with open(temp_wav_file, "wb") as f:
        f.write(speaker_wav_contents)
    speaker_wav_path = str(temp_wav_file)

    # Generate audio
    audio_array = pheme_client.infer(
        text, 
        prompt_file_path=speaker_wav_path, 
        initial_text=speaker_txt, 
        temperature=args.temperature, 
        top_k=args.top_k
    )
    # Save audio_array to file
    temp_output_file = Path("/results_pheme") / "output.wav"
    sf.write(temp_output_file.name, audio_array, args.target_sample_rate)
    output_path = temp_output_file.name
    return FileResponse(output_path, media_type="audio/wav", filename="output.wav")

@app.get("/info")
def info():
    return {
        "versions": ["Pheme"],
        "requires_text": [True],
    }

@app.get("/ready")
def ready():
    return "ready"