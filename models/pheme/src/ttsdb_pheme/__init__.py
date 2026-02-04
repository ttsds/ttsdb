from __future__ import annotations

# ruff: noqa: E402
import logging
import os
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from einops import rearrange
from librosa.util import normalize
from ttsdb_core import (
    AudioOutput,
    VoiceCloningTTSBase,
    get_variant_checkpoint_dir,
    setup_vendor_path,
)

setup_vendor_path("ttsdb_pheme")

import constants as c
from data.collation import get_text_semantic_token_collater
from data.semantic_dataset import TextTokenizer
from modules.s2a_model import Pheme as PhemeModel
from modules.speech_tokenizer import SpeechTokenizer
from modules.vocoder import VocoderType
from pyannote.audio import Inference, Model
from transformers import GenerationConfig, T5ForConditionalGeneration

__all__ = ["Pheme"]

MAX_TOKEN_COUNT = 100

logging.basicConfig(level=logging.DEBUG)


def _ensure_torchaudio_info() -> None:
    import torchaudio

    if hasattr(torchaudio, "info"):
        return

    def _info(path: str):
        with sf.SoundFile(path) as handle:
            return types.SimpleNamespace(
                num_frames=len(handle),
                sample_rate=handle.samplerate,
                num_channels=handle.channels,
            )

    torchaudio.info = _info


class _PhemeEngine:
    def __init__(
        self,
        *,
        text_tokens_file: Path,
        t2s_path: Path,
        generation_config_dir: Path | None,
        s2a_path: Path,
        sp_tokenizer_dir: Path,
        pyannote_model_path: Path,
        device: torch.device,
        target_sample_rate: int,
    ):
        self.t2s_path = t2s_path
        self.generation_config_dir = generation_config_dir
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.collater = get_text_semantic_token_collater(str(text_tokens_file))
        self.phonemizer = TextTokenizer()

        self.t2s = T5ForConditionalGeneration.from_pretrained(str(t2s_path))
        self.t2s.to(self.device)
        self.t2s.eval()

        self.s2a = PhemeModel.load_from_checkpoint(str(s2a_path))
        self.s2a.to(device=self.device)
        self.s2a.eval()

        vocoder = VocoderType["SPEECHTOKENIZER"].get_vocoder(
            str(sp_tokenizer_dir / "SpeechTokenizer.pt"),
            str(sp_tokenizer_dir / "config.json"),
        )
        self.vocoder = vocoder.to(self.device)
        self.vocoder.eval()

        pyannote_model = Model.from_pretrained(str(pyannote_model_path))
        self.spkr_embedding = Inference(
            pyannote_model,
            window="whole",
            device=self.device,
        )

        self.speech_tokenizer = SpeechTokenizer(
            ckpt_path=str(sp_tokenizer_dir / "SpeechTokenizer.pt"),
            config_path=str(sp_tokenizer_dir / "config.json"),
            device=self.device,
        )

    @staticmethod
    def _lazy_decode(decoder_output, symbol_table):
        semantic_tokens = map(lambda x: symbol_table[x], decoder_output)
        semantic_tokens = [int(x) for x in semantic_tokens if x.isdigit()]
        return np.array(semantic_tokens)

    def _load_speaker_emb(self, prompt_path: Path) -> np.ndarray:
        wav, _ = sf.read(prompt_path)
        audio = normalize(wav) * 0.95
        speaker_emb = self.spkr_embedding(
            {
                "waveform": torch.FloatTensor(audio).unsqueeze(0),
                "sample_rate": self.target_sample_rate,
            }
        ).reshape(1, -1)
        return speaker_emb

    def build_prompt_features(
        self, prompt_path: Path, work_dir: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        input_dir = work_dir / "input"
        output_dir = work_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.speech_tokenizer.encode_file(
            folder_path=str(input_dir),
            destination_folder=str(output_dir),
            filename=prompt_path.name,
        )

        acoustic_path = output_dir / "acoustic" / f"{prompt_path.stem}.npy"
        semantic_path = output_dir / "semantic" / f"{prompt_path.stem}.npy"

        acoustic_prompt = np.load(acoustic_path).squeeze().T
        semantic_prompt = np.load(semantic_path).squeeze()
        return acoustic_prompt, semantic_prompt

    def infer_text(
        self,
        text: str,
        semantic_prompt: np.ndarray,
        sampling_config: GenerationConfig,
    ) -> np.ndarray:
        phones_seq = self.phonemizer(text)[0]
        input_ids = self.collater([phones_seq])
        input_ids = input_ids.type(torch.IntTensor).to(self.device)

        labels = [str(lbl) for lbl in semantic_prompt]
        labels = self.collater([labels])[:, :-1]
        decoder_input_ids = labels.to(self.device).long()
        logging.debug("decoder_input_ids: %s", decoder_input_ids)

        counts = 1e10
        while counts > MAX_TOKEN_COUNT:
            output_ids = self.t2s.generate(
                input_ids,
                decoder_input_ids=decoder_input_ids,
                generation_config=sampling_config,
            ).sequences

            _, counts = torch.unique_consecutive(output_ids, return_counts=True)
            counts = max(counts).item()

        output_semantic = self._lazy_decode(output_ids[0], self.collater.idx2token)
        return output_semantic[len(semantic_prompt) :].reshape(1, -1)

    def infer_acoustic(
        self,
        output_semantic: np.ndarray,
        acoustic_prompt: np.ndarray,
        semantic_prompt: np.ndarray,
        prompt_path: Path,
    ) -> torch.Tensor:
        semantic_tokens = output_semantic.reshape(1, -1)
        acoustic_tokens = np.full([semantic_tokens.shape[1], 7], fill_value=c.PAD)

        acoustic_tokens = np.concatenate([acoustic_prompt, acoustic_tokens], axis=0)
        semantic_tokens = np.concatenate([semantic_prompt[None], semantic_tokens], axis=1)

        acoustic_tokens = np.pad(acoustic_tokens, [[1, 0], [0, 0]], constant_values=c.SPKR_1)
        semantic_tokens = np.pad(semantic_tokens, [[0, 0], [1, 0]], constant_values=c.SPKR_1)

        speaker_emb = None
        if self.s2a.hp.use_spkr_emb:
            speaker_emb = self._load_speaker_emb(prompt_path)
            speaker_emb = np.repeat(speaker_emb, semantic_tokens.shape[1], axis=0)
            speaker_emb = torch.from_numpy(speaker_emb).to(self.device)

        acoustic_tokens = torch.from_numpy(acoustic_tokens).unsqueeze(0).to(self.device).long()
        semantic_tokens = torch.from_numpy(semantic_tokens).to(self.device).long()
        start_index = int(acoustic_prompt.shape[0])
        start_t = torch.tensor([start_index], dtype=torch.long, device=self.device)
        length = torch.tensor([semantic_tokens.shape[1]], dtype=torch.long, device=self.device)

        codes = self.s2a.model.inference(
            acoustic_tokens,
            semantic_tokens,
            start_t=start_t,
            length=length,
            maskgit_inference=True,
            speaker_emb=speaker_emb,
        )

        synth_codes = codes[:, :, start_index:]
        synth_codes = rearrange(synth_codes, "b c t -> c b t")
        return synth_codes

    def generate_audio(
        self,
        text: str,
        semantic_prompt: np.ndarray,
        acoustic_prompt: np.ndarray,
        prompt_path: Path,
        sampling_config: GenerationConfig,
    ) -> np.ndarray:
        start_time = time.time()
        output_semantic = self.infer_text(text, semantic_prompt, sampling_config)
        logging.debug("semantic_tokens: %s", time.time() - start_time)

        start_time = time.time()
        codes = self.infer_acoustic(output_semantic, acoustic_prompt, semantic_prompt, prompt_path)
        logging.debug("acoustic_tokens: %s", time.time() - start_time)

        start_time = time.time()
        audio_array = self.vocoder.decode(codes)
        audio_array = rearrange(audio_array, "1 1 T -> T").cpu().numpy()
        logging.debug("vocoder time: %s", time.time() - start_time)

        return audio_array

    @torch.no_grad()
    def infer(
        self,
        *,
        text: str,
        ref_text: str,
        semantic_prompt: np.ndarray,
        acoustic_prompt: np.ndarray,
        prompt_path: Path,
        temperature: float,
        top_k: int,
        max_new_tokens: int,
    ) -> np.ndarray:
        sampling_config = None
        if self.generation_config_dir and self.generation_config_dir.exists():
            sampling_config = GenerationConfig.from_pretrained(
                str(self.generation_config_dir),
                top_k=top_k,
                num_beams=1,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=1,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )

        if sampling_config is None:
            sampling_config = GenerationConfig(
                top_k=top_k,
                num_beams=1,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=1,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )

        combined_text = f"{ref_text} {text}".strip()
        return self.generate_audio(
            combined_text,
            semantic_prompt,
            acoustic_prompt,
            prompt_path,
            sampling_config,
        )


class Pheme(VoiceCloningTTSBase):
    """Pheme voice cloning TTS model."""

    _package_name = "ttsdb_pheme"
    SAMPLE_RATE = 16000

    def _iter_torch_modules(self):
        modules = []
        for candidate in [
            getattr(self, "t2s", None),
            getattr(self, "s2a", None),
            getattr(self, "vocoder", None),
        ]:
            if isinstance(candidate, torch.nn.Module):
                modules.append(candidate)
        return modules

    def _load_model(self, load_path: str):
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

        _ensure_torchaudio_info()

        base = Path(load_path)
        variant_dir = get_variant_checkpoint_dir(base, config=self.model_config)

        shared = base / "shared"
        text_tokens_file = shared / "uslm" / "USLM_libritts" / "unique_text_tokens.k2symbols"
        sp_tokenizer_dir = shared / "speechtokenizer" / "speechtokenizer_hubert_avg"
        pyannote_model_path = shared / "pyannote_embedding" / "pytorch_model.bin"

        t2s_path = variant_dir / "t2s"
        s2a_path = variant_dir / "s2a.ckpt"
        generation_config_dir = None
        if (variant_dir / "generation_config.json").exists():
            generation_config_dir = variant_dir
        elif (base / "generation_config.json").exists():
            generation_config_dir = base

        force_cpu = os.environ.get("PHEME_FORCE_CPU", "").lower() in {"1", "true", "yes"}
        device = torch.device("cpu")
        if torch.cuda.is_available() and not force_cpu:
            try:
                major, _minor = torch.cuda.get_device_capability()
                if major >= 7:
                    device = torch.device("cuda")
            except Exception:
                device = torch.device("cuda")

        try:
            self.engine = _PhemeEngine(
                text_tokens_file=text_tokens_file,
                t2s_path=t2s_path,
                generation_config_dir=generation_config_dir,
                s2a_path=s2a_path,
                sp_tokenizer_dir=sp_tokenizer_dir,
                pyannote_model_path=pyannote_model_path,
                device=device,
                target_sample_rate=self.SAMPLE_RATE,
            )
        except RuntimeError as exc:
            if device.type == "cuda" and "no kernel image" in str(exc).lower():
                device = torch.device("cpu")
                self.engine = _PhemeEngine(
                    text_tokens_file=text_tokens_file,
                    t2s_path=t2s_path,
                    generation_config_dir=generation_config_dir,
                    s2a_path=s2a_path,
                    sp_tokenizer_dir=sp_tokenizer_dir,
                    pyannote_model_path=pyannote_model_path,
                    device=device,
                    target_sample_rate=self.SAMPLE_RATE,
                )
            else:
                raise

        self.device = device

        self.t2s = self.engine.t2s
        self.s2a = self.engine.s2a
        self.vocoder = self.engine.vocoder

        return self.t2s

    def _synthesize(self, text, reference_audio, reference_sample_rate, **kwargs) -> AudioOutput:
        ref_text = kwargs.get("text_reference") or kwargs.get("reference_text") or ""
        if not str(ref_text).strip():
            raise ValueError("text_reference is required for Pheme")

        ref = np.asarray(reference_audio, dtype=np.float32)
        if ref.ndim > 1:
            ref = ref.mean(axis=0)

        if reference_sample_rate != self.SAMPLE_RATE:
            import torchaudio

            ref_tensor = torch.tensor(ref, dtype=torch.float32).unsqueeze(0)
            ref_tensor = torchaudio.functional.resample(
                ref_tensor, reference_sample_rate, self.SAMPLE_RATE
            )
            ref = ref_tensor.squeeze(0).cpu().numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_dir = tmpdir_path / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = input_dir / "prompt.wav"
            sf.write(prompt_path, ref, self.SAMPLE_RATE)

            acoustic_prompt, semantic_prompt = self.engine.build_prompt_features(
                prompt_path, tmpdir_path
            )

            audio_array = self.engine.infer(
                text=str(text),
                ref_text=str(ref_text),
                semantic_prompt=semantic_prompt,
                acoustic_prompt=acoustic_prompt,
                prompt_path=prompt_path,
                temperature=float(kwargs.get("temperature", 0.7)),
                top_k=int(kwargs.get("top_k", 210)),
                max_new_tokens=int(kwargs.get("max_new_tokens", 750)),
            )

        return audio_array, self.SAMPLE_RATE
