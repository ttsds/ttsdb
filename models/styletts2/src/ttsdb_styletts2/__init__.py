from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from ttsdb_core import AudioOutput, VoiceCloningTTSBase, setup_vendor_path, vendor_context

setup_vendor_path("ttsdb_styletts2")

__all__ = ["StyleTTS2"]


class StyleTTS2(VoiceCloningTTSBase):
    """StyleTTS2 voice cloning TTS model."""

    _package_name = "ttsdb_styletts2"
    SAMPLE_RATE = 24000

    def _load_model(self, load_path: str):
        import nltk
        import phonemizer
        import torch
        import torchaudio
        import yaml
        from nltk.data import find as nltk_find
        from nltk.tokenize import word_tokenize

        nltk_data_dir = Path(load_path) / ".nltk_data"
        nltk_data_dir.mkdir(parents=True, exist_ok=True)
        if str(nltk_data_dir) not in nltk.data.path:
            nltk.data.path.insert(0, str(nltk_data_dir))

        for resource in ("punkt", "punkt_tab"):
            try:
                nltk_find(f"tokenizers/{resource}")
            except LookupError:
                nltk.download(resource, download_dir=str(nltk_data_dir), quiet=True)

        with vendor_context("ttsdb_styletts2", cwd=True):
            from Modules.diffusion.sampler import (  # type: ignore
                ADPM2Sampler,
                DiffusionSampler,
                KarrasSchedule,
            )
            from text_utils import TextCleaner  # type: ignore
            from utils import recursive_munch  # type: ignore
            from Utils.PLBERT.util import load_plbert  # type: ignore

            from models import build_model, load_ASR_models, load_F0_models  # type: ignore

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_cleaner = TextCleaner()

        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        )
        self.mean, self.std = -4, 4

        self.phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )

        weights_root = Path(load_path)
        model_dir = weights_root / "Models" / "LibriTTS"
        config = yaml.safe_load(open(model_dir / "config.yml"))
        self.model_params = recursive_munch(config["model_params"])

        with vendor_context("ttsdb_styletts2", cwd=True):
            self.text_aligner = load_ASR_models(
                config.get("ASR_path"),
                config.get("ASR_config"),
            )
            self.pitch_extractor = load_F0_models(config.get("F0_path"))
            self.plbert = load_plbert(config.get("PLBERT_dir"))

        self.model = build_model(
            self.model_params, self.text_aligner, self.pitch_extractor, self.plbert
        )
        _ = [self.model[key].eval().to(self.device) for key in self.model]

        params = torch.load(model_dir / "epochs_2nd_00020.pth", map_location=self.device)["net"]
        for key in self.model:
            if key in params:
                try:
                    self.model[key].load_state_dict(params[key])
                except Exception:
                    from collections import OrderedDict

                    state_dict = OrderedDict((k[7:], v) for k, v in params[key].items())
                    self.model[key].load_state_dict(state_dict, strict=False)

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )

        return self.model

    def _preprocess(self, wave):
        import torch

        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def _compute_style(self, path: Path):
        import librosa
        import torch

        wave, sr = librosa.load(path, sr=24000)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = self._preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)

    def _inference(self, text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1):
        import torch
        import torch.nn.functional as F
        from nltk.tokenize import word_tokenize
        from utils import length_to_mask  # type: ignore

        text = str(text).strip()
        ps = self.phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = " ".join(ps)
        tokens = self.text_cleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,
                num_steps=diffusion_steps,
            ).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data)).to(self.device)
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0)
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = t_en @ pred_aln_trg.unsqueeze(0)
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()

    def _synthesize(self, text, reference_audio, reference_sample_rate, **kwargs) -> AudioOutput:
        import tempfile

        import torch
        import torchaudio

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "ref.wav"
            torchaudio.save(
                str(ref_path), torch.tensor(reference_audio).unsqueeze(0), reference_sample_rate
            )

            ref_s = self._compute_style(ref_path)
            audio_output = self._inference(str(text), ref_s)
            output_path = Path(tmpdir) / "output.wav"
            torchaudio.save(output_path, torch.tensor(audio_output).unsqueeze(0), 24000)
            audio, sr = sf.read(output_path)

        return np.asarray(audio), int(sr)
