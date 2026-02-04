from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
from ttsdb_core import (
    AudioOutput,
    VoiceCloningTTSBase,
    get_variant_checkpoint_dir,
    setup_vendor_path,
)

setup_vendor_path("ttsdb_hierspeech")

__all__ = ["HierSpeech"]


def _intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def _add_blank_token(text: list[int]) -> torch.Tensor:
    text_norm = _intersperse(text, 0)
    return torch.LongTensor(text_norm)


class HierspeechSynthesizer:
    def __init__(
        self,
        hierspeech_ckpt: str,
        text2w2v_ckpt: str,
        config_hierspeech: str,
        config_text2w2v: str,
        device: str = "cuda",
    ):
        import utils
        from hierspeechpp_speechsynthesizer import SynthesizerTrn as HierSpeechSynth
        from Mels_preprocess import MelSpectrogramFixed
        from ttv_v1.t2w2v_transformer import SynthesizerTrn as Text2W2VSynth

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.hierspeech_ckpt = hierspeech_ckpt
        self.text2w2v_ckpt = text2w2v_ckpt

        self.hps_hierspeech = utils.get_hparams_from_file(config_hierspeech)
        self.hps_text2w2v = utils.get_hparams_from_file(config_text2w2v)

        hps = self.hps_hierspeech
        self.mel_fn = MelSpectrogramFixed(
            sample_rate=hps.data.sampling_rate,
            n_fft=hps.data.filter_length,
            win_length=hps.data.win_length,
            hop_length=hps.data.hop_length,
            f_min=hps.data.mel_fmin,
            f_max=hps.data.mel_fmax,
            n_mels=hps.data.n_mel_channels,
            window_fn=torch.hann_window,
        ).to(self.device)

        self.net_g = HierSpeechSynth(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
        ).to(self.device)
        self.net_g.load_state_dict(torch.load(self.hierspeech_ckpt, map_location=self.device))
        self.net_g.eval()

        hps_t2w2v = self.hps_text2w2v
        self.text2w2v = Text2W2VSynth(
            hps_t2w2v.data.filter_length // 2 + 1,
            hps_t2w2v.train.segment_size // hps_t2w2v.data.hop_length,
            **hps_t2w2v.model,
        ).to(self.device)
        self.text2w2v.load_state_dict(torch.load(self.text2w2v_ckpt, map_location=self.device))
        self.text2w2v.eval()

    def infer(
        self,
        text: str,
        prompt_audio_path: str,
        denoise_ratio: float = 0.0,
        noise_scale_ttv: float = 0.333,
        noise_scale_vc: float = 0.333,
        scale_norm: str = "prompt",
    ):
        from ttv_v1.text import text_to_sequence

        text_seq = text_to_sequence(str(text), ["english_cleaners2"])
        token = _add_blank_token(text_seq).unsqueeze(0).to(self.device)
        token_length = torch.LongTensor([token.size(-1)]).to(self.device)

        audio, sample_rate = torchaudio.load(prompt_audio_path)
        audio = audio[:1, :]
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(
                audio, sample_rate, 16000, resampling_method="kaiser_window"
            )

        if scale_norm == "prompt":
            prompt_audio_max = torch.max(audio.abs())

        ori_prompt_len = audio.shape[-1]
        multiple_of = 1600
        needed_pad = (ori_prompt_len // multiple_of + 1) * multiple_of - ori_prompt_len
        audio = torch.nn.functional.pad(audio, (0, needed_pad), mode="constant")

        if denoise_ratio == 0.0:
            audio = torch.cat([audio.to(self.device), audio.to(self.device)], dim=0)
        else:
            audio = torch.cat([audio.to(self.device), audio.to(self.device)], dim=0)

        audio = audio[:, :ori_prompt_len]

        src_mel = self.mel_fn(audio)
        src_length = torch.LongTensor([src_mel.size(2)]).to(self.device)
        src_length2 = torch.cat([src_length, src_length], dim=0)

        with torch.no_grad():
            w2v_x, pitch = self.text2w2v.infer_noise_control(
                token,
                token_length,
                src_mel,
                src_length2,
                noise_scale=noise_scale_ttv,
                denoise_ratio=denoise_ratio,
            )

            pitch[pitch < torch.log(torch.tensor([55]).to(self.device))] = 0

            new_src_len = torch.LongTensor([w2v_x.size(2)]).to(self.device)
            converted_audio = self.net_g.voice_conversion_noise_control(
                w2v_x,
                new_src_len,
                src_mel,
                src_length2,
                pitch,
                noise_scale=noise_scale_vc,
                denoise_ratio=denoise_ratio,
            )

        converted_audio = converted_audio.squeeze(0)
        max_abs_value = torch.max(torch.abs(converted_audio))
        if scale_norm == "prompt":
            scale_factor = 32767.0 * prompt_audio_max / (max_abs_value + 1e-8)
        else:
            scale_factor = 32767.0 * 0.999 / (max_abs_value + 1e-8)

        converted_audio = converted_audio * scale_factor
        out_audio_np = converted_audio.detach().cpu().numpy().astype(np.int16)

        return out_audio_np, 16000


class HierSpeech(VoiceCloningTTSBase):
    """HierSpeech++ voice cloning TTS model."""

    _package_name = "ttsdb_hierspeech"
    SAMPLE_RATE = 16000
    SAMPLE_RATE = 16000

    def _load_model(self, load_path: str):
        variant = (self.model_config.variant if self.model_config else None) or "v1"
        base = Path(load_path)
        variant_dir = get_variant_checkpoint_dir(base, config=self.model_config)

        if variant in {"v1", "v1.1"}:
            hierspeech_dir = variant_dir / "hierspeechpp_eng_kor"
            ckpt = hierspeech_dir / "hierspeechpp_v1_ckpt.pth"
            cfg = hierspeech_dir / "config.json"
        elif variant == "lt460":
            hierspeech_dir = variant_dir / "hierspeechpp_libritts460"
            ckpt = hierspeech_dir / "hierspeechpp_lt460_ckpt.pth"
            cfg = hierspeech_dir / "config.json"
        elif variant == "lt960":
            hierspeech_dir = variant_dir / "hierspeechpp_libritts960"
            ckpt = hierspeech_dir / "hierspeechpp_lt960_ckpt.pth"
            cfg = hierspeech_dir / "config.json"
        else:
            raise ValueError(f"Unknown HierSpeech variant: {variant}")

        ttv_dir = base / "shared" / "ttv_libritts_v1"
        ttv_ckpt = ttv_dir / "ttv_lt960_ckpt.pth"
        ttv_cfg = ttv_dir / "config.json"

        self.hierspeech = HierspeechSynthesizer(
            hierspeech_ckpt=str(ckpt),
            config_hierspeech=str(cfg),
            text2w2v_ckpt=str(ttv_ckpt),
            config_text2w2v=str(ttv_cfg),
            device=str(self.device),
        )
        return self.hierspeech

    def _synthesize(self, text, reference_audio, reference_sample_rate, **kwargs) -> AudioOutput:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "ref.wav"
            torchaudio.save(
                str(tmp_path), torch.tensor(reference_audio).unsqueeze(0), reference_sample_rate
            )

            audio, sr = self.hierspeech.infer(
                text=str(text),
                prompt_audio_path=str(tmp_path),
                noise_scale_ttv=0.333,
                noise_scale_vc=0.333,
                denoise_ratio=0.0,
                scale_norm="prompt",
            )

        audio = np.asarray(audio)
        if audio.ndim > 1:
            audio = audio.squeeze()
        if audio.ndim > 1:
            audio = audio[0]
        audio = audio.astype(np.float32) / 32768.0
        return audio, int(sr)
