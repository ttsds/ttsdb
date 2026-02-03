from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from ttsdb_core import (
    AudioOutput,
    VoiceCloningTTSBase,
    get_vendor_path,
    setup_vendor_path,
    vendor_context,
)

setup_vendor_path("ttsdb_vevo")

__all__ = ["Vevo"]

LANG_MAP = {
    "eng": "en",
    "en": "en",
    "zho": "zh",
    "zh": "zh",
    "jpn": "ja",
    "ja": "ja",
    "kor": "ko",
    "ko": "ko",
    "deu": "de",
    "de": "de",
    "fra": "fr",
    "fr": "fr",
}


class Vevo(VoiceCloningTTSBase):
    """Vevo voice cloning TTS model."""

    _package_name = "ttsdb_vevo"
    SAMPLE_RATE = 24000

    def _load_model(self, load_path: str):
        import os

        import torch

        vendor_path = get_vendor_path("ttsdb_vevo")
        os.environ.setdefault("WORK_DIR", str(vendor_path))

        with vendor_context("ttsdb_vevo", cwd=True, env={"WORK_DIR": str(vendor_path)}):
            from models.svc.vevosing.vevosing_utils import VevosingInferencePipeline, save_audio

        base = Path(load_path)
        shared = base / "shared"
        base = base / self._variant
        hubert_ckpt = shared / "hubert" / "hubert_fairseq_large_ll60k.pth"
        hubert_cache = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        hubert_cache.mkdir(parents=True, exist_ok=True)
        if hubert_ckpt.exists():
            (hubert_cache / hubert_ckpt.name).write_bytes(hubert_ckpt.read_bytes())

        prosody_tokenizer_ckpt_path = str(base / "tokenizer" / "prosody_fvq512_6.25hz")
        content_style_tokenizer_ckpt_path = str(base / "tokenizer" / "contentstyle_fvq16384_12.5hz")
        ar_cfg_path = str(
            vendor_path / Path("models/svc/vevosing/config/ar_emilia101k_singnet7k.json")
        )
        ar_ckpt_path = str(base / "contentstyle_modeling" / "ar_emilia101k_singnet7k")
        fmt_cfg_source = vendor_path / Path(
            "models/svc/vevosing/config/fm_emilia101k_singnet7k.json"
        )
        fmt_cfg_cache_dir = base / ".cache"
        fmt_cfg_cache_dir.mkdir(parents=True, exist_ok=True)
        fmt_cfg_path = fmt_cfg_cache_dir / "fm_emilia101k_singnet7k.json"
        if not fmt_cfg_path.exists():
            cfg_text = fmt_cfg_source.read_text()
            whisper_stats_path = str(
                vendor_path / Path("models/svc/vevosing/config/whisper_stats.pt")
            )
            cfg_text = cfg_text.replace(
                '"whisper_stats_path": "models/svc/vevosing/config/whisper_stats.pt"',
                f'"whisper_stats_path": "{whisper_stats_path}"',
            )
            fmt_cfg_path.write_text(cfg_text)
        fmt_ckpt_path = str(base / "acoustic_modeling" / "fm_emilia101k_singnet7k")
        vocoder_cfg_path = str(vendor_path / Path("models/svc/vevosing/config/vocoder.json"))
        vocoder_ckpt_path = str(base / "acoustic_modeling" / "Vocoder")

        force_cpu = os.environ.get("VEVO_FORCE_CPU", "").lower() in {"1", "true", "yes"}
        device = torch.device("cpu")
        if torch.cuda.is_available() and not force_cpu:
            device = torch.device("cuda")

        try:
            self.vevo = VevosingInferencePipeline(
                prosody_tokenizer_ckpt_path=prosody_tokenizer_ckpt_path,
                content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
                ar_cfg_path=ar_cfg_path,
                ar_ckpt_path=ar_ckpt_path,
                fmt_cfg_path=str(fmt_cfg_path),
                fmt_ckpt_path=fmt_ckpt_path,
                vocoder_cfg_path=vocoder_cfg_path,
                vocoder_ckpt_path=vocoder_ckpt_path,
                device=device,
            )
        except RuntimeError as exc:
            if "no kernel image is available" not in str(exc):
                raise
            device = torch.device("cpu")
            self.vevo = VevosingInferencePipeline(
                prosody_tokenizer_ckpt_path=prosody_tokenizer_ckpt_path,
                content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
                ar_cfg_path=ar_cfg_path,
                ar_ckpt_path=ar_ckpt_path,
                fmt_cfg_path=str(fmt_cfg_path),
                fmt_ckpt_path=fmt_ckpt_path,
                vocoder_cfg_path=vocoder_cfg_path,
                vocoder_ckpt_path=vocoder_ckpt_path,
                device=device,
            )

        return self.vevo

    def _synthesize(self, text, reference_audio, reference_sample_rate, **kwargs) -> AudioOutput:
        ref_text = kwargs.get("text_reference") or kwargs.get("reference_text") or ""
        language = kwargs.get("language") or "eng"
        lang = LANG_MAP.get(str(language), str(language))
        vendor_path = get_vendor_path("ttsdb_vevo")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            if isinstance(reference_audio, str | Path):
                ref_path = Path(reference_audio)
            else:
                ref_path = tmpdir_path / "ref.wav"
                sf.write(ref_path, reference_audio, reference_sample_rate)

            from models.svc.vevosing.vevosing_utils import save_audio

            with vendor_context("ttsdb_vevo", cwd=True, env={"WORK_DIR": str(vendor_path)}):
                gen_audio = self.vevo.inference_ar_and_fm(
                    task="synthesis",
                    src_wav_path=None,
                    src_text=str(text),
                    style_ref_wav_path=str(ref_path),
                    timbre_ref_wav_path=str(ref_path),
                    style_ref_wav_text=str(ref_text),
                    src_text_language=lang,
                    style_ref_wav_text_language=lang,
                )

            output_path = tmpdir_path / "vevo_out.wav"
            save_audio(gen_audio, output_path=str(output_path))
            audio, sr = sf.read(output_path)

        return audio.astype(np.float32), int(sr)
