from __future__ import annotations

# ruff: noqa: E402
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from ttsdb_core import AudioOutput, VoiceCloningTTSBase, setup_vendor_path

setup_vendor_path("ttsdb_indextts")

from indextts.infer_v2 import IndexTTS2

__all__ = ["IndexTTS"]

LANG_ALIASES = {
    "eng": "en_US",
    "en": "en_US",
    "zho": "zh_CN",
    "zh": "zh_CN",
}


class IndexTTS(VoiceCloningTTSBase):
    """IndexTTS v2 voice cloning TTS model."""

    _package_name = "ttsdb_indextts"
    SAMPLE_RATE = 22050

    def _load_model(self, load_path: str):
        weights_dir = Path(load_path)
        cfg_path = weights_dir / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"IndexTTS config.yaml not found at {cfg_path}")

        os.environ.setdefault("HF_HUB_CACHE", str(weights_dir / "hf_cache"))
        os.environ.setdefault("TTSDB_VENDOR_ASSETS_DIR", str(weights_dir / "vendor_assets"))

        use_fp16 = bool(self.init_kwargs.get("fp16", False))
        use_deepspeed = bool(self.init_kwargs.get("deepspeed", False))
        use_cuda_kernel = self.init_kwargs.get("cuda_kernel")
        use_torch_compile = bool(self.init_kwargs.get("torch_compile", False))

        model = IndexTTS2(
            model_dir=str(weights_dir),
            cfg_path=str(cfg_path),
            use_fp16=use_fp16,
            device=str(self.device),
            use_deepspeed=use_deepspeed,
            use_cuda_kernel=use_cuda_kernel,
            use_torch_compile=use_torch_compile,
        )
        return model

    def _synthesize(
        self, text: str, reference_audio: np.ndarray, reference_sample_rate: int, **kwargs
    ) -> AudioOutput:
        language = kwargs.get("language") or "eng"
        _ = LANG_ALIASES.get(str(language), str(language))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            ref_path = tmpdir_path / "reference.wav"
            sf.write(ref_path, reference_audio, reference_sample_rate)

            max_text_tokens = int(kwargs.get("max_text_tokens_per_segment", 120))

            result = self.model.infer(
                spk_audio_prompt=str(ref_path),
                text=str(text),
                output_path=None,
                emo_audio_prompt=None,
                emo_alpha=float(kwargs.get("emo_alpha", 1.0)),
                emo_vector=None,
                use_emo_text=False,
                emo_text=None,
                use_random=False,
                verbose=bool(kwargs.get("verbose", False)),
                max_text_tokens_per_segment=max_text_tokens,
            )

            if isinstance(result, tuple) and len(result) == 2:
                sr, audio = result
                audio = np.asarray(audio)
                sr = int(sr)
            else:
                output_path = tmpdir_path / "output.wav"
                if result is None or not output_path.exists():
                    raise RuntimeError("IndexTTS inference failed to produce audio.")
                audio, sr = sf.read(output_path)

            if audio.dtype.kind in {"i", "u"}:
                max_int = np.iinfo(audio.dtype).max
                audio = audio.astype(np.float32) / max_int
            else:
                audio = audio.astype(np.float32)

            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 1.0:
                audio = audio / peak

        return np.asarray(audio), int(sr)
