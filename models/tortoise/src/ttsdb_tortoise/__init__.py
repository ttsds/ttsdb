# Generated at 2026-01-29T19:21:50Z from templates/init/__init__.py.j2

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from ttsdb_core import VoiceCloningTTSBase, AudioOutput, setup_vendor_path

# Vendored upstream repo (run `just fetch tortoise` / `just setup tortoise`).
setup_vendor_path("ttsdb_tortoise")

__all__ = ["TorToise"]


class TorToise(VoiceCloningTTSBase):
    """Tortoise voice cloning TTS model.
    """

    _package_name = "ttsdb_tortoise"
    SAMPLE_RATE = 24000

    def __init__(self, *args, **kwargs):
        self.device_str: str | None = None
        self.tts = None
        super().__init__(*args, **kwargs)

    def _resolve_weights_dir(self, load_path: str) -> Path:
        base = Path(load_path)
        if base.exists():
            return base
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=load_path))

    def _load_model(self, load_path: str):
        # Make sure numba cache is writable (librosa/numba use cache=True in some paths).
        weights_dir = self._resolve_weights_dir(load_path)
        try:
            cache_dir = weights_dir / ".numba_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
        except Exception:
            pass

        # Expose local checkpoint dir for any patched vendor code to use.
        os.environ.setdefault("TTSDB_TORTOISE_CHECKPOINTS_DIR", str(weights_dir / "checkpoints"))

        try:
            import torch
            from tortoise import api
        except Exception as e:
            raise RuntimeError(
                "Failed to import tortoise. Ensure `just fetch tortoise` has vendored the repo "
                "and dependencies are installed."
            ) from e

        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"

        models_dir = weights_dir / ".models"
        if not models_dir.exists():
            raise FileNotFoundError(
                f"Tortoise model weights not found at {models_dir}. "
                "Run: just hf-weights-prepare tortoise"
            )

        self.tts = api.TextToSpeech(kv_cache=True, models_dir=str(models_dir))
        return self.tts

    def _synthesize(
        self,
        text: str,
        reference_audio: np.ndarray,
        reference_sample_rate: int,
        **kwargs,
    ) -> AudioOutput:
        if self.tts is None:
            raise RuntimeError("Model is not loaded")

        try:
            import torch
            import torchaudio
            from tortoise import utils
        except Exception as e:
            raise RuntimeError("Missing tortoise runtime dependencies (torch/torchaudio).") from e

        # Save speaker reference as temporary wav, resampled to 22050 (matches Cog predictor)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name

        try:
            wav = torch.tensor(reference_audio, dtype=torch.float32).unsqueeze(0)
            if reference_sample_rate != 22050:
                wav = torchaudio.transforms.Resample(reference_sample_rate, 22050)(wav)
            torchaudio.save(tmp_path, wav, 22050)

            reference_clips = [utils.audio.load_audio(tmp_path, 22050)]
            pcm_audio = self.tts.tts_with_preset(
                text, voice_samples=reference_clips, preset="fast"
            )

            # pcm_audio is a batch of audio tensors; return first as numpy.
            out = pcm_audio[0].detach().cpu().numpy()
            # Ensure we return a 1D mono waveform (soundfile expects (frames,) or (frames, channels)).
            out = np.asarray(out).squeeze()
            return out.astype(np.float32), self.SAMPLE_RATE
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
