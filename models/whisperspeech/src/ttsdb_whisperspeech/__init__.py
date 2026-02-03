from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from ttsdb_core import AudioOutput, VoiceCloningTTSBase

__all__ = ["WhisperSpeech"]

LANG_MAP = {
    "eng": "en",
    "en": "en",
    "pol": "pl",
    "pl": "pl",
    "deu": "de",
    "de": "de",
    "fra": "fr",
    "fr": "fr",
    "ita": "it",
    "it": "it",
    "nld": "nl",
    "nl": "nl",
    "spa": "es",
    "es": "es",
    "por": "pt",
    "pt": "pt",
}


class WhisperSpeech(VoiceCloningTTSBase):
    """WhisperSpeech voice cloning TTS model."""

    _package_name = "ttsdb_whisperspeech"
    SAMPLE_RATE = 24000

    def _load_model(self, load_path: str):
        import torch
        from whisperspeech.pipeline import Pipeline

        self.pipelines = {
            "small": Pipeline(
                t2s_ref="whisperspeech/whisperspeech:t2s-v1.95-small-8lang.model",
                s2a_ref="whisperspeech/whisperspeech:s2a-v1.95-medium-7lang.model",
            )
        }

        return self.pipelines

    def _synthesize(self, text, reference_audio, reference_sample_rate, **kwargs) -> AudioOutput:
        import tempfile

        import torch
        import torchaudio

        variant = (self.model_config.variant if self.model_config else None) or "small"
        if variant != "small":
            raise ValueError(f"Unsupported WhisperSpeech variant: {variant}")
        pipe = self.pipelines["small"]

        language = kwargs.get("language") or "en"
        lang = LANG_MAP.get(str(language), str(language))

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "ref.wav"
            torchaudio.save(
                str(ref_path), torch.tensor(reference_audio).unsqueeze(0), reference_sample_rate
            )

            output_path = Path(tmpdir) / "output.wav"
            pipe.generate_to_file(
                str(output_path), text=str(text), lang=lang, speaker=str(ref_path)
            )

            audio, sr = sf.read(output_path)

        return np.asarray(audio), int(sr)
