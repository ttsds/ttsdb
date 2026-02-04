from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from ttsdb_core import AudioOutput, VoiceCloningTTSBase, get_variant_checkpoint_dir

__all__ = ["XTTS"]

LANG_MAP = {
    "eng": "en",
    "spa": "es",
    "fra": "fr",
    "deu": "de",
    "ita": "it",
    "por": "pt",
    "pol": "pl",
    "tur": "tr",
    "rus": "ru",
    "nld": "nl",
    "ces": "cs",
    "ara": "ar",
    "zho": "zh",
    "jpn": "ja",
    "hun": "hu",
    "kor": "ko",
    "hin": "hi",
}


class XTTS(VoiceCloningTTSBase):
    """XTTS voice cloning TTS model."""

    _package_name = "ttsdb_xtts"
    SAMPLE_RATE = 24000

    def _load_model(self, load_path: str):
        import torch
        from TTS.api import TTS

        variant = (self.model_config.variant if self.model_config else None) or "v2"
        if variant != "v2":
            raise ValueError(f"Unsupported XTTS variant: {variant}")

        variant_dir = get_variant_checkpoint_dir(Path(load_path), config=self.model_config)
        config_path = variant_dir / "config.json"
        use_cuda = torch.cuda.is_available()
        self.tts = TTS(model_path=str(variant_dir), config_path=str(config_path), gpu=use_cuda)
        self.tts.to("cuda" if use_cuda else "cpu")
        return self.tts

    def _synthesize(self, text, reference_audio, reference_sample_rate, **kwargs) -> AudioOutput:
        import tempfile

        import torch

        language = kwargs.get("language") or "en"
        lang = LANG_MAP.get(str(language), str(language))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            ref_path = tmpdir_path / "ref.wav"
            audio_tensor = torch.tensor(reference_audio).detach().cpu().numpy()
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.squeeze()
            sf.write(str(ref_path), audio_tensor, reference_sample_rate)

            output_path = tmpdir_path / "output.wav"
            self.tts.tts_to_file(
                text=str(text),
                file_path=str(output_path),
                speaker_wav=str(ref_path),
                language=lang,
            )

            audio, sr = sf.read(output_path)

        return np.asarray(audio), int(sr)
