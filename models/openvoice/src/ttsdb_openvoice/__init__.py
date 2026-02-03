from __future__ import annotations

# ruff: noqa: E402
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from ttsdb_core import AudioOutput, VoiceCloningTTSBase, setup_vendor_path

setup_vendor_path("ttsdb_openvoice")

__all__ = ["OpenVoice"]

LANG_ALIASES = {
    "eng": "eng",
    "en": "eng",
    "zho": "zho",
    "zh": "zho",
    "spa": "spa",
    "es": "spa",
    "fra": "fra",
    "fr": "fra",
}

LANG_CONFIG = {
    "eng": {
        "melo_lang": "EN",
        "melo_dir": "openvoice_en",
        "ses": "en-default.pth",
        "speaker": "EN-Default",
    },
    "zho": {
        "melo_lang": "ZH",
        "melo_dir": "openvoice_zh",
        "ses": "zh.pth",
        "speaker": "ZH",
    },
    "spa": {
        "melo_lang": "ES",
        "melo_dir": "openvoice_es",
        "ses": "es.pth",
        "speaker": "ES",
    },
    "fra": {
        "melo_lang": "FR",
        "melo_dir": "openvoice_fr",
        "ses": "fr.pth",
        "speaker": "FR",
    },
}


def _ensure_unidic() -> None:
    try:
        import unidic
    except Exception:
        return

    dicdir = None
    try:
        dicdir = Path(unidic.DICDIR)
    except Exception:
        dicdir = None

    if dicdir is not None and (dicdir / "mecabrc").exists():
        return

    try:
        import unidic_lite

        lite_dir = Path(unidic_lite.DICDIR)
        if (lite_dir / "mecabrc").exists():
            unidic.DICDIR = str(lite_dir)
            return
    except Exception:
        pass


def _ensure_torch_hub_cache(base_dir: Path) -> None:
    home_dir = base_dir / ".cache_home"
    home_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(home_dir)

    torch_home = home_dir / ".cache" / "torch"
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)

    try:
        from unidic import download

        download()
    except Exception:
        pass


class OpenVoice(VoiceCloningTTSBase):
    """OpenVoice v2 voice cloning TTS model."""

    _package_name = "ttsdb_openvoice"

    def _load_model(self, load_path: str):
        try:
            import nltk

            try:
                nltk.data.find("taggers/averaged_perceptron_tagger_eng")
            except LookupError:
                nltk.download("averaged_perceptron_tagger_eng")
        except Exception:
            pass

        _ensure_unidic()
        _ensure_torch_hub_cache(Path(load_path))

        from melo.api import TTS
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter

        self.se_extractor = se_extractor

        device = str(self.device)
        weights_dir = Path(load_path)

        openvoice_dir = weights_dir / "openvoice"
        if not openvoice_dir.exists():
            raise FileNotFoundError(
                "OpenVoice weights not found. Expected directory: " f"{openvoice_dir}"
            )

        converter_config = openvoice_dir / "converter" / "config.json"
        converter_ckpt = openvoice_dir / "converter" / "checkpoint.pth"
        if not converter_config.exists() or not converter_ckpt.exists():
            raise FileNotFoundError(
                "OpenVoice converter files not found. Expected at "
                f"{converter_config} and {converter_ckpt}."
            )

        self.tone_color_converter = ToneColorConverter(str(converter_config), device=device)
        self.tone_color_converter.load_ckpt(str(converter_ckpt))

        self.models: dict[str, TTS] = {}
        self.source_ses: dict[str, torch.Tensor] = {}

        for lang, cfg in LANG_CONFIG.items():
            model_dir = weights_dir / cfg["melo_dir"]
            config_path = model_dir / "config.json"
            ckpt_path = model_dir / "checkpoint.pth"
            if not config_path.exists() or not ckpt_path.exists():
                raise FileNotFoundError(
                    f"MeloTTS files not found for {lang}. Expected at "
                    f"{config_path} and {ckpt_path}."
                )

            self.models[lang] = TTS(
                language=cfg["melo_lang"],
                device=device,
                config_path=str(config_path),
                ckpt_path=str(ckpt_path),
            )

            ses_path = openvoice_dir / "base_speakers" / "ses" / cfg["ses"]
            if not ses_path.exists():
                raise FileNotFoundError(f"OpenVoice speaker embedding not found: {ses_path}")
            self.source_ses[lang] = torch.load(ses_path, map_location=device)

        return self.models

    def _iter_torch_modules(self) -> list[torch.nn.Module]:
        modules: list[torch.nn.Module] = []
        if hasattr(self, "tone_color_converter"):
            converter_model = getattr(self.tone_color_converter, "model", None)
            if isinstance(converter_model, torch.nn.Module):
                modules.append(converter_model)

        for model in getattr(self, "models", {}).values():
            model_module = getattr(model, "model", None)
            if isinstance(model_module, torch.nn.Module):
                modules.append(model_module)

        return modules

    def _synthesize(
        self, text: str, reference_audio: np.ndarray, reference_sample_rate: int, **kwargs
    ) -> AudioOutput:
        language = kwargs.get("language") or "eng"
        lang = LANG_ALIASES.get(str(language), str(language))
        if lang not in self.models:
            raise ValueError(f"Unsupported language: {language}")

        model = self.models[lang]
        speaker_ids = model.hps.data.spk2id
        speaker_key = LANG_CONFIG[lang]["speaker"]
        speaker_id = None
        if isinstance(speaker_ids, dict):
            speaker_id = speaker_ids.get(speaker_key)
        elif hasattr(speaker_ids, "get"):
            try:
                speaker_id = speaker_ids.get(speaker_key)
            except Exception:
                speaker_id = None
        if speaker_id is None and hasattr(speaker_ids, "items"):
            try:
                speaker_id = dict(speaker_ids.items()).get(speaker_key)
            except Exception:
                speaker_id = None
        if speaker_id is None and hasattr(speaker_ids, "to_dict"):
            try:
                speaker_id = speaker_ids.to_dict().get(speaker_key)
            except Exception:
                speaker_id = None
        if speaker_id is None and hasattr(speaker_ids, "__dict__"):
            try:
                speaker_id = speaker_ids.__dict__.get(speaker_key)
            except Exception:
                speaker_id = None
        if speaker_id is None:
            raise KeyError(f"Speaker id '{speaker_key}' not found for language {lang}.")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            ref_path = tmpdir_path / "reference.wav"
            sf.write(ref_path, reference_audio, reference_sample_rate)

            target_se, _ = self.se_extractor.get_se(
                str(ref_path),
                self.tone_color_converter,
                target_dir=str(tmpdir_path),
                vad=True,
            )

            src_path = tmpdir_path / "src.wav"
            model.tts_to_file(
                text=str(text),
                speaker_id=speaker_id,
                output_path=str(src_path),
                speed=1.0,
            )

            output_path = tmpdir_path / "output.wav"
            self.tone_color_converter.convert(
                audio_src_path=str(src_path),
                src_se=self.source_ses[lang],
                tgt_se=target_se,
                output_path=str(output_path),
                message="@TTSDB",
            )

            audio, sr = sf.read(output_path)

        return np.asarray(audio), int(sr)
