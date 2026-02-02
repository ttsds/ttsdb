"""F5-TTS voice cloning TTS model."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from ttsdb_core import AudioInput, AudioOutput, VoiceCloningTTSBase, setup_vendor_path

# Add vendored upstream F5-TTS code to sys.path (run `just fetch f5-tts` first).
# This is safe even if the vendor directory is missing; imports only happen when
# the model is loaded.
setup_vendor_path("ttsdb_f5_tts")

__all__ = ["F5TTS"]


class F5TTS(VoiceCloningTTSBase):
    """F5-TTS voice cloning TTS model.

    F5-TTS is a non-autoregressive DiT Transformer text-to-speech model.
    This wrapper uses *vendored* upstream code from the SWivid/F5-TTS repo.

    Config is accessible via `self.model_config`.

    Example:
        >>> model = F5TTS(model_id="SWivid/F5-TTS")
        >>> audio, sr = model.synthesize(
        ...     text="Hello world",
        ...     reference_audio="speaker.wav",
        ...     text_reference="This is the speaker reference text."
        ... )

    Args:
        model_path: Path to local model directory containing checkpoint files.
        model_id: HuggingFace model identifier (e.g., "SWivid/F5-TTS").
        device: Device to run the model on ('cpu', 'cuda', 'cuda:0', etc.).
    """

    _package_name = "ttsdb_f5_tts"

    # Output sample rate for F5-TTS
    SAMPLE_RATE = 24000

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_id: str | None = None,
        device: str | torch.device | None = None,
        **kwargs,
    ):
        self._weights_base: str | None = None
        self._mel_spec_type: str | None = None
        self.vocoder = None
        super().__init__(model_path=model_path, model_id=model_id, device=device, **kwargs)

    def _iter_torch_modules(self):
        """Ensure both base model and vocoder follow `.to()` / `.eval()`."""

        modules = []
        try:
            import torch

            if isinstance(getattr(self, "model", None), torch.nn.Module):
                modules.append(self.model)
            if isinstance(getattr(self, "vocoder", None), torch.nn.Module):
                modules.append(self.vocoder)
        except Exception:
            pass
        return modules

    def _load_model(self, load_path: str):
        """Load F5-TTS model following the upstream Space-style pipeline."""

        base = Path(load_path)
        if not base.exists():
            raise FileNotFoundError(
                f"Model path not found: {load_path}. "
                "Pass `model_path=` pointing at prepared weights or use `model_id=` to "
                "let ttsdb_core download weights before loading."
            )

        self._weights_base = str(base)

        # Numba (pulled in by librosa) tries to write its cache next to site-packages by
        # default, which may be read-only in some environments (like Spaces containers).
        # Point it at a writable directory under the resolved weights folder.
        try:
            cache_dir = base / ".numba_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
        except Exception:
            pass

        # Get variant directory name
        # Standard structure: weights/{variant}/ (e.g., weights/base/, weights/alt/)
        # prepare_weights.py renames HF subdirs to variant names for consistency
        variant = self.model_config.variant if self.model_config else None

        exp_name = str(self.init_kwargs.get("exp_name") or variant or "base")

        exp_dir = base / exp_name
        model_cfg = self._load_variant_model_config(exp_dir)

        # Use the same loader for all variants.
        return self._load_model_common(base, exp_name, model_cfg)

    def _load_variant_model_config(self, exp_dir: Path) -> dict:
        config_path = exp_dir / "f5_model_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing model config at {config_path}")
        with open(config_path) as f:
            return json.load(f)

    def _load_model_common(self, base: Path, exp_name: str, model_cfg: dict):
        """Load F5-TTS model using the upstream inference pipeline."""
        try:
            from f5_tts.infer.utils_infer import load_model, load_vocoder
        except Exception as e:
            raise RuntimeError("Failed to import F5-TTS dependencies (utils_infer).") from e

        from f5_tts.model import DiT
        from ttsdb_core import find_checkpoint

        model_cls = DiT
        mel_spec_type = model_cfg.get("mel_spec_type", "vocos")
        self._mel_spec_type = str(self.init_kwargs.get("mel_spec_type", mel_spec_type))

        exp_dir = base / exp_name
        ckpt_path = find_checkpoint(exp_dir)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint found in {exp_dir}. Expected model_*.pt or model_*.safetensors"
            )

        vocos_dir = None
        candidate = base / "shared" / "vocos-mel-24khz"
        if candidate.exists():
            vocos_dir = candidate

        self.vocoder = load_vocoder(
            self._mel_spec_type,
            vocos_dir is not None,
            str(vocos_dir) if vocos_dir is not None else "",
            self.device,
        )

        model_cfg_no_mel_spec = model_cfg.copy()
        model_cfg_no_mel_spec.pop("mel_spec_type", None)
        self.model = load_model(
            model_cls,
            model_cfg_no_mel_spec,
            str(ckpt_path),
            mel_spec_type=self._mel_spec_type,
            ode_method="euler",
            use_ema=True,
            device=str(self.device),
        )
        return self.model

    def _synthesize(
        self,
        text: str,
        reference_audio: np.ndarray,
        reference_sample_rate: int,
        text_reference: str = "",
        **kwargs,
    ) -> AudioOutput:
        """Synthesize speech from text using F5-TTS.

        Args:
            text: Input text to synthesize.
            reference_audio: Reference audio as numpy array for voice cloning.
            reference_sample_rate: Sample rate of reference audio.
            text_reference: Transcript of the reference audio.
            **kwargs: Additional parameters (unused).

        Returns:
            Tuple of (audio_array, sample_rate).
        """
        return self._synthesize_common(
            text=text,
            reference_audio=reference_audio,
            reference_sample_rate=reference_sample_rate,
            text_reference=text_reference,
            **kwargs,
        )

    def _synthesize_common(
        self,
        text: str,
        reference_audio: np.ndarray,
        reference_sample_rate: int,
        text_reference: str = "",
        **kwargs,
    ) -> AudioOutput:
        if not text_reference:
            raise ValueError("text_reference is required for F5-TTS")

        try:
            from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
        except Exception as e:
            raise RuntimeError("Missing F5-TTS runtime deps (utils_infer).") from e

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        ref_file = tmp_path
        try:
            sf.write(tmp_path, reference_audio, reference_sample_rate)
            ref_file, ref_text = preprocess_ref_audio_text(
                tmp_path, text_reference, show_info=lambda *_args, **_kwargs: None
            )

            target_rms = float(self.init_kwargs.get("target_rms", 0.1))
            nfe_step = int(self.init_kwargs.get("nfe_step", 32))
            cfg_strength = float(self.init_kwargs.get("cfg_strength", 2.0))
            sway_sampling_coef = float(self.init_kwargs.get("sway_sampling_coef", -1.0))
            speed = float(self.init_kwargs.get("speed", 1.0))

            wave, sr, _ = infer_process(
                ref_file,
                ref_text,
                text,
                self.model,
                self.vocoder,
                mel_spec_type=self._mel_spec_type or "vocos",
                target_rms=target_rms,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                speed=speed,
                device=str(self.device),
            )
            return wave.astype(np.float32), int(sr)
        finally:
            # If preprocessing returned a different cached path, clean up the temp file.
            if ref_file != tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
