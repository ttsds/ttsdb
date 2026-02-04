from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ttsdb_core import AudioOutput, VoiceCloningTTSBase, setup_vendor_path

__all__ = ["VibeVoice"]


class VibeVoice(VoiceCloningTTSBase):
    """VibeVoice voice cloning TTS model (single speaker)."""

    _package_name = "ttsdb_vibevoice"
    SAMPLE_RATE = 24000

    def _load_model_components(self):
        load_path = self._resolve_model_path()
        self.model = self._load_model(str(load_path))
        for m in self._iter_torch_modules():
            m.to(self.device)
            m.eval()

    def _load_model(self, load_path: str):
        setup_vendor_path("ttsdb_vibevoice")

        from vvembed.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vvembed.processor.vibevoice_processor import VibeVoiceProcessor

        model_dir = Path(load_path)
        tokenizer_dir = model_dir / "shared" / "qwen_tokenizer"

        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch_dtype,
        )

        processor_kwargs: dict[str, object] = {
            "trust_remote_code": True,
            "local_files_only": True,
        }
        if tokenizer_dir.exists():
            processor_kwargs["language_model_pretrained_name"] = str(tokenizer_dir)

        self.processor = VibeVoiceProcessor.from_pretrained(str(model_dir), **processor_kwargs)
        return model

    def _preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != self.SAMPLE_RATE:
            audio = self._resample_audio(audio, sample_rate, self.SAMPLE_RATE)
        return audio.astype(np.float32)

    def _synthesize(
        self, text: str, reference_audio: np.ndarray, reference_sample_rate: int, **kwargs
    ) -> AudioOutput:
        if self.processor is None:
            raise RuntimeError("VibeVoice processor is not initialized")

        diffusion_steps = int(kwargs.get("diffusion_steps", 20))
        cfg_scale = float(kwargs.get("cfg_scale", 1.3))
        seed = int(kwargs.get("seed", 42))
        use_sampling = bool(kwargs.get("use_sampling", False))
        temperature = float(kwargs.get("temperature", 0.95))
        top_p = float(kwargs.get("top_p", 0.95))

        formatted_text = f"Speaker 1: {str(text).strip()}"
        voice_samples = [reference_audio]

        inputs = self.processor(
            [formatted_text],
            voice_samples=[voice_samples],
            return_tensors="pt",
            return_attention_mask=True,
        )

        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
        }

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        self.model.set_ddpm_inference_steps(diffusion_steps)

        with torch.no_grad():
            if use_sampling:
                output = self.model.generate(
                    **inputs,
                    tokenizer=self.processor.tokenizer,
                    cfg_scale=cfg_scale,
                    max_new_tokens=None,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                output = self.model.generate(
                    **inputs,
                    tokenizer=self.processor.tokenizer,
                    cfg_scale=cfg_scale,
                    max_new_tokens=None,
                    do_sample=False,
                )

        if not hasattr(output, "speech_outputs") or not output.speech_outputs:
            raise RuntimeError("VibeVoice did not return speech outputs")

        speech_tensor = output.speech_outputs[0]
        if speech_tensor is None:
            raise RuntimeError("VibeVoice returned empty speech output")

        if isinstance(speech_tensor, list):
            speech_tensor = torch.cat(speech_tensor, dim=-1)

        if speech_tensor.dim() > 1:
            speech_tensor = speech_tensor.squeeze()

        audio = speech_tensor.detach().cpu().float().numpy()
        return np.asarray(audio), int(self.SAMPLE_RATE)
