from __future__ import annotations

# ruff: noqa: E402
import io
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from ttsdb_core import AudioOutput, VoiceCloningTTSBase, setup_vendor_path

setup_vendor_path("ttsdb_fish_speech")

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    WrappedGenerateResponse,
    generate_long,
    init_model,
)
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

__all__ = ["FishSpeech"]


class _DirectLlamaQueue:
    def __init__(self, model, decode_one_token):
        self._model = model
        self._decode_one_token = decode_one_token

    def put(self, item: GenerateRequest) -> None:
        kwargs = item.request
        response_queue = item.response_queue

        try:
            for chunk in generate_long(
                model=self._model, decode_one_token=self._decode_one_token, **kwargs
            ):
                response_queue.put(WrappedGenerateResponse(status="success", response=chunk))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            response_queue.put(WrappedGenerateResponse(status="error", response=exc))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class FishSpeech(VoiceCloningTTSBase):
    """Fish Speech voice cloning TTS model (OpenAudio S1 Mini-style)."""

    _package_name = "ttsdb_fish_speech"
    SAMPLE_RATE = 44100

    def _load_model(self, load_path: str):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HYDRA_FULL_ERROR", "1")
        os.environ.setdefault("EINX_FILTER_TRACEBACK", "false")

        compile_model = bool(self.init_kwargs.get("compile", False))
        half = bool(self.init_kwargs.get("half", False))
        precision = torch.half if half else torch.bfloat16
        decoder_config_name = self.init_kwargs.get("decoder_config_name", "modded_dac_vq")

        base_path = Path(load_path)
        variant = self.variant
        checkpoint_dir = base_path
        if variant:
            candidate = base_path / str(variant)
            if (candidate / "config.json").is_file():
                checkpoint_dir = candidate

        if not (checkpoint_dir / "config.json").is_file():
            raise FileNotFoundError(
                "Fish Speech checkpoint config.json not found. Expected at "
                f"{checkpoint_dir / 'config.json'}."
            )

        llama_checkpoint_path = self.init_kwargs.get("llama_checkpoint_path", str(checkpoint_dir))
        decoder_checkpoint_path = self.init_kwargs.get(
            "decoder_checkpoint_path", str(checkpoint_dir / "codec.pth")
        )

        self.llama_model, self.decode_one_token = init_model(
            llama_checkpoint_path,
            self.device,
            precision,
            compile=compile_model,
        )
        with torch.device(self.device):
            self.llama_model.setup_caches(
                max_batch_size=1,
                max_seq_len=self.llama_model.config.max_seq_len,
                dtype=next(self.llama_model.parameters()).dtype,
            )
        self.llama_queue = _DirectLlamaQueue(self.llama_model, self.decode_one_token)

        print(decoder_checkpoint_path, decoder_config_name)
        self.decoder_model = load_decoder_model(
            config_name=decoder_config_name,
            checkpoint_path=decoder_checkpoint_path,
            device=str(self.device),
        )

        self.inference_engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.decoder_model,
            compile=compile_model,
            precision=precision,
        )

        return self.inference_engine

    def _iter_torch_modules(self) -> list[torch.nn.Module]:
        modules: list[torch.nn.Module] = []
        if hasattr(self, "decoder_model") and isinstance(self.decoder_model, torch.nn.Module):
            modules.append(self.decoder_model)
        if hasattr(self, "llama_model") and isinstance(self.llama_model, torch.nn.Module):
            modules.append(self.llama_model)
        return modules

    def _synthesize(
        self, text: str, reference_audio: np.ndarray, reference_sample_rate: int, **kwargs
    ) -> AudioOutput:
        references = self._build_reference_list(reference_audio, reference_sample_rate, **kwargs)

        req = ServeTTSRequest(
            text=str(text),
            references=references,
            reference_id=kwargs.get("reference_id") or None,
            max_new_tokens=int(kwargs.get("max_new_tokens", 1024)),
            chunk_length=int(kwargs.get("chunk_length", 200)),
            top_p=float(kwargs.get("top_p", 0.7)),
            repetition_penalty=float(kwargs.get("repetition_penalty", 1.1)),
            temperature=float(kwargs.get("temperature", 0.7)),
            seed=int(kwargs.get("seed")) if kwargs.get("seed") is not None else None,
            use_memory_cache=kwargs.get("use_memory_cache", "off"),
        )

        for result in self.inference_engine.inference(req):
            if result.code == "final":
                sample_rate, audio = result.audio
                return audio, int(sample_rate)
            if result.code == "error":
                raise RuntimeError(str(result.error))

        raise RuntimeError("Fish Speech inference returned no audio")

    def _build_reference_list(
        self, reference_audio: np.ndarray, reference_sample_rate: int, **kwargs
    ) -> list[ServeReferenceAudio]:
        reference_text = kwargs.get("text_reference") or kwargs.get("reference_text") or ""
        if reference_audio is None or len(reference_audio) == 0:
            return []

        audio_bytes = self._audio_to_wav_bytes(reference_audio, reference_sample_rate)
        return [ServeReferenceAudio(audio=audio_bytes, text=str(reference_text))]

    @staticmethod
    def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
        array = np.asarray(audio)
        if array.ndim == 2 and array.shape[0] in (1, 2) and array.shape[0] < array.shape[1]:
            array = array.T

        buffer = io.BytesIO()
        sf.write(buffer, array, samplerate=int(sample_rate), format="WAV", subtype="PCM_16")
        return buffer.getvalue()
