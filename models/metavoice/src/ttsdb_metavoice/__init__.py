from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from ttsdb_core import AudioOutput, VoiceCloningTTSBase, setup_vendor_path

setup_vendor_path("ttsdb_metavoice")

__all__ = ["Metavoice"]


class Metavoice(VoiceCloningTTSBase):
    """MetaVoice voice cloning TTS model."""

    _package_name = "ttsdb_metavoice"
    SAMPLE_RATE = 24000

    def _load_model(self, load_path: str):
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        try:
            import torch._dynamo

            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass

        import fam.llm.fast_inference as fast_inference
        import fam.llm.fast_inference_utils as fast_inference_utils
        import fam.llm.inference as inference
        import torch

        def _default_dtype() -> str:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return "bfloat16"
            return "float16"

        fast_inference.get_default_dtype = _default_dtype

        original_load_model = fast_inference_utils._load_model

        def _load_model_with_precision(
            checkpoint_path,
            spk_emb_ckpt_path,
            device,
            precision,
            quantisation_mode=None,
        ):
            model, tokenizer, smodel = original_load_model(
                checkpoint_path,
                spk_emb_ckpt_path,
                device,
                precision,
                quantisation_mode=quantisation_mode,
            )
            model = model.to(device=device, dtype=precision)
            return model, tokenizer, smodel

        fast_inference_utils._load_model = _load_model_with_precision

        def _skip_check_audio_file(*_args, **_kwargs):
            return None

        fast_inference.check_audio_file = _skip_check_audio_file

        cache_root = Path(load_path).parent / ".cache" / "fam"
        cache_root.mkdir(parents=True, exist_ok=True)

        def _get_cached_embedding(local_file_path: str, spkemb_model):
            if not os.path.exists(local_file_path):
                raise FileNotFoundError(f"File {local_file_path} not found!")

            cache_name = (
                "embedding_" + hashlib.md5(local_file_path.encode("utf-8")).hexdigest() + ".pt"
            )
            cache_path = cache_root / cache_name

            if not cache_path.exists():
                spk_emb = spkemb_model.embed_utterance_from_file(
                    local_file_path, numpy=False
                ).unsqueeze(0)
                torch.save(spk_emb, cache_path)
            else:
                spk_emb = torch.load(cache_path)

            return spk_emb

        inference.get_cached_embedding = _get_cached_embedding
        fast_inference.get_cached_embedding = _get_cached_embedding

        # Monkeypatch snapshot_download to accept local paths.
        def _local_snapshot(repo_id: str, *args, **kwargs):
            return repo_id

        fast_inference.snapshot_download = _local_snapshot

        try:
            import fam.llm.utils as utils

            if hasattr(utils, "threshold_s"):
                utils.threshold_s = 1
        except Exception:
            pass

        self.metavoice = fast_inference.TTS(model_name=str(load_path))
        return self.metavoice

    def _synthesize(self, text, reference_audio, reference_sample_rate, **kwargs) -> AudioOutput:
        import tempfile

        import torch
        import torchaudio

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "ref.wav"
            audio_tensor = torch.tensor(reference_audio).unsqueeze(0)
            torchaudio.save(str(ref_path), audio_tensor, reference_sample_rate)

            wav_path = self.metavoice.synthesise(
                text=str(text),
                spk_ref_path=str(ref_path),
                top_p=0.95,
                guidance_scale=3.0,
                temperature=1.0,
            )

            audio, sr = sf.read(wav_path)

        return np.asarray(audio), int(sr)
