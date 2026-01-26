"""MaskGCT voice cloning TTS model."""

import os
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import safetensors.torch
import soundfile as sf
from accelerate import load_checkpoint_and_dispatch

from ttsdb_core import VoiceCloningTTSBase, AudioInput, AudioOutput, setup_vendor_path, vendor_context

setup_vendor_path("ttsdb_maskgct")

__all__ = ["MaskGCT", "MaskGCTModels"]

# Amphion environment configuration
_AMPHION_ENV = {"WORK_DIR": "{vendor_path}"}

# Mapping from ISO 639-3 codes to MaskGCT internal codes (ISO 639-1)
# MaskGCT uses 2-letter codes: en, zh, ja, ko, fr, de
ISO_639_3_TO_MASKGCT = {
    "eng": "en",  # English
    "zho": "zh",  # Chinese
    "cmn": "zh",  # Mandarin Chinese -> zh
    "jpn": "ja",  # Japanese
    "kor": "ko",  # Korean
    "fra": "fr",  # French
    "deu": "de",  # German
    # Also accept the 2-letter codes directly for backwards compatibility
    "en": "en",
    "zh": "zh",
    "ja": "ja",
    "ko": "ko",
    "fr": "fr",
    "de": "de",
}


class MaskGCTModels:
    """Container for all MaskGCT model components."""
    
    def __init__(
        self,
        semantic_model,
        semantic_codec,
        codec_encoder,
        codec_decoder,
        t2s_model,
        s2a_model_1layer,
        s2a_model_full,
        semantic_mean,
        semantic_std,
    ):
        self.semantic_model = semantic_model
        self.semantic_codec = semantic_codec
        self.codec_encoder = codec_encoder
        self.codec_decoder = codec_decoder
        self.t2s_model = t2s_model
        self.s2a_model_1layer = s2a_model_1layer
        self.s2a_model_full = s2a_model_full
        self.semantic_mean = semantic_mean
        self.semantic_std = semantic_std
    
    def to(self, device):
        """Move all models to the specified device."""
        self.semantic_model = self.semantic_model.to(device)
        self.semantic_codec = self.semantic_codec.to(device)
        self.codec_encoder = self.codec_encoder.to(device)
        self.codec_decoder = self.codec_decoder.to(device)
        self.t2s_model = self.t2s_model.to(device)
        self.s2a_model_1layer = self.s2a_model_1layer.to(device)
        self.s2a_model_full = self.s2a_model_full.to(device)
        return self
    
    def eval(self):
        """Set all models to evaluation mode."""
        self.semantic_model.eval()
        self.semantic_codec.eval()
        self.codec_encoder.eval()
        self.codec_decoder.eval()
        self.t2s_model.eval()
        self.s2a_model_1layer.eval()
        self.s2a_model_full.eval()
        return self
    
    def train(self):
        """Set all models to training mode."""
        self.semantic_model.train()
        self.semantic_codec.train()
        self.codec_encoder.train()
        self.codec_decoder.train()
        self.t2s_model.train()
        self.s2a_model_1layer.train()
        self.s2a_model_full.train()
        return self


class MaskGCT(VoiceCloningTTSBase):
    """MaskGCT voice cloning TTS model.
    
    MaskGCT is a zero-shot text-to-speech model using masked generative codec 
    transformer from Amphion.
    
    Config is accessible via `self.model_config`.
    
    Example:
        >>> model = MaskGCT(model_path="/path/to/maskgct/checkpoints")
        >>> audio, sr = model.synthesize(
        ...     text="Hello world",
        ...     reference_audio="speaker.wav",
        ...     text_reference="This is the speaker reference text.",
        ...     language="eng"  # ISO 639-3 code (or "en" for backwards compatibility)
        ... )
    
    Args:
        model_path: Path to local model directory containing checkpoint files.
        model_id: HuggingFace model identifier (e.g., "amphion/MaskGCT").
        device: Device to run the model on ('cpu', 'cuda', 'cuda:0', etc.).
    """
    
    _package_name = "ttsdb_maskgct"
    
    # Output sample rate for MaskGCT
    SAMPLE_RATE = 24000

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_id: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs
    ):
        self._inference_pipeline = None
        super().__init__(model_path=model_path, model_id=model_id, device=device, **kwargs)

    def _load_model(self, load_path: str) -> MaskGCTModels:
        """Load all MaskGCT model components.
        
        Args:
            load_path: Path to model checkpoints directory or HuggingFace ID.
            
        Returns:
            MaskGCTModels container with all model components.
        """
        with vendor_context(self._package_name, cwd=True, env=_AMPHION_ENV) as vendor_path:
            # Import Amphion utilities
            from models.tts.maskgct.maskgct_utils import (
                build_semantic_model,
                build_semantic_codec,
                build_acoustic_codec,
                build_t2s_model,
                build_s2a_model,
            )
            from utils.util import load_config
            
            # Load config
            cfg_path = vendor_path / "models/tts/maskgct/config/maskgct.json"
            cfg = load_config(str(cfg_path))
            
            # Build models
            semantic_model, semantic_mean, semantic_std = build_semantic_model(self.device)
            semantic_codec = build_semantic_codec(cfg.model.semantic_codec, self.device)
            codec_encoder, codec_decoder = build_acoustic_codec(
                cfg.model.acoustic_codec, self.device
            )
            t2s_model = build_t2s_model(cfg.model.t2s_model, self.device)
            s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, self.device)
            s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, self.device)
            
            # Resolve checkpoint path
            if os.path.isdir(load_path):
                base = load_path
            else:
                # Assume HuggingFace model ID - download checkpoints
                from huggingface_hub import snapshot_download
                base = snapshot_download(repo_id=load_path)
            
            # Load weights
            use_gpu = self.device.type == "cuda"
            
            if use_gpu:
                safetensors.torch.load_model(
                    semantic_codec, os.path.join(base, "semantic_codec/model.safetensors")
                )
                safetensors.torch.load_model(
                    codec_encoder, os.path.join(base, "acoustic_codec/model.safetensors")
                )
                safetensors.torch.load_model(
                    codec_decoder, os.path.join(base, "acoustic_codec/model_1.safetensors")
                )
                safetensors.torch.load_model(
                    t2s_model, os.path.join(base, "t2s_model/model.safetensors")
                )
                safetensors.torch.load_model(
                    s2a_model_1layer, os.path.join(base, "s2a_model/s2a_model_1layer/model.safetensors")
                )
                safetensors.torch.load_model(
                    s2a_model_full, os.path.join(base, "s2a_model/s2a_model_full/model.safetensors")
                )
            else:
                # Use accelerate for CPU loading
                load_checkpoint_and_dispatch(
                    semantic_codec,
                    os.path.join(base, "semantic_codec/model.safetensors"),
                    device_map={"": "cpu"},
                )
                load_checkpoint_and_dispatch(
                    codec_encoder,
                    os.path.join(base, "acoustic_codec/model.safetensors"),
                    device_map={"": "cpu"},
                )
                load_checkpoint_and_dispatch(
                    codec_decoder,
                    os.path.join(base, "acoustic_codec/model_1.safetensors"),
                    device_map={"": "cpu"},
                )
                load_checkpoint_and_dispatch(
                    t2s_model,
                    os.path.join(base, "t2s_model/model.safetensors"),
                    device_map={"": "cpu"},
                )
                load_checkpoint_and_dispatch(
                    s2a_model_1layer,
                    os.path.join(base, "s2a_model/s2a_model_1layer/model.safetensors"),
                    device_map={"": "cpu"},
                )
                load_checkpoint_and_dispatch(
                    s2a_model_full,
                    os.path.join(base, "s2a_model/s2a_model_full/model.safetensors"),
                    device_map={"": "cpu"},
                )
            
            return MaskGCTModels(
                semantic_model=semantic_model,
                semantic_codec=semantic_codec,
                codec_encoder=codec_encoder,
                codec_decoder=codec_decoder,
                t2s_model=t2s_model,
                s2a_model_1layer=s2a_model_1layer,
                s2a_model_full=s2a_model_full,
                semantic_mean=semantic_mean,
                semantic_std=semantic_std,
            )
    
    def _get_inference_pipeline(self):
        """Get or create the MaskGCT inference pipeline."""
        if self._inference_pipeline is None:
            with vendor_context(self._package_name, cwd=True, env=_AMPHION_ENV):
                from models.tts.maskgct.maskgct_utils import MaskGCT_Inference_Pipeline
                
                self._inference_pipeline = MaskGCT_Inference_Pipeline(
                    self.model.semantic_model,
                    self.model.semantic_codec,
                    self.model.codec_encoder,
                    self.model.codec_decoder,
                    self.model.t2s_model,
                    self.model.s2a_model_1layer,
                    self.model.s2a_model_full,
                    self.model.semantic_mean,
                    self.model.semantic_std,
                    self.device,
                )
        
        return self._inference_pipeline

    def _synthesize(
        self,
        text: str,
        reference_audio: np.ndarray,
        reference_sample_rate: int,
        text_reference: str = "",
        language: str = "eng",
        target_language: Optional[str] = None,
        target_len: Optional[int] = None,
        **kwargs
    ) -> AudioOutput:
        """Synthesize speech from text using MaskGCT.
        
        Args:
            text: Input text to synthesize.
            reference_audio: Reference audio as numpy array for voice cloning.
            reference_sample_rate: Sample rate of reference audio.
            text_reference: Transcript of the reference audio.
            language: Language code for the reference audio (ISO 639-3 or 639-1).
                     Supported: eng/en, zho/zh, jpn/ja, kor/ko, fra/fr, deu/de.
            target_language: Language code for the target text. If None, uses 
                           the same as `language`.
            target_len: Target length for the generated audio. If None, 
                       determined automatically.
            **kwargs: Additional parameters (unused).
            
        Returns:
            Tuple of (audio_array, sample_rate).
        """
        if target_language is None:
            target_language = language
        
        # Map ISO 639-3 codes to MaskGCT internal codes (ISO 639-1)
        language = ISO_639_3_TO_MASKGCT.get(language, language)
        target_language = ISO_639_3_TO_MASKGCT.get(target_language, target_language)
        
        # MaskGCT expects a file path, so we need to save the reference audio
        # to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, reference_audio, reference_sample_rate)
        
        try:
            with vendor_context(self._package_name, cwd=True, env=_AMPHION_ENV):
                pipeline = self._get_inference_pipeline()
                
                recovered_audio = pipeline.maskgct_inference(
                    tmp_path,
                    text_reference,
                    text,
                    language,
                    target_language,
                    target_len=target_len,
                )
            
            # Convert to numpy array if needed
            if isinstance(recovered_audio, torch.Tensor):
                recovered_audio = recovered_audio.cpu().numpy()
            
            return recovered_audio, self.SAMPLE_RATE
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)