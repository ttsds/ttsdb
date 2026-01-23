"""Base class for voice cloning TTS models using PyTorch."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional, Tuple

import numpy as np
import torch
import soundfile as sf
from scipy import signal
from transformers import AutoTokenizer, AutoProcessor

from .config import ModelConfig
from .vendor import setup_vendor_path, get_vendor_path, vendor_context


AudioInput = Union[str, Path, np.ndarray, torch.Tensor]
AudioOutput = Tuple[np.ndarray, int]

__all__ = [
    "VoiceCloningTTSBase",
    "AudioInput",
    "AudioOutput",
    "ModelConfig",
    "setup_vendor_path",
    "get_vendor_path",
    "vendor_context",
]


class VoiceCloningTTSBase(ABC):
    """Abstract base class for voice cloning Text-to-Speech models.
    
    This class provides a foundation for implementing voice cloning TTS models
    using PyTorch. Models can be loaded from disk or HuggingFace Hub.
    
    Subclasses should implement:
    - `_load_model()`: Load the model architecture and weights
    - `_synthesize()`: Core synthesis logic
    - `_preprocess_audio()`: Preprocess reference audio if needed
    - `_postprocess_audio()`: Postprocess generated audio if needed
    """
    
    # Override in subclass to set the package name for config loading
    _package_name: Optional[str] = None
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_id: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs
    ):
        """Initialize the voice cloning TTS model.
        
        Args:
            model_path: Path to local model directory. If provided, loads from disk.
            model_id: HuggingFace model identifier (e.g., "microsoft/speecht5_tts").
                     If provided and model_path is None, loads from HuggingFace Hub.
            device: Device to run the model on ('cpu', 'cuda', 'cuda:0', etc.).
                   If None, uses 'cuda' if available, else 'cpu'.
            **kwargs: Additional model-specific initialization parameters.
        """
        if model_path is None and model_id is None:
            raise ValueError("Either model_path or model_id must be provided")
        
        self.model_path = Path(model_path) if model_path else None
        self.model_id = model_id
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.init_kwargs = kwargs
        
        # Load model config from package
        self.model_config: Optional[ModelConfig] = None
        if self._package_name:
            try:
                self.model_config = ModelConfig.from_package(self._package_name)
            except Exception:
                pass  # Config loading is optional
        
        # Load model components
        self._load_model_components()
    
    def _setup_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Set up the computation device.
        
        Args:
            device: Device specification.
            
        Returns:
            torch.device object.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _load_model_components(self):
        """Load model, tokenizer, and processor components."""
        load_path = str(self.model_path) if self.model_path else self.model_id
        
        # Load model (implemented by subclasses)
        self.model = self._load_model(load_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Try to load tokenizer and processor if they exist (optional helpers)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        except (OSError, ValueError, TypeError):
            self.tokenizer = None
        
        try:
            self.processor = AutoProcessor.from_pretrained(load_path)
        except (OSError, ValueError, TypeError):
            self.processor = None
    
    @abstractmethod
    def _load_model(self, load_path: str) -> torch.nn.Module:
        """Load the model architecture and weights.
        
        Args:
            load_path: Path to model (local or HuggingFace ID).
            
        Returns:
            Loaded PyTorch model.
        """
        pass
    
    @abstractmethod
    def _synthesize(
        self,
        text: str,
        reference_audio: np.ndarray,
        reference_sample_rate: int,
        **kwargs
    ) -> AudioOutput:
        """Core synthesis logic.
        
        Args:
            text: Input text to synthesize.
            reference_audio: Reference audio as numpy array for voice cloning.
            reference_sample_rate: Sample rate of reference audio.
            **kwargs: Additional model-specific parameters.
            
        Returns:
            Tuple of (audio_array, sample_rate).
        """
        pass
    
    def synthesize(
        self,
        text: str,
        reference_audio: AudioInput,
        reference_sample_rate: Optional[int] = None,
        output_sample_rate: Optional[int] = None,
        **kwargs
    ) -> AudioOutput:
        """Synthesize speech from text using voice cloning.
        
        Args:
            text: Input text to synthesize.
            reference_audio: Reference audio for voice cloning. Can be:
                - str/Path: Path to audio file
                - np.ndarray: Audio array (requires reference_sample_rate)
                - torch.Tensor: Audio tensor (requires reference_sample_rate)
            reference_sample_rate: Sample rate of reference audio (required if
                                  reference_audio is array/tensor).
            output_sample_rate: Desired output sample rate. If None, uses
                               model's default sample rate.
            **kwargs: Additional model-specific parameters.
            
        Returns:
            Tuple of (audio_array, sample_rate).
        """
        # Load and preprocess reference audio
        ref_audio, ref_sr = self._load_audio(reference_audio, reference_sample_rate)
        ref_audio = self._preprocess_audio(ref_audio, ref_sr)
        
        # Synthesize
        audio, sample_rate = self._synthesize(
            text,
            ref_audio,
            ref_sr,
            **kwargs
        )
        if self.model_config.metadata.sample_rate != sample_rate:
            print(f"Warning: Resampling audio from {sample_rate} to {self.model_config.metadata.sample_rate}")
            audio = self._resample_audio(audio, sample_rate, self.model_config.metadata.sample_rate)
            sample_rate = self.model_config.metadata.sample_rate
        
        # Postprocess
        audio = self._postprocess_audio(audio, sample_rate)
        
        # Resample if needed
        if output_sample_rate is not None and output_sample_rate != sample_rate:
            audio = self._resample_audio(audio, sample_rate, output_sample_rate)
            sample_rate = output_sample_rate
        
        return audio, sample_rate
    
    def _load_audio(
        self,
        audio_input: AudioInput,
        sample_rate: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """Load audio from various input formats.
        
        Args:
            audio_input: Audio input (path, numpy array, or torch tensor).
            sample_rate: Sample rate (required for array/tensor inputs).
            
        Returns:
            Tuple of (audio_array, sample_rate).
        """
        if isinstance(audio_input, (str, Path)):
            # Load from file
            path = Path(audio_input)
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")
            
            audio, sr = sf.read(str(path))
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            return audio.astype(np.float32), sr
        
        elif isinstance(audio_input, np.ndarray):
            if sample_rate is None:
                raise ValueError("sample_rate must be provided for numpy array input")
            # Convert to mono if stereo
            if len(audio_input.shape) > 1:
                audio_input = np.mean(audio_input, axis=1)
            return audio_input.astype(np.float32), sample_rate
        
        elif isinstance(audio_input, torch.Tensor):
            if sample_rate is None:
                raise ValueError("sample_rate must be provided for torch tensor input")
            # Convert to numpy
            audio_np = audio_input.detach().cpu().numpy()
            # Convert to mono if stereo
            if len(audio_np.shape) > 1:
                audio_np = np.mean(audio_np, axis=1)
            return audio_np.astype(np.float32), sample_rate
        
        else:
            raise TypeError(
                f"Unsupported audio input type: {type(audio_input)}. "
                "Supported types: str, Path, np.ndarray, torch.Tensor"
            )
    
    def _preprocess_audio(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Preprocess reference audio before synthesis.
        
        Subclasses can override this for model-specific preprocessing
        (e.g., normalization, trimming, feature extraction).
        
        Args:
            audio: Audio array.
            sample_rate: Sample rate.
            
        Returns:
            Preprocessed audio array.
        """
        return audio
    
    def _postprocess_audio(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Postprocess generated audio after synthesis.
        
        Subclasses can override this for model-specific postprocessing
        (e.g., normalization, denoising).
        
        Args:
            audio: Audio array.
            sample_rate: Sample rate.
            
        Returns:
            Postprocessed audio array.
        """
        # Ensure audio is in valid range [-1, 1]
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = np.clip(audio, -1.0, 1.0)
        return audio
    
    def _resample_audio(
        self,
        audio: np.ndarray,
        input_sr: int,
        output_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate.
        
        Uses scipy.signal.resample for resampling. Subclasses can override
        this method to use model-specific resampling (e.g., librosa.resample).
        
        Args:
            audio: Audio array.
            input_sr: Input sample rate.
            output_sr: Output sample rate.
            
        Returns:
            Resampled audio array.
        """
        if input_sr == output_sr:
            return audio
        
        # Calculate number of output samples
        num_samples = int(len(audio) * output_sr / input_sr)
        return signal.resample(audio, num_samples).astype(np.float32)
    
    def save_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_path: Union[str, Path],
        format: str = "WAV"
    ):
        """Save audio to file.
        
        Args:
            audio: Audio array.
            sample_rate: Sample rate.
            output_path: Path to save audio file.
            format: Audio format (WAV, FLAC, etc.).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, sample_rate, format=format)
    
    def to(self, device: Union[str, torch.device]):
        """Move model to specified device.
        
        Args:
            device: Target device.
        """
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
        return self
    
    def train(self):
        """Set model to training mode."""
        if self.model is not None:
            self.model.train()
        return self
    
    def __call__(
        self,
        text: str,
        reference_audio: AudioInput,
        reference_sample_rate: Optional[int] = None,
        **kwargs
    ) -> AudioOutput:
        """Call the model for synthesis.
        
        Args:
            text: Input text.
            reference_audio: Reference audio for voice cloning.
            reference_sample_rate: Sample rate of reference audio.
            **kwargs: Additional parameters.
            
        Returns:
            Tuple of (audio_array, sample_rate).
        """
        return self.synthesize(
            text,
            reference_audio,
            reference_sample_rate,
            **kwargs
        )
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit."""
        pass
