"""Base class for voice cloning TTS models using PyTorch."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal

try:
    # Optional helpers. Some environments may have an older `transformers`
    # version installed; keep core importable even if these names don't exist.
    from transformers import AutoProcessor, AutoTokenizer
except Exception:  # pragma: no cover
    AutoProcessor = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

from .config import ModelConfig
from .vendor import get_vendor_path, setup_vendor_path, vendor_context
from .weights import find_checkpoint, get_variant_checkpoint_dir, resolve_weights_path

AudioInput = str | Path | np.ndarray | torch.Tensor
AudioOutput = tuple[np.ndarray, int]

__all__ = [
    "VoiceCloningTTSBase",
    "AudioInput",
    "AudioOutput",
    "ModelConfig",
    "setup_vendor_path",
    "get_vendor_path",
    "vendor_context",
    "find_checkpoint",
    "get_variant_checkpoint_dir",
    "resolve_weights_path",
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
    _package_name: str | None = None

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_id: str | None = None,
        device: str | torch.device | None = None,
        variant: str | None = None,
        **kwargs,
    ):
        """Initialize the voice cloning TTS model.

        Args:
            model_path: Path to local model directory. If provided, loads from disk.
            model_id: HuggingFace model identifier (e.g., "microsoft/speecht5_tts").
                     If provided and model_path is None, loads from HuggingFace Hub.
            device: Device to run the model on ('cpu', 'cuda', 'cuda:0', etc.).
                   If None, uses 'cuda' if available, else 'cpu'.
            variant: Model variant to use (e.g., "v1", "base"). If None, uses the
                    default variant specified in config, or base config if no variants.
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
        self._variant = variant

        # Load model config from package (with variant if specified)
        self.model_config: ModelConfig | None = None
        if self._package_name:
            try:
                self.model_config = ModelConfig.from_package(self._package_name, variant=variant)
            except Exception:
                pass  # Config loading is optional

        # Load model components
        self._load_model_components()

    @property
    def variant(self) -> str | None:
        """Currently loaded variant name."""
        if self.model_config is not None:
            return self.model_config.variant
        return self._variant

    def _setup_device(self, device: str | torch.device | None) -> torch.device:
        """Set up the computation device.

        Args:
            device: Device specification.

        Returns:
            torch.device object.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)

    def _resolve_model_path(self) -> Path:
        """Resolve a local path for model weights."""
        if self.model_path is not None:
            return self.model_path

        if self.model_id is None:
            raise ValueError("Either model_path or model_id must be provided")

        return resolve_weights_path(self.model_id)

    def _load_model_components(self):
        """Load model, tokenizer, and processor components."""
        load_path = self._resolve_model_path()

        # Load model (implemented by subclasses)
        self.model = self._load_model(str(load_path))
        for m in self._iter_torch_modules():
            # Only torch modules are moved/eval'ed. This allows wrappers that return
            # non-Module objects (e.g. service objects) while still supporting
            # multi-module models (e.g. model + vocoder).
            m.to(self.device)
            m.eval()

        # Try to load tokenizer and processor if they exist (optional helpers)
        if AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(str(load_path))
            except (OSError, ValueError, TypeError):
                self.tokenizer = None
        else:
            self.tokenizer = None

        if AutoProcessor is not None:
            try:
                self.processor = AutoProcessor.from_pretrained(str(load_path))
            except (OSError, ValueError, TypeError):
                self.processor = None
        else:
            self.processor = None

    @abstractmethod
    def _load_model(self, load_path: str):
        """Load the model architecture and weights.

        Args:
            load_path: Path to model (local or HuggingFace ID).

        Returns:
            A loaded model object. Typically a `torch.nn.Module`, but wrappers may
            return other objects (e.g. a container/service) as long as they also
            override `_iter_torch_modules()` to expose any underlying torch modules.
        """
        pass

    def _iter_torch_modules(self) -> list[torch.nn.Module]:
        """Return all torch modules that should follow `.to()/.eval()/.train()`.

        Default: only `self.model` if it is a `torch.nn.Module`.
        Subclasses can override to include e.g. vocoders, tokenizers with torch
        weights, or other auxiliary modules.
        """

        modules: list[torch.nn.Module] = []
        if isinstance(self.model, torch.nn.Module):
            modules.append(self.model)
        return modules

    @abstractmethod
    def _synthesize(
        self, text: str, reference_audio: np.ndarray, reference_sample_rate: int, **kwargs
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
        reference_sample_rate: int | None = None,
        output_sample_rate: int | None = None,
        **kwargs,
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
        audio, sample_rate = self._synthesize(text, ref_audio, ref_sr, **kwargs)
        if self.model_config.metadata.sample_rate != sample_rate:
            print(
                f"Warning: Resampling audio from {sample_rate} to {self.model_config.metadata.sample_rate}"
            )
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
        self, audio_input: AudioInput, sample_rate: int | None = None
    ) -> tuple[np.ndarray, int]:
        """Load audio from various input formats.

        Args:
            audio_input: Audio input (path, numpy array, or torch tensor).
            sample_rate: Sample rate (required for array/tensor inputs).

        Returns:
            Tuple of (audio_array, sample_rate).
        """
        if isinstance(audio_input, str | Path):
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

    def _preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
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

    def _postprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
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

    def _resample_audio(self, audio: np.ndarray, input_sr: int, output_sr: int) -> np.ndarray:
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
        self, audio: np.ndarray, sample_rate: int, output_path: str | Path, format: str = "WAV"
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

    def to(self, device: str | torch.device):
        """Move model to specified device.

        Args:
            device: Target device.
        """
        self.device = torch.device(device)
        for m in self._iter_torch_modules():
            m.to(self.device)
        return self

    def eval(self):
        """Set model to evaluation mode."""
        for m in self._iter_torch_modules():
            m.eval()
        return self

    def train(self):
        """Set model to training mode."""
        for m in self._iter_torch_modules():
            m.train()
        return self

    def __call__(
        self,
        text: str,
        reference_audio: AudioInput,
        reference_sample_rate: int | None = None,
        **kwargs,
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
        return self.synthesize(text, reference_audio, reference_sample_rate, **kwargs)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit."""
        pass
