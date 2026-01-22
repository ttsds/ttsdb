"""Abstract base classes for TTS and voice conversion models."""

from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path

import numpy as np


class BaseTTS(ABC):
    """Abstract base class for Text-to-Speech models.

    Args:
        model_id: HuggingFace model identifier (e.g., "microsoft/speecht5_tts")
    """

    def __init__(self, model_id: str):
        """Initialize the TTS model with a HuggingFace model ID.

        Args:
            model_id: HuggingFace model identifier
        """
        self.model_id = model_id

    @abstractmethod
    def synthesize(
        self,
        text: str,
        speaker_reference: str = None,
        text_reference: str = None,
        language: str = None,
        **kwargs,
    ) -> tuple[np.ndarray, int]:
        """Synthesize speech from text.

        Args:
            text: Input text to synthesize
            speaker_reference: Optional speaker reference audio file path
            text_reference: Optional text reference audio file path
            language: Optional language code
                        If None, should return audio data as bytes.
            **kwargs: Additional model-specific parameters

        Returns:
            Audio data as numpy array and sampling rate
        """
        pass

    def start(self):
        """Start the TTS model, if necessary."""
        pass

    def stop(self):
        """Stop the TTS model, if necessary."""
        pass

    def __enter__(self):
        """Enter the context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager."""
        self.stop()

    def __call__(self, *args, **kwargs):
        """Call the TTS model."""
        return self.synthesize(*args, **kwargs)


class BaseVoiceConversion(ABC):
    """Abstract base class for Voice Conversion models.

    This class defines the interface for voice conversion models that use HuggingFace model IDs.
    Subclasses should implement the convert method to convert voice characteristics
    from source audio to target audio.

    Args:
        model_id: HuggingFace model identifier (e.g., "huggingface/voice-conversion-model")
    """

    def __init__(self, model_id: str):
        """Initialize the voice conversion model with a HuggingFace model ID.

        Args:
            model_id: HuggingFace model identifier
        """
        self.model_id = model_id

    @abstractmethod
    def convert(
        self,
        source_audio: Union[str, Path, bytes],
        target_audio: Union[str, Path, bytes],
        output_path: Union[str, Path, None] = None,
        **kwargs,
    ) -> Union[bytes, Path, str]:
        """Convert voice from source audio to match target audio characteristics.

        Args:
            source_audio: Source audio file path or audio data (to be converted)
            target_audio: Target audio file path or audio data (voice reference)
            output_path: Optional path to save the output audio file.
                        If None, should return audio data as bytes.
            **kwargs: Additional model-specific parameters

        Returns:
            Converted audio data as bytes, or path to saved audio file if output_path is provided
        """
        pass
