from ttsdb_core import VoiceCloningTTSBase, AudioInput, AudioOutput, setup_vendor_path

# Uncomment to add vendored code to path (run `just fetch f5-tts` first)
# setup_vendor_path("ttsdb_f5_tts")
# from external_module import SomeClass  # Now you can import vendored code


class F5TTS(VoiceCloningTTSBase):
    """F5-TTS voice cloning TTS model.
    
    Config is automatically loaded and accessible via `self.model_config`.
    
    Example:
        >>> model = F5TTS(model_id="...")
        >>> print(model.model_config.metadata.name)
        'F5-TTS'
    """
    
    _package_name = "ttsdb_f5_tts"

    def _load_model(self, load_path):
        # TODO: Load the model from load_path (local path or HuggingFace ID)
        raise NotImplementedError("_load_model must be implemented")

    def _synthesize(self, text, reference_audio, reference_sample_rate, **kwargs):
        # TODO: Implement synthesis logic
        # Returns: tuple of (audio_array, sample_rate)
        raise NotImplementedError("_synthesize must be implemented")
