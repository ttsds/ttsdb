"""Tests for MaskGCT."""

import os
import pytest
from ttsdb_core import ModelConfig, vendor_context, get_vendor_path


class TestConfig:
    """Test config loading."""
    
    def test_config_loads_from_package(self):
        """Config should load from the package."""
        config = ModelConfig.from_package("ttsdb_maskgct")
        assert config is not None
    
    def test_config_has_metadata(self):
        """Config should have metadata section."""
        config = ModelConfig.from_package("ttsdb_maskgct")
        assert "metadata" in config
        assert config.metadata.name == "MaskGCT"
    
    def test_config_has_sample_rate(self):
        """Config should specify sample rate."""
        config = ModelConfig.from_package("ttsdb_maskgct")
        assert config.metadata.sample_rate == 24000
    
    def test_config_has_code_root(self):
        """Config should specify code root for vendor path."""
        config = ModelConfig.from_package("ttsdb_maskgct")
        assert config.code.root == "."


class TestModelImport:
    """Test model class imports."""
    
    def test_model_class_importable(self):
        """Model class should be importable."""
        from ttsdb_maskgct import MaskGCT
        assert MaskGCT is not None
    
    def test_model_has_package_name(self):
        """Model should have _package_name set."""
        from ttsdb_maskgct import MaskGCT
        assert MaskGCT._package_name == "ttsdb_maskgct"
    
    def test_model_has_sample_rate(self):
        """Model should have SAMPLE_RATE constant."""
        from ttsdb_maskgct import MaskGCT
        assert MaskGCT.SAMPLE_RATE == 24000
    
    def test_maskgct_models_container_importable(self):
        """MaskGCTModels container should be importable."""
        from ttsdb_maskgct import MaskGCTModels
        assert MaskGCTModels is not None


class TestMaskGCTModelsContainer:
    """Test MaskGCTModels container class."""
    
    def test_container_stores_all_components(self):
        """Container should store all model components."""
        from ttsdb_maskgct import MaskGCTModels
        
        # Create mock components
        class MockModel:
            def to(self, device):
                return self
            def eval(self):
                return self
            def train(self):
                return self
        
        models = MaskGCTModels(
            semantic_model=MockModel(),
            semantic_codec=MockModel(),
            codec_encoder=MockModel(),
            codec_decoder=MockModel(),
            t2s_model=MockModel(),
            s2a_model_1layer=MockModel(),
            s2a_model_full=MockModel(),
            semantic_mean=1.0,
            semantic_std=1.0,
        )
        
        assert models.semantic_model is not None
        assert models.semantic_codec is not None
        assert models.codec_encoder is not None
        assert models.codec_decoder is not None
        assert models.t2s_model is not None
        assert models.s2a_model_1layer is not None
        assert models.s2a_model_full is not None
        assert models.semantic_mean == 1.0
        assert models.semantic_std == 1.0
    
    def test_container_to_method(self):
        """Container to() should move all models to device."""
        from ttsdb_maskgct import MaskGCTModels
        
        moved_to = []
        
        class MockModel:
            def to(self, device):
                moved_to.append(device)
                return self
            def eval(self):
                return self
            def train(self):
                return self
        
        models = MaskGCTModels(
            semantic_model=MockModel(),
            semantic_codec=MockModel(),
            codec_encoder=MockModel(),
            codec_decoder=MockModel(),
            t2s_model=MockModel(),
            s2a_model_1layer=MockModel(),
            s2a_model_full=MockModel(),
            semantic_mean=1.0,
            semantic_std=1.0,
        )
        
        models.to("cpu")
        # 7 models should have been moved
        assert len(moved_to) == 7
        assert all(d == "cpu" for d in moved_to)
    
    def test_container_eval_method(self):
        """Container eval() should set all models to eval mode."""
        from ttsdb_maskgct import MaskGCTModels
        
        eval_called = []
        
        class MockModel:
            def __init__(self, name):
                self.name = name
            def to(self, device):
                return self
            def eval(self):
                eval_called.append(self.name)
                return self
            def train(self):
                return self
        
        models = MaskGCTModels(
            semantic_model=MockModel("semantic"),
            semantic_codec=MockModel("codec"),
            codec_encoder=MockModel("encoder"),
            codec_decoder=MockModel("decoder"),
            t2s_model=MockModel("t2s"),
            s2a_model_1layer=MockModel("s2a_1layer"),
            s2a_model_full=MockModel("s2a_full"),
            semantic_mean=1.0,
            semantic_std=1.0,
        )
        
        models.eval()
        assert len(eval_called) == 7


class TestVendorContext:
    """Test vendor_context functionality."""
    
    def test_vendor_context_restores_cwd(self):
        """vendor_context should restore working directory after exit."""
        original_cwd = os.getcwd()
        
        # Use a directory we know exists
        with vendor_context("ttsdb_maskgct", cwd=True, env=None):
            # cwd might change (or fail if vendor not set up)
            pass
        
        # Should be restored
        assert os.getcwd() == original_cwd
    
    def test_vendor_context_restores_env_vars(self):
        """vendor_context should restore environment variables after exit."""
        # Set a test env var
        test_key = "TTSDB_TEST_VAR"
        original_value = os.environ.get(test_key)
        
        try:
            with vendor_context("ttsdb_maskgct", cwd=False, env={test_key: "test_value"}):
                assert os.environ.get(test_key) == "test_value"
            
            # Should be restored
            assert os.environ.get(test_key) == original_value
        finally:
            # Clean up
            if original_value is None:
                os.environ.pop(test_key, None)
            else:
                os.environ[test_key] = original_value
    
    def test_vendor_context_template_substitution(self):
        """vendor_context should substitute {vendor_path} in env values."""
        test_key = "TTSDB_TEST_PATH"
        
        try:
            with vendor_context("ttsdb_maskgct", cwd=False, env={test_key: "{vendor_path}"}) as vendor_path:
                assert os.environ.get(test_key) == str(vendor_path)
        finally:
            os.environ.pop(test_key, None)
    
    def test_get_vendor_path_returns_path(self):
        """get_vendor_path should return a Path object."""
        from pathlib import Path
        
        vendor_path = get_vendor_path("ttsdb_maskgct")
        assert isinstance(vendor_path, Path)


# Integration tests - require model weights and Amphion to be installed
@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for MaskGCT synthesis.
    
    These tests require:
    - Amphion to be vendored in _vendor/source/Amphion
    - Model weights to be downloaded (run: just hf prepare maskgct)
    
    Run with: just test-integration maskgct
    """
    
    @pytest.fixture
    def model(self, weights_path):
        """Load the MaskGCT model from local weights directory."""
        from ttsdb_maskgct import MaskGCT
        
        if not weights_path.exists():
            pytest.skip(f"Weights not found at {weights_path}. Run: just hf prepare maskgct")
        
        return MaskGCT(model_path=str(weights_path))
    
    def test_model_loads(self, model):
        """Model should load successfully."""
        assert model is not None
        assert model.model is not None
    
    def test_synthesize_returns_audio(self, model, reference_audio, test_data):
        """Synthesis should return audio array and sample rate."""
        sentences = test_data.get("test_sentences", {}).get("en", [])
        if not sentences:
            pytest.skip("No test sentences found")
        
        text = sentences[0]["text"]
        audio, sr = model.synthesize(
            text=text,
            reference_audio=reference_audio["path"],
            text_reference=reference_audio["text"],
            language=reference_audio["language"],
        )
        
        assert audio is not None
        assert len(audio) > 0
        assert sr == 24000
    
    def test_synthesize_with_different_languages(self, model, reference_audio):
        """Synthesis should work with different language codes."""
        for lang in ["en", "zh"]:
            audio, sr = model.synthesize(
                text="Test text",
                reference_audio=reference_audio["path"],
                text_reference=reference_audio["text"],
                language=lang,
            )
            assert audio is not None
            assert sr == 24000
    
    def test_generate_audio_examples(self, model, reference_audio, test_data, audio_examples_dir):
        """Generate audio examples and save to audio_examples/ directory."""
        import soundfile as sf
        
        sentences = test_data.get("test_sentences", {}).get("en", [])
        if not sentences:
            pytest.skip("No test sentences found")
        
        for i, sentence in enumerate(sentences):
            text = sentence["text"]
            audio, sr = model.synthesize(
                text=text,
                reference_audio=reference_audio["path"],
                text_reference=reference_audio["text"],
                language=reference_audio["language"],
            )
            
            # Save the audio
            output_path = audio_examples_dir / f"en_test_{i+1:03d}.wav"
            sf.write(str(output_path), audio, sr)
            print(f"Saved: {output_path}")
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0