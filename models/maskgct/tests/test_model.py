"""Tests for MaskGCT."""

import os

import pytest
from ttsdb_core.testing import BaseModelConfigTests, BaseModelImportTests, BaseModelIntegrationTests
from ttsdb_core.vendor import get_vendor_path, vendor_context


class TestConfig(BaseModelConfigTests):
    PACKAGE_NAME = "ttsdb_maskgct"
    EXPECTED_MODEL_NAME = "MaskGCT"
    EXPECTED_SAMPLE_RATE = 24000
    EXPECTED_CODE_ROOT = "."


class TestModelImport(BaseModelImportTests):
    PACKAGE_NAME = "ttsdb_maskgct"
    MODEL_CLASS_PATH = "ttsdb_maskgct.MaskGCT"
    EXPECTED_SAMPLE_RATE_CONST = 24000


class TestMaskGCTSpecificImports:
    """MaskGCT-only import tests (non-generic)."""

    def test_maskgct_models_container_importable(self):
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
        vendor_path = get_vendor_path("ttsdb_maskgct")
        if not vendor_path.exists():
            pytest.skip("Vendored code not present; run: just fetch maskgct")

        with vendor_context("ttsdb_maskgct", cwd=True, env=None):
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
class TestModelIntegration(BaseModelIntegrationTests):
    PACKAGE_NAME = "ttsdb_maskgct"
    MODEL_CLASS_PATH = "ttsdb_maskgct.MaskGCT"
    WEIGHTS_PREPARE_HINT = "maskgct"
    EXPECTED_SAMPLE_RATE = 24000

    def test_synthesize_with_different_languages(self, synthesis_results):
        """Synthesis should work with different language codes."""
        for lang in ["eng", "zho"]:
            if (lang, 0) not in synthesis_results:
                continue
            audio, sr = synthesis_results[(lang, 0)]
            assert audio is not None
            assert sr == 24000