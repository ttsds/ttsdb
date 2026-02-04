"""Tests for ttsdb_core.config module."""

import pytest
import yaml
from ttsdb_core.config import ModelConfig, _deep_merge


class TestDeepMerge:
    """Tests for _deep_merge helper function."""

    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 20, "z": 30}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}

    def test_override_replaces_non_dict(self):
        base = {"a": {"x": 1}, "b": [1, 2, 3]}
        override = {"b": [4, 5]}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1}, "b": [4, 5]}


class TestModelConfigVariants:
    """Tests for ModelConfig variant support."""

    @pytest.fixture
    def config_with_variants(self, tmp_path):
        """Create a temporary config file with variants.

        Note: Model-specific config details (like HF subdir mappings) are now
        in prepare_weights.py, not config.yaml. Variants in config only contain
        identifiers and metadata overrides.
        """
        config_data = {
            "metadata": {
                "name": "TestModel",
                "sample_rate": 24000,
            },
            "weights": {
                "url": "https://huggingface.co/test/model",
                "license": "mit",
            },
            "variants": {
                "default": "base",
                "base": {
                    # Variants can override any config section
                    "metadata": {
                        "description": "Base variant",
                    },
                },
                "large": {
                    "metadata": {
                        "description": "Large variant",
                        "num_parameters": 1000,
                    },
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        return config_path

    @pytest.fixture
    def config_without_variants(self, tmp_path):
        """Create a temporary config file without variants."""
        config_data = {
            "metadata": {
                "name": "SimpleModel",
                "sample_rate": 16000,
            },
            "weights": {
                "url": "https://huggingface.co/test/simple",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        return config_path

    def test_load_default_variant(self, config_with_variants):
        config = ModelConfig.from_yaml(config_with_variants)
        assert config.variant == "base"
        assert config.metadata.description == "Base variant"

    def test_load_specific_variant(self, config_with_variants):
        config = ModelConfig.from_yaml(config_with_variants, variant="large")
        assert config.variant == "large"
        assert config.metadata.description == "Large variant"
        # Check that variant-specific metadata is merged
        assert config.metadata.num_parameters == 1000
        # Check that base metadata is preserved
        assert config.metadata.name == "TestModel"
        assert config.metadata.sample_rate == 24000

    def test_load_explicit_default_variant(self, config_with_variants):
        config = ModelConfig.from_yaml(config_with_variants, variant="base")
        assert config.variant == "base"
        assert config.metadata.description == "Base variant"

    def test_unknown_variant_raises(self, config_with_variants):
        with pytest.raises(ValueError, match="Unknown variant 'nonexistent'"):
            ModelConfig.from_yaml(config_with_variants, variant="nonexistent")

    def test_available_variants(self, config_with_variants):
        config = ModelConfig.from_yaml(config_with_variants)
        assert set(config.available_variants) == {"base", "large"}

    def test_default_variant_property(self, config_with_variants):
        config = ModelConfig.from_yaml(config_with_variants)
        assert config.default_variant == "base"

    def test_config_without_variants(self, config_without_variants):
        config = ModelConfig.from_yaml(config_without_variants)
        assert config.variant is None
        assert config.available_variants == []
        assert config.default_variant is None
        assert config.metadata.name == "SimpleModel"

    def test_config_without_variants_ignores_variant_param(self, config_without_variants):
        # When no variants defined, variant param should be ignored (returns base config)
        config = ModelConfig.from_yaml(config_without_variants, variant="anything")
        assert config.variant is None
        assert config.metadata.name == "SimpleModel"


class TestModelConfigAttributeAccess:
    """Tests for ModelConfig attribute access."""

    def test_dict_access(self, tmp_path):
        config_data = {"metadata": {"name": "Test"}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = ModelConfig.from_yaml(config_path)
        assert config["metadata"]["name"] == "Test"

    def test_attribute_access(self, tmp_path):
        config_data = {"metadata": {"name": "Test", "sample_rate": 24000}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = ModelConfig.from_yaml(config_path)
        assert config.metadata.name == "Test"
        assert config.metadata.sample_rate == 24000

    def test_missing_attribute_raises(self, tmp_path):
        config_data = {"metadata": {"name": "Test"}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = ModelConfig.from_yaml(config_path)
        with pytest.raises(AttributeError, match="no attribute 'nonexistent'"):
            _ = config.nonexistent
