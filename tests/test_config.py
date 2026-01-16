"""Tests for configuration management utilities."""

import pytest
import yaml
import os
from pathlib import Path
from src.utils.config import Config


class TestConfigLoading:
    """Test configuration file loading."""

    def test_load_default_config(self, config_dir):
        """Should load default config.yaml."""
        config = Config(config_dir / "config.yaml")

        assert config is not None
        assert config.raw is not None
        assert isinstance(config.raw, dict)

    def test_load_custom_config_path(self, sample_config):
        """Should load config from custom path."""
        config = Config(sample_config)

        assert config.get("project.name") == "Test Project"
        assert config.get("project.version") == "0.1.0"

    def test_invalid_config_path_raises(self):
        """Should raise error for non-existent config file."""
        with pytest.raises(FileNotFoundError):
            Config("nonexistent/config.yaml")

    def test_config_get_nested_keys(self, sample_config):
        """Should access nested keys using dot notation."""
        config = Config(sample_config)

        assert config.get("models.diffusion_ts.seq_length") == 32
        assert config.get("models.diffusion_ts.feature_dim") == 12
        assert config.get("models.moirai.model_size") == "small"

    def test_config_get_with_default(self, sample_config):
        """Should return default for non-existent keys."""
        config = Config(sample_config)

        assert config.get("nonexistent.key", "default") == "default"
        assert config.get("models.nonexistent", 42) == 42


class TestEnvironmentOverrides:
    """Test environment variable overrides."""

    def test_use_gpu_env_override(self, sample_config, monkeypatch):
        """Should override use_gpu from environment."""
        monkeypatch.setenv("USE_GPU", "true")

        config = Config(sample_config)
        assert config.get("device.use_gpu") is True

        monkeypatch.setenv("USE_GPU", "false")
        config = Config(sample_config)
        assert config.get("device.use_gpu") is False

    def test_cuda_visible_devices_override(self, sample_config, monkeypatch):
        """Should override gpu_id from CUDA_VISIBLE_DEVICES."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

        config = Config(sample_config)
        assert config.get("device.gpu_id") == 1

    def test_env_override_precedence(self, sample_config, monkeypatch):
        """Environment variables should take precedence over config file."""
        # Config file has use_gpu: false
        monkeypatch.setenv("USE_GPU", "true")

        config = Config(sample_config)
        assert config.get("device.use_gpu") is True  # ENV overrides file

    def test_boolean_env_parsing(self, sample_config, monkeypatch):
        """Should correctly parse boolean environment variables."""
        # Test various boolean string formats
        for value in ["true", "True", "TRUE"]:
            monkeypatch.setenv("USE_GPU", value)
            config = Config(sample_config)
            assert config.get("device.use_gpu") is True

        for value in ["false", "False", "FALSE"]:
            monkeypatch.setenv("USE_GPU", value)
            config = Config(sample_config)
            assert config.get("device.use_gpu") is False


class TestConfigValidation:
    """Test configuration validation and edge cases."""

    def test_invalid_yaml_raises(self, tmp_path):
        """Should raise error for malformed YAML."""
        bad_config = tmp_path / "bad_config.yaml"
        bad_config.write_text("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            Config(bad_config)

    def test_config_raw_property(self, sample_config):
        """Should provide access to raw config dict."""
        config = Config(sample_config)

        raw = config.raw
        assert isinstance(raw, dict)
        assert "project" in raw
        assert "models" in raw

    def test_deep_nested_get(self, tmp_path):
        """Should handle deeply nested keys."""
        deep_config = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {"value": "deep"}
                    }
                }
            }
        }

        config_path = tmp_path / "deep_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(deep_config, f)

        config = Config(config_path)
        assert config.get("level1.level2.level3.level4.value") == "deep"

    def test_get_nonexistent_key_returns_default(self, sample_config):
        """Should return default for completely missing keys."""
        config = Config(sample_config)

        assert config.get("missing") is None
        assert config.get("missing.nested.key") is None
        assert config.get("missing.nested.key", "fallback") == "fallback"

    def test_config_reload(self, tmp_path):
        """Should reflect config file at initialization time."""
        config_path = tmp_path / "test_config.yaml"

        # Write initial config
        initial = {"value": 1}
        with open(config_path, "w") as f:
            yaml.dump(initial, f)

        config = Config(config_path)
        assert config.get("value") == 1

        # Modify file (config should still have old value)
        modified = {"value": 2}
        with open(config_path, "w") as f:
            yaml.dump(modified, f)

        # Config object holds initial state
        assert config.get("value") == 1

        # New Config object gets new value
        new_config = Config(config_path)
        assert new_config.get("value") == 2

    def test_empty_config_file(self, tmp_path):
        """Should handle empty config file."""
        empty_config = tmp_path / "empty.yaml"
        empty_config.write_text("")

        config = Config(empty_config)
        # Empty YAML returns None, which should be handled
        assert config.get("any.key") is None

    def test_config_with_special_characters(self, tmp_path):
        """Should handle keys with special characters."""
        special_config = {
            "key-with-dashes": "value1",
            "key_with_underscores": "value2",
            "keyWithCamelCase": "value3"
        }

        config_path = tmp_path / "special.yaml"
        with open(config_path, "w") as f:
            yaml.dump(special_config, f)

        config = Config(config_path)
        assert config.get("key-with-dashes") == "value1"
        assert config.get("key_with_underscores") == "value2"
        assert config.get("keyWithCamelCase") == "value3"
