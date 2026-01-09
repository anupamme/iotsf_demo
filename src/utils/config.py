"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Any
import os


class Config:
    """Configuration manager for the application."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._apply_env_overrides()

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _apply_env_overrides(self):
        """Override config with environment variables."""
        if os.getenv("USE_GPU") is not None:
            self._config["device"]["use_gpu"] = (
                os.getenv("USE_GPU").lower() == "true"
            )
        if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
            self._config["device"]["gpu_id"] = int(
                os.getenv("CUDA_VISIBLE_DEVICES", 0)
            )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Example:
            config.get("models.moirai.model_size")

        Args:
            key: Configuration key in dot notation
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    @property
    def raw(self) -> dict:
        """Get raw configuration dictionary."""
        return self._config
