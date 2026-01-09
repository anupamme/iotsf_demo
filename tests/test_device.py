"""Tests for device utility functions."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.device import get_device, is_cuda_available, get_device_info


class TestGPUUtils:
    def test_get_device_returns_device(self):
        """Test that get_device returns a torch device."""
        device = get_device()
        assert device is not None
        assert str(device) in ["cuda:0", "cpu"]

    def test_is_cuda_available_returns_bool(self):
        """Test that is_cuda_available returns a boolean."""
        result = is_cuda_available()
        assert isinstance(result, bool)

    def test_get_device_info_returns_dict(self):
        """Test that get_device_info returns a dictionary."""
        info = get_device_info()
        assert isinstance(info, dict)
        assert "cuda_available" in info
        assert "device_count" in info
        assert "pytorch_version" in info

    def test_cpu_fallback(self):
        """Test that CPU fallback works when GPU is disabled."""
        device = get_device(use_gpu=False)
        assert str(device) == "cpu"
