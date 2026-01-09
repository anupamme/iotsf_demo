"""GPU/CPU device detection and management utilities."""

import torch
import logging
from typing import Union

logger = logging.getLogger(__name__)


def get_device(use_gpu: bool = True, gpu_id: int = 0) -> torch.device:
    """
    Get PyTorch device (GPU or CPU).

    Args:
        use_gpu: Whether to use GPU if available
        gpu_id: GPU device ID to use

    Returns:
        torch.device: Selected device
    """
    if use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.2f} GB"
        )
    else:
        device = torch.device("cpu")
        if use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but not available. Falling back to CPU.")
        else:
            logger.info("Using CPU")
    return device


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def get_device_info() -> dict:
    """Get detailed device information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "pytorch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    return info
