"""
PyTorch Dataset for Moirai fine-tuning.

Converts time-series samples to PyTorch format compatible with Moirai's forecasting API.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict


class MoiraiFineTuneDataset(Dataset):
    """
    PyTorch Dataset for Moirai fine-tuning.

    Returns context/target pairs for forecasting task:
    - Context: First `context_length` timesteps (used for conditioning)
    - Target: Next `prediction_length` timesteps (what model should predict)

    Args:
        data: Time-series data of shape (n_samples, seq_length, n_features)
        context_length: Number of timesteps for context window (default: 96)
        prediction_length: Number of timesteps to predict (default: 32)

    Returns:
        Dictionary with:
        - 'context': Tensor of shape (context_length, n_features)
        - 'target': Tensor of shape (prediction_length, n_features)
        - 'past_is_pad': Boolean tensor indicating valid timesteps (all True)
    """

    def __init__(
        self,
        data: np.ndarray,
        context_length: int = 96,
        prediction_length: int = 32
    ):
        """
        Initialize dataset.

        Args:
            data: NumPy array of shape (n_samples, seq_length, n_features)
            context_length: Length of context window
            prediction_length: Length of prediction window
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be numpy array, got {type(data)}")

        if data.ndim != 3:
            raise ValueError(f"data must be 3D (n_samples, seq_length, n_features), got shape {data.shape}")

        if data.shape[1] < context_length + prediction_length:
            raise ValueError(
                f"Sequence length ({data.shape[1]}) must be >= "
                f"context_length + prediction_length ({context_length + prediction_length})"
            )

        self.data = torch.FloatTensor(data)
        self.context_length = context_length
        self.prediction_length = prediction_length

        self.n_samples = len(data)
        self.n_features = data.shape[2]

    def __len__(self) -> int:
        """Return number of samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with context, target, and past_is_pad tensors
        """
        sample = self.data[idx]  # Shape: (seq_length, n_features)

        # Split into context and target
        context = sample[:self.context_length]  # Shape: (context_length, n_features)
        target = sample[
            self.context_length:self.context_length + self.prediction_length
        ]  # Shape: (prediction_length, n_features)

        # All timesteps are valid (no padding, all values observed)
        past_is_pad = torch.zeros(self.context_length, dtype=torch.bool)
        past_observed_target = torch.ones(self.context_length, self.n_features, dtype=torch.bool)

        # All values are observed (not missing)
        past_observed_target = torch.ones(self.context_length, self.n_features, dtype=torch.bool)

        return {
            'context': context,
            'target': target,
            'past_is_pad': past_is_pad,
            'past_observed_target': past_observed_target
        }

    def get_sample_info(self) -> Dict[str, int]:
        """Return dataset information."""
        return {
            'n_samples': self.n_samples,
            'context_length': self.context_length,
            'prediction_length': self.prediction_length,
            'n_features': self.n_features,
            'total_length': self.data.shape[1]
        }
