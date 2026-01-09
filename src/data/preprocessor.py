"""
Data Preprocessor for IoT Network Traffic

Converts raw network flow features into normalized time-series format.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, Literal
import pickle
from pathlib import Path
from loguru import logger


class TrafficPreprocessor:
    """
    Preprocessor for network traffic data.

    Handles feature normalization and maintains scaler state for
    consistent preprocessing across training and inference.
    """

    def __init__(self, scaler_type: Literal['standard', 'minmax'] = 'standard'):
        """
        Initialize preprocessor.

        Args:
            scaler_type: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self._fitted = False

    def fit(self, data: np.ndarray) -> 'TrafficPreprocessor':
        """
        Fit the scaler on training data.

        Args:
            data: Array of shape (n_samples, n_features) or
                  (n_samples, seq_length, n_features)

        Returns:
            Self for method chaining
        """
        # Flatten if 3D for fitting
        original_shape = data.shape
        if data.ndim == 3:
            data_flat = data.reshape(-1, data.shape[-1])
        else:
            data_flat = data

        # Initialize scaler
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
            logger.info("Using StandardScaler (zero mean, unit variance)")
        else:
            self.scaler = MinMaxScaler()
            logger.info("Using MinMaxScaler (range [0, 1])")

        # Fit scaler
        self.scaler.fit(data_flat)
        self._fitted = True

        logger.success(f"Scaler fitted on {len(data_flat)} samples")
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.

        Args:
            data: Array of shape (n_samples, n_features) or
                  (n_samples, seq_length, n_features)

        Returns:
            Normalized data of same shape
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        original_shape = data.shape

        # Flatten if 3D
        if data.ndim == 3:
            n_samples, seq_len, n_features = data.shape
            data_flat = data.reshape(-1, n_features)
        else:
            data_flat = data

        # Transform
        normalized = self.scaler.transform(data_flat)

        # Reshape back if needed
        if len(original_shape) == 3:
            normalized = normalized.reshape(original_shape)

        return normalized

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data in one step."""
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.

        Args:
            data: Normalized array

        Returns:
            Data in original scale
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        original_shape = data.shape

        # Flatten if 3D
        if data.ndim == 3:
            data_flat = data.reshape(-1, data.shape[-1])
        else:
            data_flat = data

        # Inverse transform
        original_scale = self.scaler.inverse_transform(data_flat)

        # Reshape back if needed
        if len(original_shape) == 3:
            original_scale = original_scale.reshape(original_shape)

        return original_scale

    def save(self, path: str):
        """
        Save fitted scaler to disk.

        Args:
            path: File path to save scaler (e.g., 'preprocessor.pkl')
        """
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted scaler")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'scaler_type': self.scaler_type
            }, f)

        logger.success(f"Scaler saved to {save_path}")

    def load(self, path: str):
        """
        Load fitted scaler from disk.

        Args:
            path: File path to load scaler from
        """
        load_path = Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {load_path}")

        with open(load_path, 'rb') as f:
            saved_data = pickle.load(f)

        self.scaler = saved_data['scaler']
        self.scaler_type = saved_data['scaler_type']
        self._fitted = True

        logger.success(f"Scaler loaded from {load_path}")


def create_sequences(
    data: np.ndarray,
    seq_length: int,
    stride: int = 1
) -> np.ndarray:
    """
    Convert raw flow records into overlapping time-series sequences.

    Args:
        data: Array of shape (n_records, n_features)
        seq_length: Desired sequence length
        stride: Step size between sequences (default=1 for max overlap)

    Returns:
        Array of shape (n_sequences, seq_length, n_features)

    Example:
        >>> data = np.random.randn(1000, 12)  # 1000 flows, 12 features
        >>> sequences = create_sequences(data, seq_length=128, stride=64)
        >>> sequences.shape
        (14, 128, 12)  # 14 sequences of length 128
    """
    if len(data) < seq_length:
        raise ValueError(
            f"Data length ({len(data)}) is less than sequence length ({seq_length})"
        )

    n_records, n_features = data.shape
    n_sequences = (n_records - seq_length) // stride + 1

    sequences = np.zeros((n_sequences, seq_length, n_features))

    for i in range(n_sequences):
        start_idx = i * stride
        end_idx = start_idx + seq_length
        sequences[i] = data[start_idx:end_idx]

    logger.info(
        f"Created {n_sequences} sequences of length {seq_length} "
        f"(stride={stride}) from {n_records} records"
    )

    return sequences


def train_val_test_split(
    data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    shuffle: bool = True,
    random_state: int = 42
) -> Tuple:
    """
    Split data into train/validation/test sets.

    Args:
        data: Input data array
        labels: Optional labels array
        train_ratio: Fraction for training (default 0.7)
        val_ratio: Fraction for validation (default 0.15)
        shuffle: Whether to shuffle before splitting
        random_state: Random seed for reproducibility

    Returns:
        If labels provided: (X_train, X_val, X_test, y_train, y_val, y_test)
        If no labels: (X_train, X_val, X_test)
    """
    n_samples = len(data)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)

    # Calculate split indices
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train = data[train_idx]
    X_val = data[val_idx]
    X_test = data[test_idx]

    logger.info(
        f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}"
    )

    if labels is not None:
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        y_test = labels[test_idx]
        return X_train, X_val, X_test, y_train, y_val, y_test

    return X_train, X_val, X_test
