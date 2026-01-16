"""
Base IDS Interface

Defines the abstract interface that all traditional IDS methods must implement.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional


class BaseIDS(ABC):
    """
    Abstract base class for traditional Intrusion Detection Systems.

    All IDS implementations must inherit from this class and implement
    the required methods: fit(), predict(), and predict_proba().
    """

    def __init__(self, seq_length: int = 128, feature_dim: int = 12):
        """
        Initialize the IDS.

        Args:
            seq_length: Length of time series sequences (default: 128)
            feature_dim: Number of features per timestep (default: 12)
        """
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self._fitted = False

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> 'BaseIDS':
        """
        Train the IDS on benign traffic.

        Args:
            X_train: Training data of shape (n_samples, seq_length, feature_dim)
            y_train: Optional labels (not used by unsupervised methods)

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels for test data.

        Args:
            X: Test data of shape (n_samples, seq_length, feature_dim)

        Returns:
            Binary predictions of shape (n_samples,) where 0=benign, 1=attack
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores for test data.

        Args:
            X: Test data of shape (n_samples, seq_length, feature_dim)

        Returns:
            Anomaly scores of shape (n_samples,) in range [0, 1]
            where higher scores indicate higher likelihood of attack
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict:
        """
        Get configuration dictionary for reproducibility.

        Returns:
            Dictionary containing all configuration parameters
        """
        pass

    def _check_fitted(self):
        """Check if the IDS has been fitted."""
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before prediction. "
                "Call fit() first."
            )

    def _validate_input(self, X: np.ndarray):
        """
        Validate input data shape.

        Args:
            X: Input data

        Raises:
            ValueError: If input shape is incorrect
        """
        if X.ndim != 3:
            raise ValueError(
                f"Expected 3D input (n_samples, seq_length, feature_dim), "
                f"got {X.ndim}D array"
            )

        if X.shape[1] != self.seq_length:
            raise ValueError(
                f"Expected seq_length={self.seq_length}, got {X.shape[1]}"
            )

        if X.shape[2] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {X.shape[2]}"
            )
