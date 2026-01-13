"""
Threshold-based IDS

Simple percentile-based anomaly detection. Flags traffic as malicious
if extracted features exceed pre-computed thresholds.
"""

import numpy as np
from typing import Dict, Optional
from loguru import logger

from .base import BaseIDS
from .feature_extraction import extract_batch_features


class ThresholdIDS(BaseIDS):
    """
    Threshold-based Intrusion Detection System.

    Uses percentile-based thresholds computed from benign traffic.
    Flags samples as attacks if too many features exceed thresholds.

    This is the simplest baseline - easy to understand but limited in
    detecting sophisticated attacks.
    """

    def __init__(
        self,
        seq_length: int = 128,
        feature_dim: int = 12,
        percentile: float = 95,
        violation_threshold: int = 3
    ):
        """
        Initialize Threshold IDS.

        Args:
            seq_length: Length of time series sequences
            feature_dim: Number of features per timestep
            percentile: Percentile for threshold computation (default: 95)
            violation_threshold: Number of features that must exceed
                               thresholds to flag as attack (default: 3)
        """
        super().__init__(seq_length, feature_dim)
        self.percentile = percentile
        self.violation_threshold = violation_threshold
        self.thresholds = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> 'ThresholdIDS':
        """
        Compute thresholds from benign training data.

        Args:
            X_train: Benign traffic of shape (n_samples, seq_length, feature_dim)
            y_train: Ignored (unsupervised method)

        Returns:
            Self for method chaining
        """
        self._validate_input(X_train)

        logger.info(f"Computing {self.percentile}th percentile thresholds...")

        # Extract statistical features
        X_features = extract_batch_features(X_train)

        # Compute percentile thresholds for each feature
        self.thresholds = np.percentile(X_features, self.percentile, axis=0)

        self._fitted = True
        logger.success(
            f"ThresholdIDS fitted on {len(X_train)} samples. "
            f"Computed {len(self.thresholds)} thresholds."
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels.

        Args:
            X: Test data of shape (n_samples, seq_length, feature_dim)

        Returns:
            Binary predictions (0=benign, 1=attack) of shape (n_samples,)
        """
        scores = self.predict_proba(X)
        return (scores >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores based on threshold violations.

        Args:
            X: Test data of shape (n_samples, seq_length, feature_dim)

        Returns:
            Anomaly scores of shape (n_samples,) in range [0, 1]
        """
        self._check_fitted()
        self._validate_input(X)

        # Extract features
        X_features = extract_batch_features(X)

        # Count threshold violations for each sample
        violations = np.sum(X_features > self.thresholds, axis=1)

        # Normalize to [0, 1] based on violation_threshold
        # If violations >= violation_threshold, score = 1.0
        # Otherwise, score = violations / violation_threshold
        scores = np.minimum(violations / self.violation_threshold, 1.0)

        return scores

    def get_config(self) -> Dict:
        """Get configuration dictionary."""
        return {
            'method': 'threshold',
            'seq_length': self.seq_length,
            'feature_dim': self.feature_dim,
            'percentile': self.percentile,
            'violation_threshold': self.violation_threshold,
            'fitted': self._fitted
        }
