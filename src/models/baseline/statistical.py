"""
Statistical IDS

Uses Z-score and IQR (Interquartile Range) methods for anomaly detection.
Flags samples that are statistical outliers compared to benign traffic.
"""

import numpy as np
from typing import Dict, Optional
from loguru import logger

from .base import BaseIDS
from .feature_extraction import extract_batch_features


class StatisticalIDS(BaseIDS):
    """
    Statistical Intrusion Detection System.

    Combines two statistical outlier detection methods:
    1. Z-score: Detects samples with extreme standardized deviations
    2. IQR: Detects samples outside the interquartile range

    Flags as attack if EITHER method triggers.
    """

    def __init__(
        self,
        seq_length: int = 128,
        feature_dim: int = 12,
        z_score_threshold: float = 3.0,
        iqr_multiplier: float = 1.5
    ):
        """
        Initialize Statistical IDS.

        Args:
            seq_length: Length of time series sequences
            feature_dim: Number of features per timestep
            z_score_threshold: Z-score threshold for anomaly (default: 3.0 = 99.7%)
            iqr_multiplier: IQR multiplier for outlier detection (default: 1.5)
        """
        super().__init__(seq_length, feature_dim)
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier

        # Learned parameters
        self.mean = None
        self.std = None
        self.q1 = None
        self.q3 = None
        self.iqr = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> 'StatisticalIDS':
        """
        Compute statistical parameters from benign training data.

        Args:
            X_train: Benign traffic of shape (n_samples, seq_length, feature_dim)
            y_train: Ignored (unsupervised method)

        Returns:
            Self for method chaining
        """
        self._validate_input(X_train)

        logger.info("Computing statistical parameters (mean, std, IQR)...")

        # Extract statistical features
        X_features = extract_batch_features(X_train)

        # Compute Gaussian parameters for Z-score
        self.mean = np.mean(X_features, axis=0)
        self.std = np.std(X_features, axis=0)

        # Avoid division by zero
        self.std = np.maximum(self.std, 1e-6)

        # Compute IQR parameters
        self.q1 = np.percentile(X_features, 25, axis=0)
        self.q3 = np.percentile(X_features, 75, axis=0)
        self.iqr = self.q3 - self.q1

        self._fitted = True
        logger.success(
            f"StatisticalIDS fitted on {len(X_train)} samples. "
            f"Z-score threshold: {self.z_score_threshold}, "
            f"IQR multiplier: {self.iqr_multiplier}"
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels using statistical outlier detection.

        Args:
            X: Test data of shape (n_samples, seq_length, feature_dim)

        Returns:
            Binary predictions (0=benign, 1=attack) of shape (n_samples,)
        """
        scores = self.predict_proba(X)
        return (scores >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores using Z-score and IQR methods.

        Args:
            X: Test data of shape (n_samples, seq_length, feature_dim)

        Returns:
            Anomaly scores of shape (n_samples,) in range [0, 1]
        """
        self._check_fitted()
        self._validate_input(X)

        # Extract features
        X_features = extract_batch_features(X)

        # Method 1: Z-score detection
        z_scores = np.abs((X_features - self.mean) / self.std)
        max_z_scores = np.max(z_scores, axis=1)  # Max Z-score across all features

        # Normalize Z-scores to [0, 1]
        # If max Z-score >= threshold, assign score close to 1
        z_score_normalized = np.minimum(max_z_scores / self.z_score_threshold, 1.0)

        # Method 2: IQR outlier detection
        lower_bound = self.q1 - self.iqr_multiplier * self.iqr
        upper_bound = self.q3 + self.iqr_multiplier * self.iqr

        # Count IQR violations per sample
        iqr_violations = np.sum(
            (X_features < lower_bound) | (X_features > upper_bound),
            axis=1
        )

        # Normalize IQR violations (>= 1 violation means potential anomaly)
        iqr_score_normalized = np.minimum(iqr_violations / 3.0, 1.0)

        # Combine methods: Use maximum of both scores (OR logic)
        combined_scores = np.maximum(z_score_normalized, iqr_score_normalized)

        return combined_scores

    def get_config(self) -> Dict:
        """Get configuration dictionary."""
        return {
            'method': 'statistical',
            'seq_length': self.seq_length,
            'feature_dim': self.feature_dim,
            'z_score_threshold': self.z_score_threshold,
            'iqr_multiplier': self.iqr_multiplier,
            'fitted': self._fitted
        }
