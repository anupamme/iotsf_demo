"""
ML-based IDS using Isolation Forest

Uses machine learning for unsupervised anomaly detection.
Isolation Forest is designed to isolate anomalies rather than profile normal data.
"""

import numpy as np
from typing import Dict, Optional
from loguru import logger
from sklearn.ensemble import IsolationForest

from .base import BaseIDS
from .feature_extraction import extract_batch_features


class MLBasedIDS(BaseIDS):
    """
    ML-based Intrusion Detection System using Isolation Forest.

    Isolation Forest is an unsupervised algorithm that explicitly isolates
    anomalies instead of profiling normal points. It works well for
    high-dimensional data and is faster than One-Class SVM.

    Key advantages:
    - No assumptions about data distribution
    - Efficient for high-dimensional data
    - Explicitly designed for anomaly detection
    """

    def __init__(
        self,
        seq_length: int = 128,
        feature_dim: int = 12,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Initialize ML-based IDS.

        Args:
            seq_length: Length of time series sequences
            feature_dim: Number of features per timestep
            contamination: Expected proportion of anomalies (default: 0.05 = 5%)
            n_estimators: Number of trees in the forest (default: 100)
            random_state: Random seed for reproducibility
        """
        super().__init__(seq_length, feature_dim)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        # Initialize Isolation Forest
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
            verbose=0
        )

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> 'MLBasedIDS':
        """
        Train Isolation Forest on benign traffic.

        Args:
            X_train: Benign traffic of shape (n_samples, seq_length, feature_dim)
            y_train: Ignored (unsupervised method)

        Returns:
            Self for method chaining
        """
        self._validate_input(X_train)

        logger.info(
            f"Training Isolation Forest with {self.n_estimators} estimators, "
            f"contamination={self.contamination}..."
        )

        # Extract statistical features
        X_features = extract_batch_features(X_train)

        # Train Isolation Forest
        self.model.fit(X_features)

        self._fitted = True
        logger.success(
            f"MLBasedIDS fitted on {len(X_train)} samples with "
            f"{X_features.shape[1]} features"
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels using Isolation Forest.

        Args:
            X: Test data of shape (n_samples, seq_length, feature_dim)

        Returns:
            Binary predictions (0=benign, 1=attack) of shape (n_samples,)
        """
        self._check_fitted()
        self._validate_input(X)

        # Extract features
        X_features = extract_batch_features(X)

        # Predict using Isolation Forest
        # Returns: -1 for outliers (attacks), +1 for inliers (benign)
        predictions = self.model.predict(X_features)

        # Convert to binary: -1 -> 1 (attack), +1 -> 0 (benign)
        binary_predictions = (predictions == -1).astype(int)

        return binary_predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores using Isolation Forest.

        Args:
            X: Test data of shape (n_samples, seq_length, feature_dim)

        Returns:
            Anomaly scores of shape (n_samples,) in range [0, 1]
        """
        self._check_fitted()
        self._validate_input(X)

        # Extract features
        X_features = extract_batch_features(X)

        # Get anomaly scores
        # Isolation Forest returns negative scores for outliers
        # More negative = more anomalous
        raw_scores = self.model.score_samples(X_features)

        # Normalize to [0, 1]
        # Raw scores are typically in range [-0.5, 0.5]
        # We map them so that:
        # - 0.5 (normal) -> 0.0
        # - -0.5 (anomaly) -> 1.0
        normalized_scores = -raw_scores  # Flip sign
        normalized_scores = np.clip(normalized_scores, 0, 1)  # Clip to [0, 1]

        return normalized_scores

    def get_config(self) -> Dict:
        """Get configuration dictionary."""
        return {
            'method': 'ml_based',
            'seq_length': self.seq_length,
            'feature_dim': self.feature_dim,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'fitted': self._fitted
        }
