"""
Combined Baseline IDS

Ensemble method that aggregates multiple traditional IDS approaches
using weighted voting to improve overall detection performance.
"""

import numpy as np
from typing import Dict, Optional
from loguru import logger

from .base import BaseIDS
from .threshold import ThresholdIDS
from .statistical import StatisticalIDS
from .signature import SignatureIDS
from .ml_based import MLBasedIDS


class CombinedBaselineIDS(BaseIDS):
    """
    Combined Baseline IDS using weighted ensemble voting.

    Aggregates four traditional detection methods:
    1. ThresholdIDS - Simple percentile-based detection
    2. SignatureIDS - Pattern matching for known attacks
    3. StatisticalIDS - Z-score and IQR outlier detection
    4. MLBasedIDS - Isolation Forest anomaly detection

    Each method has a weight that reflects its relative importance.
    Final score is the weighted average of individual method scores.

    Default weights prioritize signature matching (high precision for
    known attacks) while maintaining balance across other methods.
    """

    def __init__(
        self,
        seq_length: int = 128,
        feature_dim: int = 12,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize Combined Baseline IDS.

        Args:
            seq_length: Length of time series sequences
            feature_dim: Number of features per timestep
            weights: Optional custom weights for each method.
                    Default: {'threshold': 0.15, 'signature': 0.35,
                             'statistical': 0.25, 'ml': 0.25}
        """
        super().__init__(seq_length, feature_dim)

        # Default weights (can be tuned via validation set)
        self.weights = weights or {
            'threshold': 0.15,    # Simple baseline
            'signature': 0.35,    # High precision for known attacks
            'statistical': 0.25,  # Good for outliers
            'ml': 0.25           # General anomaly detection
        }

        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0):
            logger.warning(
                f"Weights sum to {weight_sum:.3f}, normalizing to 1.0"
            )
            # Normalize weights
            for key in self.weights:
                self.weights[key] /= weight_sum

        # Initialize individual IDS methods
        self.methods = {
            'threshold': ThresholdIDS(seq_length, feature_dim),
            'signature': SignatureIDS(seq_length, feature_dim),
            'statistical': StatisticalIDS(seq_length, feature_dim),
            'ml': MLBasedIDS(seq_length, feature_dim)
        }

        logger.info(
            f"CombinedBaselineIDS initialized with weights: {self.weights}"
        )

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> 'CombinedBaselineIDS':
        """
        Train all constituent IDS methods.

        Args:
            X_train: Benign traffic of shape (n_samples, seq_length, feature_dim)
            y_train: Optional labels (not used by unsupervised methods)

        Returns:
            Self for method chaining
        """
        self._validate_input(X_train)

        logger.info(f"Training {len(self.methods)} IDS methods...")

        # Train each method
        for name, method in self.methods.items():
            logger.info(f"Training {name}...")
            method.fit(X_train, y_train)

        self._fitted = True
        logger.success(
            f"CombinedBaselineIDS fitted on {len(X_train)} samples"
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels using weighted ensemble.

        Args:
            X: Test data of shape (n_samples, seq_length, feature_dim)

        Returns:
            Binary predictions (0=benign, 1=attack) of shape (n_samples,)
        """
        scores = self.predict_proba(X)
        return (scores >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute ensemble anomaly scores using weighted voting.

        Args:
            X: Test data of shape (n_samples, seq_length, feature_dim)

        Returns:
            Anomaly scores of shape (n_samples,) in range [0, 1]
        """
        self._check_fitted()
        self._validate_input(X)

        # Get scores from all methods
        method_scores = {}
        for name, method in self.methods.items():
            method_scores[name] = method.predict_proba(X)

        # Compute weighted average
        weighted_scores = np.zeros(len(X))
        for name, scores in method_scores.items():
            weighted_scores += scores * self.weights[name]

        return weighted_scores

    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual method.

        Useful for analysis and debugging.

        Args:
            X: Test data of shape (n_samples, seq_length, feature_dim)

        Returns:
            Dictionary mapping method name to predictions
        """
        self._check_fitted()
        self._validate_input(X)

        predictions = {}
        for name, method in self.methods.items():
            predictions[name] = method.predict(X)

        return predictions

    def get_individual_scores(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get probability scores from each individual method.

        Useful for analysis and debugging.

        Args:
            X: Test data of shape (n_samples, seq_length, feature_dim)

        Returns:
            Dictionary mapping method name to probability scores
        """
        self._check_fitted()
        self._validate_input(X)

        scores = {}
        for name, method in self.methods.items():
            scores[name] = method.predict_proba(X)

        return scores

    def get_config(self) -> Dict:
        """Get configuration dictionary."""
        return {
            'method': 'combined',
            'seq_length': self.seq_length,
            'feature_dim': self.feature_dim,
            'weights': self.weights,
            'fitted': self._fitted,
            'constituent_methods': {
                name: method.get_config()
                for name, method in self.methods.items()
            }
        }
