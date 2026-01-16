"""Baseline IDS using threshold-based anomaly detection."""

import numpy as np
from typing import Dict, Optional
from loguru import logger


class BaselineIDS:
    """
    Threshold-based Intrusion Detection System.

    Detects anomalies by comparing samples against learned statistical
    thresholds from benign traffic. Flags samples where features exceed
    mean ± k*std boundaries.

    Attributes:
        threshold: Anomaly score threshold for classification (0-1)
        n_std: Number of standard deviations for threshold (default: 3.0)
        feature_means_: Learned mean values per feature
        feature_stds_: Learned standard deviation per feature
        is_fitted_: Whether the detector has been trained
    """

    def __init__(self, threshold: float = 0.3, n_std: float = 3.0):
        """
        Initialize threshold-based IDS.

        Args:
            threshold: Anomaly score threshold for binary classification.
                      Samples with score > threshold are classified as attacks.
            n_std: Number of standard deviations to define normal range.
                   Common values: 2.0 (95%), 3.0 (99.7%)
        """
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        if n_std <= 0:
            raise ValueError("n_std must be positive")

        self.threshold = threshold
        self.n_std = n_std
        self.feature_means_: Optional[np.ndarray] = None
        self.feature_stds_: Optional[np.ndarray] = None
        self.is_fitted_ = False

        logger.info(f"Initialized BaselineIDS (threshold={threshold}, n_std={n_std})")

    def fit(self, benign_samples: np.ndarray) -> 'BaselineIDS':
        """
        Learn normal behavior statistics from benign samples.

        Args:
            benign_samples: Benign traffic samples.
                          Shape: (n_samples, seq_length, n_features)
                          or (n_samples * seq_length, n_features)

        Returns:
            self: Fitted detector
        """
        if benign_samples.ndim == 3:
            # Flatten time dimension: (n_samples, seq_length, n_features)
            # -> (n_samples * seq_length, n_features)
            n_samples, seq_length, n_features = benign_samples.shape
            benign_flat = benign_samples.reshape(-1, n_features)
        elif benign_samples.ndim == 2:
            benign_flat = benign_samples
            n_features = benign_samples.shape[1]
        else:
            raise ValueError(
                f"benign_samples must be 2D or 3D, got shape {benign_samples.shape}"
            )

        # Compute statistics across all time steps
        self.feature_means_ = np.mean(benign_flat, axis=0)
        self.feature_stds_ = np.std(benign_flat, axis=0)

        # Handle features with zero variance
        self.feature_stds_ = np.maximum(self.feature_stds_, 1e-8)

        self.is_fitted_ = True

        logger.info(
            f"Fitted BaselineIDS on {benign_samples.shape[0]} benign samples "
            f"({n_features} features)"
        )
        logger.debug(f"Feature means range: [{self.feature_means_.min():.3f}, "
                    f"{self.feature_means_.max():.3f}]")
        logger.debug(f"Feature stds range: [{self.feature_stds_.min():.3f}, "
                    f"{self.feature_stds_.max():.3f}]")

        return self

    def detect(self, samples: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect anomalies in samples.

        Args:
            samples: Traffic samples to analyze.
                    Shape: (n_samples, seq_length, n_features)

        Returns:
            Dictionary containing:
                - predictions: Binary predictions (0=benign, 1=attack). Shape: (n_samples,)
                - scores: Anomaly scores in [0, 1]. Shape: (n_samples,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted before calling detect()")

        if samples.ndim != 3:
            raise ValueError(
                f"samples must be 3D (n_samples, seq_length, n_features), "
                f"got shape {samples.shape}"
            )

        n_samples, seq_length, n_features = samples.shape

        # Compute anomaly score for each sample
        scores = np.zeros(n_samples)

        for i in range(n_samples):
            sample = samples[i]  # Shape: (seq_length, n_features)

            # Compute z-scores for all features at all time steps
            z_scores = np.abs(
                (sample - self.feature_means_) / self.feature_stds_
            )

            # Count features exceeding threshold at each time step
            exceeds_threshold = z_scores > self.n_std

            # Anomaly score = proportion of (feature, time) pairs exceeding threshold
            scores[i] = exceeds_threshold.mean()

        # Binary predictions based on threshold
        predictions = (scores > self.threshold).astype(int)

        logger.info(f"Detected {predictions.sum()}/{n_samples} anomalies")
        logger.debug(f"Anomaly scores: {scores}")

        return {
            'predictions': predictions,
            'scores': scores
        }

    def get_params(self) -> Dict:
        """Get detector parameters and statistics."""
        params = {
            'threshold': self.threshold,
            'n_std': self.n_std,
            'is_fitted': self.is_fitted_
        }

        if self.is_fitted_:
            params.update({
                'n_features': len(self.feature_means_),
                'mean_feature_mean': float(self.feature_means_.mean()),
                'mean_feature_std': float(self.feature_stds_.mean())
            })

        return params
