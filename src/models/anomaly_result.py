"""
Anomaly Detection Result Data Structure

This module defines the AnomalyResult dataclass that encapsulates
the output of Moirai-based anomaly detection.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class AnomalyResult:
    """
    Result of anomaly detection on a time-series sequence.

    This dataclass encapsulates all information about the detection,
    including predictions, confidence intervals, anomaly scores, and metadata.

    Attributes:
        predictions: Predicted values of shape (seq_length, n_features)
        actuals: Observed values of shape (seq_length, n_features)
        confidence_lower: Lower bound of confidence interval (seq_length, n_features)
        confidence_upper: Upper bound of confidence interval (seq_length, n_features)
        anomaly_scores: Per-timestep anomaly scores [0-1] of shape (seq_length,)
        is_anomaly: Binary flags indicating anomalies of shape (seq_length,)
        threshold: Detection threshold used (0-1)
        feature_contributions: Per-feature anomaly contribution (seq_length, n_features)
                              Normalized to sum to 1 across features per timestep
        metadata: Additional information (model_size, inference_time, etc.)
    """

    predictions: np.ndarray
    actuals: np.ndarray
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray
    anomaly_scores: np.ndarray
    is_anomaly: np.ndarray
    threshold: float
    feature_contributions: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the data structure after initialization."""
        # Validate shapes
        if self.predictions.shape != self.actuals.shape:
            raise ValueError(
                f"Predictions shape {self.predictions.shape} doesn't match "
                f"actuals shape {self.actuals.shape}"
            )

        if self.predictions.shape != self.confidence_lower.shape:
            raise ValueError(
                f"Predictions shape {self.predictions.shape} doesn't match "
                f"confidence_lower shape {self.confidence_lower.shape}"
            )

        if self.predictions.shape != self.confidence_upper.shape:
            raise ValueError(
                f"Predictions shape {self.predictions.shape} doesn't match "
                f"confidence_upper shape {self.confidence_upper.shape}"
            )

        seq_length = self.predictions.shape[0]
        if self.anomaly_scores.shape != (seq_length,):
            raise ValueError(
                f"Anomaly scores shape {self.anomaly_scores.shape} doesn't match "
                f"expected ({seq_length},)"
            )

        if self.is_anomaly.shape != (seq_length,):
            raise ValueError(
                f"is_anomaly shape {self.is_anomaly.shape} doesn't match "
                f"expected ({seq_length},)"
            )

        # Validate threshold
        if not 0 <= self.threshold <= 1:
            raise ValueError(f"Threshold must be in [0, 1], got {self.threshold}")

        # Validate feature_contributions if provided
        if self.feature_contributions is not None:
            if self.feature_contributions.shape != self.predictions.shape:
                raise ValueError(
                    f"Feature contributions shape {self.feature_contributions.shape} "
                    f"doesn't match predictions shape {self.predictions.shape}"
                )

    @property
    def seq_length(self) -> int:
        """Return the sequence length."""
        return self.predictions.shape[0]

    @property
    def n_features(self) -> int:
        """Return the number of features."""
        return self.predictions.shape[1]

    @property
    def n_anomalies(self) -> int:
        """Return the number of detected anomalies."""
        return int(self.is_anomaly.sum())

    @property
    def anomaly_rate(self) -> float:
        """Return the proportion of timesteps flagged as anomalous."""
        return float(self.n_anomalies) / self.seq_length

    def get_anomalous_timesteps(self) -> np.ndarray:
        """
        Get indices of timesteps flagged as anomalous.

        Returns:
            Array of timestep indices where anomalies were detected
        """
        return np.where(self.is_anomaly)[0]

    def get_top_anomalous_features(self, timestep: int, top_k: int = 3) -> np.ndarray:
        """
        Get the top-k features contributing to anomaly at a given timestep.

        Args:
            timestep: Timestep index
            top_k: Number of top features to return

        Returns:
            Array of feature indices sorted by contribution (descending)
        """
        if self.feature_contributions is None:
            raise ValueError("Feature contributions not available")

        if timestep < 0 or timestep >= self.seq_length:
            raise ValueError(
                f"Timestep {timestep} out of range [0, {self.seq_length})"
            )

        contributions = self.feature_contributions[timestep]
        top_indices = np.argsort(contributions)[::-1][:top_k]
        return top_indices

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the detection result.

        Returns:
            Dictionary with summary statistics
        """
        return {
            'seq_length': self.seq_length,
            'n_features': self.n_features,
            'n_anomalies': self.n_anomalies,
            'anomaly_rate': self.anomaly_rate,
            'threshold': self.threshold,
            'mean_anomaly_score': float(self.anomaly_scores.mean()),
            'max_anomaly_score': float(self.anomaly_scores.max()),
            'min_anomaly_score': float(self.anomaly_scores.min()),
            'metadata': self.metadata
        }
