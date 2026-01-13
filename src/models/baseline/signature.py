"""
Signature-based IDS

Pattern matching for known attack signatures (Mirai, DDoS).
This method is effective for well-known attacks but fails on hard-negatives.
"""

import numpy as np
from typing import Dict, Optional
from loguru import logger

from .base import BaseIDS
from .feature_extraction import (
    extract_structured_features,
    compute_asymmetry_ratio,
    detect_periodicity
)


class SignatureIDS(BaseIDS):
    """
    Signature-based Intrusion Detection System.

    Detects known attack patterns:
    1. Mirai: Very high packet rates with low bytes per packet
    2. DDoS: Extremely high flow packet rate with traffic asymmetry

    Does NOT detect hard-negative attacks (by design, to show limitations).
    """

    def __init__(
        self,
        seq_length: int = 128,
        feature_dim: int = 12,
        mirai_pkt_rate_threshold: float = 1000.0,
        mirai_byte_avg_threshold: float = 100.0,
        ddos_pkt_rate_threshold: float = 500.0,
        ddos_asymmetry_threshold: float = 10.0
    ):
        """
        Initialize Signature IDS.

        Args:
            seq_length: Length of time series sequences
            feature_dim: Number of features per timestep
            mirai_pkt_rate_threshold: Min packet rate for Mirai detection (pkts/sec)
            mirai_byte_avg_threshold: Max avg bytes for Mirai detection
            ddos_pkt_rate_threshold: Min packet rate for DDoS detection (pkts/sec)
            ddos_asymmetry_threshold: Min fwd/bwd ratio for DDoS detection
        """
        super().__init__(seq_length, feature_dim)
        self.mirai_pkt_rate_threshold = mirai_pkt_rate_threshold
        self.mirai_byte_avg_threshold = mirai_byte_avg_threshold
        self.ddos_pkt_rate_threshold = ddos_pkt_rate_threshold
        self.ddos_asymmetry_threshold = ddos_asymmetry_threshold

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> 'SignatureIDS':
        """
        Signature-based detection doesn't require training.

        Args:
            X_train: Training data (not used, signatures are pre-defined)
            y_train: Labels (not used)

        Returns:
            Self for method chaining
        """
        self._fitted = True
        logger.info("SignatureIDS initialized (no training required)")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels using signature matching.

        Args:
            X: Test data of shape (n_samples, seq_length, feature_dim)

        Returns:
            Binary predictions (0=benign, 1=attack) of shape (n_samples,)
        """
        scores = self.predict_proba(X)
        return (scores >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores based on signature matches.

        Args:
            X: Test data of shape (n_samples, seq_length, feature_dim)

        Returns:
            Anomaly scores of shape (n_samples,) in range [0, 1]
        """
        self._check_fitted()
        self._validate_input(X)

        n_samples = X.shape[0]
        scores = np.zeros(n_samples)

        for i in range(n_samples):
            sequence = X[i]

            # Check each signature
            mirai_score = self._check_mirai_signature(sequence)
            ddos_score = self._check_ddos_signature(sequence)

            # Return max score across all signatures (OR logic)
            scores[i] = max(mirai_score, ddos_score)

        return scores

    def _check_mirai_signature(self, sequence: np.ndarray) -> float:
        """
        Check for Mirai attack signature.

        Mirai characteristics:
        - Very high packet rate (>1000 pkts/sec)
        - Low bytes per packet (<100 bytes)
        - UDP-heavy flooding

        Args:
            sequence: Time series of shape (seq_length, feature_dim)

        Returns:
            Signature match score in [0, 1]
        """
        # Extract features using structured interface for maintainability
        features = extract_structured_features(sequence)

        # Access features by name - much more readable and maintainable
        fwd_pkt_rate_mean = features['fwd_pkts_per_sec']['mean']
        fwd_byte_avg_mean = features['fwd_byts_b_avg']['mean']

        # Check Mirai conditions
        high_pkt_rate = fwd_pkt_rate_mean > self.mirai_pkt_rate_threshold
        low_byte_avg = fwd_byte_avg_mean < self.mirai_byte_avg_threshold

        if high_pkt_rate and low_byte_avg:
            # Strong Mirai signature match
            return 0.9
        elif high_pkt_rate:
            # Partial match (high rate but not low bytes)
            return 0.5
        else:
            return 0.0

    def _check_ddos_signature(self, sequence: np.ndarray) -> float:
        """
        Check for DDoS attack signature.

        DDoS characteristics:
        - Extremely high flow packet rate (>500 pkts/sec)
        - High traffic asymmetry (fwd >> bwd)
        - Short duration with high packet count

        Args:
            sequence: Time series of shape (seq_length, feature_dim)

        Returns:
            Signature match score in [0, 1]
        """
        # Extract features using structured interface
        features = extract_structured_features(sequence)

        # Access flow packet rate by name - clear and maintainable
        flow_pkt_rate_mean = features['flow_pkts_per_sec']['mean']

        # Compute asymmetry ratio (using raw sequence, indices still needed here)
        # Feature indices: 1=fwd_pkts_tot, 2=bwd_pkts_tot (from FEATURE_NAMES)
        asymmetry = compute_asymmetry_ratio(
            sequence,
            fwd_idx=1,  # fwd_pkts_tot
            bwd_idx=2   # bwd_pkts_tot
        )

        # Check DDoS conditions
        high_flow_rate = flow_pkt_rate_mean > self.ddos_pkt_rate_threshold
        high_asymmetry = asymmetry > self.ddos_asymmetry_threshold

        if high_flow_rate and high_asymmetry:
            # Strong DDoS signature match
            return 0.9
        elif high_flow_rate:
            # Partial match (high rate but not asymmetric)
            return 0.6
        elif high_asymmetry:
            # Partial match (asymmetric but not high rate)
            return 0.4
        else:
            return 0.0

    def get_config(self) -> Dict:
        """Get configuration dictionary."""
        return {
            'method': 'signature',
            'seq_length': self.seq_length,
            'feature_dim': self.feature_dim,
            'mirai_pkt_rate_threshold': self.mirai_pkt_rate_threshold,
            'mirai_byte_avg_threshold': self.mirai_byte_avg_threshold,
            'ddos_pkt_rate_threshold': self.ddos_pkt_rate_threshold,
            'ddos_asymmetry_threshold': self.ddos_asymmetry_threshold,
            'fitted': self._fitted
        }
