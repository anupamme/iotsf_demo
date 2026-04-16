"""
One-Class SVM IDS baseline.

Uses sklearn's OneClassSVM with RBF kernel for unsupervised anomaly detection.
Follows the same interface as MLBasedIDS (Isolation Forest).
"""

import numpy as np
from typing import Dict, Optional
from loguru import logger
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from .base import BaseIDS
from .feature_extraction import extract_batch_features


class OCSVMIDS(BaseIDS):
    """
    One-Class SVM Intrusion Detection System.

    Fits an RBF-kernel One-Class SVM on statistical features extracted
    from benign traffic windows.
    """

    def __init__(
        self,
        seq_length: int = 128,
        feature_dim: int = 12,
        nu: float = 0.05,
        kernel: str = "rbf",
        gamma: str = "scale",
        random_state: int = 42,
    ):
        super().__init__(seq_length, feature_dim)
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state

        self.model = OneClassSVM(
            kernel=kernel, nu=nu, gamma=gamma,
        )
        self._scaler = StandardScaler()

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "OCSVMIDS":
        self._validate_input(X_train)

        logger.info(
            f"Training One-Class SVM (kernel={self.kernel}, nu={self.nu}, gamma={self.gamma})..."
        )

        X_features = extract_batch_features(X_train)
        X_features = self._scaler.fit_transform(X_features)
        self.model.fit(X_features)

        self._fitted = True
        logger.success(
            f"OCSVMIDS fitted on {len(X_train)} samples with "
            f"{X_features.shape[1]} features"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        self._validate_input(X)

        X_features = extract_batch_features(X)
        X_features = self._scaler.transform(X_features)
        predictions = self.model.predict(X_features)
        return (predictions == -1).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        self._validate_input(X)

        X_features = extract_batch_features(X)
        X_features = self._scaler.transform(X_features)

        raw_scores = self.model.score_samples(X_features)
        normalized_scores = -raw_scores
        lo, hi = normalized_scores.min(), normalized_scores.max()
        if hi - lo > 1e-10:
            normalized_scores = (normalized_scores - lo) / (hi - lo)
        else:
            normalized_scores = np.full_like(normalized_scores, 0.5)
        return normalized_scores

    def get_config(self) -> Dict:
        return {
            "method": "ocsvm",
            "seq_length": self.seq_length,
            "feature_dim": self.feature_dim,
            "nu": self.nu,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "fitted": self._fitted,
        }
