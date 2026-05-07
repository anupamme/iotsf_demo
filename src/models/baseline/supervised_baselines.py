"""
Supervised Baselines for NeurIPS 2026 reviewer response (W7, Q4).

These classifiers are trained on the same labeled stealth-controlled negatives
used by HNIDS condition D. This isolates the "labeled attack data helps" effect
from the foundation-model contribution. If XGBoost on 72 hand-crafted features
matches HNIDS, the temporal modeling adds nothing beyond the labeled data.

Three classifiers on the 72-dim statistical feature vector (12 features × 6 stats):
  - SupervisedLogReg: Logistic Regression
  - SupervisedXGBoost: Gradient Boosting (XGBClassifier or sklearn GBM fallback)
  - SupervisedMLP: Small 2-layer MLP
"""

import numpy as np
from typing import Dict, Optional
from loguru import logger

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .base import BaseIDS
from .feature_extraction import extract_batch_features

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False
    logger.warning("xgboost not installed; SupervisedXGBoost will fall back to sklearn GradientBoostingClassifier")


class SupervisedLogReg(BaseIDS):
    """
    Logistic Regression trained on labeled benign + stealth-controlled negatives.
    Input: 72-dim statistical features (12 network features × 6 statistics).
    """

    def __init__(self, seq_length: int = 128, feature_dim: int = 12,
                 random_state: int = 42, C: float = 1.0):
        super().__init__(seq_length, feature_dim)
        self.random_state = random_state
        self.C = C
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=1000, random_state=random_state,
                                       class_weight="balanced")),
        ])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "SupervisedLogReg":
        self._validate_input(X_train)
        X_feat = extract_batch_features(X_train)
        self.model.fit(X_feat, y_train)
        self._fitted = True
        logger.info(f"SupervisedLogReg fitted on {len(X_train)} samples ({int(y_train.sum())} attack)")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(extract_batch_features(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict_proba(extract_batch_features(X))[:, 1]

    def get_config(self) -> Dict:
        return {"method": "supervised_logreg", "C": self.C,
                "seq_length": self.seq_length, "feature_dim": self.feature_dim}


class SupervisedXGBoost(BaseIDS):
    """
    Gradient Boosting (XGBoost if available, sklearn GBM fallback) trained on
    labeled benign + stealth-controlled negatives.
    """

    def __init__(self, seq_length: int = 128, feature_dim: int = 12,
                 random_state: int = 42, n_estimators: int = 100,
                 max_depth: int = 6, learning_rate: float = 0.1):
        super().__init__(seq_length, feature_dim)
        self.random_state = random_state
        self.scaler = StandardScaler()
        if _HAS_XGBOOST:
            self.clf = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                eval_metric="logloss",
                verbosity=0,
                use_label_encoder=False,
            )
        else:
            self.clf = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
            )
        self._use_xgboost = _HAS_XGBOOST

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "SupervisedXGBoost":
        self._validate_input(X_train)
        X_feat = self.scaler.fit_transform(extract_batch_features(X_train))
        self.clf.fit(X_feat, y_train)
        self._fitted = True
        kind = "XGBoost" if self._use_xgboost else "GradientBoosting"
        logger.info(f"Supervised{kind} fitted on {len(X_train)} samples ({int(y_train.sum())} attack)")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_feat = self.scaler.transform(extract_batch_features(X))
        return self.clf.predict(X_feat)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_feat = self.scaler.transform(extract_batch_features(X))
        return self.clf.predict_proba(X_feat)[:, 1]

    def get_config(self) -> Dict:
        return {"method": "supervised_xgboost" if self._use_xgboost else "supervised_gbm",
                "seq_length": self.seq_length, "feature_dim": self.feature_dim}


class SupervisedMLP(BaseIDS):
    """
    Small 2-layer MLP trained on labeled benign + stealth-controlled negatives.
    Input: 72-dim statistical features. Architecture: 72 -> 128 -> 32 -> 1.
    """

    def __init__(self, seq_length: int = 128, feature_dim: int = 12,
                 random_state: int = 42, hidden_layer_sizes=(128, 32),
                 max_iter: int = 500):
        super().__init__(seq_length, feature_dim)
        from sklearn.neural_network import MLPClassifier
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1,
            )),
        ])
        self.random_state = random_state

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "SupervisedMLP":
        self._validate_input(X_train)
        X_feat = extract_batch_features(X_train)
        self.model.fit(X_feat, y_train)
        self._fitted = True
        logger.info(f"SupervisedMLP fitted on {len(X_train)} samples ({int(y_train.sum())} attack)")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(extract_batch_features(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict_proba(extract_batch_features(X))[:, 1]

    def get_config(self) -> Dict:
        return {"method": "supervised_mlp", "seq_length": self.seq_length,
                "feature_dim": self.feature_dim}
