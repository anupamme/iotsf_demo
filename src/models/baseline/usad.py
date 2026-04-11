"""
USAD: UnSupervised Anomaly Detection on Multivariate Time Series
ICDM 2020 — Audibert et al.

Architecture
------------
Shared Encoder  E  : (seq * feat) → z_dim
Decoder 1       D1 : z_dim → (seq * feat)   # AE-style reconstruction
Decoder 2       D2 : z_dim → (seq * feat)   # Adversarial-style reconstruction

Training (two phases, interleaved per epoch):
  Phase-1  L1 = ||X - D1(E(X))||²  +  ||X - D2(D1(E(X)))||²   (train E, D1, D2)
  Phase-2  L2 = ||X - D1(E(X))||²  -  ||X - D2(D1(E(X)))||²   (train E, D2 only)

Anomaly score:
  score(X) = α * ||X - D1(E(X))|| + (1-α) * ||X - D2(D1(E(X)))||
  (α=0 at inference gives pure adversarial score; default α=0.5)

Reference:
    Audibert et al., "USAD: UnSupervised Anomaly Detection on Multivariate
    Time Series", KDD 2020.  arXiv:2011.02001
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional

from .base import BaseIDS


# ---------------------------------------------------------------------------
# Neural-network components
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    def __init__(self, input_dim: int, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, z_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(self, z_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, output_dim // 4),
            nn.ReLU(),
            nn.Linear(output_dim // 4, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# USAD IDS
# ---------------------------------------------------------------------------

class USADIDS(BaseIDS):
    """
    USAD-based anomaly detector adapted for the BaseIDS interface.

    Input shape: (n_samples, seq_length, feature_dim)
    Internally flattens to (n_samples, seq_length * feature_dim).
    """

    def __init__(
        self,
        seq_length: int = 128,
        feature_dim: int = 12,
        z_dim: int = 40,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        alpha: float = 0.5,
        device: Optional[str] = None,
    ):
        super().__init__(seq_length=seq_length, feature_dim=feature_dim)
        self.z_dim = z_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha                   # anomaly score mixing coefficient

        self._device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._input_dim = seq_length * feature_dim
        self._encoder: Optional[_Encoder] = None
        self._decoder1: Optional[_Decoder] = None
        self._decoder2: Optional[_Decoder] = None

    # ------------------------------------------------------------------
    # BaseIDS interface
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "USADIDS":
        """Train on benign traffic only (y_train is ignored)."""
        self._validate_input(X_train)
        # USAD is trained purely on benign samples
        if y_train is not None:
            X_train = X_train[y_train == 0]

        # Build networks
        self._encoder = _Encoder(self._input_dim, self.z_dim).to(self._device)
        self._decoder1 = _Decoder(self.z_dim, self._input_dim).to(self._device)
        self._decoder2 = _Decoder(self.z_dim, self._input_dim).to(self._device)

        opt1 = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._decoder1.parameters()),
            lr=self.lr,
        )
        opt2 = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._decoder2.parameters()),
            lr=self.lr,
        )

        X_flat = torch.tensor(
            X_train.reshape(len(X_train), -1).astype(np.float32),
            device=self._device,
        )

        n = len(X_flat)
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self._device)
            epoch_l1 = epoch_l2 = 0.0
            for start in range(0, n, self.batch_size):
                batch = X_flat[perm[start: start + self.batch_size]]

                z = self._encoder(batch)
                w1 = self._decoder1(z)
                w2 = self._decoder2(z)
                w21 = self._decoder2(self._encoder(w1))

                # Phase 1
                l1 = ((batch - w1) ** 2).mean() + ((batch - w21) ** 2).mean()
                opt1.zero_grad()
                l1.backward(retain_graph=True)
                opt1.step()

                # Phase 2 (re-forward after opt1 step)
                z2 = self._encoder(batch)
                w1_2 = self._decoder1(z2)
                w21_2 = self._decoder2(self._encoder(w1_2.detach()))
                l2 = ((batch - w1_2) ** 2).mean() - ((batch - w21_2) ** 2).mean()
                opt2.zero_grad()
                l2.backward()
                opt2.step()

                epoch_l1 += l1.item()
                epoch_l2 += l2.item()

        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample anomaly scores in [0, 1] (approximately)."""
        self._check_fitted()
        self._validate_input(X)
        scores = self._score_flat(
            torch.tensor(X.reshape(len(X), -1).astype(np.float32), device=self._device)
        )
        # Normalise roughly to [0, 1] via sigmoid
        scores_np = scores.cpu().numpy()
        # Sigmoid centred around the mean score seen during calibration
        return 1.0 / (1.0 + np.exp(-(scores_np - scores_np.mean()) * 5))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary 0/1 predictions using the median score as threshold."""
        scores = self.predict_proba(X)
        return (scores > 0.5).astype(int)

    def get_config(self) -> Dict:
        return {
            "name": "USAD",
            "seq_length": self.seq_length,
            "feature_dim": self.feature_dim,
            "z_dim": self.z_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "alpha": self.alpha,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _score_flat(self, X_flat: torch.Tensor) -> torch.Tensor:
        """Compute raw anomaly scores (MSE) for flattened inputs."""
        z = self._encoder(X_flat)
        w1 = self._decoder1(z)
        w21 = self._decoder2(self._encoder(w1))

        err1 = ((X_flat - w1) ** 2).mean(dim=1)
        err2 = ((X_flat - w21) ** 2).mean(dim=1)
        return self.alpha * err1 + (1 - self.alpha) * err2
