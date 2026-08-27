"""
TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time
Series Data
VLDB 2022 — Tuli et al.

Key ideas
----------
1. Context-aware self-conditioning Transformer encoder
2. Two decoders trained adversarially: W1 reconstructs faithfully,
   W2 reconstructs with a focus-score amplification of anomalous regions
3. The focus score is derived from the reconstruction error of W1, so
   each training epoch forces W2 to attend to harder-to-reconstruct regions

Training (n epochs, loss scaled by epoch index e):
  L1 = (1/e) * ||X - W1(Z, X)||   +  (1 - 1/e) * ||X - W2(Z, X)||   (E, W1)
  L2 = (1/e) * ||X - W1(Z, X)||   -  (1 - 1/e) * ||X - W2(Z, X)||   (E, W2)

Anomaly score (inference):
  score = ||X - W2(Z, X)||   (focus-score weighted reconstruction error)

Reference:
    Tuli et al., "TranAD: Deep Transformer Networks for Anomaly Detection
    in Multivariate Time Series Data", VLDB 2022.  arXiv:2201.07284
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional

from .base import BaseIDS


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# TranAD components
# ---------------------------------------------------------------------------

class _TransformerEncoder(nn.Module):
    """Shared Transformer encoder (context-aware)."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dim_ff: int):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class _Decoder(nn.Module):
    """Transformer decoder that uses the encoder output as memory."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dim_ff: int, out_dim: int):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        out = self.decoder(tgt, memory)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# TranAD IDS
# ---------------------------------------------------------------------------

class TranADIDS(BaseIDS):
    """
    TranAD-based anomaly detector for the BaseIDS interface.

    Input shape expected: (n_samples, seq_length, feature_dim)
    """

    def __init__(
        self,
        seq_length: int = 128,
        feature_dim: int = 12,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 128,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-4,
        device: Optional[str] = None,
    ):
        super().__init__(seq_length=seq_length, feature_dim=feature_dim)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_ff = dim_ff
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        if device:
            self._device = torch.device(device)
        elif torch.cuda.is_available():
            self._device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self._device = torch.device('mps')
        else:
            self._device = torch.device('cpu')
        self._input_proj: Optional[nn.Linear] = None
        self._pos_enc: Optional[_PositionalEncoding] = None
        self._enc: Optional[_TransformerEncoder] = None
        self._dec1: Optional[_Decoder] = None
        self._dec2: Optional[_Decoder] = None

    # ------------------------------------------------------------------
    # BaseIDS interface
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "TranADIDS":
        """Train on benign traffic (unsupervised; y_train ignored)."""
        self._validate_input(X_train)
        if y_train is not None:
            X_train = X_train[y_train == 0]

        self._build_model()

        all_params = (
            list(self._input_proj.parameters())
            + list(self._pos_enc.parameters())
            + list(self._enc.parameters())
        )
        opt1 = torch.optim.AdamW(all_params + list(self._dec1.parameters()), lr=self.lr)
        opt2 = torch.optim.AdamW(all_params + list(self._dec2.parameters()), lr=self.lr)

        X_t = torch.tensor(X_train.astype(np.float32), device=self._device)  # (N, T, F)
        n = len(X_t)

        for epoch in range(1, self.epochs + 1):
            perm = torch.randperm(n, device=self._device)
            # Epoch-scaling factor from the paper
            scale = 1.0 / epoch

            for start in range(0, n, self.batch_size):
                batch = X_t[perm[start: start + self.batch_size]]   # (B, T, F)

                z, x_hat1, x_hat2 = self._forward(batch)

                err1 = ((batch - x_hat1) ** 2).mean()
                err2 = ((batch - x_hat2) ** 2).mean()

                # Phase 1: train encoder + decoder1
                l1 = scale * err1 + (1 - scale) * err2
                opt1.zero_grad()
                l1.backward(retain_graph=True)
                opt1.step()

                # Phase 2: train encoder + decoder2
                z2, x_hat1_2, x_hat2_2 = self._forward(batch)
                err1_2 = ((batch - x_hat1_2) ** 2).mean()
                err2_2 = ((batch - x_hat2_2) ** 2).mean()
                l2 = scale * err1_2 - (1 - scale) * err2_2
                opt2.zero_grad()
                l2.backward()
                opt2.step()

        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        self._validate_input(X)
        X_t = torch.tensor(X.astype(np.float32), device=self._device)
        with torch.no_grad():
            _, _, x_hat2 = self._forward(X_t)
            err = ((X_t - x_hat2) ** 2).mean(dim=(1, 2))   # (B,)
        scores = err.cpu().numpy()
        # Normalise via min-max to approximately [0, 1]
        lo, hi = scores.min(), scores.max()
        if hi > lo:
            scores = (scores - lo) / (hi - lo)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.predict_proba(X)
        return (scores > 0.5).astype(int)

    def get_config(self) -> Dict:
        return {
            "name": "TranAD",
            "seq_length": self.seq_length,
            "feature_dim": self.feature_dim,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self):
        F = self.feature_dim
        d = self.d_model
        self._input_proj = nn.Linear(F, d).to(self._device)
        self._pos_enc = _PositionalEncoding(d).to(self._device)
        self._enc = _TransformerEncoder(d, self.n_heads, self.n_layers, self.dim_ff).to(self._device)
        self._dec1 = _Decoder(d, self.n_heads, self.n_layers, self.dim_ff, F).to(self._device)
        self._dec2 = _Decoder(d, self.n_heads, self.n_layers, self.dim_ff, F).to(self._device)

    def _forward(self, x: torch.Tensor):
        """
        x: (B, T, F)
        Returns (memory, x_hat1, x_hat2)
        """
        emb = self._pos_enc(self._input_proj(x))   # (B, T, d_model)
        memory = self._enc(emb)                     # (B, T, d_model)
        x_hat1 = self._dec1(emb, memory)            # (B, T, F)
        x_hat2 = self._dec2(emb, memory)            # (B, T, F)
        return memory, x_hat1, x_hat2
