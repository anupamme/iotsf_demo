"""
Anomaly Transformer
ICLR 2022 — Xu et al.

Key idea: Association Discrepancy
----------------------------------
Each Transformer layer computes two attention distributions:
  * Series Association  P  — standard scaled dot-product self-attention
  * Prior  Association  Q  — learnable Gaussian kernel, biased toward
                             neighbouring timesteps (continuity prior)

Anomaly Criterion: samples with high ||KL(P||Q) + KL(Q||P)||
are anomalous, because their attention pattern deviates strongly
from the neighbourhood prior (they are not well-explained by local context).

Training:
  Minimax objective — the network simultaneously tries to
  (a) maximise association discrepancy (encoder forces anomalies to be
      attended differently) and
  (b) minimise reconstruction error subject to the discrepancy constraint.

For simplicity we use a single-phase alternating optimisation:
  L_total = L_rec - λ * L_assoc_disc

Reference:
    Xu et al., "Anomaly Transformer: Time Series Anomaly Detection with
    Association Discrepancy", ICLR 2022.  arXiv:2110.02642
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .base import BaseIDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


def _gaussian_prior(seq_len: int, sigma: float = 3.0, device: torch.device = None) -> torch.Tensor:
    """
    Return a (seq_len, seq_len) Gaussian prior matrix where entry (i, j)
    is proportional to N(j; i, sigma²).  Rows are normalised to sum to 1.
    """
    idx = torch.arange(seq_len, dtype=torch.float32, device=device)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2   # (T, T)
    prior = torch.exp(-dist / (2 * sigma ** 2))
    prior = prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return prior   # (T, T)


# ---------------------------------------------------------------------------
# Association-aware Transformer layer
# ---------------------------------------------------------------------------

class _AnomalyAttentionLayer(nn.Module):
    """
    Computes both series attention P and Gaussian prior Q,
    returns reconstructed features and the association discrepancy.
    """

    def __init__(self, d_model: int, n_heads: int, sigma: float = 3.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.sigma = sigma

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        # Learnable sigma per head
        self.sigma_param = nn.Parameter(torch.ones(n_heads) * sigma)

    def forward(self, x: torch.Tensor):
        """
        x : (B, T, d_model)
        Returns
            context : (B, T, d_model)
            disc    : scalar — mean KL(P||Q) + KL(Q||P) over batch/heads
        """
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_head

        Q = self.W_q(x).view(B, T, H, Dh).transpose(1, 2)   # (B, H, T, Dh)
        K = self.W_k(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, Dh).transpose(1, 2)

        # Series association (standard softmax attention)
        scale = math.sqrt(Dh)
        P = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / scale, dim=-1)   # (B, H, T, T)

        # Prior association — one Gaussian kernel per head
        priors = []
        for h in range(H):
            sig = self.sigma_param[h].abs().clamp(min=0.5)
            priors.append(_gaussian_prior(T, sigma=float(sig), device=x.device))
        Q_prior = torch.stack(priors, dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, T, T)

        # Association discrepancy: symmetric KL
        eps = 1e-8
        kl_pq = (P * (torch.log(P + eps) - torch.log(Q_prior + eps))).sum(dim=-1)   # (B, H, T)
        kl_qp = (Q_prior * (torch.log(Q_prior + eps) - torch.log(P + eps))).sum(dim=-1)
        disc = (kl_pq + kl_qp).mean()

        # Context
        context = torch.matmul(P, V)                          # (B, H, T, Dh)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out(context), disc


class _AnomalyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, sigma: float = 3.0):
        super().__init__()
        self.attn = _AnomalyAttentionLayer(d_model, n_heads, sigma)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        ctx, disc = self.attn(x)
        x = self.norm1(x + ctx)
        x = self.norm2(x + self.ff(x))
        return x, disc


# ---------------------------------------------------------------------------
# Anomaly Transformer IDS
# ---------------------------------------------------------------------------

class AnomalyTransformerIDS(BaseIDS):
    """
    Anomaly Transformer adapted for the BaseIDS interface.

    Input shape: (n_samples, seq_length, feature_dim)
    """

    def __init__(
        self,
        seq_length: int = 128,
        feature_dim: int = 12,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 128,
        lambda_assoc: float = 3.0,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-4,
        sigma: float = 3.0,
        device: Optional[str] = None,
    ):
        super().__init__(seq_length=seq_length, feature_dim=feature_dim)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_ff = dim_ff
        self.lambda_assoc = lambda_assoc
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.sigma = sigma

        self._device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    # BaseIDS interface
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "AnomalyTransformerIDS":
        self._validate_input(X_train)
        if y_train is not None:
            X_train = X_train[y_train == 0]

        self._build_model()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        X_t = torch.tensor(X_train.astype(np.float32), device=self._device)
        n = len(X_t)

        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self._device)
            for start in range(0, n, self.batch_size):
                batch = X_t[perm[start: start + self.batch_size]]

                x_hat, disc = self._model(batch)

                rec_loss = F.mse_loss(x_hat, batch)
                # Minimax: maximise discrepancy while minimising reconstruction
                loss = rec_loss - self.lambda_assoc * disc

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        self._validate_input(X)
        X_t = torch.tensor(X.astype(np.float32), device=self._device)
        scores_list = []

        with torch.no_grad():
            for start in range(0, len(X_t), self.batch_size):
                batch = X_t[start: start + self.batch_size]
                x_hat, _ = self._model(batch)
                # Per-sample reconstruction error
                err = ((batch - x_hat) ** 2).mean(dim=(1, 2))
                scores_list.append(err)

        scores = torch.cat(scores_list).cpu().numpy()
        lo, hi = scores.min(), scores.max()
        if hi > lo:
            scores = (scores - lo) / (hi - lo)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(int)

    def get_config(self) -> Dict:
        return {
            "name": "AnomalyTransformer",
            "seq_length": self.seq_length,
            "feature_dim": self.feature_dim,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "lambda_assoc": self.lambda_assoc,
            "epochs": self.epochs,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self):
        F = self.feature_dim
        d = self.d_model

        class _Net(nn.Module):
            def __init__(self, seq_len, feat_dim, d_model, n_heads, n_layers, dim_ff, sigma):
                super().__init__()
                self.input_proj = nn.Linear(feat_dim, d_model)
                self.pos_enc = _PositionalEncoding(d_model, max_len=seq_len + 10)
                self.layers = nn.ModuleList(
                    [_AnomalyTransformerBlock(d_model, n_heads, dim_ff, sigma)
                     for _ in range(n_layers)]
                )
                self.out_proj = nn.Linear(d_model, feat_dim)

            def forward(self, x):
                h = self.pos_enc(self.input_proj(x))
                total_disc = torch.tensor(0.0, device=x.device)
                for layer in self.layers:
                    h, disc = layer(h)
                    total_disc = total_disc + disc
                return self.out_proj(h), total_disc / len(self.layers)

        self._model = _Net(
            self.seq_length, F, d, self.n_heads, self.n_layers, self.dim_ff, self.sigma
        ).to(self._device)
