"""
PatchTST for Reconstruction-Based Anomaly Detection
Adapted from: Nie et al., "A Time Series is Worth 64 Words: Long-term
Forecasting with Transformers", ICLR 2023.  arXiv:2211.14730

Adaptation for anomaly detection
---------------------------------
PatchTST was designed for time-series forecasting.  Here we adapt it for
unsupervised anomaly detection via masked-patch reconstruction:

Training:
  1. Split each (T, F) sequence into overlapping patches of size P.
  2. Randomly mask a fraction of patches (e.g. 50%).
  3. The Transformer encodes all patches (masked tokens replaced by a
     learnable [MASK] embedding) and reconstructs every patch.
  4. Training loss: MSE on the masked patches only.

Inference (anomaly scoring):
  1. Reconstruct ALL patches (no masking).
  2. Per-sample anomaly score = mean squared reconstruction error
     over all patches and features.
  3. Higher score ↔ harder to reconstruct ↔ more anomalous.

Reference:
    Nie et al., "A Time Series is Worth 64 Words", ICLR 2023.
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
# Positional encoding (learnable, as in PatchTST)
# ---------------------------------------------------------------------------

class _LearnablePositionalEncoding(nn.Module):
    def __init__(self, n_patches: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(n_patches, d_model)
        nn.init.trunc_normal_(self.pe.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_patches, d_model)
        pos = torch.arange(x.size(1), device=x.device)
        return x + self.pe(pos)


# ---------------------------------------------------------------------------
# PatchTST Anomaly Detector
# ---------------------------------------------------------------------------

class PatchTSTAnomalyIDS(BaseIDS):
    """
    PatchTST-based unsupervised anomaly detector for the BaseIDS interface.

    Parameters
    ----------
    patch_size   : timesteps per patch (default 16 → 8 patches for T=128)
    patch_stride : stride between patches (default 8, gives overlap)
    d_model      : Transformer embedding dimension
    n_heads      : number of attention heads
    n_layers     : number of Transformer encoder layers
    mask_ratio   : fraction of patches to mask during training (default 0.50)
    epochs       : training epochs
    batch_size   : mini-batch size
    lr           : Adam learning rate
    """

    def __init__(
        self,
        seq_length: int = 128,
        feature_dim: int = 12,
        patch_size: int = 16,
        patch_stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 128,
        mask_ratio: float = 0.50,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ):
        super().__init__(seq_length=seq_length, feature_dim=feature_dim)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_ff = dim_ff
        self.mask_ratio = mask_ratio
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

        # Compute number of patches
        self._n_patches = (seq_length - patch_size) // patch_stride + 1
        self._patch_dim = patch_size * feature_dim   # flattened patch

        self._model: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    # BaseIDS interface
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "PatchTSTAnomalyIDS":
        """Train on benign traffic via masked-patch reconstruction."""
        self._validate_input(X_train)
        if y_train is not None:
            X_train = X_train[y_train == 0]

        self._build_model()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        X_t = torch.tensor(X_train.astype(np.float32), device=self._device)  # (N, T, F)
        n = len(X_t)

        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self._device)
            for start in range(0, n, self.batch_size):
                batch = X_t[perm[start: start + self.batch_size]]  # (B, T, F)

                # Patchify
                patches = self._patchify(batch)                    # (B, n_patches, patch_dim)

                # Random mask
                B, NP, PD = patches.shape
                n_mask = max(1, int(NP * self.mask_ratio))
                mask_ids = torch.randperm(NP, device=self._device)[:n_mask]
                mask = torch.zeros(NP, dtype=torch.bool, device=self._device)
                mask[mask_ids] = True

                # Replace masked patches with learnable token
                patches_in = patches.clone()
                patches_in[:, mask, :] = self._model.mask_token

                # Forward
                patches_out = self._model(patches_in)              # (B, n_patches, patch_dim)

                # Loss only on masked patches
                loss = F.mse_loss(patches_out[:, mask, :], patches[:, mask, :])

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
        scores = []

        with torch.no_grad():
            for start in range(0, len(X_t), self.batch_size):
                batch = X_t[start: start + self.batch_size]
                patches = self._patchify(batch)
                patches_out = self._model(patches)
                err = ((patches - patches_out) ** 2).mean(dim=(1, 2))
                scores.append(err)

        scores = torch.cat(scores).cpu().numpy()
        lo, hi = scores.min(), scores.max()
        if hi > lo:
            scores = (scores - lo) / (hi - lo)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(int)

    def get_config(self) -> Dict:
        return {
            "name": "PatchTST-Anomaly",
            "seq_length": self.seq_length,
            "feature_dim": self.feature_dim,
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "mask_ratio": self.mask_ratio,
            "epochs": self.epochs,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, F)
        Returns : (B, n_patches, patch_size * F)
        """
        patches = []
        for i in range(self._n_patches):
            start = i * self.patch_stride
            end = start + self.patch_size
            p = x[:, start:end, :]                            # (B, patch_size, F)
            patches.append(p.reshape(p.size(0), -1))          # (B, patch_size * F)
        return torch.stack(patches, dim=1)                     # (B, n_patches, patch_dim)

    def _build_model(self):
        NP = self._n_patches
        PD = self._patch_dim
        d = self.d_model

        class _PatchTSTNet(nn.Module):
            def __init__(self, n_patches, patch_dim, d_model, n_heads, n_layers, dim_ff):
                super().__init__()
                self.mask_token = nn.Parameter(torch.zeros(patch_dim))
                nn.init.trunc_normal_(self.mask_token, std=0.02)

                self.patch_proj = nn.Linear(patch_dim, d_model)
                self.pos_enc = _LearnablePositionalEncoding(n_patches, d_model)

                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=dim_ff,
                    dropout=0.1,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
                self.head = nn.Linear(d_model, patch_dim)

            def forward(self, patches: torch.Tensor) -> torch.Tensor:
                # patches: (B, n_patches, patch_dim)
                h = self.pos_enc(self.patch_proj(patches))     # (B, n_patches, d_model)
                h = self.encoder(h)
                return self.head(h)                            # (B, n_patches, patch_dim)

        self._model = _PatchTSTNet(NP, PD, d, self.n_heads, self.n_layers, self.dim_ff).to(self._device)
