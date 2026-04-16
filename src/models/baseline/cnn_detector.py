"""
From-scratch 1D-CNN anomaly detector (Phase 3 baseline).

Provides the same interface as MoiraiAnomalyDetector so it can be dropped into
run_ablation.py with zero changes to the evaluation loop.  Designed to test
whether Moirai's pre-training provides any benefit over a lightweight supervised
model trained from scratch on the same data.

Architecture
------------
Encoder: 3-layer 1D-CNN over (seq_len=128, n_features=12) input
  Conv1d(12→64, k=7) → BatchNorm → ReLU
  Conv1d(64→128, k=5) → BatchNorm → ReLU
  Conv1d(128→128, k=3) → BatchNorm → ReLU
  GlobalAvgPool → 128-dim embedding

Projection head: ProjectionHead(128→64→32) — same class used by Moirai conditions

Anomaly scoring: mean squared reconstruction error via a symmetric decoder
  (mirrors the encoder in reverse); score = per-sample MSE averaged over time.

Training: NLL surrogate (MSE reconstruction) + SupCon on projection embeddings,
  exactly matching the CombinedLoss used for conditions B–D.

Usage in run_ablation.py
--------------------------
    from src.models.baseline.cnn_detector import CNNAnomalyDetector
    det = CNNAnomalyDetector()
    det.initialize()
    det.fine_tune_supervised(train_data, train_labels, val_data, val_labels, ...)
    result = det.detect_anomalies(sample, threshold=0.0, method='nll')
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from ..losses import SupervisedContrastiveLoss
from ..projection_head import ProjectionHead
from ..anomaly_result import AnomalyResult


# ---------------------------------------------------------------------------
# CNN Encoder / Decoder
# ---------------------------------------------------------------------------

class _CNNEncoder(nn.Module):
    """3-layer 1D-CNN encoder: (B, 12, 128) → (B, 128)."""

    def __init__(self, n_features: int = 12, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, n_features) → (B, embed_dim)."""
        # Conv1d expects (B, C, L)
        x = x.permute(0, 2, 1)          # (B, n_features, seq_len)
        features = self.net(x)           # (B, embed_dim, seq_len)
        return features.mean(dim=-1)     # global avg pool → (B, embed_dim)


class _CNNDecoder(nn.Module):
    """Mirror decoder: (B, embed_dim) → (B, seq_len, n_features) for reconstruction."""

    def __init__(self, n_features: int = 12, seq_len: int = 128, embed_dim: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.project = nn.Linear(embed_dim, embed_dim * seq_len)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, n_features, kernel_size=7, padding=3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, embed_dim) → (B, seq_len, n_features)."""
        B = z.shape[0]
        h = self.project(z).view(B, -1, self.seq_len)   # (B, embed_dim, seq_len)
        out = self.net(h)                                 # (B, n_features, seq_len)
        return out.permute(0, 2, 1)                       # (B, seq_len, n_features)


class _CNNAutoencoder(nn.Module):
    """Full autoencoder + projection head."""

    def __init__(self, n_features: int = 12, seq_len: int = 128, embed_dim: int = 128):
        super().__init__()
        self.encoder = _CNNEncoder(n_features, embed_dim)
        self.decoder = _CNNDecoder(n_features, seq_len, embed_dim)
        self.proj_head = ProjectionHead(
            input_dim=embed_dim, hidden_dim=64, output_dim=32
        )

    def forward(self, x: torch.Tensor):
        """Returns (recon, embedding, projection)."""
        z = self.encoder(x)
        recon = self.decoder(z)
        proj = self.proj_head(z)
        return recon, z, proj


# ---------------------------------------------------------------------------
# Public detector class (MoiraiAnomalyDetector-compatible interface)
# ---------------------------------------------------------------------------

class CNNAnomalyDetector:
    """
    From-scratch 1D-CNN detector with NLL(MSE)+SupCon training.

    Exposes the same interface as MoiraiAnomalyDetector so it plugs directly
    into run_ablation.py:
        det = CNNAnomalyDetector()
        det.initialize()
        det.fine_tune_supervised(...)
        result = det.detect_anomalies(sample, threshold=0.0, method='nll')
    """

    def __init__(
        self,
        n_features: int = 12,
        seq_len: int = 128,
        embed_dim: int = 128,
        **kwargs,          # absorbs unused MoiraiAnomalyDetector kwargs (model_size etc.)
    ):
        self.n_features = n_features
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self._model: Optional[_CNNAutoencoder] = None
        self._device = torch.device("cpu")
        self._initialized = False
        self._threshold: float = 0.0

    def initialize(self, **kwargs):
        """Build and initialise the CNN model (no pre-training)."""
        self._model = _CNNAutoencoder(
            n_features=self.n_features,
            seq_len=self.seq_len,
            embed_dim=self.embed_dim,
        ).to(self._device)
        self._initialized = True
        n_params = sum(p.numel() for p in self._model.parameters())
        logger.info(f"[CNN] Initialized from scratch — {n_params:,} parameters")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fine_tune_supervised(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: np.ndarray,
        val_labels: np.ndarray,
        n_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        contrastive_weight: float = 0.5,
        temperature: float = 0.07,
        checkpoint_dir: str = "models/cnn_supervised",
        early_stopping_patience: int = 5,
        early_stopping_criterion: str = "nll",
        freeze_encoder: str = "none",
    ) -> Dict[str, List[float]]:
        """Train NLL(MSE) + SupCon, identical calling convention to Moirai."""
        if not self._initialized:
            raise RuntimeError("Call initialize() first")

        model = self._model
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        supcon = SupervisedContrastiveLoss(temperature=temperature)

        train_X = torch.tensor(train_data, dtype=torch.float32)
        train_y = torch.tensor(train_labels, dtype=torch.long)
        val_X = torch.tensor(val_data, dtype=torch.float32)
        val_y = torch.tensor(val_labels, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(train_X, train_y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        history: Dict[str, List[float]] = {
            "train_loss": [], "train_nll": [], "train_contrastive": [], "val_loss": []
        }
        best_val = float("inf")
        patience_ctr = 0

        logger.info(
            f"[CNN] Training {n_epochs} epochs | "
            f"{len(train_data)} train ({(train_labels==0).sum()} benign, "
            f"{(train_labels==1).sum()} attack) | "
            f"λ={contrastive_weight}, τ={temperature}"
        )

        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = epoch_nll = epoch_con = 0.0
            n_batches = 0

            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                recon, _z, proj = model(xb)

                # NLL surrogate: MSE reconstruction loss
                nll_loss = F.mse_loss(recon, xb)

                # SupCon on projection
                con_loss = supcon(proj, yb) if contrastive_weight > 0 else torch.tensor(0.0)

                total = nll_loss + contrastive_weight * con_loss
                optimizer.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += total.item()
                epoch_nll += nll_loss.item()
                epoch_con += con_loss.item() if isinstance(con_loss, torch.Tensor) else con_loss
                n_batches += 1

            # Validation
            model.eval()
            with torch.no_grad():
                val_recon, _vz, _vp = model(val_X.to(self._device))
                val_nll = F.mse_loss(val_recon, val_X.to(self._device)).item()

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_nll = epoch_nll / max(n_batches, 1)
            avg_con = epoch_con / max(n_batches, 1)

            history["train_loss"].append(avg_loss)
            history["train_nll"].append(avg_nll)
            history["train_contrastive"].append(avg_con)
            history["val_loss"].append(val_nll)

            logger.info(
                f"[CNN] Epoch {epoch}/{n_epochs} | "
                f"loss={avg_loss:.4f} (nll={avg_nll:.4f}, con={avg_con:.4f}) | "
                f"val_nll={val_nll:.4f}"
            )

            # Early stopping on val NLL
            if val_nll < best_val - 1e-5:
                best_val = val_nll
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= early_stopping_patience:
                    logger.info(f"[CNN] Early stopping at epoch {epoch}")
                    break

        model.eval()
        return history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect_anomalies(
        self,
        sample: np.ndarray,
        threshold: float = 0.0,
        method: str = "nll",
    ) -> AnomalyResult:
        """
        Score a single sample using MSE reconstruction error as NLL surrogate.

        sample: (seq_len, n_features) — same shape as Moirai input.
        Returns AnomalyResult with anomaly_scores attribute (array of per-timestep MSE).
        The `threshold` parameter must be in [0, 1] per AnomalyResult contract;
        run_ablation.py always calls with threshold=0.0 (scores calibrated externally).
        """
        if not self._initialized or self._model is None:
            raise RuntimeError("Call initialize() first")

        self._model.eval()
        with torch.no_grad():
            x = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, n_feat)
            recon, _z, _proj = self._model(x.to(self._device))
            # Per-timestep reconstruction error, averaged over features
            mse_per_ts = ((recon - x.to(self._device)) ** 2).mean(dim=-1).squeeze(0)
            scores = mse_per_ts.cpu().numpy()          # (seq_len,)
            recon_np = recon.squeeze(0).cpu().numpy()  # (seq_len, n_features)

        clamp = np.clip(threshold, 0.0, 1.0)   # ensure [0, 1] contract
        is_anomaly_arr = (scores > scores.mean()).astype(bool)   # per-timestep flag

        return AnomalyResult(
            predictions=recon_np,
            actuals=sample,
            confidence_lower=recon_np - scores[:, None],
            confidence_upper=recon_np + scores[:, None],
            anomaly_scores=scores,
            is_anomaly=is_anomaly_arr,
            threshold=clamp,
            metadata={"mean_mse": float(scores.mean()), "method": "cnn_mse"},
        )
