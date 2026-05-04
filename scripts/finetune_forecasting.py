#!/usr/bin/env python3
"""
Contrastive Fine-Tuning for Forecasting + Catastrophic Forgetting Diagnosis

Fine-tunes Moirai on ETTh2 with NLL + optional temporal contrastive loss,
then measures whether fine-tuning degrades zero-shot forecasting ability
(catastrophic forgetting).

Conditions:
  A: Zero-shot (no fine-tuning) — baseline
  B: NLL-only fine-tuning — standard supervised approach
  C: NLL + Temporal SupCon — contrastive fine-tuning (mirrors IoT setup)
  D: Frozen encoder + linear head — upper bound on representation preservation

Diagnosis metrics:
  - Zero-shot MSE on held-out val set at each epoch
  - CKA between pre-trained and fine-tuned encoder representations
  - Weight drift (L2 distance from pre-trained weights)

Usage:
    python scripts/finetune_forecasting.py \
        --data-path data/forecasting/ETTh2.csv \
        --condition B \
        --epochs 20 \
        --results-dir results/forecasting_finetune
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.forecasting_loader import ETTh1Loader, get_forecasting_loader
from src.models.losses import SupervisedContrastiveLoss


def _patch_packed_scaler_for_mps():
    """Patch uni2ts PackedScaler to skip .double() on MPS (unsupported dtype).

    The original forward does target.double() for numerical precision in
    mean/variance computation, then casts back to float32.  On MPS, float64
    is unsupported.  Float32 precision is sufficient for our use case.
    """
    try:
        from uni2ts.module.packed_scaler import PackedScaler

        _original_forward = PackedScaler.forward

        def _forward_no_double(self, target, observed_mask=None, sample_id=None,
                               variate_id=None):
            if observed_mask is None:
                observed_mask = torch.ones_like(target, dtype=torch.bool)
            if sample_id is None:
                sample_id = torch.zeros(
                    target.shape[:-1], dtype=torch.long, device=target.device)
            if variate_id is None:
                variate_id = torch.zeros(
                    target.shape[:-1], dtype=torch.long, device=target.device)

            # Skip .double() — compute in float32 (sufficient precision)
            loc, scale = self._get_loc_scale(
                target.float(), observed_mask, sample_id, variate_id)
            return loc.float(), scale.float()

        PackedScaler.forward = _forward_no_double
        logger.info("Patched PackedScaler to skip .double() for MPS compatibility")
    except ImportError:
        pass

    # Patch attention: MPS requires contiguous tensors for scaled_dot_product_attention
    try:
        from uni2ts.module.attention import GroupedQueryAttention
        _original_gqa_forward = GroupedQueryAttention.forward

        def _gqa_forward_contiguous(self, query, key, value, attn_mask=None,
                                     query_var_id=None, kv_var_id=None,
                                     query_time_id=None, kv_time_id=None):
            from einops import rearrange, repeat

            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

            query = self.q_norm(rearrange(
                query, "... q_len (group hpg dim) -> ... group hpg q_len dim",
                group=self.num_groups, hpg=self.heads_per_group))
            key = self.k_norm(repeat(
                key, "... kv_len (group dim) -> ... group hpg kv_len dim",
                group=self.num_groups, hpg=self.heads_per_group))
            value = repeat(
                value, "... kv_len (group dim) -> ... group hpg kv_len dim",
                group=self.num_groups, hpg=self.heads_per_group)

            query_var_id, kv_var_id = self._get_var_id(query, key, query_var_id, kv_var_id)
            query_time_id, kv_time_id = self._get_time_id(
                query, key, query_time_id, kv_time_id)
            attn_mask = self._update_attn_mask(
                attn_mask, query, key,
                query_var_id=query_var_id, kv_var_id=kv_var_id,
                query_time_id=query_time_id, kv_time_id=kv_time_id)
            query, key = self._qk_proj(
                query, key,
                query_var_id=query_var_id, kv_var_id=kv_var_id,
                query_time_id=query_time_id, kv_time_id=kv_time_id)

            # MPS fix: ensure contiguous tensors for scaled_dot_product_attention
            out = F.scaled_dot_product_attention(
                query.contiguous(), key.contiguous(), value.contiguous(),
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout_p,
                scale=self.softmax_scale)
            out = rearrange(out, "... group hpg q_len dim -> ... q_len (group hpg dim)")
            return self.out_proj(out)

        GroupedQueryAttention.forward = _gqa_forward_contiguous
        logger.info("Patched GroupedQueryAttention for MPS contiguous tensors")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Temporal contrastive label assignment
# ---------------------------------------------------------------------------

def assign_temporal_labels(n_sequences: int, n_clusters: int = 8) -> np.ndarray:
    """
    Assign pseudo-labels based on temporal position for contrastive learning.

    Divides time series into temporal clusters. Windows within the same
    cluster are treated as positive pairs; windows from different clusters
    are negatives. This encourages the encoder to learn temporal-regime-aware
    representations.

    Args:
        n_sequences: Total number of sequences
        n_clusters: Number of temporal clusters (labels)

    Returns:
        labels: (n_sequences,) integer pseudo-labels
    """
    labels = np.zeros(n_sequences, dtype=np.int64)
    cluster_size = n_sequences // n_clusters
    for i in range(n_clusters):
        start = i * cluster_size
        end = (i + 1) * cluster_size if i < n_clusters - 1 else n_sequences
        labels[start:end] = i
    return labels


# ---------------------------------------------------------------------------
# CKA computation
# ---------------------------------------------------------------------------

def linear_CKA(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear CKA (Centered Kernel Alignment) between two representations.

    CKA measures similarity between representations — 1.0 means identical,
    0.0 means completely different. Used to track how much the encoder
    representations change during fine-tuning.

    Args:
        X: (n, d1) representation matrix
        Y: (n, d2) representation matrix

    Returns:
        CKA similarity score in [0, 1]
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y

    hsic_xy = np.trace(XtX @ YtY)  # Simplified: ||X^T Y||_F^2
    hsic_xx = np.trace(XtX @ XtX)
    hsic_yy = np.trace(YtY @ YtY)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


def linear_probe_r2(
    reps_train: np.ndarray,
    reps_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    alpha: float = 1.0,
    probe_type: str = 'ridge',
    mlp_layers: int = 1,
    gbm_depth: int = 6,
):
    """Fit a probe on frozen representations and return val R-squared.

    probe_type='ridge' (default): Ridge regression on mean-pooled reps.
    probe_type='mlp'             : sklearn MLPRegressor, hidden=(64,)*mlp_layers.
    probe_type='linear_forecaster': Ridge with stronger regularisation on
                                    (possibly flattened sequence) reps;
                                    matches reviewer Q2's "frozen-encoder
                                    linear forecaster head" ask.
    probe_type='all'             : dict with all three (MLP at mlp_layers depth).
    mlp_layers: number of hidden layers for MLP probe (each of width 64).
    """
    from sklearn.linear_model import Ridge
    # If reps are 3D (N, T, D) flatten sequence axis for the linear-forecaster
    def _flat(x):
        return x.reshape(len(x), -1)
    reps_tr_flat = _flat(reps_train)
    reps_va_flat = _flat(reps_val)
    y_tr = y_train.reshape(len(y_train), -1)
    y_va = y_val.reshape(len(y_val), -1)

    def _fit_ridge():
        reg = Ridge(alpha=alpha).fit(reps_tr_flat, y_tr)
        return float(reg.score(reps_va_flat, y_va))

    def _fit_mlp():
        from sklearn.neural_network import MLPRegressor
        reg = MLPRegressor(
            hidden_layer_sizes=tuple([64] * int(mlp_layers)),
            max_iter=500,
            alpha=1e-3,
            random_state=0,
            early_stopping=True,
            validation_fraction=0.1,
        ).fit(reps_tr_flat, y_tr)
        return float(reg.score(reps_va_flat, y_va))

    def _fit_linear_forecaster():
        # Stronger linear probe: Ridge with sweep over alpha, pick best-val.
        # Mirrors the "train a linear forecaster head frozen" recipe.
        best_r2 = -np.inf
        for a in (0.01, 0.1, 1.0, 10.0, 100.0):
            reg = Ridge(alpha=a).fit(reps_tr_flat, y_tr)
            r2 = float(reg.score(reps_va_flat, y_va))
            if r2 > best_r2:
                best_r2 = r2
        return best_r2

    def _fit_gbm():
        # Non-linear expressive probe: HistGradientBoostingRegressor per output dim.
        from sklearn.ensemble import HistGradientBoostingRegressor
        preds_val = np.zeros_like(y_va)
        for j in range(y_tr.shape[1]):
            gbm = HistGradientBoostingRegressor(
                max_iter=150, max_depth=gbm_depth, learning_rate=0.05,
                early_stopping=True, validation_fraction=0.1, random_state=0)
            gbm.fit(reps_tr_flat, y_tr[:, j])
            preds_val[:, j] = gbm.predict(reps_va_flat)
        ss_res = ((preds_val - y_va) ** 2).sum()
        ss_tot = ((y_va - y_va.mean(axis=0)) ** 2).sum()
        if ss_tot < 1e-12:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    if probe_type == 'ridge':
        return _fit_ridge()
    if probe_type == 'mlp':
        return _fit_mlp()
    if probe_type == 'linear_forecaster':
        return _fit_linear_forecaster()
    if probe_type == 'gbm':
        return _fit_gbm()
    if probe_type == 'all':
        return {
            'ridge': _fit_ridge(),
            'mlp': _fit_mlp(),
            'linear_forecaster': _fit_linear_forecaster(),
        }
    return {'ridge': _fit_ridge(), 'mlp': _fit_mlp()}


def compute_weight_drift(model, pretrained_params: dict) -> float:
    """Compute L2 distance between current and pre-trained weights."""
    total_drift = 0.0
    n_params = 0
    for name, param in model.named_parameters():
        if name in pretrained_params:
            diff = (param.data - pretrained_params[name]).float()
            total_drift += diff.norm().item() ** 2
            n_params += param.numel()
    return float(np.sqrt(total_drift))


# ---------------------------------------------------------------------------
# EWC: Fisher Information Matrix
# ---------------------------------------------------------------------------

def compute_fisher_diagonal(model, data_loader, horizon, device, n_samples=200):
    """
    Compute diagonal Fisher Information Matrix for EWC regularization.

    Runs the pre-trained model on training data, computes NLL gradients,
    and squares them to estimate parameter importance.
    """
    from src.models.moirai_detector import _apply_uni2ts_gradient_patch, UNI2TS_AVAILABLE
    if UNI2TS_AVAILABLE:
        _apply_uni2ts_gradient_patch()

    fisher = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    model.eval()
    count = 0

    for batch_idx, (context_batch, target_batch, labels_batch) in enumerate(data_loader):
        if count >= n_samples:
            break
        context_batch = context_batch.to(device)
        target_batch = target_batch.to(device)
        b = context_batch.shape[0]

        full_target = torch.cat([context_batch, target_batch], dim=1)
        seq_len = full_target.shape[1]
        n_feat = full_target.shape[2]
        observed = torch.ones(b, seq_len, n_feat, dtype=torch.bool, device=device)
        is_pad = torch.zeros(b, seq_len, dtype=torch.bool, device=device)

        model.zero_grad()
        try:
            per_sample_nll = model._val_loss(
                patch_size=32, target=full_target,
                observed_target=observed, is_pad=is_pad,
            )
            loss = per_sample_nll.mean()
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) * b
            count += b
        except Exception as e:
            logger.warning(f"Fisher batch {batch_idx} failed: {e}")
            continue

    if count > 0:
        for name in fisher:
            fisher[name] /= count

    logger.info(f"Fisher diagonal computed from {count} samples")
    return fisher


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_forecasting(
    model,
    context: torch.Tensor,
    target: np.ndarray,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    horizon: int,
    batch_size: int = 8,
    num_samples: int = 20,
    device: str = 'cpu'
) -> dict:
    """
    Evaluate Moirai forecasting performance (MSE/MAE on normalized scale).

    Uses median of forecast samples for robust point prediction.
    """
    model.eval()
    all_preds = []
    extended_lookback = context.shape[1]

    with torch.no_grad():
        for i in range(0, len(context), batch_size):
            batch_ctx = context[i:i+batch_size].to(device)
            b = batch_ctx.shape[0]
            past_obs = torch.ones_like(batch_ctx, dtype=torch.bool)
            past_pad = torch.zeros(b, extended_lookback, dtype=torch.bool, device=device)

            samples = model.forward(
                past_target=batch_ctx,
                past_observed_target=past_obs,
                past_is_pad=past_pad,
                num_samples=num_samples
            )
            median_pred = samples.median(dim=1).values.cpu().numpy()
            all_preds.append(median_pred)

    predictions = np.concatenate(all_preds, axis=0)

    # Squeeze trailing singleton dim in univariate mode (pred: (B,H); target: (B,H,1)).
    if predictions.ndim == 2 and target.ndim == 3 and target.shape[-1] == 1:
        target = target.squeeze(-1)
    elif predictions.ndim == 3 and predictions.shape[-1] == 1 and target.ndim == 2:
        predictions = predictions.squeeze(-1)

    # Normalize both using training statistics (mean/std shapes (D,) or scalar).
    mean_arr = np.asarray(train_mean).reshape(-1)
    std_arr = np.asarray(train_std).reshape(-1)
    if mean_arr.size == 1:
        mean_arr = mean_arr.item()
        std_arr = std_arr.item()
    pred_norm = (predictions - mean_arr) / std_arr
    tgt_norm = (target - mean_arr) / std_arr

    mse = float(np.mean((pred_norm - tgt_norm) ** 2))
    mae = float(np.mean(np.abs(pred_norm - tgt_norm)))
    return {'mse': mse, 'mae': mae}


def extract_representations(
    model,
    data: torch.Tensor,
    encoder_hook_fn,
    batch_size: int = 32,
    device: str = 'cpu',
    max_samples: int = 500,
    keep_sequence: bool = False,
) -> np.ndarray:
    """Extract encoder representations for CKA / probing.

    keep_sequence=False (default): mean-pool over sequence → (N, D).
    keep_sequence=True           : return (N, T, D) for linear-forecaster probing.
    """
    model.eval()
    captured = {}

    def hook(module, input, output):
        captured['out'] = output

    module = model.module
    if hasattr(module, 'base_model'):
        encoder = module.base_model.model.encoder
    else:
        encoder = module.encoder

    handle = encoder.register_forward_hook(hook)

    all_reps = []
    n = min(len(data), max_samples)
    data_subset = data[:n]

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = data_subset[i:i+batch_size].to(device)
            b = batch.shape[0]
            seq_len = batch.shape[1]
            past_obs = torch.ones_like(batch, dtype=torch.bool)
            past_pad = torch.zeros(b, seq_len, dtype=torch.bool, device=device)

            try:
                model.forward(
                    past_target=batch,
                    past_observed_target=past_obs,
                    past_is_pad=past_pad,
                    num_samples=2
                )
            except Exception:
                pass

            if 'out' in captured:
                rep = captured['out']
                if isinstance(rep, tuple):
                    rep = rep[0]
                if keep_sequence:
                    all_reps.append(rep.cpu().numpy())
                else:
                    rep_pooled = rep.mean(dim=1).cpu().numpy()
                    all_reps.append(rep_pooled)

    handle.remove()

    if all_reps:
        return np.concatenate(all_reps, axis=0)
    return np.zeros((0, 1))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    horizon: int,
    device: str,
    contrastive_weight: float = 0.0,
    projection_head: nn.Module = None,
    contrastive_loss_fn: nn.Module = None,
    captured_embeddings: dict = None,
    freeze_encoder: bool = False,
    pretrained_params: dict = None,
    l2sp_weight: float = 0.0,
    ewc_lambda: float = 0.0,
    fisher: dict = None,
) -> dict:
    """Train one epoch of NLL (+ optional contrastive/L2-SP/EWC) fine-tuning."""
    model.train()
    if projection_head is not None:
        projection_head.train()

    epoch_nll = 0.0
    epoch_cont = 0.0
    epoch_total = 0.0
    batch_count = 0

    for batch_idx, (context_batch, target_batch, labels_batch) in enumerate(train_loader):
        context_batch = context_batch.to(device)
        target_batch = target_batch.to(device)
        labels_batch = labels_batch.to(device)
        b = context_batch.shape[0]

        # Concatenate context (96) + target (96) for NLL computation = 192 timesteps
        # This matches the IoT fine-tuning pattern in moirai_detector.py
        full_target = torch.cat([context_batch, target_batch], dim=1)
        seq_len = full_target.shape[1]
        n_feat = full_target.shape[2]
        observed = torch.ones(b, seq_len, n_feat, dtype=torch.bool, device=device)
        is_pad = torch.zeros(b, seq_len, dtype=torch.bool, device=device)

        try:
            # Use patch_size=32 (standard for Moirai small, matches IoT code)
            per_sample_nll = model._val_loss(
                patch_size=32,
                target=full_target,
                observed_target=observed,
                is_pad=is_pad,
            )
            nll_loss = per_sample_nll.mean()  # reduce to scalar

            total_loss = nll_loss

            # Optional contrastive loss
            cont_loss = torch.tensor(0.0, device=device)
            if contrastive_weight > 0 and projection_head is not None and captured_embeddings:
                if 'encoder' in captured_embeddings:
                    enc = captured_embeddings['encoder']
                    if isinstance(enc, tuple):
                        enc = enc[0]
                    # Pool over sequence dimension
                    pooled = enc.mean(dim=1)
                    projected = projection_head(pooled)
                    cont_loss = contrastive_loss_fn(projected, labels_batch)
                    total_loss = nll_loss + contrastive_weight * cont_loss

            # L2-SP regularization: penalize drift from pre-trained weights
            if l2sp_weight > 0 and pretrained_params is not None:
                l2sp_loss = sum(
                    (p - pretrained_params[name]).pow(2).sum()
                    for name, p in model.named_parameters()
                    if p.requires_grad and name in pretrained_params
                )
                total_loss = total_loss + l2sp_weight * l2sp_loss

            # EWC regularization: Fisher-weighted drift penalty
            if ewc_lambda > 0 and fisher is not None and pretrained_params is not None:
                ewc_loss = sum(
                    (fisher[name] * (p - pretrained_params[name]).pow(2)).sum()
                    for name, p in model.named_parameters()
                    if p.requires_grad and name in fisher and name in pretrained_params
                )
                total_loss = total_loss + ewc_lambda * ewc_loss

        except Exception as e:
            logger.warning(f"Batch {batch_idx} failed: {e}")
            continue

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + (list(projection_head.parameters()) if projection_head else []),
            1.0
        )
        optimizer.step()

        epoch_nll += nll_loss.item()
        epoch_cont += cont_loss.item()
        epoch_total += total_loss.item()
        batch_count += 1

    if batch_count == 0:
        return {'nll': 0, 'contrastive': 0, 'total': 0}

    return {
        'nll': epoch_nll / batch_count,
        'contrastive': epoch_cont / batch_count,
        'total': epoch_total / batch_count,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Contrastive fine-tuning + forgetting diagnosis")
    parser.add_argument('--data-path', default='data/forecasting/ETTh2.csv')
    parser.add_argument('--horizon', type=int, default=96)
    parser.add_argument('--condition', required=True, choices=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                        help="A=zero-shot, B=NLL-only, C=NLL+SupCon, D=frozen encoder, E=LoRA, F=L2-SP, G=EWC")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--contrastive-weight', type=float, default=0.5)
    parser.add_argument('--n-temporal-clusters', type=int, default=8)
    parser.add_argument('--max-train-samples', type=int, default=1000)
    parser.add_argument('--model-size', default='small', choices=['small', 'base', 'large'],
                        help="Moirai model size")
    parser.add_argument('--l2sp-weight', type=float, default=0.0,
                        help="L2-SP regularization weight (condition F)")
    parser.add_argument('--ewc-lambda', type=float, default=0.0,
                        help="EWC regularization weight (condition G)")
    parser.add_argument('--lora-rank', type=int, default=8, help="LoRA rank (condition E)")
    parser.add_argument('--lora-alpha', type=int, default=16, help="LoRA alpha (condition E)")
    parser.add_argument('--unfreeze-top-n-layers', type=int, default=0,
                        help="For condition D: unfreeze top N transformer layers (0=full freeze, default).")
    parser.add_argument('--lora-target-modules', nargs='+', default=None,
                        help="LoRA target module names (condition E, default q_proj v_proj out_proj). Used for reviewer Q4 ablation.")
    parser.add_argument('--results-dir', default='results/forecasting_finetune')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-every', type=int, default=1,
                        help="Evaluate forgetting every N epochs")
    parser.add_argument('--max-eval-sequences', type=int, default=300,
                        help="Max sequences for periodic evaluation")
    parser.add_argument('--features', default='M', choices=['M', 'S', 'MS'],
                        help="M=multivariate, S=univariate (OT target only), MS=multivariate->univariate")
    parser.add_argument('--early-stopping', action='store_true',
                        help="Restore encoder weights to best-val-MSE checkpoint before final eval/probe (V17).")
    parser.add_argument('--probe-type', default='ridge',
                        choices=['ridge', 'mlp', 'both', 'linear_forecaster', 'all'],
                        help="Linear-probe regressor type: ridge, mlp, both (ridge+mlp), linear_forecaster (Ridge over sequence), or all (reviewer Q2 comparison)")
    parser.add_argument('--probe-mlp-layers', default='1',
                        help="Comma-separated MLP hidden-layer depths to sweep (e.g. '1,2,5'). Each depth k uses hidden_layer_sizes=(64,)*k. Emits r2_delta_mlp_k{k} fields.")
    parser.add_argument('--save-best-encoder', action='store_true',
                        help="When --early-stopping is set, also persist best_state to {results_dir}/best_encoder.pt for future re-probing.")
    parser.add_argument('--deterministic', action='store_true',
                        help="Enable deterministic training: fixed seeds for DataLoader workers, PYTHONHASHSEED, and torch.use_deterministic_algorithms(True, warn_only=True).")
    args = parser.parse_args()

    # Patch uni2ts for MPS compatibility before any model loading
    if args.device == 'mps' or (args.device == 'auto' and torch.backends.mps.is_available()):
        _patch_packed_scaler_for_mps()

    if args.deterministic:
        import os, random as _random
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        _random.seed(args.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    lookback = 96
    horizon = args.horizon
    # n_features determined by features mode; set after loader inspection

    condition_names = {
        'A': 'Zero-shot', 'B': 'NLL-only', 'C': 'NLL+SupCon',
        'D': 'Frozen encoder', 'E': 'LoRA', 'F': 'L2-SP', 'G': 'EWC'
    }
    logger.info(f"Condition {args.condition}: {condition_names[args.condition]}")

    # Load data
    loader = get_forecasting_loader(args.data_path, lookback_window=lookback, forecast_horizon=horizon, features=args.features)
    train_df, val_df, test_df = loader.get_splits()

    # Univariate mode uses OT target only; multivariate uses all feature columns.
    feature_cols = ['OT'] if args.features == 'S' else loader.FEATURE_COLUMNS
    n_features = len(feature_cols)
    logger.info(f"Features mode={args.features}, n_features={n_features}")
    train_vals = train_df[feature_cols].values
    val_vals = val_df[feature_cols].values
    test_vals = test_df[feature_cols].values

    train_mean = train_vals.mean(axis=0)
    train_std = train_vals.std(axis=0) + 1e-8

    # Create sequences for Moirai
    # For TRAINING (NLL loss): use context_length (96) + prediction_length (96) = 192
    #   This matches the IoT fine-tuning pattern in moirai_detector.py
    # For EVALUATION (forward/inference): use extended_lookback (192) for patch_size='auto'
    extended_lookback = lookback + horizon  # for evaluation

    def make_train_sequences(data, ctx_len, hz):
        """Create (context, target) pairs for NLL training."""
        X, y = [], []
        total = ctx_len + hz
        for i in range(len(data) - total + 1):
            X.append(data[i:i+ctx_len])
            y.append(data[i+ctx_len:i+total])
        return np.array(X), np.array(y)

    def make_eval_sequences(data, ext_lb, hz):
        """Create extended lookback sequences for inference evaluation."""
        X, y = [], []
        total = ext_lb + hz
        for i in range(len(data) - total + 1):
            X.append(data[i:i+ext_lb])
            y.append(data[i+ext_lb:i+total])
        return np.array(X), np.array(y)

    X_train, y_train = make_train_sequences(train_vals, lookback, horizon)
    X_val_eval, y_val_eval_raw = make_eval_sequences(val_vals, extended_lookback, horizon)
    X_test_eval, y_test_eval_raw = make_eval_sequences(test_vals, extended_lookback, horizon)

    # Subsample training data
    if args.max_train_samples > 0 and args.max_train_samples < len(X_train):
        indices = np.random.choice(len(X_train), size=args.max_train_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    logger.info(f"Train: {len(X_train)}, Val(eval): {len(X_val_eval)}, Test(eval): {len(X_test_eval)}")

    # Load model
    from src.models.moirai_detector import MoiraiAnomalyDetector
    detector = MoiraiAnomalyDetector(
        model_size=args.model_size,
        context_length=lookback,
        prediction_length=horizon,
        target_dim=n_features,
        num_samples=20,
        device=args.device
    )
    detector.initialize()
    model = detector.model

    # Store pre-trained weights for CKA and drift computation
    pretrained_params = {name: param.data.clone() for name, param in model.named_parameters()}

    # Apply LoRA if condition E
    if args.condition == 'E':
        from src.models.lora_adapter import apply_lora
        model = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha,
                           target_modules=args.lora_target_modules)

    # Subsample eval data for speed
    eval_limit = args.max_eval_sequences
    X_val_eval_t = torch.from_numpy(X_val_eval[:eval_limit]).float()
    y_val_eval_sub = y_val_eval_raw[:eval_limit]

    # Compute zero-shot baseline
    logger.info("Computing zero-shot baseline...")
    zeroshot_metrics = evaluate_forecasting(
        model, X_val_eval_t, y_val_eval_sub, train_mean, train_std,
        horizon, device=args.device
    )
    logger.info(f"Zero-shot: MSE={zeroshot_metrics['mse']:.6f}, MAE={zeroshot_metrics['mae']:.6f}")

    # Extract pre-trained representations
    pretrained_reps = extract_representations(model, X_val_eval_t, None, device=args.device)
    logger.info(f"Pre-trained reps shape: {pretrained_reps.shape}")

    # Condition A: just return zero-shot results
    if args.condition == 'A':
        results = {
            'condition': 'A',
            'zeroshot_mse': zeroshot_metrics['mse'],
            'zeroshot_mae': zeroshot_metrics['mae'],
            'horizon': horizon,
            'seed': args.seed,
        }
        output_path = results_dir / f'condition_A_h{horizon}_s{args.seed}.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved to {output_path}")
        return

    # Setup training
    # Assign temporal labels for contrastive learning
    train_labels = assign_temporal_labels(len(X_train), args.n_temporal_clusters)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
        torch.from_numpy(train_labels).long()
    )
    _dl_generator = torch.Generator()
    _dl_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        generator=_dl_generator if args.deterministic else None,
    )

    # Setup optimizer
    freeze_encoder = (args.condition == 'D')
    use_contrastive = (args.condition == 'C')
    use_l2sp = (args.condition == 'F' and args.l2sp_weight > 0)
    use_ewc = (args.condition == 'G' and args.ewc_lambda > 0)
    fisher_diag = None

    # Compute Fisher diagonal for EWC before any fine-tuning
    if use_ewc:
        logger.info("Computing Fisher Information Matrix for EWC...")
        fisher_diag = compute_fisher_diagonal(
            model, train_loader, horizon, args.device, n_samples=200
        )

    if freeze_encoder:
        encoder = model.module.encoder if not hasattr(model.module, 'base_model') else model.module.base_model.model.encoder
        n_top = getattr(args, 'unfreeze_top_n_layers', 0)
        if n_top > 0:
            n_layers = len(encoder.layers)
            unfreeze_from = n_layers - n_top
            for name, param in encoder.named_parameters():
                if 'layers.' in name:
                    layer_idx = int(name.split('layers.')[1].split('.')[0])
                    param.requires_grad = (layer_idx >= unfreeze_from)
                else:
                    param.requires_grad = False
            n_frozen = sum(1 for p in encoder.parameters() if not p.requires_grad)
            n_total = sum(1 for p in encoder.parameters())
            logger.info(f"Partial freeze: {n_frozen}/{n_total} encoder params frozen, top {n_top} layers trainable")
        else:
            for param in encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder frozen — only head parameters will be updated")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Contrastive setup
    projection_head = None
    contrastive_loss_fn = None
    captured_embeddings = {}

    if use_contrastive:
        d_model = model.module.d_model  # 384 for small
        projection_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(args.device)
        contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.07)
        trainable_params += list(projection_head.parameters())

        # Register hook for encoder embeddings
        encoder_module = model.module.encoder if not hasattr(model.module, 'base_model') else model.module.base_model.model.encoder
        encoder_module.register_forward_hook(
            lambda mod, inp, out: captured_embeddings.update({'encoder': out})
        )
        logger.info(f"Contrastive learning enabled: weight={args.contrastive_weight}, "
                    f"clusters={args.n_temporal_clusters}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop with forgetting diagnosis
    history = {
        'train_nll': [], 'train_contrastive': [], 'train_total': [],
        'val_mse': [], 'val_mae': [],
        'cka': [], 'weight_drift': [],
        'epoch': [],
    }

    # Record epoch 0 (pre-trained)
    history['epoch'].append(0)
    history['val_mse'].append(zeroshot_metrics['mse'])
    history['val_mae'].append(zeroshot_metrics['mae'])
    history['cka'].append(1.0)
    history['weight_drift'].append(0.0)
    history['train_nll'].append(0.0)
    history['train_contrastive'].append(0.0)
    history['train_total'].append(0.0)

    logger.info(f"\nStarting training for {args.epochs} epochs...")
    logger.info(f"{'Epoch':>6} {'NLL':>8} {'Cont':>8} {'MSE':>8} {'CKA':>6} {'Drift':>8}")
    logger.info("-" * 52)

    best_val_mse = float('inf')
    best_epoch = 0
    best_state = None
    if args.early_stopping:
        logger.info("Early stopping enabled: will restore best-val-MSE checkpoint before final eval.")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Apply gradient patch
        from src.models.moirai_detector import _apply_uni2ts_gradient_patch, UNI2TS_AVAILABLE
        if UNI2TS_AVAILABLE:
            _apply_uni2ts_gradient_patch()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, horizon, args.device,
            contrastive_weight=args.contrastive_weight if use_contrastive else 0.0,
            projection_head=projection_head,
            contrastive_loss_fn=contrastive_loss_fn,
            captured_embeddings=captured_embeddings,
            freeze_encoder=freeze_encoder,
            pretrained_params=pretrained_params if (use_l2sp or use_ewc) else None,
            l2sp_weight=args.l2sp_weight if use_l2sp else 0.0,
            ewc_lambda=args.ewc_lambda if use_ewc else 0.0,
            fisher=fisher_diag,
        )
        scheduler.step()

        history['train_nll'].append(train_metrics['nll'])
        history['train_contrastive'].append(train_metrics['contrastive'])
        history['train_total'].append(train_metrics['total'])
        history['epoch'].append(epoch)

        # Periodic evaluation
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = evaluate_forecasting(
                model, X_val_eval_t, y_val_eval_sub, train_mean, train_std,
                horizon, device=args.device
            )
            history['val_mse'].append(val_metrics['mse'])
            history['val_mae'].append(val_metrics['mae'])

            # CKA
            current_reps = extract_representations(model, X_val_eval_t, None, device=args.device)
            if len(pretrained_reps) > 0 and len(current_reps) > 0:
                n = min(len(pretrained_reps), len(current_reps))
                cka = linear_CKA(pretrained_reps[:n], current_reps[:n])
            else:
                cka = 0.0
            history['cka'].append(cka)

            # Weight drift
            drift = compute_weight_drift(model, pretrained_params)
            history['weight_drift'].append(drift)

            elapsed = time.time() - t0
            logger.info(f"{epoch:>6d} {train_metrics['nll']:>8.4f} {train_metrics['contrastive']:>8.4f} "
                       f"{val_metrics['mse']:>8.4f} {cka:>6.3f} {drift:>8.2f}  ({elapsed:.1f}s)")

            if args.early_stopping and val_metrics['mse'] < best_val_mse:
                best_val_mse = val_metrics['mse']
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            history['val_mse'].append(None)
            history['val_mae'].append(None)
            history['cka'].append(None)
            history['weight_drift'].append(None)

    # Capture final-epoch metrics before any early-stopping restore (for honest reporting).
    final_epoch_val_mse = history['val_mse'][-1]
    final_epoch_cka = history['cka'][-1]
    final_epoch_drift = history['weight_drift'][-1]
    final_epoch_forgetting = (final_epoch_val_mse - zeroshot_metrics['mse']) / zeroshot_metrics['mse'] * 100

    # If early stopping is enabled, restore encoder weights to best-val-MSE checkpoint.
    early_stopping_info = {'enabled': bool(args.early_stopping)}
    if args.early_stopping and best_state is not None:
        logger.info(f"Restoring best-val-MSE checkpoint: epoch {best_epoch}, val_mse={best_val_mse:.6f}")
        device_state = {k: v.to(args.device) for k, v in best_state.items()}
        model.load_state_dict(device_state)
        # Recompute CKA and drift against the restored state for the "final" history slot.
        restored_reps = extract_representations(model, X_val_eval_t, None, device=args.device)
        if len(pretrained_reps) > 0 and len(restored_reps) > 0:
            n = min(len(pretrained_reps), len(restored_reps))
            restored_cka = linear_CKA(pretrained_reps[:n], restored_reps[:n])
        else:
            restored_cka = 0.0
        restored_drift = compute_weight_drift(model, pretrained_params)
        restored_val_metrics = evaluate_forecasting(
            model, X_val_eval_t, y_val_eval_sub, train_mean, train_std,
            horizon, device=args.device
        )
        history['val_mse'][-1] = restored_val_metrics['mse']
        history['val_mae'][-1] = restored_val_metrics['mae']
        history['cka'][-1] = restored_cka
        history['weight_drift'][-1] = restored_drift
        early_stopping_info.update({
            'best_epoch': int(best_epoch),
            'best_val_mse': float(best_val_mse),
            'restored_val_mse': float(restored_val_metrics['mse']),
            'restored_cka': float(restored_cka),
            'restored_weight_drift': float(restored_drift),
            'final_epoch_val_mse': float(final_epoch_val_mse),
            'final_epoch_cka': float(final_epoch_cka),
            'final_epoch_weight_drift': float(final_epoch_drift),
            'final_epoch_forgetting_pct': float(final_epoch_forgetting),
        })
        if args.save_best_encoder:
            enc_path = Path(args.results_dir) / 'best_encoder.pt'
            torch.save(best_state, enc_path)
            early_stopping_info['best_encoder_path'] = str(enc_path)
            logger.info(f"Saved best-val-MSE encoder to {enc_path}")

    # Final test evaluation (on possibly-restored weights).
    X_test_eval_t = torch.from_numpy(X_test_eval[:eval_limit]).float()
    test_metrics = evaluate_forecasting(
        model, X_test_eval_t, y_test_eval_raw[:eval_limit], train_mean, train_std,
        horizon, device=args.device
    )

    # Compute forgetting metric (using current history slot — restored if ES, else final-epoch).
    forgetting = (history['val_mse'][-1] - zeroshot_metrics['mse']) / zeroshot_metrics['mse'] * 100

    # Linear probing: functional measure of feature preservation
    # Use eval sequences (extended lookback) for both train and val probing
    probe_results = {}
    try:
        # Use first 300 eval sequences as "probe train", next 200 as "probe val"
        # Both use extended_lookback format that the model expects
        X_all_eval = torch.from_numpy(
            np.concatenate([X_val_eval[:300], X_test_eval[:200]], axis=0)
        ).float()
        y_all_eval = np.concatenate([y_val_eval_raw[:300], y_test_eval_raw[:200]], axis=0)

        n_probe_train = min(300, len(X_val_eval))
        n_probe_val = min(200, len(X_test_eval))
        X_probe_train_t = torch.from_numpy(X_val_eval[:n_probe_train]).float()
        X_probe_val_t = torch.from_numpy(X_test_eval[:n_probe_val]).float()
        y_probe_train = y_val_eval_raw[:n_probe_train]
        y_probe_val = y_test_eval_raw[:n_probe_val]

        logger.info(f"Linear probing: train={n_probe_train}, val={n_probe_val}")

        # Fine-tuned reps (current model state)
        ft_reps_train = extract_representations(model, X_probe_train_t, None, device=args.device, max_samples=n_probe_train)
        ft_reps_val = extract_representations(model, X_probe_val_t, None, device=args.device, max_samples=n_probe_val)
        logger.info(f"FT reps: train={ft_reps_train.shape}, val={ft_reps_val.shape}")

        # Pre-trained reps: reload pre-trained weights temporarily
        current_state = {k: v.clone() for k, v in model.state_dict().items()}
        restore_dict = {}
        for name in pretrained_params:
            if name in dict(model.named_parameters()):
                restore_dict[name] = pretrained_params[name]
        if restore_dict:
            model.load_state_dict({**model.state_dict(), **restore_dict}, strict=False)

        pt_reps_train = extract_representations(model, X_probe_train_t, None, device=args.device, max_samples=n_probe_train)
        pt_reps_val = extract_representations(model, X_probe_val_t, None, device=args.device, max_samples=n_probe_val)
        logger.info(f"PT reps: train={pt_reps_train.shape}, val={pt_reps_val.shape}")

        # Restore fine-tuned weights
        model.load_state_dict(current_state)

        if len(ft_reps_train) > 0 and len(pt_reps_train) > 0:
            probe_results['pretrained_r2'] = linear_probe_r2(pt_reps_train, pt_reps_val, y_probe_train, y_probe_val)
            probe_results['finetuned_r2'] = linear_probe_r2(ft_reps_train, ft_reps_val, y_probe_train, y_probe_val)
            probe_results['r2_delta'] = probe_results['finetuned_r2'] - probe_results['pretrained_r2']
            logger.info(f"Linear probe (ridge): pretrained R²={probe_results['pretrained_r2']:.4f}, "
                       f"finetuned R²={probe_results['finetuned_r2']:.4f}, "
                       f"Δ={probe_results['r2_delta']:+.4f}")
            if args.probe_type in ('mlp', 'both', 'all'):
                try:
                    probe_depths = [int(k) for k in str(args.probe_mlp_layers).split(',') if k.strip()]
                except Exception:
                    probe_depths = [1]
                if not probe_depths:
                    probe_depths = [1]
                # First depth becomes the "default" MLP result for backward compat.
                for idx, k_depth in enumerate(probe_depths):
                    pt_r2 = linear_probe_r2(
                        pt_reps_train, pt_reps_val, y_probe_train, y_probe_val,
                        probe_type='mlp', mlp_layers=k_depth)
                    ft_r2 = linear_probe_r2(
                        ft_reps_train, ft_reps_val, y_probe_train, y_probe_val,
                        probe_type='mlp', mlp_layers=k_depth)
                    d_r2 = ft_r2 - pt_r2
                    probe_results[f'pretrained_r2_mlp_k{k_depth}'] = pt_r2
                    probe_results[f'finetuned_r2_mlp_k{k_depth}'] = ft_r2
                    probe_results[f'r2_delta_mlp_k{k_depth}'] = d_r2
                    if idx == 0:
                        probe_results['pretrained_r2_mlp'] = pt_r2
                        probe_results['finetuned_r2_mlp'] = ft_r2
                        probe_results['r2_delta_mlp'] = d_r2
                    logger.info(
                        f"Linear probe (mlp k={k_depth}): pretrained R²={pt_r2:.4f}, "
                        f"finetuned R²={ft_r2:.4f}, Δ={d_r2:+.4f}")
            if args.probe_type in ('linear_forecaster', 'all'):
                # Re-extract sequence-level reps for linear-forecaster probe (reviewer Q2).
                ft_reps_train_seq = extract_representations(
                    model, X_probe_train_t, None, device=args.device,
                    max_samples=n_probe_train, keep_sequence=True)
                ft_reps_val_seq = extract_representations(
                    model, X_probe_val_t, None, device=args.device,
                    max_samples=n_probe_val, keep_sequence=True)
                current_state2 = {k: v.clone() for k, v in model.state_dict().items()}
                restore_dict2 = {name: pretrained_params[name]
                                 for name in pretrained_params
                                 if name in dict(model.named_parameters())}
                if restore_dict2:
                    model.load_state_dict({**model.state_dict(), **restore_dict2}, strict=False)
                pt_reps_train_seq = extract_representations(
                    model, X_probe_train_t, None, device=args.device,
                    max_samples=n_probe_train, keep_sequence=True)
                pt_reps_val_seq = extract_representations(
                    model, X_probe_val_t, None, device=args.device,
                    max_samples=n_probe_val, keep_sequence=True)
                model.load_state_dict(current_state2)
                probe_results['pretrained_r2_lf'] = linear_probe_r2(
                    pt_reps_train_seq, pt_reps_val_seq, y_probe_train, y_probe_val,
                    probe_type='linear_forecaster')
                probe_results['finetuned_r2_lf'] = linear_probe_r2(
                    ft_reps_train_seq, ft_reps_val_seq, y_probe_train, y_probe_val,
                    probe_type='linear_forecaster')
                probe_results['r2_delta_lf'] = (
                    probe_results['finetuned_r2_lf'] - probe_results['pretrained_r2_lf'])
                logger.info(f"Linear probe (linear_forecaster): pretrained R²={probe_results['pretrained_r2_lf']:.4f}, "
                           f"finetuned R²={probe_results['finetuned_r2_lf']:.4f}, "
                           f"Δ={probe_results['r2_delta_lf']:+.4f}")
    except Exception as e:
        logger.warning(f"Linear probing failed: {e}")
        import traceback
        traceback.print_exc()

    results = {
        'condition': args.condition,
        'condition_name': condition_names[args.condition],
        'model_size': args.model_size,
        'horizon': horizon,
        'seed': args.seed,
        'epochs': args.epochs,
        'max_train_samples': args.max_train_samples,
        'l2sp_weight': args.l2sp_weight if use_l2sp else 0.0,
        'ewc_lambda': args.ewc_lambda if use_ewc else 0.0,
        'lora_rank': args.lora_rank if args.condition == 'E' else 0,
        'zeroshot_mse': zeroshot_metrics['mse'],
        'zeroshot_mae': zeroshot_metrics['mae'],
        'final_val_mse': history['val_mse'][-1],
        'final_val_mae': history['val_mae'][-1],
        'test_mse': test_metrics['mse'],
        'test_mae': test_metrics['mae'],
        'final_cka': history['cka'][-1],
        'final_weight_drift': history['weight_drift'][-1],
        'forgetting_pct': forgetting,
        'early_stopping': early_stopping_info,
        'linear_probe': probe_results,
        'history': history,
    }

    output_path = results_dir / f'condition_{args.condition}_h{horizon}_s{args.seed}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: Condition {args.condition}, horizon={horizon}")
    logger.info(f"{'='*60}")
    logger.info(f"Zero-shot MSE:  {zeroshot_metrics['mse']:.6f}")
    logger.info(f"Final val MSE:  {history['val_mse'][-1]:.6f}")
    logger.info(f"Test MSE:       {test_metrics['mse']:.6f}")
    logger.info(f"Forgetting:     {forgetting:+.1f}%")
    logger.info(f"Final CKA:      {history['cka'][-1]:.3f}")
    logger.info(f"Weight drift:   {history['weight_drift'][-1]:.2f}")
    if args.early_stopping:
        logger.info(f"[ES] Best epoch: {best_epoch}  (final-epoch forg.={final_epoch_forgetting:+.1f}%)")
    if probe_results:
        logger.info(f"Probe R² (PT):  {probe_results.get('pretrained_r2', 'N/A')}")
        logger.info(f"Probe R² (FT):  {probe_results.get('finetuned_r2', 'N/A')}")
    logger.info(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
