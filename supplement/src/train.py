"""Training loop, evaluation, and auxiliary training utilities."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def assign_temporal_labels(n_sequences: int, n_clusters: int = 8) -> np.ndarray:
    """Assign pseudo-labels by temporal position for contrastive learning."""
    labels = np.zeros(n_sequences, dtype=np.int64)
    cluster_size = n_sequences // n_clusters
    for i in range(n_clusters):
        start = i * cluster_size
        end = (i + 1) * cluster_size if i < n_clusters - 1 else n_sequences
        labels[start:end] = i
    return labels


def evaluate_forecasting(
    model,
    context: torch.Tensor,
    target: np.ndarray,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    horizon: int,
    batch_size: int = 8,
    num_samples: int = 20,
    device: str = "cpu",
) -> dict:
    """Evaluate Moirai forecasting (MSE/MAE on normalized scale)."""
    model.eval()
    all_preds = []
    extended_lookback = context.shape[1]

    with torch.no_grad():
        for i in range(0, len(context), batch_size):
            batch_ctx = context[i : i + batch_size].to(device)
            b = batch_ctx.shape[0]
            past_obs = torch.ones_like(batch_ctx, dtype=torch.bool)
            past_pad = torch.zeros(
                b, extended_lookback, dtype=torch.bool, device=device
            )

            samples = model.forward(
                past_target=batch_ctx,
                past_observed_target=past_obs,
                past_is_pad=past_pad,
                num_samples=num_samples,
            )
            median_pred = samples.median(dim=1).values.cpu().numpy()
            all_preds.append(median_pred)

    predictions = np.concatenate(all_preds, axis=0)

    if predictions.ndim == 2 and target.ndim == 3 and target.shape[-1] == 1:
        target = target.squeeze(-1)
    elif predictions.ndim == 3 and predictions.shape[-1] == 1 and target.ndim == 2:
        predictions = predictions.squeeze(-1)

    mean_arr = np.asarray(train_mean).reshape(-1)
    std_arr = np.asarray(train_std).reshape(-1)
    if mean_arr.size == 1:
        mean_arr = mean_arr.item()
        std_arr = std_arr.item()
    pred_norm = (predictions - mean_arr) / std_arr
    tgt_norm = (target - mean_arr) / std_arr

    mse = float(np.mean((pred_norm - tgt_norm) ** 2))
    mae = float(np.mean(np.abs(pred_norm - tgt_norm)))
    return {"mse": mse, "mae": mae}


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

    for batch_idx, (context_batch, target_batch, labels_batch) in enumerate(
        train_loader
    ):
        context_batch = context_batch.to(device)
        target_batch = target_batch.to(device)
        labels_batch = labels_batch.to(device)
        b = context_batch.shape[0]

        full_target = torch.cat([context_batch, target_batch], dim=1)
        seq_len = full_target.shape[1]
        n_feat = full_target.shape[2]
        observed = torch.ones(b, seq_len, n_feat, dtype=torch.bool, device=device)
        is_pad = torch.zeros(b, seq_len, dtype=torch.bool, device=device)

        try:
            per_sample_nll = model._val_loss(
                patch_size=32,
                target=full_target,
                observed_target=observed,
                is_pad=is_pad,
            )
            nll_loss = per_sample_nll.mean()
            total_loss = nll_loss

            cont_loss = torch.tensor(0.0, device=device)
            if (
                contrastive_weight > 0
                and projection_head is not None
                and captured_embeddings
            ):
                if "encoder" in captured_embeddings:
                    enc = captured_embeddings["encoder"]
                    if isinstance(enc, tuple):
                        enc = enc[0]
                    pooled = enc.mean(dim=1)
                    projected = projection_head(pooled)
                    cont_loss = contrastive_loss_fn(projected, labels_batch)
                    total_loss = nll_loss + contrastive_weight * cont_loss

            if l2sp_weight > 0 and pretrained_params is not None:
                l2sp_loss = sum(
                    (p - pretrained_params[name]).pow(2).sum()
                    for name, p in model.named_parameters()
                    if p.requires_grad and name in pretrained_params
                )
                total_loss = total_loss + l2sp_weight * l2sp_loss

            if ewc_lambda > 0 and fisher is not None and pretrained_params is not None:
                ewc_loss = sum(
                    (fisher[name] * (p - pretrained_params[name]).pow(2)).sum()
                    for name, p in model.named_parameters()
                    if p.requires_grad
                    and name in fisher
                    and name in pretrained_params
                )
                total_loss = total_loss + ewc_lambda * ewc_loss

        except Exception:
            continue

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters())
            + (list(projection_head.parameters()) if projection_head else []),
            1.0,
        )
        optimizer.step()

        epoch_nll += nll_loss.item()
        epoch_cont += cont_loss.item()
        epoch_total += total_loss.item()
        batch_count += 1

    if batch_count == 0:
        return {"nll": 0, "contrastive": 0, "total": 0}

    return {
        "nll": epoch_nll / batch_count,
        "contrastive": epoch_cont / batch_count,
        "total": epoch_total / batch_count,
    }


def compute_fisher_diagonal(model, data_loader, horizon, device, n_samples=200):
    """Compute diagonal Fisher Information Matrix for EWC regularization."""
    from .models import apply_uni2ts_gradient_patch

    apply_uni2ts_gradient_patch()

    fisher = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    model.eval()
    count = 0

    for batch_idx, (context_batch, target_batch, labels_batch) in enumerate(
        data_loader
    ):
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
                patch_size=32,
                target=full_target,
                observed_target=observed,
                is_pad=is_pad,
            )
            loss = per_sample_nll.mean()
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) * b
            count += b
        except Exception:
            continue

    if count > 0:
        for name in fisher:
            fisher[name] /= count

    return fisher
