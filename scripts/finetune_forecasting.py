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

from src.data.forecasting_loader import ETTh1Loader
from src.models.losses import SupervisedContrastiveLoss


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

    # Normalize both using training statistics
    pred_norm = (predictions - train_mean) / train_std
    tgt_norm = (target - train_mean) / train_std

    mse = float(np.mean((pred_norm - tgt_norm) ** 2))
    mae = float(np.mean(np.abs(pred_norm - tgt_norm)))
    return {'mse': mse, 'mae': mae}


def extract_representations(
    model,
    data: torch.Tensor,
    encoder_hook_fn,
    batch_size: int = 32,
    device: str = 'cpu',
    max_samples: int = 500
) -> np.ndarray:
    """Extract encoder representations for CKA computation."""
    model.eval()
    captured = {}

    def hook(module, input, output):
        captured['out'] = output

    # Find encoder
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

            # Run forward to trigger hook
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
                # Pool over sequence dimension
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
) -> dict:
    """Train one epoch of NLL (+ optional contrastive) fine-tuning."""
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
    parser.add_argument('--condition', required=True, choices=['A', 'B', 'C', 'D'],
                        help="A=zero-shot, B=NLL-only, C=NLL+SupCon, D=frozen encoder")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--contrastive-weight', type=float, default=0.5)
    parser.add_argument('--n-temporal-clusters', type=int, default=8)
    parser.add_argument('--max-train-samples', type=int, default=1000)
    parser.add_argument('--results-dir', default='results/forecasting_finetune')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-every', type=int, default=1,
                        help="Evaluate forgetting every N epochs")
    parser.add_argument('--max-eval-sequences', type=int, default=300,
                        help="Max sequences for periodic evaluation")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    lookback = 96
    horizon = args.horizon
    n_features = 7

    logger.info(f"Condition {args.condition}: "
               f"{'Zero-shot' if args.condition == 'A' else 'NLL-only' if args.condition == 'B' else 'NLL+SupCon' if args.condition == 'C' else 'Frozen encoder'}")

    # Load data
    loader = ETTh1Loader(args.data_path, lookback_window=lookback, forecast_horizon=horizon, features='M')
    train_df, val_df, test_df = loader.get_splits()

    train_vals = train_df[loader.FEATURE_COLUMNS].values
    val_vals = val_df[loader.FEATURE_COLUMNS].values
    test_vals = test_df[loader.FEATURE_COLUMNS].values

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
        model_size='small',
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Setup optimizer
    freeze_encoder = (args.condition == 'D')
    use_contrastive = (args.condition == 'C')

    if freeze_encoder:
        # Freeze encoder weights
        encoder = model.module.encoder if not hasattr(model.module, 'base_model') else model.module.base_model.model.encoder
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
        else:
            history['val_mse'].append(None)
            history['val_mae'].append(None)
            history['cka'].append(None)
            history['weight_drift'].append(None)

    # Final test evaluation
    X_test_eval_t = torch.from_numpy(X_test_eval[:eval_limit]).float()
    test_metrics = evaluate_forecasting(
        model, X_test_eval_t, y_test_eval_raw[:eval_limit], train_mean, train_std,
        horizon, device=args.device
    )

    # Compute forgetting metric
    forgetting = (history['val_mse'][-1] - zeroshot_metrics['mse']) / zeroshot_metrics['mse'] * 100

    results = {
        'condition': args.condition,
        'horizon': horizon,
        'seed': args.seed,
        'epochs': args.epochs,
        'max_train_samples': args.max_train_samples,
        'zeroshot_mse': zeroshot_metrics['mse'],
        'zeroshot_mae': zeroshot_metrics['mae'],
        'final_val_mse': history['val_mse'][-1],
        'final_val_mae': history['val_mae'][-1],
        'test_mse': test_metrics['mse'],
        'test_mae': test_metrics['mae'],
        'final_cka': history['cka'][-1],
        'final_weight_drift': history['weight_drift'][-1],
        'forgetting_pct': forgetting,
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
    logger.info(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
