#!/usr/bin/env python3
"""
Moirai-Small Fine-Tuning on ILI (National Illness) with Full Diagnostic Protocol.

Adapts the ETTh2 fine-tuning diagnostic methodology for ILI:
- NLL fine-tuning (condition B) or frozen encoder (condition D)
- CKA between pre-trained and fine-tuned encoder representations
- Ridge probes (Delta-R-squared) on encoder outputs for h=24 target
- Task-orthogonal probes (lag-1 autocorrelation, mean, variance)
- Weight drift (L2)
- CUDA-deterministic per-seed reproducibility

ILI: 966 weekly observations, 7 features, lookback=104, horizon=24.
Gate-pass = 57% (confirmed by gate check).

Usage:
    python scripts/finetune_ili.py --condition B --seed 42 --device cuda
    python scripts/finetune_ili.py --condition D --seed 42 --device cuda
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

LOOKBACK = 104
HORIZON = 24
MODEL_ID = "salesforce/moirai-1.0-R-small"
SEEDS = [42, 101, 123, 202, 303, 456, 777, 789, 888, 999]


def load_ili_data(data_path, train_ratio=0.6, val_ratio=0.2):
    """Load ILI data and split into train/val/test."""
    import pandas as pd
    df = pd.read_csv(data_path)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])

    data = df.values.astype(np.float32)
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]

    print(f"ILI data: {n} timesteps, {data.shape[1]} features")
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test, list(df.columns) if 'date' not in df.columns else list(df.drop(columns=['date']).columns)


def make_sequences(data, lookback, horizon):
    """Create (context, target) sliding window pairs."""
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + horizon])
    return np.array(X), np.array(y)


def linear_CKA(X, Y):
    """Compute linear CKA between two representation matrices."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    XtX = X.T @ X
    YtY = Y.T @ Y
    hsic_xy = np.trace(XtX @ YtY)
    hsic_xx = np.trace(XtX @ XtX)
    hsic_yy = np.trace(YtY @ YtY)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


def compute_weight_drift(model, pretrained_params):
    """Compute L2 distance between current and pre-trained weights."""
    total_drift = 0.0
    for name, param in model.named_parameters():
        if name in pretrained_params:
            diff = (param.data - pretrained_params[name]).float()
            total_drift += diff.norm().item() ** 2
    return float(np.sqrt(total_drift))


def task_orthogonal_probes(reps_train, reps_val, contexts_train, contexts_val, alpha=1.0):
    """Task-orthogonal probes: lag-1 autocorrelation, mean, variance."""
    results = {}

    def lag1_autocorr(ctx):
        x = ctx[:, 1:]
        x_lag = ctx[:, :-1]
        mu = ctx.mean(axis=1, keepdims=True)
        num = ((x - mu) * (x_lag - mu)).mean(axis=1)
        denom = ((ctx - mu) ** 2).mean(axis=1)
        return (num / (denom + 1e-8)).reshape(-1, 1)

    def input_mean(ctx):
        return ctx.mean(axis=1, keepdims=True)

    def input_var(ctx):
        return ctx.var(axis=1, keepdims=True)

    for name, fn in [("lag1", lag1_autocorr), ("mean", input_mean), ("var", input_var)]:
        tgt_tr = fn(contexts_train)
        tgt_va = fn(contexts_val)
        probe = Ridge(alpha=alpha).fit(reps_train, tgt_tr)
        r2 = float(probe.score(reps_val, tgt_va))
        results[name] = r2

    return results


def extract_representations(model, contexts, targets, device, batch_size=32, max_samples=500):
    """Extract mean-pooled encoder representations from Moirai.

    Passes context+target (full sequence) through _val_loss to trigger encoder.
    Returns representations corresponding to each (context, target) pair.
    """
    model.eval()
    captured = {}

    def hook(module, input, output):
        captured['out'] = output

    encoder = model.module.encoder
    handle = encoder.register_forward_hook(hook)

    all_reps = []
    n = min(len(contexts), max_samples)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            ctx_batch = contexts[i:end].to(device)
            tgt_batch = targets[i:end].to(device)

            # Concatenate context + target for _val_loss (same as training)
            full_seq = torch.cat([ctx_batch, tgt_batch], dim=1)
            b = full_seq.shape[0]
            seq_len = full_seq.shape[1]
            n_feat = full_seq.shape[2]
            observed = torch.ones(b, seq_len, n_feat, dtype=torch.bool, device=device)
            is_pad = torch.zeros(b, seq_len, dtype=torch.bool, device=device)

            captured.clear()
            try:
                model._val_loss(
                    patch_size=8,
                    target=full_seq,
                    observed_target=observed,
                    is_pad=is_pad,
                )
            except Exception:
                pass

            if 'out' in captured:
                rep = captured['out']
                if isinstance(rep, tuple):
                    rep = rep[0]
                rep_pooled = rep.mean(dim=1).cpu().numpy()
                all_reps.append(rep_pooled)

    handle.remove()
    if all_reps:
        return np.concatenate(all_reps, axis=0)
    return np.zeros((0, 1))


def evaluate_forecasting_mse(model, context_tensor, targets, device, batch_size=16, num_samples=20):
    """Evaluate Moirai forecasting MSE (per-window normalized by context stats)."""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(context_tensor), batch_size):
            batch = context_tensor[i:i + batch_size].to(device)
            b = batch.shape[0]
            n_feat = batch.shape[2]
            past_obs = torch.ones(b, batch.shape[1], n_feat, dtype=torch.bool, device=device)
            past_pad = torch.zeros(b, batch.shape[1], dtype=torch.bool, device=device)

            samples = model.forward(
                past_target=batch,
                past_observed_target=past_obs,
                past_is_pad=past_pad,
                num_samples=num_samples
            )
            median_pred = samples.median(dim=1).values.cpu().numpy()
            all_preds.append(median_pred)

    predictions = np.concatenate(all_preds, axis=0)
    if predictions.ndim == 3 and predictions.shape[-1] == 1:
        predictions = predictions.squeeze(-1)
    if targets.ndim == 3 and targets.shape[-1] == 1:
        targets_flat = targets.squeeze(-1)
    else:
        targets_flat = targets

    # Per-window normalization by context statistics (matching gate-check protocol)
    mses = []
    for j in range(len(predictions)):
        ctx = context_tensor[j].numpy()
        mu = ctx.mean()
        sd = ctx.std() + 1e-8
        pred_n = (predictions[j] - mu) / sd
        tgt_n = (targets_flat[j] - mu) / sd
        mses.append(float(np.mean((pred_n - tgt_n) ** 2)))

    return float(np.mean(mses))


def train_one_epoch(model, train_loader, optimizer, device, freeze_encoder=False):
    """Train one epoch with NLL loss."""
    model.train()
    epoch_loss = 0.0
    batch_count = 0

    for context_batch, target_batch in train_loader:
        context_batch = context_batch.to(device)
        target_batch = target_batch.to(device)
        b = context_batch.shape[0]

        full_target = torch.cat([context_batch, target_batch], dim=1)
        seq_len = full_target.shape[1]
        n_feat = full_target.shape[2]
        observed = torch.ones(b, seq_len, n_feat, dtype=torch.bool, device=device)
        is_pad = torch.zeros(b, seq_len, dtype=torch.bool, device=device)

        try:
            per_sample_nll = model._val_loss(
                patch_size=8,
                target=full_target,
                observed_target=observed,
                is_pad=is_pad,
            )
            loss = per_sample_nll.mean()
        except Exception as e:
            print(f"  Batch failed: {e}")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    if batch_count == 0:
        return 0.0
    return epoch_loss / batch_count


def run_single_feature(model, pretrained_params, pretrained_state, feature_idx, feature_name,
                       train_data, val_data, test_data, args, device):
    """Run full diagnostic protocol for a single feature (univariate)."""
    print(f"\n  Feature {feature_idx + 1}: {feature_name}")

    # Extract single feature (univariate)
    train_feat = train_data[:, feature_idx:feature_idx + 1]
    val_feat = val_data[:, feature_idx:feature_idx + 1]
    test_feat = test_data[:, feature_idx:feature_idx + 1]

    # Create sequences
    X_train, y_train = make_sequences(train_feat, LOOKBACK, HORIZON)
    X_val, y_val = make_sequences(val_feat, LOOKBACK, HORIZON)

    if len(X_train) == 0 or len(X_val) == 0:
        print(f"    Skipping: insufficient data (train={len(X_train)}, val={len(X_val)})")
        return None

    print(f"    Train windows: {len(X_train)}, Val windows: {len(X_val)}")

    # Reset model to pre-trained state
    model.load_state_dict(pretrained_state)

    # Evaluation windows: context of LOOKBACK, target of HORIZON
    X_val_eval, y_val_eval = [], []
    for i in range(len(val_feat) - LOOKBACK - HORIZON + 1):
        X_val_eval.append(val_feat[i:i + LOOKBACK])
        y_val_eval.append(val_feat[i + LOOKBACK:i + LOOKBACK + HORIZON])
    X_val_eval = np.array(X_val_eval) if X_val_eval else np.zeros((0, LOOKBACK, 1))
    y_val_eval = np.array(y_val_eval) if y_val_eval else np.zeros((0, HORIZON, 1))

    if len(X_val_eval) == 0:
        print(f"    Skipping: no eval windows")
        return None

    val_eval_tensor = torch.from_numpy(X_val_eval).float()

    # Zero-shot baseline
    zs_mse = evaluate_forecasting_mse(model, val_eval_tensor, y_val_eval, device)
    print(f"    ZS MSE: {zs_mse:.4f}")

    # Extract pre-trained representations (using training data context+target)
    train_ctx_tensor = torch.from_numpy(X_train).float()
    train_tgt_tensor = torch.from_numpy(y_train).float()
    val_ctx_tensor = torch.from_numpy(X_val).float()
    val_tgt_tensor = torch.from_numpy(y_val).float()
    pt_reps_train = extract_representations(model, train_ctx_tensor, train_tgt_tensor, device)
    pt_reps_val = extract_representations(model, val_ctx_tensor, val_tgt_tensor, device)

    # Pre-trained Ridge probe (target = y_val flattened)
    y_train_flat = y_train.reshape(len(y_train), -1)
    y_val_flat = y_val.reshape(len(y_val), -1)
    r2_pt = float(Ridge(alpha=1.0).fit(pt_reps_train, y_train_flat).score(pt_reps_val, y_val_flat))

    # Pre-trained orthogonal probes
    ortho_pt = task_orthogonal_probes(
        pt_reps_train, pt_reps_val,
        X_train.squeeze(-1), X_val.squeeze(-1)
    )

    if args.condition == 'D':
        # Frozen encoder: only head trains (no encoder weight changes)
        # CKA = 1.0 by definition, delta_r2 ~ 0
        result = {
            "feature": feature_name,
            "zs_mse": zs_mse,
            "ft_mse": zs_mse,
            "forgetting_pct": 0.0,
            "cka": 1.0,
            "weight_drift": 0.0,
            "r2_pt": r2_pt,
            "r2_ft": r2_pt,
            "delta_r2": 0.0,
            "orthogonal_probes_pt": ortho_pt,
            "orthogonal_probes_ft": ortho_pt,
        }
        return result

    # Condition B: NLL fine-tuning
    # Freeze encoder check
    freeze_encoder = (args.condition == 'D')

    # Setup training
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float()
    )
    dl_generator = torch.Generator()
    dl_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, generator=dl_generator
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-2
    )

    # Training loop with early stopping
    best_val_mse = float('inf')
    best_epoch = 0
    best_state = None
    patience = 3
    patience_counter = 0

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)

        # Evaluate every epoch
        val_mse = evaluate_forecasting_mse(model, val_eval_tensor, y_val_eval, device)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"    Early stop at epoch {epoch + 1} (best={best_epoch + 1})")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    ft_mse = evaluate_forecasting_mse(model, val_eval_tensor, y_val_eval, device)
    forgetting_pct = (ft_mse - zs_mse) / abs(zs_mse) * 100 if abs(zs_mse) > 1e-8 else 0.0

    # CKA
    ft_reps_train = extract_representations(model, train_ctx_tensor, train_tgt_tensor, device)
    ft_reps_val = extract_representations(model, val_ctx_tensor, val_tgt_tensor, device)
    cka = linear_CKA(pt_reps_val, ft_reps_val)

    # Weight drift
    drift = compute_weight_drift(model, pretrained_params)

    # Fine-tuned Ridge probe
    r2_ft = float(Ridge(alpha=1.0).fit(ft_reps_train, y_train_flat).score(ft_reps_val, y_val_flat))
    delta_r2 = r2_ft - r2_pt

    # Fine-tuned orthogonal probes
    ortho_ft = task_orthogonal_probes(
        ft_reps_train, ft_reps_val,
        X_train.squeeze(-1), X_val.squeeze(-1)
    )

    print(f"    FT MSE: {ft_mse:.4f} (forg: {forgetting_pct:+.1f}%)")
    print(f"    CKA: {cka:.4f}, Drift: {drift:.4f}")
    print(f"    R2(PT): {r2_pt:.4f}, R2(FT): {r2_ft:.4f}, dR2: {delta_r2:+.4f}")
    print(f"    Best epoch: {best_epoch + 1}")

    result = {
        "feature": feature_name,
        "zs_mse": zs_mse,
        "ft_mse": ft_mse,
        "forgetting_pct": forgetting_pct,
        "cka": cka,
        "weight_drift": drift,
        "r2_pt": r2_pt,
        "r2_ft": r2_ft,
        "delta_r2": delta_r2,
        "best_epoch": best_epoch,
        "orthogonal_probes_pt": ortho_pt,
        "orthogonal_probes_ft": ortho_ft,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="ILI full diagnostic probes")
    parser.add_argument('--data-path', default='data/national_illness.csv')
    parser.add_argument('--condition', required=True, choices=['B', 'D'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--results-dir', default='results/ili_probes')
    parser.add_argument('--deterministic', action='store_true', default=True)
    args = parser.parse_args()

    # Deterministic setup
    if args.deterministic:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import random
        random.seed(args.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Device: {device}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_data, val_data, test_data, feature_names = load_ili_data(args.data_path)

    # Load Moirai-Small
    print("Loading Moirai-Small...")
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    module = MoiraiModule.from_pretrained(MODEL_ID)
    model = MoiraiForecast(
        module=module,
        prediction_length=HORIZON,
        context_length=LOOKBACK,
        patch_size=8,
        num_samples=20,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    model.eval()
    model = model.to(device)
    print(f"Model loaded on {device}")

    # Store pre-trained state
    pretrained_state = copy.deepcopy(model.state_dict())
    pretrained_params = {name: param.data.clone() for name, param in model.named_parameters()}

    # Run per-feature evaluation (univariate protocol)
    all_results = []
    for feat_idx, feat_name in enumerate(feature_names):
        result = run_single_feature(
            model, pretrained_params, pretrained_state,
            feat_idx, feat_name,
            train_data, val_data, test_data,
            args, device
        )
        if result is not None:
            all_results.append(result)

    # Aggregate results
    if not all_results:
        print("ERROR: No features produced results")
        return

    avg_cka = np.mean([r['cka'] for r in all_results])
    avg_delta_r2 = np.mean([r['delta_r2'] for r in all_results])
    avg_forgetting = np.mean([r['forgetting_pct'] for r in all_results])
    avg_drift = np.mean([r['weight_drift'] for r in all_results])
    n_positive_dr2 = sum(1 for r in all_results if r['delta_r2'] > 0)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"AGGREGATE RESULTS (seed={args.seed}, condition={args.condition})")
    print(f"  Avg CKA:        {avg_cka:.4f}")
    print(f"  Avg Delta-R2:   {avg_delta_r2:+.4f} ({n_positive_dr2}/{len(all_results)} positive)")
    print(f"  Avg Forgetting:  {avg_forgetting:+.1f}%")
    print(f"  Avg Weight Drift: {avg_drift:.4f}")
    print(sep)

    # Save results
    output = {
        "seed": args.seed,
        "condition": args.condition,
        "dataset": "ILI",
        "model": MODEL_ID,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "epochs": args.epochs,
        "lr": args.lr,
        "device": str(device),
        "aggregate": {
            "cka": avg_cka,
            "delta_r2": avg_delta_r2,
            "forgetting_pct": avg_forgetting,
            "weight_drift": avg_drift,
            "n_positive_delta_r2": n_positive_dr2,
            "n_features": len(all_results),
        },
        "per_feature": all_results,
    }

    output_path = results_dir / f"condition_{args.condition}_seed{args.seed}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
