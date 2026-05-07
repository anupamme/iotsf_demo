#!/usr/bin/env python3
"""
Zero-Shot Moirai Gate Check on ILI (National Illness) Dataset

Evaluates Moirai-Small zero-shot vs Linear baseline on ILI forecasting
to determine gate-pass percentage for the pre-registered prediction.

ILI: 7 features, ~966 weekly observations, horizon=24 (standard).
Gate-pass = 1 - (ZS_MSE / Linear_MSE) * 100%

Usage:
    python scripts/ili_gate_check.py --data-path data/national_illness.csv
"""

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path

def load_ili_data(data_path, lookback=96, horizon=24, test_ratio=0.2):
    """Load ILI data and create test windows."""
    import pandas as pd
    df = pd.read_csv(data_path)
    # Drop date column if present
    if 'date' in df.columns:
        df = df.drop(columns=['date'])

    data = df.values.astype(np.float32)
    n_total = len(data)
    n_test = int(n_total * test_ratio)

    # Use last test_ratio as test set
    test_start = n_total - n_test

    # Create sliding windows from test portion
    contexts = []
    targets = []
    for i in range(test_start - lookback, n_total - lookback - horizon + 1):
        if i < 0:
            continue
        ctx = data[i:i+lookback]  # (lookback, 7)
        tgt = data[i+lookback:i+lookback+horizon]  # (horizon, 7)
        contexts.append(ctx)
        targets.append(tgt)

    print(f"ILI data: {n_total} timesteps, {data.shape[1]} features")
    print(f"Test windows: {len(contexts)} (lookback={lookback}, horizon={horizon})")
    return contexts, targets, data


def linear_baseline_mse(contexts, targets):
    """Simple per-feature linear regression baseline."""
    from sklearn.linear_model import Ridge

    n_windows = len(contexts)
    lookback = contexts[0].shape[0]
    horizon = targets[0].shape[0]
    n_features = contexts[0].shape[1]

    # Stack
    X = np.array(contexts).reshape(n_windows, -1)  # (N, lookback*features)
    Y = np.array(targets).reshape(n_windows, -1)   # (N, horizon*features)

    # Train on first 80% of windows, test on last 20%
    n_train = int(n_windows * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    # Normalize per-window
    mu = X_train.mean(axis=1, keepdims=True)
    sd = X_train.std(axis=1, keepdims=True) + 1e-8
    X_train_n = (X_train - mu) / sd

    mu_test = X_test.mean(axis=1, keepdims=True)
    sd_test = X_test.std(axis=1, keepdims=True) + 1e-8
    X_test_n = (X_test - mu_test) / sd_test

    # Normalize targets by context stats
    Y_train_flat = Y_train.reshape(n_train, horizon, n_features)
    ctx_mu = np.array(contexts[:n_train]).mean(axis=1, keepdims=True)  # (n_train, 1, features)
    ctx_sd = np.array(contexts[:n_train]).std(axis=1, keepdims=True) + 1e-8
    Y_train_n = ((Y_train_flat - ctx_mu) / ctx_sd).reshape(n_train, -1)

    Y_test_flat = Y_test.reshape(len(X_test), horizon, n_features)
    ctx_mu_t = np.array(contexts[n_train:]).mean(axis=1, keepdims=True)
    ctx_sd_t = np.array(contexts[n_train:]).std(axis=1, keepdims=True) + 1e-8
    Y_test_n = ((Y_test_flat - ctx_mu_t) / ctx_sd_t).reshape(len(X_test), -1)

    # Fit ridge
    reg = Ridge(alpha=1.0)
    reg.fit(X_train_n, Y_train_n)
    Y_pred = reg.predict(X_test_n)

    mse = float(np.mean((Y_pred - Y_test_n) ** 2))
    print(f"Linear baseline MSE (normalized): {mse:.4f}")
    return mse


def moirai_zs_mse(contexts, targets, model_id="salesforce/moirai-1.0-R-small"):
    """Moirai zero-shot forecasting MSE."""
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    horizon = targets[0].shape[0]
    n_features = contexts[0].shape[1]
    n_windows = len(contexts)

    # Use last 20% as test (same split as linear)
    n_train = int(n_windows * 0.8)
    test_contexts = contexts[n_train:]
    test_targets = targets[n_train:]

    # Load model
    print(f"Loading Moirai from {model_id}...")
    module = MoiraiModule.from_pretrained(model_id)

    # Moirai expects univariate or treats each feature separately
    # We evaluate per-feature and average (standard protocol)
    all_mses = []

    for feat_idx in range(n_features):
        feat_contexts = [ctx[:, feat_idx] for ctx in test_contexts]
        feat_targets = [tgt[:, feat_idx] for tgt in test_targets]

        # Create forecast pipeline
        model = MoiraiForecast(
            module=module,
            prediction_length=horizon,
            context_length=contexts[0].shape[0],
            patch_size="auto",
            num_samples=20,
        )

        # Predict in batches
        from einops import rearrange
        import torch

        batch_size = 32
        preds = []

        for i in range(0, len(feat_contexts), batch_size):
            batch_ctx = feat_contexts[i:i+batch_size]
            # Stack into tensor
            ctx_tensor = torch.tensor(np.array(batch_ctx), dtype=torch.float32).unsqueeze(-1)

            # Use the model's predict method
            forecast = model(
                past_target=ctx_tensor,
                past_observed_target=torch.ones_like(ctx_tensor[..., 0:1]),
            )
            # forecast samples shape: (batch, n_samples, horizon)
            median_pred = forecast.quantile(0.5).detach().cpu().numpy()  # (batch, horizon)
            preds.append(median_pred)

        preds = np.concatenate(preds, axis=0)  # (n_test, horizon)

        # Compute normalized MSE per window
        for j in range(len(feat_contexts)):
            ctx = feat_contexts[j]
            mu = ctx.mean()
            sd = ctx.std() + 1e-8
            pred_n = (preds[j] - mu) / sd
            tgt_n = (feat_targets[j] - mu) / sd
            all_mses.append(float(np.mean((pred_n - tgt_n) ** 2)))

    mse = float(np.mean(all_mses))
    print(f"Moirai ZS MSE (normalized, {n_features} features): {mse:.4f}")
    return mse


def moirai_zs_mse_simple(contexts, targets, model_id="salesforce/moirai-1.0-R-small"):
    """Simplified Moirai ZS eval using uni2ts DataLoader approach."""
    try:
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    except ImportError:
        print("ERROR: uni2ts not installed. Install with: pip install 'uni2ts[all]'")
        sys.exit(1)

    horizon = targets[0].shape[0]
    n_features = contexts[0].shape[1]
    n_windows = len(contexts)

    # Use last 20% as test
    n_train = int(n_windows * 0.8)
    test_contexts = contexts[n_train:]
    test_targets = targets[n_train:]
    n_test = len(test_contexts)

    print(f"Loading Moirai from {model_id}...")
    module = MoiraiModule.from_pretrained(model_id)

    lookback = contexts[0].shape[0]
    model = MoiraiForecast(
        module=module,
        prediction_length=horizon,
        context_length=lookback,
        patch_size="auto",
        num_samples=20,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Evaluate per-feature (univariate protocol, standard for Moirai benchmarks)
    all_mses = []
    batch_size = 16

    for feat_idx in range(n_features):
        print(f"  Feature {feat_idx+1}/{n_features}...")
        feat_preds = []

        for i in range(0, n_test, batch_size):
            batch_end = min(i + batch_size, n_test)
            batch_ctx = np.array([test_contexts[j][:, feat_idx] for j in range(i, batch_end)])

            ctx_tensor = torch.tensor(batch_ctx, dtype=torch.float32).unsqueeze(-1).to(device)
            past_observed = torch.ones(ctx_tensor.shape[0], lookback, 1, device=device)

            with torch.no_grad():
                forecast = model(
                    past_target=ctx_tensor,
                    past_observed_target=past_observed,
                )
            # Get median
            median = forecast.quantile(0.5).cpu().numpy()  # (batch, horizon)
            feat_preds.append(median)

        feat_preds = np.concatenate(feat_preds, axis=0)  # (n_test, horizon)

        # Normalized MSE per window
        for j in range(n_test):
            ctx_vals = test_contexts[j][:, feat_idx]
            mu = ctx_vals.mean()
            sd = ctx_vals.std() + 1e-8
            pred_n = (feat_preds[j] - mu) / sd
            tgt_n = (test_targets[j][:, feat_idx] - mu) / sd
            all_mses.append(float(np.mean((pred_n - tgt_n) ** 2)))

    mse = float(np.mean(all_mses))
    print(f"Moirai ZS MSE (normalized, avg over {n_features} features): {mse:.4f}")
    return mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="salesforce/moirai-1.0-R-small")
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--output", type=str, default="results/ili_gate_check.json")
    args = parser.parse_args()

    print(f"=== ILI Gate Check: Moirai-Small ZS vs Linear ===")
    print(f"Data: {args.data_path}")
    print(f"Lookback: {args.lookback}, Horizon: {args.horizon}")
    print()

    # Load data
    contexts, targets, raw_data = load_ili_data(
        args.data_path, lookback=args.lookback, horizon=args.horizon
    )

    # Linear baseline
    print("\n--- Linear Baseline ---")
    linear_mse = linear_baseline_mse(contexts, targets)

    # Moirai zero-shot
    print("\n--- Moirai Zero-Shot ---")
    zs_mse = moirai_zs_mse_simple(contexts, targets, model_id=args.model_id)

    # Gate-pass calculation
    gate_pass_pct = (1 - zs_mse / linear_mse) * 100

    print(f"\n{'='*50}")
    print(f"RESULTS:")
    print(f"  Linear MSE: {linear_mse:.4f}")
    print(f"  Moirai ZS MSE: {zs_mse:.4f}")
    print(f"  Gate-pass: {gate_pass_pct:.1f}%")
    print(f"  Threshold (20%): {'PASS' if gate_pass_pct >= 20 else 'FAIL'}")
    print(f"{'='*50}")

    # Save results
    results = {
        "dataset": "ILI (national_illness)",
        "model": args.model_id,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "linear_mse": linear_mse,
        "moirai_zs_mse": zs_mse,
        "gate_pass_pct": gate_pass_pct,
        "gate_pass": gate_pass_pct >= 20,
        "n_features": raw_data.shape[1],
        "n_timesteps": len(raw_data),
        "n_test_windows": len(contexts) - int(len(contexts) * 0.8),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
