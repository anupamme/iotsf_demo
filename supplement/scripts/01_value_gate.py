#!/usr/bin/env python3
"""Compute value-gate metric: R^2_task = 1 - MSE_ZS / MSE_Linear.

Determines whether pre-trained model provides value over linear baseline.
Gate-pass (R^2_task > 0) means preservation-focused analysis is warranted.

Usage:
    python scripts/01_value_gate.py --config configs/etth2_small_n500.yaml
    python scripts/01_value_gate.py --config configs/ili_small.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_ett_data, load_ili_data, make_eval_sequences, linear_baseline_mse
from src.metrics import compute_r2_task
from src.models import load_moirai
from src.train import evaluate_forecasting
from src.utils import load_config, set_seed, get_device


def moirai_zeroshot_mse(model, contexts, targets, train_mean, train_std, horizon, device):
    """Compute zero-shot MSE for Moirai on given windows."""
    context_t = torch.from_numpy(np.array(contexts)).float()
    target_arr = np.array(targets)
    metrics = evaluate_forecasting(
        model, context_t, target_arr, train_mean, train_std,
        horizon, device=str(device),
    )
    return metrics["mse"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device(args.device)

    dataset = getattr(cfg, "dataset", "ETTh2")
    lookback = cfg.lookback
    horizon = cfg.horizon

    if dataset == "ILI":
        train, val, test, feat_cols = load_ili_data(cfg.data_path)
    else:
        train, val, test, feat_cols = load_ett_data(
            cfg.data_path, features=cfg.features
        )

    train_mean = train.mean(axis=0)
    train_std = train.std(axis=0) + 1e-8

    extended_lookback = lookback + horizon
    X_val, y_val = make_eval_sequences(val, extended_lookback, horizon)

    contexts = [X_val[i] for i in range(len(X_val))]
    targets = [y_val[i] for i in range(len(y_val))]

    print(f"Dataset: {dataset}, features={len(feat_cols)}, horizon={horizon}")
    print(f"Eval windows: {len(contexts)}")

    # Linear baseline
    lin_mse = linear_baseline_mse(contexts, targets)
    print(f"Linear baseline MSE: {lin_mse:.6f}")

    # Moirai zero-shot
    model = load_moirai(
        model_size=cfg.model_size,
        context_length=lookback,
        prediction_length=horizon,
        target_dim=len(feat_cols),
        device=str(device),
    )
    zs_mse = moirai_zeroshot_mse(
        model, contexts, targets, train_mean, train_std, horizon, device
    )
    print(f"Moirai zero-shot MSE: {zs_mse:.6f}")

    # Value gate
    r2_task = compute_r2_task(zs_mse, lin_mse)
    gate_pass = r2_task > 0

    print(f"\nR^2_task = 1 - {zs_mse:.6f}/{lin_mse:.6f} = {r2_task:.4f}")
    print(f"Gate pass: {'YES' if gate_pass else 'NO'}")
    print(f"Improvement over linear: {r2_task * 100:.1f}%")


if __name__ == "__main__":
    main()
