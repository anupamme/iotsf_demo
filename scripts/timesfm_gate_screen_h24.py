#!/usr/bin/env python3
"""
TimesFM-2.5-200M gate screen at h=24 (matching Chronos moderate-gate protocol).

TimesFM failed the gate at h=96 (-13.8% on ETTh2, -13.9% on ETTh1).
This script tests whether it passes at the shorter horizon h=24 where
Chronos-T5-Small shows moderate gate-pass (42-52%).

Usage:
    python scripts/timesfm_gate_screen_h24.py --device cuda
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

DATASETS = {
    "ETTh2": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
        "target_col": "OT",
        "n_train": 8640,
        "n_val": 2880,
    },
    "ETTh1": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "target_col": "OT",
        "n_train": 8640,
        "n_val": 2880,
    },
    "ETTm1": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        "target_col": "OT",
        "n_train": 34560,
        "n_val": 11520,
    },
    "ETTm2": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
        "target_col": "OT",
        "n_train": 34560,
        "n_val": 11520,
    },
}

LOOKBACK = 96
HORIZON = 24


def download_dataset(name, cache_dir="/tmp/gate_screen_data"):
    os.makedirs(cache_dir, exist_ok=True)
    cfg = DATASETS[name]
    local_path = os.path.join(cache_dir, f"{name}.csv")
    if not os.path.exists(local_path):
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(cfg["url"], local_path)
    return local_path


def load_series(name, cache_dir="/tmp/gate_screen_data"):
    import pandas as pd
    cfg = DATASETS[name]
    path = download_dataset(name, cache_dir)
    df = pd.read_csv(path)
    values = df[cfg["target_col"]].values.astype(np.float64)
    n_train = cfg["n_train"]
    n_val = cfg["n_val"]
    train = values[:n_train]
    val = values[n_train:n_train + n_val]
    test = values[n_train + n_val:]
    return train, val, test


def build_windows(series, lookback, horizon, max_windows=None, seed=42):
    n_total = len(series) - lookback - horizon + 1
    if n_total <= 0:
        return np.empty((0, lookback)), np.empty((0, horizon))
    contexts = np.array([series[i:i+lookback] for i in range(n_total)])
    targets = np.array([series[i+lookback:i+lookback+horizon] for i in range(n_total)])
    if max_windows and len(contexts) > max_windows:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(contexts), max_windows, replace=False)
        contexts, targets = contexts[idx], targets[idx]
    return contexts, targets


def linear_baseline_mse(contexts, targets):
    mses = []
    for i in range(len(contexts)):
        ctx = contexts[i]
        tgt = targets[i]
        mu = ctx.mean()
        sd = ctx.std() + 1e-8
        x = np.arange(len(ctx)).reshape(-1, 1)
        reg = LinearRegression().fit(x, ctx)
        x_pred = np.arange(len(ctx), len(ctx) + len(tgt)).reshape(-1, 1)
        pred = reg.predict(x_pred)
        pred_n = (pred - mu) / sd
        tgt_n = (tgt - mu) / sd
        mses.append(float(np.mean((pred_n - tgt_n) ** 2)))
    return float(np.mean(mses))


def timesfm_predict(model, contexts, horizon):
    inputs = [contexts[i].astype(np.float32) for i in range(len(contexts))]
    point, _ = model.forecast(horizon=horizon, inputs=inputs)
    return np.asarray(point)[:, :horizon]


def timesfm_mse(model, contexts, targets, horizon):
    preds = timesfm_predict(model, contexts, horizon)
    mses = []
    for i in range(len(contexts)):
        mu = contexts[i].mean()
        sd = contexts[i].std() + 1e-8
        pred_n = (preds[i] - mu) / sd
        tgt_n = (targets[i] - mu) / sd
        mses.append(float(np.mean((pred_n - tgt_n) ** 2)))
    return float(np.mean(mses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model-id', default='google/timesfm-2.5-200m-pytorch')
    parser.add_argument('--max-windows', type=int, default=200)
    parser.add_argument('--results-dir', default='results/timesfm_gate_h24')
    args = parser.parse_args()

    import timesfm

    print("=" * 60)
    print("TimesFM Gate Screen — h=24, lookback=96")
    print("=" * 60)

    print(f"\nLoading {args.model_id}...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(args.model_id)
    fc = timesfm.ForecastConfig(
        max_context=LOOKBACK,
        max_horizon=HORIZON,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,
        fix_quantile_crossing=True,
    )
    model.compile(fc)
    print("Model loaded and compiled")

    results = {}
    for name in DATASETS:
        print(f"\n--- {name} (h={HORIZON}, lb={LOOKBACK}) ---")
        try:
            train, val, test = load_series(name)
            ctx_val, tgt_val = build_windows(val, LOOKBACK, HORIZON,
                                             max_windows=args.max_windows, seed=42)
            if len(ctx_val) < 10:
                print(f"  Too few windows ({len(ctx_val)}), skipping")
                results[name] = {"status": "skipped"}
                continue

            zs_mse = timesfm_mse(model, ctx_val, tgt_val, HORIZON)
            lin_mse = linear_baseline_mse(ctx_val, tgt_val)
            gate_pct = (lin_mse - zs_mse) / lin_mse * 100 if lin_mse > 1e-10 else 0.0
            regime = "gate-fail" if gate_pct < 20 else ("moderate" if gate_pct < 45 else "strong")

            print(f"  ZS MSE: {zs_mse:.4f}  Linear MSE: {lin_mse:.4f}")
            print(f"  Gate improvement: {gate_pct:.1f}% → {regime.upper()}")

            results[name] = {
                "zs_mse": zs_mse,
                "linear_mse": lin_mse,
                "gate_improvement_pct": gate_pct,
                "regime": regime,
                "n_windows": len(ctx_val),
                "horizon": HORIZON,
                "lookback": LOOKBACK,
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"status": "error", "error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("TIMESFM GATE SCREEN SUMMARY (h=24)")
    print("=" * 60)
    gate_pass = []
    for name, r in results.items():
        if "gate_improvement_pct" in r:
            marker = " <<<" if r["regime"] in ("moderate", "strong") else ""
            print(f"  {name:12s}: {r['gate_improvement_pct']:+6.1f}%  [{r['regime']}]{marker}")
            if r["regime"] in ("moderate", "strong"):
                gate_pass.append(name)
        else:
            print(f"  {name:12s}: {r.get('status', 'unknown')}")

    print(f"\nGate-passing cells (≥20%): {gate_pass if gate_pass else 'NONE'}")
    if not gate_pass:
        print("  TimesFM gate-fails at h=24 too — consistent with h=96 result.")
        print("  This confirms TimesFM (like MOMENT) lacks transferable ZS value on ETT.")

    # Save
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "gate_screen_h24.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_dir / 'gate_screen_h24.json'}")


if __name__ == "__main__":
    main()
