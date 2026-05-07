"""EWC diagonal Fisher diagnostic (reviewer concern 5).

Recomputes the diagonal Fisher Information Matrix used by condition G (EWC)
across multiple seeds, with different seed-dependent 500-sample subsampling,
and reports the across-seed coefficient of variation (std/mean) of per-parameter
Fisher values. If CoV is large, the "noisy Fisher at n=500" claim in §5 is
empirically grounded; if small, the claim should be revised.

Usage:
    python scripts/ewc_fisher_diagnostic.py \\
        --seeds 42 123 456 \\
        --max-train-samples 500 --horizon 96 \\
        --out results/v12_ewc_fisher.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.finetune_forecasting import (
    _patch_packed_scaler_for_mps,
)
from src.data.forecasting_loader import ETTh1Loader
from src.models.moirai_detector import MoiraiAnomalyDetector


def compute_fisher_diagonal_v2(model, data_loader, horizon, device, patch_size=16, n_samples=200):
    """Version that uses patch_size=16 to match Moirai-Small training default."""
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
    n_failed = 0

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
                patch_size=patch_size, target=full_target,
                observed_target=observed, is_pad=is_pad,
            )
            loss = per_sample_nll.mean()
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) * b
            count += b
        except Exception as e:
            n_failed += 1
            if n_failed <= 2:
                print(f"Fisher batch {batch_idx} patch_size={patch_size} failed: {e}")
            continue

    if count > 0:
        for name in fisher:
            fisher[name] /= count
    print(f"  Fisher: {count} samples OK, {n_failed} batches failed")
    return fisher, count


def make_train_sequences(data, ctx_len, hz):
    X, Y = [], []
    total = ctx_len + hz
    for i in range(len(data) - total + 1):
        X.append(data[i : i + ctx_len])
        Y.append(data[i + ctx_len : i + total])
    return np.asarray(X), np.asarray(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    ap.add_argument("--max-train-samples", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--data-path", default="data/forecasting/ETTh2.csv")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model-size", default="small")
    ap.add_argument("--n-fisher-samples", type=int, default=200)
    ap.add_argument("--out", default="results/v12_ewc_fisher.json")
    args = ap.parse_args()

    if args.device == "mps":
        _patch_packed_scaler_for_mps()

    loader = ETTh1Loader(args.data_path, lookback_window=96, forecast_horizon=args.horizon, features="M")
    tr, va, te = loader.get_splits()
    feature_cols = loader.FEATURE_COLUMNS
    train_vals = tr[feature_cols].values
    # Normalise
    mu = train_vals.mean(axis=0); sd = train_vals.std(axis=0) + 1e-8
    Xfull, Yfull = make_train_sequences(train_vals, 96, args.horizon)

    # Load Moirai once, store pretrained params
    detector = MoiraiAnomalyDetector(model_size=args.model_size, context_length=96, num_samples=20, device=args.device)
    detector.initialize()
    model = detector.model
    pretrained_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Fisher per seed
    fishers = {}
    for seed in args.seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        idx = np.random.choice(len(Xfull), args.max_train_samples, replace=False)
        X = Xfull[idx]; Y = Yfull[idx]
        ds = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(Y).float(),
            torch.zeros(len(X), dtype=torch.long),
        )
        dl = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)
        # Reset model to pretrained
        model.load_state_dict(pretrained_state)
        print(f"[seed {seed}] computing Fisher...")
        # Try multiple patch sizes; Moirai-Small default is 32 but its patch-size
        # selection depends on seq_len. Try 32 first, fall back to 16/64/8.
        best_fish = None
        best_count = 0
        for ps in (32, 16, 64, 8, 128):
            fish, cnt = compute_fisher_diagonal_v2(model, dl, args.horizon, args.device, patch_size=ps, n_samples=args.n_fisher_samples)
            if cnt > best_count:
                best_fish = fish
                best_count = cnt
                print(f"  patch_size={ps}: Fisher over {cnt} samples (best so far)")
                if cnt >= args.n_fisher_samples // 2:
                    break
        if best_fish is None:
            print(f"[seed {seed}] ALL patch sizes failed; skipping.")
            continue
        fish = best_fish
        # Flatten per-parameter magnitudes for diagnostic
        flat = torch.cat([v.flatten().float().cpu() for v in fish.values()]).numpy()
        fishers[str(seed)] = flat
        print(f"[seed {seed}] Fisher: n_params={len(flat)}, mean={flat.mean():.2e}, median={np.median(flat):.2e}, "
              f"max={flat.max():.2e}, frac>0={float(np.mean(flat > 0)):.3f}")

    # Across-seed CoV
    keys = list(fishers.keys())
    stacked = np.stack([fishers[k] for k in keys], axis=0)  # (K, P)
    mean_per_p = stacked.mean(axis=0)
    std_per_p = stacked.std(axis=0)
    cov_per_p = np.where(mean_per_p > 1e-30, std_per_p / (mean_per_p + 1e-30), np.nan)
    # Filter out zero-mean params (inactive)
    valid = mean_per_p > 1e-15
    cov_valid = cov_per_p[valid]

    # Condition-number proxy: max/min across valid params, per seed
    per_seed_cond = {}
    for k in keys:
        v = fishers[k]
        vvalid = v[v > 1e-15]
        if len(vvalid) > 0:
            per_seed_cond[k] = float(vvalid.max() / vvalid.min())

    summary = {
        "seeds": list(map(int, args.seeds)),
        "n_fisher_samples": args.n_fisher_samples,
        "max_train_samples": args.max_train_samples,
        "horizon": args.horizon,
        "n_params_total": int(stacked.shape[1]),
        "n_params_active": int(valid.sum()),
        "cov_across_seeds_median": float(np.median(cov_valid)) if len(cov_valid) else None,
        "cov_across_seeds_p90": float(np.percentile(cov_valid, 90)) if len(cov_valid) else None,
        "cov_across_seeds_p99": float(np.percentile(cov_valid, 99)) if len(cov_valid) else None,
        "frac_cov_gt_1": float(np.mean(cov_valid > 1.0)) if len(cov_valid) else None,
        "per_seed_condition_number": per_seed_cond,
        "per_seed_fisher_stats": {
            k: {
                "mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "max": float(np.max(v)),
                "frac_nonzero": float(np.mean(v > 0)),
            }
            for k, v in fishers.items()
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:")
    for k, v in summary.items():
        if k != "per_seed_fisher_stats" and k != "per_seed_condition_number":
            print(f"  {k}: {v}")
    print(f"  per_seed_condition_number: {per_seed_cond}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
