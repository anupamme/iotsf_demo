#!/usr/bin/env python3
"""Run linear probes on encoder representations to measure task-native Delta R^2.

Computes:
  - R^2 on pre-trained representations (frozen encoder utility)
  - R^2 on fine-tuned representations
  - Delta R^2 = post - pre (positive = restructuring helped task performance)

Usage:
    python scripts/04_run_probes.py --results-dir runs/
    python scripts/04_run_probes.py --config configs/etth2_small_n500.yaml \
        --seed 42 --device cuda
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_ett_data, load_ili_data, make_eval_sequences
from src.models import load_moirai, extract_representations
from src.probes import linear_probe_r2
from src.utils import load_config, set_seed, get_device, load_results


def summarize_from_results(results_dir: str):
    """Print probe summary from pre-computed results."""
    results = load_results(results_dir)
    if not results:
        print(f"No results in {results_dir}")
        return

    print(f"{'Cond':<6} {'Seed':>6} {'R2_pre':>8} {'R2_post':>9} {'Delta':>8}")
    print("-" * 42)

    for r in sorted(results, key=lambda x: (x.get("condition", ""), x.get("seed", 0))):
        pre = r.get("probe_r2_pre")
        post = r.get("probe_r2_post")
        if pre is None or post is None:
            continue
        delta = post - pre if isinstance(post, (int, float)) else None
        print(
            f"{r['condition']:<6} {r['seed']:>6} {pre:>8.4f} {post:>9.4f} "
            f"{delta:>+7.4f}" if delta is not None else ""
        )


def compute_probes(cfg, seed, device):
    """Compute probes from scratch using model."""
    set_seed(seed)

    dataset = getattr(cfg, "dataset", "ETTh2")
    if dataset == "ILI":
        train, val, test, feat_cols = load_ili_data(cfg.data_path)
    else:
        train, val, test, feat_cols = load_ett_data(cfg.data_path, features=cfg.features)

    lookback = cfg.lookback
    horizon = cfg.horizon
    extended_lookback = lookback + horizon
    X_val, y_val = make_eval_sequences(val, extended_lookback, horizon)

    model = load_moirai(
        model_size=cfg.model_size,
        context_length=lookback,
        prediction_length=horizon,
        target_dim=len(feat_cols),
        device=str(device),
    )

    X_val_t = torch.from_numpy(X_val[:300]).float()
    reps = extract_representations(model, X_val_t, device=str(device))
    y_probe = y_val[:len(reps)]

    n_train = len(reps) // 2
    for probe_type in ["ridge", "mlp", "linear_forecaster"]:
        r2 = linear_probe_r2(
            reps[:n_train], reps[n_train:],
            y_probe[:n_train], y_probe[n_train:],
            probe_type=probe_type,
        )
        print(f"  {probe_type}: R^2 = {r2:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="runs/")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        device = get_device(args.device)
        compute_probes(cfg, args.seed, device)
    else:
        summarize_from_results(args.results_dir)


if __name__ == "__main__":
    main()
