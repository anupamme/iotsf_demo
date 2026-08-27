#!/usr/bin/env python3
"""Fine-tune Moirai with full diagnostic protocol.

Conditions:
  A: Zero-shot (no fine-tuning) - baseline
  B: NLL-only fine-tuning
  C: NLL + Temporal SupCon
  D: Frozen encoder + linear head

Tracks: zero-shot MSE, CKA, weight drift, probes at each eval checkpoint.

Usage:
    python scripts/02_finetune_moirai.py --config configs/etth2_small_n500.yaml --seed 42
    python scripts/02_finetune_moirai.py --config configs/etth2_small_n500.yaml --condition B --seed 42
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cka import linear_CKA
from src.data import load_ett_data, load_ili_data, make_train_sequences, make_eval_sequences
from src.metrics import compute_forgetting_pct, compute_weight_drift
from src.models import load_moirai, extract_representations
from src.probes import linear_probe_r2
from src.train import (
    assign_temporal_labels,
    evaluate_forecasting,
    train_one_epoch,
)
from src.utils import load_config, set_seed, get_device, save_results


class SupervisedContrastiveLoss(nn.Module):
    """SupCon loss (Khosla et al., 2020)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = nn.functional.normalize(features, dim=1)
        sim = torch.mm(features, features.t()) / self.temperature
        b = features.shape[0]
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask.fill_diagonal_(False)
        pos_count = mask.sum(dim=1).clamp(min=1)
        log_prob = sim - torch.logsumexp(
            sim.masked_fill(torch.eye(b, device=sim.device).bool(), -1e9), dim=1, keepdim=True
        )
        loss = -(mask * log_prob).sum(dim=1) / pos_count
        return loss.mean()


def run_condition(cfg, condition, seed, device, results_dir):
    """Run a single (condition, seed) experiment."""
    set_seed(seed, deterministic=getattr(cfg, "deterministic", False))

    dataset = getattr(cfg, "dataset", "ETTh2")
    lookback = cfg.lookback
    horizon = cfg.horizon

    if dataset == "ILI":
        train_vals, val_vals, test_vals, feat_cols = load_ili_data(cfg.data_path)
    else:
        train_vals, val_vals, test_vals, feat_cols = load_ett_data(
            cfg.data_path, features=cfg.features
        )

    n_features = len(feat_cols)
    train_mean = train_vals.mean(axis=0)
    train_std = train_vals.std(axis=0) + 1e-8

    X_train, y_train = make_train_sequences(train_vals, lookback, horizon)
    extended_lookback = lookback + horizon
    X_val_eval, y_val_eval = make_eval_sequences(val_vals, extended_lookback, horizon)

    # Subsample training
    max_n = getattr(cfg, "max_train_samples", len(X_train))
    if max_n < len(X_train):
        idx = np.random.choice(len(X_train), size=max_n, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    # Load model
    model = load_moirai(
        model_size=cfg.model_size,
        context_length=lookback,
        prediction_length=horizon,
        target_dim=n_features,
        device=str(device),
    )
    pretrained_params = {n: p.data.clone() for n, p in model.named_parameters()}

    eval_limit = 300
    X_val_t = torch.from_numpy(X_val_eval[:eval_limit]).float()
    y_val_sub = y_val_eval[:eval_limit]

    # Zero-shot baseline
    zs_metrics = evaluate_forecasting(
        model, X_val_t, y_val_sub, train_mean, train_std, horizon, device=str(device)
    )
    pretrained_reps = extract_representations(model, X_val_t, device=str(device))

    if condition == "A":
        results = {
            "condition": "A",
            "horizon": horizon,
            "seed": seed,
            "max_train_samples": max_n,
            "zeroshot_mse": zs_metrics["mse"],
            "zeroshot_mae": zs_metrics["mae"],
        }
        save_results(results, str(results_dir / f"condition_A_h{horizon}_s{seed}.json"))
        return results

    # Setup training
    train_labels = assign_temporal_labels(len(X_train), getattr(cfg, "n_temporal_clusters", 8))
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
        torch.from_numpy(train_labels).long(),
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, generator=gen
    )

    # Condition-specific setup
    freeze_encoder = condition == "D"
    contrastive_weight = cfg.contrastive_weight if condition == "C" else 0.0

    if freeze_encoder:
        from src.models import get_encoder
        encoder = get_encoder(model)
        for p in encoder.parameters():
            p.requires_grad = False

    projection_head = None
    contrastive_loss_fn = None
    captured_embeddings = {}

    if contrastive_weight > 0:
        d_model = 256
        projection_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, 64)
        ).to(device)
        contrastive_loss_fn = SupervisedContrastiveLoss()

        from src.models import get_encoder
        encoder = get_encoder(model)
        encoder.register_forward_hook(
            lambda m, i, o: captured_embeddings.update({"encoder": o})
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    if projection_head:
        trainable += list(projection_head.parameters())
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr)

    # Training loop
    epochs = cfg.epochs
    eval_every = getattr(cfg, "eval_every", 5)
    best_mse = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        loss_dict = train_one_epoch(
            model, train_loader, optimizer, horizon, str(device),
            contrastive_weight=contrastive_weight,
            projection_head=projection_head,
            contrastive_loss_fn=contrastive_loss_fn,
            captured_embeddings=captured_embeddings,
            freeze_encoder=freeze_encoder,
            pretrained_params=pretrained_params,
        )

        if epoch % eval_every == 0 or epoch == epochs:
            val_metrics = evaluate_forecasting(
                model, X_val_t, y_val_sub, train_mean, train_std, horizon, device=str(device)
            )
            if val_metrics["mse"] < best_mse:
                best_mse = val_metrics["mse"]
                best_state = copy.deepcopy(model.state_dict())

    # Restore best if early stopping
    if getattr(cfg, "early_stopping", False) and best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    final_metrics = evaluate_forecasting(
        model, X_val_t, y_val_sub, train_mean, train_std, horizon, device=str(device)
    )
    finetuned_reps = extract_representations(model, X_val_t, device=str(device))

    cka = linear_CKA(pretrained_reps, finetuned_reps)
    drift = compute_weight_drift(model, pretrained_params)
    forgetting = compute_forgetting_pct(zs_metrics["mse"], final_metrics["mse"])

    # Probes
    y_probe = y_val_sub[:len(pretrained_reps)]
    n_train_probe = len(pretrained_reps) // 2
    probe_pre = linear_probe_r2(
        pretrained_reps[:n_train_probe], pretrained_reps[n_train_probe:],
        y_probe[:n_train_probe], y_probe[n_train_probe:],
    )
    probe_post = linear_probe_r2(
        finetuned_reps[:n_train_probe], finetuned_reps[n_train_probe:],
        y_probe[:n_train_probe], y_probe[n_train_probe:],
    )

    results = {
        "condition": condition,
        "horizon": horizon,
        "seed": seed,
        "max_train_samples": max_n,
        "zeroshot_mse": zs_metrics["mse"],
        "zeroshot_mae": zs_metrics["mae"],
        "final_val_mse": final_metrics["mse"],
        "final_val_mae": final_metrics["mae"],
        "final_cka": cka,
        "final_weight_drift": drift,
        "forgetting_pct": forgetting,
        "probe_r2_pre": probe_pre,
        "probe_r2_post": probe_post,
        "probe_delta_r2": probe_post - probe_pre if isinstance(probe_post, float) else None,
    }
    save_results(results, str(results_dir / f"condition_{condition}_h{horizon}_s{seed}.json"))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", default=None, help="Override: run single condition")
    parser.add_argument("--seed", type=int, default=None, help="Override: run single seed")
    parser.add_argument("--seeds-file", default=None, help="Path to seeds file (one per line)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--results-dir", default="runs/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    conditions = [args.condition] if args.condition else getattr(cfg, "conditions", ["A", "B", "C", "D"])

    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds_file:
        seeds = [int(s.strip()) for s in open(args.seeds_file) if s.strip()]
    else:
        seeds = [42]

    for cond in conditions:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  Condition {cond}, seed {seed}")
            print(f"{'='*60}")
            result = run_condition(cfg, cond, seed, device, results_dir)
            print(f"  MSE: {result.get('final_val_mse', result.get('zeroshot_mse')):.6f}")
            if "forgetting_pct" in result:
                print(f"  Forgetting: {result['forgetting_pct']:+.1f}%")
            if "final_cka" in result:
                print(f"  CKA: {result['final_cka']:.4f}")


if __name__ == "__main__":
    main()
