#!/usr/bin/env python3
"""
Gradient Analysis Script for NeurIPS Reviewer Q5

Answers: "Are SupCon loss gradients actually non-zero during training?"
Runs Condition C (Gaussian-noise SupCon) for a few epochs with gradient logging
and reports per-epoch gradient norms for the projection head and encoder.

Usage:
    python -m scripts.gradient_analysis [--epochs 5] [--seed 42]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from loguru import logger
from src.models import MoiraiAnomalyDetector


def main():
    parser = argparse.ArgumentParser(
        description="Gradient analysis for SupCon training (NeurIPS Q5)"
    )
    parser.add_argument(
        "--synthetic-dir", type=str, default="data/synthetic",
        help="Directory containing benign_samples.npy"
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output", type=str, default="results/gradient_analysis.json",
        help="Output JSON path (default: results/gradient_analysis.json)"
    )
    args = parser.parse_args()

    # ---- Load data (mirrors _fine_tune Condition C in run_ablation.py) --------
    synth = Path(args.synthetic_dir)
    benign_path = synth / "benign_samples.npy"
    if not benign_path.exists():
        logger.error(f"Benign samples not found at {benign_path}")
        sys.exit(1)

    benign = np.load(benign_path)
    logger.info(f"Loaded benign samples: {benign.shape}")

    # Condition C: Gaussian-noise negatives
    rng = np.random.default_rng(args.seed)
    attacks = benign + rng.normal(0, 0.3, benign.shape)
    logger.info(f"Generated Gaussian-noise negatives: {attacks.shape}")

    # Cap to 200 samples total for fast diagnostics
    max_samples = 200
    all_X = np.concatenate([benign, attacks])
    all_y = np.array([0] * len(benign) + [1] * len(attacks))

    if len(all_X) > max_samples:
        rng_cap = np.random.default_rng(args.seed)
        cap_idx = rng_cap.choice(len(all_X), size=max_samples, replace=False)
        all_X = all_X[cap_idx]
        all_y = all_y[cap_idx]
        logger.info(f"Capped to {max_samples} samples for diagnostics")

    # 85/15 train/val split
    rng_split = np.random.default_rng(1)
    perm = rng_split.permutation(len(all_X))
    n_val = max(4, int(0.15 * len(all_X)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_data, train_labels = all_X[train_idx], all_y[train_idx]
    val_data, val_labels = all_X[val_idx], all_y[val_idx]

    logger.info(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples")

    # ---- Initialize detector ---------------------------------------------------
    detector = MoiraiAnomalyDetector(
        model_size="small",
        context_length=96,
        prediction_length=32,
    )
    detector.initialize()
    logger.info("MoiraiAnomalyDetector initialized")

    # ---- Run training with gradient logging ------------------------------------
    history = detector.fine_tune_supervised(
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        n_epochs=args.epochs,
        batch_size=16,
        learning_rate=1e-4,
        contrastive_weight=0.5,
        log_gradients=True,
    )

    # ---- Collect results -------------------------------------------------------
    grad_hist = history.get("gradient_history", [])
    n_epochs_run = len(history.get("train_loss", []))

    results = {
        "config": {
            "model_size": "small",
            "context_length": 96,
            "prediction_length": 32,
            "epochs_requested": args.epochs,
            "epochs_completed": n_epochs_run,
            "total_samples": max_samples,
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "seed": args.seed,
        },
        "per_epoch": [],
    }

    for i in range(n_epochs_run):
        epoch_entry = {
            "epoch": i + 1,
            "nll_loss": history["train_nll"][i],
            "supcon_loss": history["train_contrastive"][i],
            "total_loss": history["train_loss"][i],
            "val_loss": history["val_loss"][i] if i < len(history["val_loss"]) else None,
        }
        if i < len(grad_hist):
            epoch_entry["proj_grad_norm"] = grad_hist[i]["proj_grad_norm"]
            epoch_entry["encoder_grad_norm"] = grad_hist[i]["encoder_grad_norm"]
            epoch_entry["cont_loss_ratio"] = grad_hist[i]["cont_loss_ratio"]
        results["per_epoch"].append(epoch_entry)

    # ---- Save JSON output ------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # ---- Print formatted table -------------------------------------------------
    print()
    print("=" * 90)
    print("GRADIENT ANALYSIS RESULTS (NeurIPS Q5: Are SupCon gradients non-zero?)")
    print("=" * 90)
    header = (
        f"{'Epoch':>5}  {'NLL Loss':>10}  {'SupCon Loss':>12}  "
        f"{'Cont/NLL':>9}  {'Proj Grad':>11}  {'Encoder Grad':>13}"
    )
    print(header)
    print("-" * 90)

    for entry in results["per_epoch"]:
        proj = entry.get("proj_grad_norm", float("nan"))
        enc = entry.get("encoder_grad_norm", float("nan"))
        ratio = entry.get("cont_loss_ratio", float("nan"))
        print(
            f"{entry['epoch']:>5}  {entry['nll_loss']:>10.6f}  "
            f"{entry['supcon_loss']:>12.6f}  {ratio:>9.4f}  "
            f"{proj:>11.6f}  {enc:>13.6f}"
        )

    print("-" * 90)

    # Summary verdict
    if grad_hist:
        avg_proj = np.mean([g["proj_grad_norm"] for g in grad_hist])
        avg_enc = np.mean([g["encoder_grad_norm"] for g in grad_hist])
        print(f"\nAvg projection head grad norm: {avg_proj:.6f}")
        print(f"Avg encoder grad norm:         {avg_enc:.6f}")
        if avg_proj > 1e-6 and avg_enc > 1e-6:
            print("\nVERDICT: SupCon gradients are NON-ZERO through both projection head and encoder.")
        else:
            print("\nWARNING: Some gradient norms are near zero -- investigate further.")
    else:
        print("\nNo gradient data collected (training may have failed).")

    print("=" * 90)


if __name__ == "__main__":
    main()
