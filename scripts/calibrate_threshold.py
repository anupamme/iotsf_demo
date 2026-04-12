#!/usr/bin/env python3
"""
Calibrate Moirai anomaly detection thresholds to fix the FPR=1.0 bug.

The base Moirai model produces mean anomaly scores around 0.77-0.82 for all
samples (benign and attack alike), so the default detection_rate_threshold=0.3
flags everything as an attack.

This script sweeps anomaly_score_threshold × detection_rate_threshold on a
balanced validation set and selects the operating point that maximises F1.

Output: results/calibrated_thresholds.json

Usage:
    python scripts/calibrate_threshold.py
    python scripts/calibrate_threshold.py --model-size base
    python scripts/calibrate_threshold.py --checkpoint models/moirai_supervised/best.pt
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from loguru import logger

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models import MoiraiAnomalyDetector
from src.evaluation.metrics import IDSMetrics


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_validation_data(synthetic_dir: str, n_benign: int = 20):
    """
    Build a balanced validation set from pre-generated synthetic data.

    Uses the synthetic benign samples plus hard-negative attacks at all stealth
    levels, then subsets to a balanced n_benign benign / n_attack attack split.

    Returns
    -------
    X_val : np.ndarray  (n_samples, 128, 12)
    y_val : np.ndarray  (n_samples,)  0=benign, 1=attack
    """
    synth = Path(synthetic_dir)
    if not synth.exists():
        raise FileNotFoundError(f"Synthetic data directory not found: {synth}")

    # Load benign
    benign_path = synth / "benign_samples.npy"
    if not benign_path.exists():
        raise FileNotFoundError(f"benign_samples.npy not found in {synth}")
    benign = np.load(benign_path)          # (n_benign, 128, 12)
    logger.info(f"Loaded {len(benign)} benign samples")

    # Load all attack files
    attack_files = list(synth.glob("*_stealth_*.npy"))
    if not attack_files:
        raise FileNotFoundError(f"No attack .npy files found in {synth}")

    attack_chunks = [np.load(p) for p in attack_files]
    attacks = np.concatenate(attack_chunks, axis=0)
    logger.info(f"Loaded {len(attacks)} attack samples from {len(attack_files)} files")

    # Balance: keep at most n_benign benign samples and the same number of attacks
    rng = np.random.default_rng(42)
    if len(benign) > n_benign:
        idx = rng.choice(len(benign), size=n_benign, replace=False)
        benign = benign[idx]

    n_attack = min(len(attacks), len(benign) * 4)   # allow more attacks than benign
    if len(attacks) > n_attack:
        idx = rng.choice(len(attacks), size=n_attack, replace=False)
        attacks = attacks[idx]

    X_val = np.concatenate([benign, attacks], axis=0)
    y_val = np.array([0] * len(benign) + [1] * len(attacks))
    logger.info(f"Validation set: {len(benign)} benign + {len(attacks)} attack = {len(X_val)} total")
    return X_val, y_val


# ---------------------------------------------------------------------------
# Calibration sweep
# ---------------------------------------------------------------------------

def collect_scores(detector, X: np.ndarray, ci_thresholds: np.ndarray) -> np.ndarray:
    """
    Run the detector on every sample and return a matrix of
    shape (n_samples, n_ci_thresholds) containing the anomaly_rate for each
    confidence-interval threshold value.

    This is done in one pass to avoid running the model multiple times per sample.
    """
    n = len(X)
    n_t = len(ci_thresholds)
    anomaly_rates = np.zeros((n, n_t), dtype=np.float32)

    for i, sample in enumerate(X):
        try:
            # detect_anomalies already returns anomaly_scores (per-timestep)
            result = detector.detect_anomalies(sample, threshold=ci_thresholds[0])
            scores = result.anomaly_scores   # shape: (T,)

            # Re-threshold on the already-computed scores for every ci level
            for j, ci in enumerate(ci_thresholds):
                anomaly_rates[i, j] = (scores > ci).mean()

        except Exception as e:
            logger.warning(f"Sample {i} error: {e}")
            # Leave as zeros (will be predicted benign)

        if (i + 1) % 10 == 0:
            logger.debug(f"  Scored {i + 1}/{n}")

    return anomaly_rates   # (n_samples, n_ci_thresholds)


def sweep_thresholds(
    anomaly_rates: np.ndarray,
    y_val: np.ndarray,
    detection_thresholds: np.ndarray,
) -> dict:
    """
    For every (ci_threshold_idx, detection_threshold) combination compute F1
    and return the best configuration.

    anomaly_rates : (n_samples, n_ci_thresholds)
    y_val         : (n_samples,)
    """
    n_ci = anomaly_rates.shape[1]
    best = {"f1": -1.0}

    for ci_idx in range(n_ci):
        rates = anomaly_rates[:, ci_idx]
        for dt in detection_thresholds:
            preds = (rates > dt).astype(int)
            metrics = IDSMetrics.compute_all_metrics(y_val, preds, rates)
            f1 = metrics["f1"]
            if f1 > best["f1"]:
                best = {
                    "f1": float(f1),
                    "ci_threshold_idx": int(ci_idx),
                    "detection_rate_threshold": float(dt),
                    "accuracy": float(metrics["accuracy"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "false_positive_rate": float(metrics["false_positive_rate"]),
                    "roc_auc": float(metrics.get("roc_auc") or 0.0),
                }
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate Moirai anomaly thresholds on a validation set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-size", default="small", choices=["small", "base", "large"])
    parser.add_argument("--checkpoint", default=None,
                        help="Optional path to fine-tuned model checkpoint")
    parser.add_argument("--synthetic-dir", default="data/synthetic",
                        help="Directory with pre-generated .npy files")
    parser.add_argument("--output-dir", default="results",
                        help="Directory to write calibrated_thresholds.json")
    parser.add_argument("--n-benign", type=int, default=20,
                        help="Number of benign validation samples")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load validation data ---
    logger.info("Loading validation data...")
    X_val, y_val = load_validation_data(args.synthetic_dir, n_benign=args.n_benign)

    # --- Initialize detector ---
    logger.info(f"Initializing Moirai ({args.model_size})...")
    detector = MoiraiAnomalyDetector(
        model_size=args.model_size,
        context_length=96,
        prediction_length=32,
        confidence_level=0.95,
    )
    detector.initialize(checkpoint_path=args.checkpoint)
    logger.info(f"Detector ready on {detector.device}")

    # --- Define sweep grids ---
    ci_thresholds = np.arange(0.50, 1.00, 0.05)           # 10 values
    detection_thresholds = np.arange(0.05, 0.70, 0.05)    # 13 values
    logger.info(f"Sweep: {len(ci_thresholds)} CI thresholds × "
                f"{len(detection_thresholds)} detection thresholds "
                f"= {len(ci_thresholds) * len(detection_thresholds)} combos")

    # --- Collect anomaly scores (single model pass per sample) ---
    logger.info("Scoring validation samples...")
    anomaly_rates = collect_scores(detector, X_val, ci_thresholds)

    # --- Find best threshold ---
    logger.info("Sweeping threshold grid...")
    best = sweep_thresholds(anomaly_rates, y_val, detection_thresholds)
    best_ci = float(ci_thresholds[best["ci_threshold_idx"]])
    best_dt = best["detection_rate_threshold"]

    logger.success(
        f"Best thresholds: CI={best_ci:.2f}, detection_rate={best_dt:.2f}  "
        f"→ F1={best['f1']:.3f}, FPR={best['false_positive_rate']:.3f}"
    )

    # --- Save results ---
    output = {
        "model_size": args.model_size,
        "checkpoint": args.checkpoint,
        "n_val_samples": len(y_val),
        "n_benign": int((y_val == 0).sum()),
        "n_attack": int((y_val == 1).sum()),
        "optimal": {
            "anomaly_score_threshold": best_ci,
            "detection_rate_threshold": best_dt,
        },
        "optimal_metrics": {
            "f1": best["f1"],
            "accuracy": best["accuracy"],
            "precision": best["precision"],
            "recall": best["recall"],
            "false_positive_rate": best["false_positive_rate"],
            "roc_auc": best["roc_auc"],
        },
        "sweep_grid": {
            "ci_thresholds": ci_thresholds.tolist(),
            "detection_thresholds": detection_thresholds.tolist(),
        },
    }

    out_file = output_path / "calibrated_thresholds.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.success(f"Calibrated thresholds saved to {out_file}")

    # Pretty-print summary
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"  Optimal CI threshold:         {best_ci:.2f}")
    print(f"  Optimal detection rate thr.:  {best_dt:.2f}")
    print(f"  F1:         {best['f1']:.3f}")
    print(f"  Accuracy:   {best['accuracy']:.3f}")
    print(f"  Precision:  {best['precision']:.3f}")
    print(f"  Recall:     {best['recall']:.3f}")
    print(f"  FPR:        {best['false_positive_rate']:.3f}")
    print(f"  ROC-AUC:    {best['roc_auc']:.3f}")
    print("=" * 60)
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
