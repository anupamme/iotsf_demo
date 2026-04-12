#!/usr/bin/env python3
"""
Calibrate Moirai anomaly detection thresholds using NLL-based scoring.

The base Moirai model's confidence-interval method produces identical scores
for all samples (0.75), making it useless for discrimination. The NLL method
produces varying scores with some separation between benign and attack traffic.

This script:
1. Scores all validation samples using NLL-based detection
2. Sweeps a threshold on the mean NLL anomaly score to maximize F1
3. Outputs calibrated threshold to results/calibrated_thresholds.json

Usage:
    python scripts/calibrate_threshold.py
    python scripts/calibrate_threshold.py --checkpoint models/moirai_supervised/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models import MoiraiAnomalyDetector
from src.evaluation.metrics import IDSMetrics


def load_validation_data(synthetic_dir: str, n_benign: int = 20):
    """
    Build a balanced validation set from pre-generated synthetic data.

    Returns
    -------
    X_val : np.ndarray  (n_samples, 128, 12)
    y_val : np.ndarray  (n_samples,)  0=benign, 1=attack
    """
    synth = Path(synthetic_dir)
    if not synth.exists():
        raise FileNotFoundError(f"Synthetic data directory not found: {synth}")

    benign_path = synth / "benign_samples.npy"
    if not benign_path.exists():
        raise FileNotFoundError(f"benign_samples.npy not found in {synth}")
    benign = np.load(benign_path)
    logger.info(f"Loaded {len(benign)} benign samples")

    attack_files = list(synth.glob("*_stealth_*.npy"))
    if not attack_files:
        raise FileNotFoundError(f"No attack .npy files found in {synth}")

    attack_chunks = [np.load(p) for p in attack_files]
    attacks = np.concatenate(attack_chunks, axis=0)
    logger.info(f"Loaded {len(attacks)} attack samples from {len(attack_files)} files")

    rng = np.random.default_rng(42)
    if len(benign) > n_benign:
        idx = rng.choice(len(benign), size=n_benign, replace=False)
        benign = benign[idx]

    n_attack = min(len(attacks), len(benign) * 4)
    if len(attacks) > n_attack:
        idx = rng.choice(len(attacks), size=n_attack, replace=False)
        attacks = attacks[idx]

    X_val = np.concatenate([benign, attacks], axis=0)
    y_val = np.array([0] * len(benign) + [1] * len(attacks))
    logger.info(f"Validation set: {len(benign)} benign + {len(attacks)} attack = {len(X_val)} total")
    return X_val, y_val


def collect_nll_scores(detector, X: np.ndarray) -> np.ndarray:
    """
    Score every sample using NLL-based detection, returning per-sample
    mean anomaly scores.
    """
    n = len(X)
    scores = np.zeros(n, dtype=np.float32)

    for i, sample in enumerate(X):
        try:
            result = detector.detect_anomalies(sample, threshold=0.5, method='nll')
            scores[i] = result.anomaly_scores.mean()
        except Exception as e:
            logger.warning(f"Sample {i} error: {e}")

        if (i + 1) % 10 == 0:
            logger.debug(f"  Scored {i + 1}/{n}")

    return scores


def sweep_threshold(scores: np.ndarray, y_val: np.ndarray) -> dict:
    """
    Sweep a threshold on raw NLL anomaly scores to maximize F1.
    Higher NLL = more anomalous.
    """
    lo, hi = np.percentile(scores, 5), np.percentile(scores, 95)
    thresholds = np.linspace(lo, hi, 200)
    best = {"f1": -1.0}

    for t in thresholds:
        preds = (scores > t).astype(int)
        metrics = IDSMetrics.compute_all_metrics(y_val, preds, scores)
        f1 = metrics["f1"]
        if f1 > best["f1"]:
            best = {
                "f1": float(f1),
                "threshold": float(t),
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "false_positive_rate": float(metrics["false_positive_rate"]),
                "roc_auc": float(metrics.get("roc_auc") or 0.0),
            }
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate Moirai anomaly thresholds (NLL method)",
    )
    parser.add_argument("--model-size", default="small", choices=["small", "base", "large"])
    parser.add_argument("--checkpoint", default=None,
                        help="Optional path to fine-tuned model checkpoint")
    parser.add_argument("--synthetic-dir", default="data/synthetic")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--n-benign", type=int, default=20)
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading validation data...")
    X_val, y_val = load_validation_data(args.synthetic_dir, n_benign=args.n_benign)

    logger.info(f"Initializing Moirai ({args.model_size})...")
    detector = MoiraiAnomalyDetector(
        model_size=args.model_size,
        context_length=96,
        prediction_length=32,
        confidence_level=0.95,
    )
    detector.initialize(checkpoint_path=args.checkpoint)
    logger.info(f"Detector ready on {detector.device}")

    logger.info("Scoring validation samples with NLL method...")
    scores = collect_nll_scores(detector, X_val)

    benign_scores = scores[y_val == 0]
    attack_scores = scores[y_val == 1]
    logger.info(f"Score distributions — benign: mean={benign_scores.mean():.4f} std={benign_scores.std():.4f} | "
                f"attack: mean={attack_scores.mean():.4f} std={attack_scores.std():.4f}")

    logger.info("Sweeping threshold...")
    best = sweep_threshold(scores, y_val)

    logger.success(
        f"Best threshold: {best['threshold']:.2f}  "
        f"→ F1={best['f1']:.3f}, FPR={best['false_positive_rate']:.3f}"
    )

    output = {
        "model_size": args.model_size,
        "checkpoint": args.checkpoint,
        "detection_method": "nll",
        "n_val_samples": len(y_val),
        "n_benign": int((y_val == 0).sum()),
        "n_attack": int((y_val == 1).sum()),
        "score_distributions": {
            "benign_mean": float(benign_scores.mean()),
            "benign_std": float(benign_scores.std()),
            "attack_mean": float(attack_scores.mean()),
            "attack_std": float(attack_scores.std()),
        },
        "optimal": {
            "anomaly_score_threshold": best["threshold"],
            "detection_rate_threshold": best["threshold"],
        },
        "optimal_metrics": {
            "f1": best["f1"],
            "accuracy": best["accuracy"],
            "precision": best["precision"],
            "recall": best["recall"],
            "false_positive_rate": best["false_positive_rate"],
            "roc_auc": best["roc_auc"],
        },
    }

    out_file = output_path / "calibrated_thresholds.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.success(f"Calibrated thresholds saved to {out_file}")

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS (NLL Method)")
    print("=" * 60)
    print(f"  Optimal threshold:  {best['threshold']:.2f}")
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
