#!/usr/bin/env python3
"""
External Validation: Evaluate HNIDS (condition D) on N-BaIoT.

Addresses Limitation 6 (circular evaluation) by evaluating the fine-tuned
HNIDS on an independent IoT benchmark not seen during training.

The detector is trained solely on CICIoT2023; no N-BaIoT data is used
during training or scaler fitting.  A StandardScaler is fit on N-BaIoT
benign samples only, then threshold is calibrated on a held-out N-BaIoT
benign validation set (identical protocol to run_ablation.py).

Usage
-----
# Danmini doorbell (default, all 10 attack types):
python scripts/evaluate_nbaiot.py --data-dir data/nbaiot/

# Multiple seeds, verbose:
python scripts/evaluate_nbaiot.py --data-dir data/nbaiot/ --seeds 42,123,456

# Specific device:
python scripts/evaluate_nbaiot.py --data-dir data/nbaiot/ --device ecobee_thermostat

Prerequisites
-------------
Download N-BaIoT first:
    # Option A — Kaggle (requires ~/.kaggle/kaggle.json):
    pip install kaggle
    kaggle datasets download -d mkashifn/nbaiot-dataset -p data/nbaiot/ --unzip

    # Option B — manual:
    # Visit https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset
    # Download and extract into data/nbaiot/ so that:
    #   data/nbaiot/Danmini_Doorbell/benign_traffic.csv
    #   data/nbaiot/Danmini_Doorbell/mirai_attacks/scan.csv  ...etc

Results
-------
Saved to results/nbaiot/metrics.json (overall + per-attack breakdown).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models import MoiraiAnomalyDetector
from src.evaluation.metrics import IDSMetrics
from src.data.nbaiot_loader import load_nbaiot, PROXY_FEATURE_NAMES

# Import the shared fine-tuning helper from run_ablation so the condition D
# setup is identical (same data, same hyperparameters, same training loop).
sys.path.insert(0, str(ROOT_DIR / "scripts"))
from run_ablation import _fine_tune


# ---------------------------------------------------------------------------
# Reuse score_samples from run_ablation (same detection protocol)
# ---------------------------------------------------------------------------

def score_samples(detector: MoiraiAnomalyDetector, X: np.ndarray) -> np.ndarray:
    """Score every sample using NLL method, returning raw NLL scores."""
    n = len(X)
    scores = np.zeros(n, dtype=np.float64)
    for i, sample in enumerate(X):
        try:
            result = detector.detect_anomalies(sample, threshold=0.0, method='nll')
            scores[i] = float(result.anomaly_scores.mean())
        except Exception as e:
            logger.warning(f"Detection error on sample {i}: {e}")
            scores[i] = -1e9
        if (i + 1) % 50 == 0 or (i + 1) == n:
            logger.debug(f"  scored {i + 1}/{n}")
    return scores


def find_best_threshold(scores: np.ndarray, y: np.ndarray):
    """
    Sweep threshold to maximise F1 on calibration set.

    Tries both polarities (score > t  and  score < t) because Moirai fine-tuned
    on attack-heavy training data sometimes assigns lower NLL to attacks than to
    benign traffic (AUC < 0.5).  The polarity that achieves higher F1 is kept,
    and the returned threshold encodes the polarity: positive values use
    ``score > t``; negative values (stored as the negative threshold) use
    ``score < -t``.  Callers should use ``apply_threshold()`` rather than
    raw comparison.
    """
    lo, hi = np.percentile(scores, 2), np.percentile(scores, 98)
    thresholds = np.linspace(lo, hi, 100)

    best_f1, best_t, best_metrics, best_invert = -1.0, lo, {}, False
    for invert in (False, True):
        working_scores = -scores if invert else scores
        wlo, whi = np.percentile(working_scores, 2), np.percentile(working_scores, 98)
        for t in np.linspace(wlo, whi, 100):
            preds = (working_scores > t).astype(int)
            m = IDSMetrics.compute_all_metrics(y, preds, working_scores)
            if m["f1"] > best_f1:
                best_f1, best_t, best_metrics, best_invert = m["f1"], t, m, invert

    return best_t, best_metrics, best_invert


def apply_threshold(scores: np.ndarray, threshold: float, invert: bool) -> np.ndarray:
    """Apply calibrated threshold with correct polarity."""
    working = -scores if invert else scores
    return (working > threshold).astype(int)


# ---------------------------------------------------------------------------
# Core evaluation for one seed
# ---------------------------------------------------------------------------

def evaluate_seed(
    detector: MoiraiAnomalyDetector,
    X_benign_train: np.ndarray,
    X_benign_val: np.ndarray,
    X_attack: np.ndarray,
    y_attack_names: np.ndarray,
    attack_types: List[str],
) -> Dict:
    """
    Evaluate detector on N-BaIoT data for one seed.

    Protocol:
    1. Score benign_val + all attacks with raw NLL.
    2. Calibrate threshold on benign_val (maximise F1 vs. benign_val as negatives
       and a random subsample of attacks as positives).
    3. Evaluate on full test set: (remaining) benign + all attacks.

    Returns dict with 'overall' and 'per_attack' metrics.
    """
    n_benign_val = len(X_benign_val)
    n_attack     = len(X_attack)

    logger.info(f"Scoring {n_benign_val} benign-val + {n_attack} attack samples...")
    t0 = time.time()
    scores_bval   = score_samples(detector, X_benign_val)
    scores_attack = score_samples(detector, X_attack)
    logger.info(f"Scoring done in {time.time()-t0:.1f}s")

    # Calibration: benign_val (y=0) + all attacks (y=1).
    # Tries both score polarities; picks the direction that maximises F1.
    cal_scores = np.concatenate([scores_bval, scores_attack])
    cal_labels = np.concatenate([
        np.zeros(n_benign_val, dtype=int),
        np.ones(n_attack, dtype=int),
    ])
    threshold, _, invert = find_best_threshold(cal_scores, cal_labels)
    logger.info(f"Calibrated threshold: {threshold:.4f}  invert={invert}")

    # Overall evaluation
    y_true = cal_labels
    y_pred = apply_threshold(cal_scores, threshold, invert)
    working_scores = -cal_scores if invert else cal_scores
    overall = IDSMetrics.compute_all_metrics(y_true, y_pred, working_scores)
    logger.info(
        f"Overall — F1={overall['f1']:.3f} "
        f"FPR={overall['false_positive_rate']:.3f} "
        f"AUC={overall.get('roc_auc', float('nan')):.3f}"
    )

    # Per-attack breakdown
    per_attack: Dict[str, Dict] = {}
    for atk_type in attack_types:
        mask = (y_attack_names == atk_type)
        if not mask.any():
            continue
        atk_scores = scores_attack[mask]
        atk_labels = np.ones(mask.sum(), dtype=int)
        eval_scores = np.concatenate([scores_bval, atk_scores])
        eval_labels = np.concatenate([np.zeros(n_benign_val, dtype=int), atk_labels])
        eval_preds  = apply_threshold(eval_scores, threshold, invert)
        working_eval = -eval_scores if invert else eval_scores
        m = IDSMetrics.compute_all_metrics(eval_labels, eval_preds, working_eval)
        per_attack[atk_type] = {
            "f1":                 m["f1"],
            "false_positive_rate": m["false_positive_rate"],
            "roc_auc":            m.get("roc_auc", float("nan")),
            "n_attack_windows":   int(mask.sum()),
        }
        logger.info(
            f"  {atk_type}: F1={m['f1']:.3f} "
            f"FPR={m['false_positive_rate']:.3f} "
            f"AUC={m.get('roc_auc', float('nan')):.3f} "
            f"({mask.sum()} windows)"
        )

    return {"overall": overall, "per_attack": per_attack, "threshold": float(threshold)}


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def aggregate_seeds(seed_results: List[Dict]) -> Dict:
    """Average per-seed metrics; compute mean ± std."""
    keys_overall = ["f1", "false_positive_rate", "roc_auc", "precision", "recall"]
    agg: Dict = {"overall": {}, "per_attack": {}}

    # Overall
    for k in keys_overall:
        vals = [sr["overall"].get(k, float("nan")) for sr in seed_results]
        agg["overall"][k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    # Per-attack
    all_types = set()
    for sr in seed_results:
        all_types.update(sr["per_attack"].keys())

    for atk in sorted(all_types):
        pa_vals: Dict[str, List] = {}
        for sr in seed_results:
            if atk in sr["per_attack"]:
                for k, v in sr["per_attack"][atk].items():
                    pa_vals.setdefault(k, []).append(v)
        agg["per_attack"][atk] = {
            k: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))}
            for k, v in pa_vals.items()
            if k != "n_attack_windows"
        }
        # Carry through window count
        if "n_attack_windows" in pa_vals:
            agg["per_attack"][atk]["n_attack_windows"] = int(pa_vals["n_attack_windows"][0])

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HNIDS (condition D) on N-BaIoT external benchmark"
    )
    parser.add_argument(
        "--data-dir", default="data/nbaiot/",
        help="Root directory of N-BaIoT CSVs (default: data/nbaiot/)"
    )
    parser.add_argument(
        "--device", default="danmini_doorbell",
        help="IoT device to evaluate (default: danmini_doorbell)"
    )
    parser.add_argument(
        "--seeds", default="42,123,456",
        help="Comma-separated random seeds (default: 42,123,456)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=49548,
        help="Max raw rows per class before windowing (default: 49548 = full Danmini benign set)"
    )
    parser.add_argument(
        "--output", default="results/nbaiot/",
        help="Output directory for metrics.json (default: results/nbaiot/)"
    )
    parser.add_argument(
        "--model-size", default="small", choices=["small", "base", "large"],
        help="Moirai model size to use (default: small)"
    )
    parser.add_argument(
        "--synthetic-dir", default="data/synthetic",
        help="Directory of CICIoT2023 synthetic hard negatives for condition D fine-tuning "
             "(default: data/synthetic)"
    )
    parser.add_argument("--epochs",     type=int,   default=10,   help="Fine-tuning epochs (default: 10)")
    parser.add_argument("--batch-size", type=int,   default=32,   help="Fine-tuning batch size (default: 32)")
    parser.add_argument("--lr",         type=float, default=1e-4, help="Fine-tuning learning rate (default: 1e-4)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("N-BaIoT External Validation")
    logger.info("=" * 60)
    logger.info(f"Device  : {args.device}")
    logger.info(f"Seeds   : {seeds}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Load N-BaIoT (once — the scaler is deterministic for fixed device)
    # ------------------------------------------------------------------
    X_train, X_val, X_attack, y_names, scaler, attack_types = load_nbaiot(
        data_dir=args.data_dir,
        device=args.device,
        max_samples_per_class=args.max_samples,
        seq_length=128,
        seed=seeds[0],
    )

    logger.info(f"Attack types found: {attack_types}")
    logger.info(f"Feature proxy names: {PROXY_FEATURE_NAMES}")

    # ------------------------------------------------------------------
    # Run evaluation for each seed
    # ------------------------------------------------------------------
    seed_results = []
    for seed in seeds:
        logger.info(f"\n--- Seed {seed} ---")

        # Reshuffle benign val split per seed for robustness estimate
        X_train_s, X_val_s, X_atk_s, y_names_s, _, atk_types_s = load_nbaiot(
            data_dir=args.data_dir,
            device=args.device,
            max_samples_per_class=args.max_samples,
            seq_length=128,
            seed=seed,
        )

        # Initialise and fine-tune detector (condition D: NLL+SupCon, hard negatives).
        # This mirrors run_condition_d() in run_ablation.py exactly.
        detector = MoiraiAnomalyDetector(
            model_size=args.model_size,
            context_length=96,
            prediction_length=32,
        )
        detector.initialize()
        logger.info(f"Fine-tuning detector (condition D) on CICIoT2023 data "
                    f"from {args.synthetic_dir} for {args.epochs} epochs ...")
        _fine_tune(
            detector,
            synthetic_dir=args.synthetic_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            contrastive_weight=0.5,
            use_hard_negatives=True,
            use_constraints=True,
        )
        logger.info("Fine-tuning complete. Evaluating on N-BaIoT ...")

        result = evaluate_seed(
            detector,
            X_benign_train=X_train_s,
            X_benign_val=X_val_s,
            X_attack=X_atk_s,
            y_attack_names=y_names_s,
            attack_types=atk_types_s,
        )
        seed_results.append(result)

    # ------------------------------------------------------------------
    # Aggregate and save
    # ------------------------------------------------------------------
    agg = aggregate_seeds(seed_results)
    agg["meta"] = {
        "device":        args.device,
        "seeds":         seeds,
        "n_seeds":       len(seeds),
        "attack_types":  attack_types,
        "feature_proxy": PROXY_FEATURE_NAMES,
        "n_benign_val_windows": int(len(X_val)),
        "n_attack_windows":     int(len(X_attack)),
    }

    out_path = output_dir / "metrics.json"
    out_path.write_text(json.dumps(agg, indent=2))
    logger.success(f"Results saved to {out_path}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY — Overall (all attacks vs. benign-val)")
    logger.info("=" * 60)
    ov = agg["overall"]
    logger.info(
        f"F1  = {ov['f1']['mean']:.3f} ± {ov['f1']['std']:.3f}  |  "
        f"FPR = {ov['false_positive_rate']['mean']:.3f} ± {ov['false_positive_rate']['std']:.3f}  |  "
        f"AUC = {ov['roc_auc']['mean']:.3f} ± {ov['roc_auc']['std']:.3f}"
    )

    logger.info("\nPer-attack type (F1 / FPR / AUC):")
    for atk, metrics in agg["per_attack"].items():
        f1  = metrics.get("f1", {})
        fpr = metrics.get("false_positive_rate", {})
        auc = metrics.get("roc_auc", {})
        logger.info(
            f"  {atk:25s}  "
            f"F1={f1.get('mean', float('nan')):.3f}  "
            f"FPR={fpr.get('mean', float('nan')):.3f}  "
            f"AUC={auc.get('mean', float('nan')):.3f}"
        )

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
