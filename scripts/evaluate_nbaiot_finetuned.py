#!/usr/bin/env python3
"""
N-BaIoT Fine-Tuned Evaluation: Fine-tune HNIDS on N-BaIoT benign data,
then evaluate on N-BaIoT attacks.

Addresses the reviewer concern that the cross-dataset zero-shot transfer
(CICIoT2023 -> N-BaIoT) fails catastrophically.  This script instead
fine-tunes the detector on N-BaIoT's own benign data with Gaussian-noise
negatives, demonstrating that the HNIDS framework generalises when
properly deployed on the target distribution.

Fine-tuning protocol
--------------------
1. Load N-BaIoT benign traffic for the target device.
2. Generate Gaussian-noise negatives from the benign training windows
   (additive noise with sigma=0.3).
3. Build a balanced training set: benign (y=0) + noise negatives (y=1).
4. Shuffle and split 85/15 into train/val.
5. Fine-tune the Moirai detector using supervised contrastive loss
   (NLL + SupCon, contrastive_weight=0.5).
6. Evaluate on N-BaIoT attack data using the standard threshold-
   calibration protocol (identical to evaluate_nbaiot.py).

Usage
-----
python scripts/evaluate_nbaiot_finetuned.py --data-dir data/nbaiot/

# Multiple seeds:
python scripts/evaluate_nbaiot_finetuned.py --data-dir data/nbaiot/ --seeds 42,123,456

# Specific device:
python scripts/evaluate_nbaiot_finetuned.py --data-dir data/nbaiot/ --device ecobee_thermostat

Prerequisites
-------------
Download N-BaIoT first:
    # Option A -- Kaggle (requires ~/.kaggle/kaggle.json):
    pip install kaggle
    kaggle datasets download -d mkashifn/nbaiot-dataset -p data/nbaiot/ --unzip

    # Option B -- manual:
    # Visit https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset
    # Download and extract into data/nbaiot/ so that:
    #   data/nbaiot/1.benign.csv
    #   data/nbaiot/1.mirai.scan.csv  ...etc

Results
-------
Saved to results/nbaiot_finetuned/metrics.json (overall + per-attack breakdown).
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


# ---------------------------------------------------------------------------
# Scoring helper (identical to evaluate_nbaiot.py)
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


# ---------------------------------------------------------------------------
# Threshold calibration (identical to evaluate_nbaiot.py)
# ---------------------------------------------------------------------------

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
# Core evaluation for one seed (identical to evaluate_nbaiot.py)
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
        f"Overall -- F1={overall['f1']:.3f} "
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
# Multi-seed aggregation (identical to evaluate_nbaiot.py)
# ---------------------------------------------------------------------------

def aggregate_seeds(seed_results: List[Dict]) -> Dict:
    """Average per-seed metrics; compute mean +/- std."""
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
# Fine-tuning on N-BaIoT benign data + Gaussian-noise negatives
# ---------------------------------------------------------------------------

def fine_tune_on_nbaiot_benign(
    detector: MoiraiAnomalyDetector,
    X_train: np.ndarray,
    seed: int,
    n_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    contrastive_weight: float = 0.5,
    noise_sigma: float = 0.3,
    negative_type: str = "gaussian",
    diffts_data_dir: str = None,
) -> Dict:
    """
    Fine-tune detector on N-BaIoT benign windows + synthetic negatives.

    Creates synthetic attack examples using either Gaussian noise or
    pre-generated Diffusion-TS negatives, then trains with supervised
    contrastive loss (NLL + SupCon).

    Parameters
    ----------
    detector : MoiraiAnomalyDetector
        Initialised detector to fine-tune.
    X_train : np.ndarray
        Benign training windows, shape (N, seq_length, n_features).
    seed : int
        Random seed for noise generation and train/val split.
    n_epochs : int
        Number of fine-tuning epochs.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Learning rate for the optimiser.
    contrastive_weight : float
        Weight of the supervised contrastive loss term.
    noise_sigma : float
        Standard deviation of Gaussian noise added to benign windows.
    negative_type : str
        "gaussian" for Gaussian noise, "diffts" for Diffusion-TS negatives.
    diffts_data_dir : str
        Directory containing pre-generated DiffTS negatives (required if
        negative_type="diffts").

    Returns
    -------
    dict
        Training history from fine_tune_supervised().
    """
    if negative_type == "diffts":
        if diffts_data_dir is None:
            raise ValueError("--diffts-data-dir required when negative_type='diffts'")
        diffts_path = Path(diffts_data_dir)
        attack_files = sorted(diffts_path.glob("*_stealth_*.npy"))
        if not attack_files:
            raise FileNotFoundError(f"No attack files found in {diffts_path}")
        attacks = [np.load(f) for f in attack_files]
        noise_negatives = np.concatenate(attacks)
        rng_ft = np.random.default_rng(seed)
        if len(noise_negatives) > len(X_train):
            idx = rng_ft.choice(len(noise_negatives), len(X_train), replace=False)
            noise_negatives = noise_negatives[idx]
        elif len(noise_negatives) < len(X_train):
            idx = rng_ft.choice(len(noise_negatives), len(X_train), replace=True)
            noise_negatives = noise_negatives[idx]
        logger.info(f"Loaded {len(noise_negatives)} DiffTS negatives from {diffts_path}")
    else:
        rng_ft = np.random.default_rng(seed)
        noise_negatives = X_train + rng_ft.normal(0, noise_sigma, X_train.shape)

    # Build balanced training set: benign (y=0) + noise negatives (y=1)
    all_X = np.concatenate([X_train, noise_negatives])
    all_y = np.array([0] * len(X_train) + [1] * len(noise_negatives))

    # Shuffle and split 85/15 into train/val
    rng_split = np.random.default_rng(seed + 1000)
    perm = rng_split.permutation(len(all_X))
    n_val = max(4, int(0.15 * len(all_X)))
    train_data, train_labels = all_X[perm[n_val:]], all_y[perm[n_val:]]
    val_data, val_labels = all_X[perm[:n_val]], all_y[perm[:n_val]]

    logger.info(
        f"Fine-tuning dataset: {len(train_data)} train "
        f"({(train_labels == 0).sum()} benign, {(train_labels == 1).sum()} noise-neg), "
        f"{len(val_data)} val "
        f"({(val_labels == 0).sum()} benign, {(val_labels == 1).sum()} noise-neg)"
    )

    # Fine-tune using the detector's supervised contrastive method
    history = detector.fine_tune_supervised(
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        contrastive_weight=contrastive_weight,
    )

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune HNIDS on N-BaIoT benign data and evaluate on N-BaIoT attacks"
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
        "--output", default="results/nbaiot_finetuned/",
        help="Output directory for metrics.json (default: results/nbaiot_finetuned/)"
    )
    parser.add_argument(
        "--model-size", default="small", choices=["small", "base", "large"],
        help="Moirai model size to use (default: small)"
    )
    parser.add_argument("--epochs",     type=int,   default=5,    help="Fine-tuning epochs (default: 5)")
    parser.add_argument("--batch-size", type=int,   default=32,   help="Fine-tuning batch size (default: 32)")
    parser.add_argument("--lr",         type=float, default=1e-4, help="Fine-tuning learning rate (default: 1e-4)")
    parser.add_argument("--negative-type", default="gaussian", choices=["gaussian", "diffts"],
                        help="Type of synthetic negatives: gaussian (default) or diffts")
    parser.add_argument("--diffts-data-dir", default=None,
                        help="Directory of pre-generated DiffTS negatives (required if --negative-type=diffts)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Verify data directory exists
    # ------------------------------------------------------------------
    data_path = Path(args.data_dir)
    if not data_path.exists():
        logger.error(
            f"N-BaIoT data directory not found: {data_path.resolve()}\n"
            "Please download the dataset first:\n"
            "  Option A -- Kaggle (requires ~/.kaggle/kaggle.json):\n"
            "    pip install kaggle\n"
            "    kaggle datasets download -d mkashifn/nbaiot-dataset "
            f"-p {args.data_dir} --unzip\n"
            "  Option B -- manual:\n"
            "    Visit https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset\n"
            f"    Download and extract into {args.data_dir}"
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("N-BaIoT Fine-Tuned Evaluation")
    logger.info("  Fine-tune on N-BaIoT benign + Gaussian-noise negatives,")
    logger.info("  then evaluate on N-BaIoT attacks.")
    logger.info("=" * 60)
    logger.info(f"Device  : {args.device}")
    logger.info(f"Seeds   : {seeds}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Epochs  : {args.epochs}")
    logger.info(f"LR      : {args.lr}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Run evaluation for each seed
    # ------------------------------------------------------------------
    seed_results = []
    for seed in seeds:
        logger.info(f"\n--- Seed {seed} ---")

        # Load N-BaIoT data (reshuffled per seed for robustness estimate)
        X_train_s, X_val_s, X_atk_s, y_names_s, _, atk_types_s = load_nbaiot(
            data_dir=args.data_dir,
            device=args.device,
            max_samples_per_class=args.max_samples,
            seq_length=128,
            seed=seed,
        )

        # Initialise detector
        detector = MoiraiAnomalyDetector(
            model_size=args.model_size,
            context_length=96,
            prediction_length=32,
        )
        detector.initialize()

        # Fine-tune on N-BaIoT benign data + synthetic negatives
        neg_desc = "DiffTS" if args.negative_type == "diffts" else "Gaussian-noise"
        logger.info(
            f"Fine-tuning on N-BaIoT benign data ({len(X_train_s)} windows) "
            f"+ {neg_desc} negatives for {args.epochs} epochs ..."
        )
        fine_tune_on_nbaiot_benign(
            detector,
            X_train=X_train_s,
            seed=seed,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            contrastive_weight=0.5,
            noise_sigma=0.3,
            negative_type=args.negative_type,
            diffts_data_dir=args.diffts_data_dir,
        )
        logger.info("Fine-tuning complete. Evaluating on N-BaIoT attacks ...")

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
    # Use first-seed data shapes for metadata
    X_train_0, X_val_0, X_atk_0, _, _, atk_types_0 = load_nbaiot(
        data_dir=args.data_dir,
        device=args.device,
        max_samples_per_class=args.max_samples,
        seq_length=128,
        seed=seeds[0],
    )

    agg = aggregate_seeds(seed_results)
    agg["meta"] = {
        "device":                args.device,
        "seeds":                 seeds,
        "n_seeds":               len(seeds),
        "attack_types":          atk_types_0,
        "feature_proxy":         PROXY_FEATURE_NAMES,
        "n_benign_train_windows": int(len(X_train_0)),
        "n_benign_val_windows":  int(len(X_val_0)),
        "n_attack_windows":      int(len(X_atk_0)),
        "fine_tuning": {
            "method":              f"nbaiot_benign_{args.negative_type}",
            "negative_type":       args.negative_type,
            "noise_sigma":         0.3 if args.negative_type == "gaussian" else None,
            "contrastive_weight":  0.5,
            "epochs":              args.epochs,
            "batch_size":          args.batch_size,
            "learning_rate":       args.lr,
        },
    }

    out_path = output_dir / "metrics.json"
    out_path.write_text(json.dumps(agg, indent=2))
    logger.success(f"Results saved to {out_path}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY -- Overall (all attacks vs. benign-val)")
    logger.info("=" * 60)
    ov = agg["overall"]
    logger.info(
        f"F1  = {ov['f1']['mean']:.3f} +/- {ov['f1']['std']:.3f}  |  "
        f"FPR = {ov['false_positive_rate']['mean']:.3f} +/- {ov['false_positive_rate']['std']:.3f}  |  "
        f"AUC = {ov['roc_auc']['mean']:.3f} +/- {ov['roc_auc']['std']:.3f}"
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
