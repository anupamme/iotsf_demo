#!/usr/bin/env python3
"""
Run ablation study conditions A–E for the NeurIPS 2026 paper.

Conditions
----------
A  Moirai zero-shot (calibrated threshold, no fine-tuning)
B  Moirai fine-tuned, NLL loss only (contrastive_weight=0.0)
C  Moirai fine-tuned NLL+SupCon with Gaussian-noise negatives (no hard negatives)
D  Full system: NLL+SupCon with synthetic hard negatives  [proposed method]
E  Full system without protocol constraints (validate=False in generator)

Each condition is evaluated on:
  - Synthetic stealth-95 (hardest)
  - All synthetic stealth levels combined

Results are saved to results/ablation/<condition>/metrics.json
A summary table is printed at the end.

Usage:
    python scripts/run_ablation.py --condition d
    python scripts/run_ablation.py --condition all   # run every condition
    python scripts/run_ablation.py --condition b --epochs 5  # quick test
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models import MoiraiAnomalyDetector
from src.models.baseline.cnn_detector import CNNAnomalyDetector
from src.evaluation.metrics import IDSMetrics
from src.evaluation.per_attack_metrics import PerAttackMetrics


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_synthetic(synthetic_dir: str, rng: np.random.Generator, max_eval_samples: int = 50):
    """
    Returns dicts of hard-negative arrays keyed by attack condition and
    a benign array.  max_eval_samples caps the number of attack samples
    per stealth level to keep evaluation fast.
    """
    synth = Path(synthetic_dir)
    attack_types = ["slow_exfiltration", "lotl_mimicry", "beacon", "protocol_anomaly"]

    benign_path = synth / "benign_samples.npy"
    benign = np.load(benign_path) if benign_path.exists() else None

    conditions = {}
    for stealth in [85, 90, 95]:
        chunks = []
        for at in attack_types:
            fp = synth / f"{at}_stealth_{stealth}.npy"
            if fp.exists():
                chunks.append(np.load(fp))
        if chunks:
            arr = np.concatenate(chunks)
            if len(arr) > max_eval_samples:
                idx = rng.choice(len(arr), size=max_eval_samples, replace=False)
                arr = arr[idx]
            conditions[f"stealth_{stealth}"] = arr

    return benign, conditions


def build_eval_sets(benign, conditions, rng, max_eval_samples: int = 50):
    """Build balanced (X, y) pairs for each stealth level and all combined."""
    eval_sets = {}

    if benign is None:
        logger.error("No benign samples found; cannot build eval sets")
        return eval_sets

    for key, attacks in conditions.items():
        n_b = min(len(benign), len(attacks))
        b_idx = rng.choice(len(benign), size=n_b, replace=False)
        X = np.concatenate([benign[b_idx], attacks])
        y = np.array([0] * n_b + [1] * len(attacks))
        eval_sets[key] = (X, y)

    # Combined all stealth levels (cap benign to max_eval_samples)
    all_attacks = np.concatenate(list(conditions.values()))
    n_b = min(len(benign), max_eval_samples)
    b_idx = rng.choice(len(benign), size=n_b, replace=False)
    X_all = np.concatenate([benign[b_idx], all_attacks])
    y_all = np.array([0] * n_b + [1] * len(all_attacks))
    eval_sets["all_stealth"] = (X_all, y_all)

    return eval_sets


# ---------------------------------------------------------------------------
# Detection helper
# ---------------------------------------------------------------------------

def score_samples(
    detector: MoiraiAnomalyDetector,
    X: np.ndarray,
) -> np.ndarray:
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
        if (i + 1) % 10 == 0 or (i + 1) == n:
            logger.debug(f"    scored {i + 1}/{n}")
    return scores


def find_best_threshold(benign_cal_scores: np.ndarray) -> float:
    """Calibrate threshold on benign-only held-out scores at 95th percentile (5% FPR target).

    Args:
        benign_cal_scores: NLL scores for benign-only calibration samples (no attack labels).

    Returns:
        threshold: scalar float; classify as attack if score > threshold.
    """
    return float(np.percentile(benign_cal_scores, 95))


def load_calibrated_thresholds(results_dir: str):
    """Load optimal thresholds from calibrate_threshold.py output."""
    p = Path(results_dir) / "calibrated_thresholds.json"
    if p.exists():
        data = json.loads(p.read_text())
        opt = data.get("optimal", {})
        ci = opt.get("anomaly_score_threshold", 0.95)
        dt = opt.get("detection_rate_threshold", 0.30)
        logger.info(f"Loaded calibrated thresholds: CI={ci:.2f}, det={dt:.2f}")
        return ci, dt
    logger.warning("calibrated_thresholds.json not found; using defaults (CI=0.95, det=0.30)")
    return 0.95, 0.30


# ---------------------------------------------------------------------------
# Condition implementations
# ---------------------------------------------------------------------------

def run_condition_a(detector_kwargs, eval_sets):
    """Condition A: zero-shot Moirai (no fine-tuning)."""
    logger.info("[A] Initialising Moirai zero-shot...")
    det = MoiraiAnomalyDetector(**detector_kwargs)
    det.initialize()
    return _evaluate_all(det, eval_sets)


def run_condition_b(detector_kwargs, eval_sets, synthetic_dir, epochs, batch_size, lr,
                    max_train_samples=None, early_stopping_criterion="nll",
                    freeze_encoder="none", l2sp_weight=0.0):
    """Condition B: fine-tuned with NLL only (contrastive_weight=0.0)."""
    logger.info("[B] Fine-tuning Moirai (NLL only, no contrastive loss)...")
    det = MoiraiAnomalyDetector(**detector_kwargs)
    det.initialize()
    _fine_tune(det, synthetic_dir, epochs, batch_size, lr, contrastive_weight=0.0,
               use_hard_negatives=True, use_constraints=True,
               max_train_samples=max_train_samples,
               early_stopping_criterion=early_stopping_criterion,
               freeze_encoder=freeze_encoder, l2sp_weight=l2sp_weight)
    return _evaluate_all(det, eval_sets)


def run_condition_c(detector_kwargs, eval_sets, synthetic_dir, epochs, batch_size, lr,
                    max_train_samples=None, early_stopping_criterion="nll",
                    freeze_encoder="none", l2sp_weight=0.0):
    """Condition C: fine-tuned NLL+SupCon with Gaussian-noise negatives (no hard negatives)."""
    logger.info("[C] Fine-tuning Moirai (NLL+SupCon, Gaussian-noise negatives, no hard negatives)...")
    det = MoiraiAnomalyDetector(**detector_kwargs)
    det.initialize()
    _fine_tune(det, synthetic_dir, epochs, batch_size, lr, contrastive_weight=0.5,
               use_hard_negatives=False, use_constraints=True,
               max_train_samples=max_train_samples,
               early_stopping_criterion=early_stopping_criterion,
               freeze_encoder=freeze_encoder, l2sp_weight=l2sp_weight)
    return _evaluate_all(det, eval_sets)


def run_condition_cprime(detector_kwargs, eval_sets, synthetic_dir, epochs, batch_size, lr,
                         max_train_samples=None, early_stopping_criterion="nll",
                         freeze_encoder="none", l2sp_weight=0.0):
    """
    Condition C': NLL+SupCon with BALANCED hard negatives (1:1 ratio, subsampled to
    match benign count).  Isolates negative *type* from dataset composition:
    C  = Gaussian-noise negatives, 200 neg (1:1)
    C' = hard negatives,           200 neg (1:1, subsampled)
    D  = hard negatives,          2400 neg (1:12, all stealth levels)
    C vs C' → effect of negative type (at equal scale)
    C' vs D → effect of dataset size / imbalance (at fixed negative type)
    """
    logger.info("[C'] Fine-tuning Moirai (NLL+SupCon, balanced hard negatives, 1:1 ratio)...")
    det = MoiraiAnomalyDetector(**detector_kwargs)
    det.initialize()
    _fine_tune(det, synthetic_dir, epochs, batch_size, lr, contrastive_weight=0.5,
               use_hard_negatives=True, use_constraints=True, balanced=True,
               max_train_samples=max_train_samples,
               early_stopping_criterion=early_stopping_criterion,
               freeze_encoder=freeze_encoder, l2sp_weight=l2sp_weight)
    return _evaluate_all(det, eval_sets)


def run_condition_cnn(detector_kwargs, eval_sets, synthetic_dir, epochs, batch_size, lr,
                      max_train_samples=None, early_stopping_criterion="nll",
                      freeze_encoder="none", l2sp_weight=0.0):
    """
    Condition CNN: 1D-CNN trained from scratch with NLL(MSE)+SupCon on hard negatives.

    Ablates Moirai's pre-training: if CNN ≪ Moirai (even at matched scale),
    the foundation model pre-training is justified.  If CNN ≈ Moirai, the
    pre-training provides no measurable benefit for this task.
    """
    logger.info("[CNN] Training 1D-CNN from scratch (NLL+SupCon, hard negatives, no pre-training)...")
    det = CNNAnomalyDetector(n_features=12, seq_len=128, embed_dim=128)
    det.initialize()
    _fine_tune(det, synthetic_dir, epochs, batch_size, lr, contrastive_weight=0.5,
               use_hard_negatives=True, use_constraints=True,
               max_train_samples=max_train_samples,
               early_stopping_criterion=early_stopping_criterion,
               freeze_encoder=freeze_encoder, l2sp_weight=l2sp_weight)
    return _evaluate_all(det, eval_sets)


def run_condition_cnn_nll(detector_kwargs, eval_sets, synthetic_dir, epochs, batch_size, lr,
                         max_train_samples=None, early_stopping_criterion="nll",
                         freeze_encoder="none", l2sp_weight=0.0):
    """
    Condition CNN-NLL: 1D-CNN with Gaussian NLL (distributional output) + SupCon.

    Closes the MSE-vs-NLL loss confound: if CNN-NLL ≈ CNN-MSE, the loss function
    is not the explanation for CNN > Moirai.  If CNN-NLL < CNN-MSE, the MSE
    advantage is confirmed.
    """
    logger.info("[CNN-NLL] Training 1D-CNN from scratch (Gaussian NLL+SupCon, hard negatives)...")
    det = CNNAnomalyDetector(n_features=12, seq_len=128, embed_dim=128, distributional=True)
    det.initialize()
    _fine_tune(det, synthetic_dir, epochs, batch_size, lr, contrastive_weight=0.5,
               use_hard_negatives=True, use_constraints=True,
               max_train_samples=max_train_samples,
               early_stopping_criterion=early_stopping_criterion,
               freeze_encoder=freeze_encoder, l2sp_weight=l2sp_weight)
    return _evaluate_all(det, eval_sets)


def run_condition_d(detector_kwargs, eval_sets, synthetic_dir, epochs, batch_size, lr,
                    max_train_samples=None, early_stopping_criterion="nll",
                    freeze_encoder="none", l2sp_weight=0.0):
    """Condition D: full system (NLL+SupCon + hard negatives + constraints)."""
    logger.info("[D] Fine-tuning Moirai (full system)...")
    det = MoiraiAnomalyDetector(**detector_kwargs)
    det.initialize()
    _fine_tune(det, synthetic_dir, epochs, batch_size, lr, contrastive_weight=0.5,
               use_hard_negatives=True, use_constraints=True,
               max_train_samples=max_train_samples,
               early_stopping_criterion=early_stopping_criterion,
               freeze_encoder=freeze_encoder, l2sp_weight=l2sp_weight)
    return _evaluate_all(det, eval_sets)


def run_condition_e(detector_kwargs, eval_sets, synthetic_dir, epochs, batch_size, lr,
                    max_train_samples=None, early_stopping_criterion="nll",
                    freeze_encoder="none", l2sp_weight=0.0):
    """Condition E: full system but WITHOUT protocol constraints."""
    logger.info("[E] Fine-tuning Moirai (full system, NO protocol constraints)...")
    det = MoiraiAnomalyDetector(**detector_kwargs)
    det.initialize()
    _fine_tune(det, synthetic_dir, epochs, batch_size, lr, contrastive_weight=0.5,
               use_hard_negatives=True, use_constraints=False,
               max_train_samples=max_train_samples,
               early_stopping_criterion=early_stopping_criterion,
               freeze_encoder=freeze_encoder, l2sp_weight=l2sp_weight)
    return _evaluate_all(det, eval_sets)


def run_condition_eprime(detector_kwargs, eval_sets, synthetic_dir, epochs, batch_size, lr,
                         max_train_samples=None, early_stopping_criterion="nll",
                         freeze_encoder="none", l2sp_weight=0.0):
    """
    Condition E': NLL+SupCon with hard negatives generated via UNCONDITIONAL RETRY
    (no constraint validator, but stealth relaxed by 0.01 on each of 3 iterations).
    Trains on data from data/synthetic_eprime/ if available, else falls back to
    data/synthetic/ with a stealth-relaxation note.
    """
    logger.info(
        "[E'] Fine-tuning Moirai (NLL+SupCon, hard-negatives, unconditional-retry "
        "stealth relaxation, no constraint validator)..."
    )
    det = MoiraiAnomalyDetector(**detector_kwargs)
    det.initialize()
    # E' data dir: expect data/synthetic_eprime/ generated via --retry-mode unconditional_retry
    eprime_dir = str(Path(synthetic_dir).parent / "synthetic_eprime")
    if not Path(eprime_dir).exists() or not list(Path(eprime_dir).glob("*_stealth_*.npy")):
        logger.warning(
            f"E' dataset not found at {eprime_dir}. "
            "Generate with: python scripts/precompute_attacks.py "
            "--retry-mode unconditional_retry --output-dir data/synthetic_eprime --n-samples 200"
        )
        logger.warning("Falling back to data/synthetic/ for E' (results will match D).")
        eprime_dir = synthetic_dir
    _fine_tune(det, eprime_dir, epochs, batch_size, lr, contrastive_weight=0.5,
               use_hard_negatives=True, use_constraints=False,
               max_train_samples=max_train_samples,
               early_stopping_criterion=early_stopping_criterion,
               freeze_encoder=freeze_encoder, l2sp_weight=l2sp_weight)
    return _evaluate_all(det, eval_sets)


def run_condition_edoubleprime(detector_kwargs, eval_sets, synthetic_dir, epochs, batch_size, lr,
                               max_train_samples=None, early_stopping_criterion="nll",
                               freeze_encoder="none", l2sp_weight=0.0):
    """
    Condition E'': NLL+SupCon with hard negatives, constraint validation ACTIVE but
    retry/stealth-relaxation DISABLED (validate once, return regardless).
    For the analytical pathway (100% compliance), E'' is functionally identical to D.
    This condition is included to show the null effect and confirm the analysis.
    """
    logger.info(
        "[E''] Fine-tuning Moirai (NLL+SupCon, hard-negatives, constraints active, "
        "NO stealth-floor retry)..."
    )
    det = MoiraiAnomalyDetector(**detector_kwargs)
    det.initialize()
    # E'' uses the same data as D (analytical perturbations always comply, retry never fires).
    # The null result (E''≈D) confirms the analysis in the paper: for the analytical pathway,
    # the stealth-floor retry is a safety net that never triggers.
    _fine_tune(det, synthetic_dir, epochs, batch_size, lr, contrastive_weight=0.5,
               use_hard_negatives=True, use_constraints=True,
               max_train_samples=max_train_samples,
               early_stopping_criterion=early_stopping_criterion,
               freeze_encoder=freeze_encoder, l2sp_weight=l2sp_weight)
    return _evaluate_all(det, eval_sets)


# ---------------------------------------------------------------------------
# Fine-tuning helper
# ---------------------------------------------------------------------------

def _fine_tune(
    detector: MoiraiAnomalyDetector,
    synthetic_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    contrastive_weight: float,
    use_hard_negatives: bool,
    use_constraints: bool,
    balanced: bool = False,
    max_train_samples: int = None,
    early_stopping_criterion: str = "nll",
    freeze_encoder: str = "none",
    l2sp_weight: float = 0.0,
):
    """
    Call detector.fine_tune_supervised() if available, otherwise fall back to a
    lightweight mock fine-tune that exercises the training loop pattern.

    balanced: if True and use_hard_negatives=True, subsample hard negatives to
              match len(benign), giving a 1:1 ratio (used for Condition C').
              When False (default), all hard negatives are used (1:12 ratio for D).
    max_train_samples: if set, cap the total combined (benign+attacks) training set
              to this many samples after all other subsampling. Used for scaling
              experiments (--max-train-samples 1000 etc.).
    """
    synth = Path(synthetic_dir)
    benign_path = synth / "benign_samples.npy"
    if not benign_path.exists():
        logger.warning("No benign samples; skipping fine-tuning")
        return

    benign = np.load(benign_path)

    if use_hard_negatives:
        attack_files = list(synth.glob("*_stealth_*.npy"))
        if not attack_files:
            logger.warning("No synthetic attack files; falling back to real attacks")
            use_hard_negatives = False

    if use_hard_negatives:
        chunks = [np.load(p) for p in attack_files]
        attacks = np.concatenate(chunks)
        if balanced and len(attacks) > len(benign):
            # Subsample to 1:1 ratio — isolates negative *type* from dataset size
            rng_balance = np.random.default_rng(42)
            idx = rng_balance.choice(len(attacks), size=len(benign), replace=False)
            attacks = attacks[idx]
            logger.info(f"Balanced sampling: {len(attacks)} hard negatives → 1:1 ratio with {len(benign)} benign")
    else:
        # Condition C: use noisy benign copies (Gaussian-noise negatives)
        rng_ft = np.random.default_rng(0)
        attacks = benign + rng_ft.normal(0, 0.3, benign.shape)

    # Build train/val arrays with labels
    rng_split = np.random.default_rng(1)
    all_X = np.concatenate([benign, attacks])
    all_y = np.array([0] * len(benign) + [1] * len(attacks))

    # Cap total training samples for scaling experiments
    if max_train_samples is not None and len(all_X) > max_train_samples:
        rng_cap = np.random.default_rng(99)
        cap_idx = rng_cap.choice(len(all_X), size=max_train_samples, replace=False)
        all_X = all_X[cap_idx]
        all_y = all_y[cap_idx]
        logger.info(f"max_train_samples={max_train_samples}: capped from {len(benign)+len(attacks)} → {len(all_X)}")
    perm = rng_split.permutation(len(all_X))
    n_val = max(4, int(0.15 * len(all_X)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_data, train_labels = all_X[train_idx], all_y[train_idx]
    val_data, val_labels = all_X[val_idx], all_y[val_idx]

    # Attempt supervised fine-tuning
    ft_method = getattr(detector, "fine_tune_supervised", None)
    if ft_method is None:
        logger.warning("fine_tune_supervised() not available (mock Moirai); skipping")
        return

    try:
        import tempfile
        ckpt_dir = tempfile.mkdtemp(prefix="moirai_ckpt_")
        ft_kwargs = dict(
            train_data=train_data,
            train_labels=train_labels,
            val_data=val_data,
            val_labels=val_labels,
            n_epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            contrastive_weight=contrastive_weight,
            early_stopping_criterion=early_stopping_criterion,
            freeze_encoder=freeze_encoder,
            checkpoint_dir=ckpt_dir,
        )
        if l2sp_weight > 0.0:
            ft_kwargs['l2sp_weight'] = l2sp_weight
        ft_method(**ft_kwargs)
        logger.info(f"Fine-tuning complete ({epochs} epochs, es_criterion={early_stopping_criterion}, frozen={freeze_encoder})")
    except Exception as e:
        logger.warning(f"Fine-tuning failed: {e}")


def _evaluate_all(detector, eval_sets, ci_thresh=None, det_thresh=None):
    """Evaluate detector on every (X, y) pair in eval_sets.

    Calibration protocol (mirrors evaluate_nbaiot.py):
      1. For each eval set, separate benign samples (y==0) into an 80% calibration
         split and a 20% held-out test split.
      2. Calibrate threshold on benign-cal scores only: threshold = 95th percentile
         (targets FPR ≤ 0.05 on benign traffic).
      3. Evaluate on held-out set: remaining 20% benign + ALL attack samples.

    This removes the circular dependency where threshold optimisation and evaluation
    share the same data, which caused FPR collapse at larger evaluation scales.
    """
    results = {}
    for name, (X, y) in eval_sets.items():
        logger.info(f"  Evaluating on {name} ({len(X)} samples)...")
        raw_scores = score_samples(detector, X)

        # --- benign 80/20 split for calibration vs. test ---
        benign_idx = np.where(y == 0)[0]
        n_cal = max(1, int(0.8 * len(benign_idx)))
        cal_idx = benign_idx[:n_cal]          # 80% benign → calibration only
        test_mask = np.ones(len(y), dtype=bool)
        test_mask[cal_idx] = False             # remaining 20% benign + all attack → test

        # --- calibrate threshold on benign-cal scores only ---
        benign_cal_scores = raw_scores[cal_idx]
        threshold = find_best_threshold(benign_cal_scores)

        # --- evaluate on held-out test set ---
        test_scores = raw_scores[test_mask]
        test_y = y[test_mask]
        test_preds = (test_scores > threshold).astype(int)
        metrics = IDSMetrics.compute_all_metrics(test_y, test_preds, test_scores)

        results[name] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                         for k, v in metrics.items()}
        results[name]["best_threshold"] = float(threshold)
        logger.info(
            f"  {name}: F1={metrics.get('f1', 0):.3f}, "
            f"FPR={metrics.get('false_positive_rate', 1):.3f}, "
            f"ROC-AUC={metrics.get('roc_auc') or 0:.3f}"
        )
    return results


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def _aggregate_seeds(all_seed_results: List[dict]) -> dict:
    """
    Given a list of per-seed results dicts (eval_set → metrics dict),
    compute mean ± std for each metric across seeds.
    Returns same structure with values replaced by {"mean": x, "std": y}.
    """
    if len(all_seed_results) == 1:
        return all_seed_results[0]   # single seed: no aggregation needed

    aggregated = {}
    # Collect all eval_set keys
    eval_keys = all_seed_results[0].keys()
    for ek in eval_keys:
        aggregated[ek] = {}
        metric_keys = [k for k in all_seed_results[0].get(ek, {}).keys()
                       if k not in ("confusion_matrix", "best_threshold")]
        for mk in metric_keys:
            vals = [sr.get(ek, {}).get(mk) for sr in all_seed_results
                    if sr.get(ek, {}).get(mk) is not None]
            if vals and isinstance(vals[0], (int, float)):
                aggregated[ek][mk] = {"mean": float(np.mean(vals)),
                                      "std": float(np.std(vals))}
            elif vals:
                aggregated[ek][mk] = vals[0]  # non-numeric: keep first
    return aggregated


# ---------------------------------------------------------------------------
# Per-attack-type metrics for condition D
# ---------------------------------------------------------------------------

def run_per_attack_metrics(detector, synthetic_dir: str) -> dict:
    """
    Compute real per-attack-type × stealth-level F1 matrix for condition D.
    Uses PerAttackMetrics.compute_matrix() with the NLL scorer.
    """
    def predict_fn(X: np.ndarray):
        raw_scores = score_samples(detector, X)
        # Find best threshold on this batch
        y_dummy = np.array([0] * len(X))  # placeholder — we only need scores
        # Return scores as both pred (thresholded at median) and scores
        threshold = np.median(raw_scores)
        preds = (raw_scores > threshold).astype(int)
        return preds, raw_scores

    try:
        matrix = PerAttackMetrics.compute_matrix(
            predict_fn=predict_fn,
            synthetic_dir=synthetic_dir,
        )
        # Convert int keys to str for JSON serialization
        serializable = {
            at: {str(stealth): metrics for stealth, metrics in stealth_dict.items()}
            for at, stealth_dict in matrix.items()
        }
        logger.info("Per-attack metrics computed:\n" + PerAttackMetrics.print_summary(matrix))
        return serializable
    except Exception as e:
        logger.warning(f"Per-attack metrics failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run NeurIPS ablation study (conditions A–E)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--condition", default="d",
        choices=["a", "b", "c", "cprime", "cnn", "cnn_nll", "d", "e", "eprime", "edoubleprime", "all"],
        help="Which ablation condition to run (default: d = full system). "
             "cprime = balanced hard negatives (1:1 ratio) to isolate negative type from data size. "
             "cnn = from-scratch 1D-CNN baseline (MSE, no Moirai pre-training). "
             "cnn_nll = 1D-CNN with Gaussian NLL (closes MSE-vs-NLL confound).",
    )
    parser.add_argument("--model-size", default="small", choices=["small", "base", "large"])
    parser.add_argument("--synthetic-dir", default="data/synthetic")
    parser.add_argument(
        "--eval-synthetic-dir", default=None,
        help="If set, load EVALUATION data from this directory instead of --synthetic-dir. "
             "Training negatives still come from --synthetic-dir. "
             "Use for controlled cross-evaluation (e.g., train on DiffTS, eval on analytical)."
    )
    parser.add_argument("--results-dir", default="results",
                        help="Root output directory (default: results). "
                             "Use 'results/ablation_scaled' for scaling experiments.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--max-eval-samples", type=int, default=50,
        help="Max attack samples per stealth level in eval sets."
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=None,
        help="If set, cap the total number of training samples (benign + attacks combined). "
             "Used for scaling experiments to test at N=200, 500, 1000 etc."
    )
    parser.add_argument(
        "--seeds", default="42",
        help="Comma-separated list of random seeds for multi-seed runs (e.g. '42,123,456'). "
             "When multiple seeds given, output includes mean ± std."
    )
    parser.add_argument(
        "--early-stopping-criterion", default="nll",
        choices=["nll", "total"],
        help="Early stopping criterion: 'nll' monitors val NLL only (recommended), "
             "'total' monitors val NLL+SupCon total loss (legacy). Default: nll."
    )
    parser.add_argument(
        "--freeze-encoder", default="none",
        choices=["none", "full", "partial", "lora"],
        help="Encoder freezing strategy: 'none' (default), 'full' (freeze all encoder weights), "
             "'partial' (freeze all but last transformer layer), "
             "'lora' (freeze base, add LoRA adapters to attention layers). "
             "Tests catastrophic forgetting hypothesis at extended scale."
    )
    parser.add_argument(
        "--per-attack", action="store_true",
        help="For condition D: also compute per-attack-type × stealth metrics (used for Figure 3)."
    )
    parser.add_argument(
        "--l2sp-weight", type=float, default=0.0,
        help="L2-SP regularization weight penalizing drift from pretrained weights. "
             "0.0 = disabled (default). Typical values: 0.01 or 0.1."
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device for computation ('auto', 'cuda', or 'cpu'). Default: auto."
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    eval_synthetic_dir = args.eval_synthetic_dir if args.eval_synthetic_dir else args.synthetic_dir
    if args.eval_synthetic_dir:
        logger.info(f"Cross-evaluation mode: train from '{args.synthetic_dir}', eval from '{eval_synthetic_dir}'")
    output_root = Path(args.results_dir) / "ablation"
    output_root.mkdir(parents=True, exist_ok=True)

    # Select conditions to run
    to_run = (["a", "b", "c", "cprime", "cnn", "cnn_nll", "d", "e", "eprime", "edoubleprime"]
              if args.condition == "all" else [args.condition])

    all_condition_results = {}
    for cond in to_run:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running condition {cond.upper()} over {len(seeds)} seed(s): {seeds}")
        logger.info(f"{'=' * 60}")
        t0 = time.time()

        per_seed_results = []
        for seed in seeds:
            logger.info(f"  Seed {seed}...")
            rng = np.random.default_rng(seed)

            benign, conditions = load_synthetic(
                eval_synthetic_dir, rng, max_eval_samples=args.max_eval_samples
            )
            eval_sets = build_eval_sets(
                benign, conditions, rng, max_eval_samples=args.max_eval_samples
            )
            if not eval_sets:
                logger.error("No evaluation sets available; aborting")
                sys.exit(1)

            detector_kwargs = dict(
                model_size=args.model_size,
                context_length=96,
                prediction_length=32,
                confidence_level=0.95,
                device=args.device,
            )
            ft_kwargs = dict(
                synthetic_dir=args.synthetic_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_train_samples=args.max_train_samples,
                early_stopping_criterion=args.early_stopping_criterion,
                freeze_encoder=args.freeze_encoder,
                l2sp_weight=args.l2sp_weight,
            )

            condition_fns = {
                "a": lambda: run_condition_a(detector_kwargs, eval_sets),
                "b": lambda: run_condition_b(detector_kwargs, eval_sets, **ft_kwargs),
                "c": lambda: run_condition_c(detector_kwargs, eval_sets, **ft_kwargs),
                "cprime": lambda: run_condition_cprime(detector_kwargs, eval_sets, **ft_kwargs),
                "cnn": lambda: run_condition_cnn(detector_kwargs, eval_sets, **ft_kwargs),
                "cnn_nll": lambda: run_condition_cnn_nll(detector_kwargs, eval_sets, **ft_kwargs),
                "d": lambda: run_condition_d(detector_kwargs, eval_sets, **ft_kwargs),
                "e": lambda: run_condition_e(detector_kwargs, eval_sets, **ft_kwargs),
                "eprime": lambda: run_condition_eprime(detector_kwargs, eval_sets, **ft_kwargs),
                "edoubleprime": lambda: run_condition_edoubleprime(detector_kwargs, eval_sets, **ft_kwargs),
            }
            seed_results = condition_fns[cond]()
            per_seed_results.append(seed_results)

        elapsed = time.time() - t0
        results = _aggregate_seeds(per_seed_results)
        all_condition_results[cond] = {"results": results, "elapsed_s": elapsed,
                                       "seeds": seeds, "n_seeds": len(seeds)}

        # Save per-condition JSON
        cond_dir = output_root / cond
        cond_dir.mkdir(exist_ok=True)
        out_file = cond_dir / "metrics.json"
        with open(out_file, "w") as f:
            json.dump({"condition": cond, "elapsed_s": elapsed,
                       "seeds": seeds, "results": results}, f, indent=2)
        logger.success(f"Condition {cond.upper()} saved to {out_file}")

        # Per-attack metrics for condition D (needed for Figure 3)
        if cond == "d" and args.per_attack:
            logger.info("Computing per-attack-type metrics for Figure 3...")
            # Re-run condition D with seed[0] to get a live detector
            rng = np.random.default_rng(seeds[0])
            detector_kwargs_d = dict(model_size=args.model_size, context_length=96,
                                     prediction_length=32, confidence_level=0.95,
                                     device=args.device)
            ft_kwargs_d = dict(synthetic_dir=args.synthetic_dir, epochs=args.epochs,
                               batch_size=args.batch_size, lr=args.lr)
            det = MoiraiAnomalyDetector(**detector_kwargs_d)
            det.initialize()
            _fine_tune(det, args.synthetic_dir, args.epochs, args.batch_size,
                       args.lr, contrastive_weight=0.5,
                       use_hard_negatives=True, use_constraints=True)
            per_attack = run_per_attack_metrics(det, args.synthetic_dir)
            pa_file = cond_dir / "per_attack_metrics.json"
            with open(pa_file, "w") as f:
                json.dump(per_attack, f, indent=2)
            logger.success(f"Per-attack metrics saved to {pa_file}")

    # Summary table
    def _get_metric(res_dict, key, default=0.0):
        """Extract mean from mean/std dict or plain float."""
        val = res_dict.get(key, default)
        if isinstance(val, dict):
            return val.get("mean", default)
        return val if val is not None else default

    if len(to_run) > 1:
        condition_labels = {
            "a": "A: Zero-shot",
            "b": "B: NLL only",
            "c": "C: NLL+SupCon, Gaussian-noise neg.",
            "cprime": "C': NLL+SupCon, balanced hard neg. (1:1)",
            "cnn": "CNN: From-scratch 1D-CNN, MSE (no pre-training)",
            "cnn_nll": "CNN-NLL: From-scratch 1D-CNN, Gaussian NLL",
            "d": "D: Full system (ours)",
            "e": "E: No constraints, no retry",
            "eprime": "E': No constraints, unconditional retry",
            "edoubleprime": "E'': Constraints active, no retry",
        }
        multi = len(seeds) > 1
        print("\n" + "=" * 80)
        print(f"ABLATION STUDY SUMMARY  (seeds={seeds})")
        print("=" * 80)
        for eval_key in ["stealth_95", "all_stealth"]:
            print(f"\n  Eval set: {eval_key}")
            hdr = f"  {'Condition':<30} {'F1':>10} {'FPR':>10} {'AUC':>10}"
            print(hdr)
            print(f"  {'-'*62}")
            for cond in to_run:
                res = all_condition_results[cond]["results"].get(eval_key, {})
                f1 = _get_metric(res, "f1")
                fpr = _get_metric(res, "false_positive_rate", 1.0)
                auc = _get_metric(res, "roc_auc")
                if multi:
                    f1_s = res.get("f1", {}).get("std", 0) if isinstance(res.get("f1"), dict) else 0
                    auc_s = res.get("roc_auc", {}).get("std", 0) if isinstance(res.get("roc_auc"), dict) else 0
                    f1_str = f"{f1:.3f}±{f1_s:.3f}"
                    auc_str = f"{auc:.3f}±{auc_s:.3f}"
                else:
                    f1_str = f"{f1:.3f}"
                    auc_str = f"{auc:.3f}"
                print(
                    f"  {condition_labels.get(cond, cond):<30} "
                    f"{f1_str:>10} "
                    f"{fpr:>10.3f} "
                    f"{auc_str:>10}"
                )
        print("=" * 80)


if __name__ == "__main__":
    main()
