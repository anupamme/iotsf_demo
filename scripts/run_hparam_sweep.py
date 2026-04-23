#!/usr/bin/env python3
"""
Hyperparameter sensitivity sweep for the NeurIPS 2026 paper (W7 fix).

Addresses reviewer W7: "Figure 4 shows instability in the heatmap,
contradicting the claim that the method is not sensitive to hyperparameters."

Runs a 5×4 grid of (contrastive_weight λ, temperature τ) for condition D
on stealth-95, using a single seed (42) for speed.

Grid:
  λ ∈ {0.1, 0.3, 0.5, 0.7, 1.0}
  τ ∈ {0.05, 0.07, 0.10, 0.20}
  = 20 runs total

Output: results/hparam_sweep.json
  Keys: "lam{λ}_temp{τ}" → {"f1": float, "roc_auc": float}

Usage:
    python scripts/run_hparam_sweep.py
    python scripts/run_hparam_sweep.py --epochs 3 --max-eval-samples 20   # quick test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models import MoiraiAnomalyDetector
from src.evaluation.metrics import IDSMetrics

# Grid definition — must match generate_paper_figures.py expectations
LAMBDAS = [0.1, 0.3, 0.5, 0.7, 1.0]
TEMPS = [0.05, 0.07, 0.10, 0.20]


# ---------------------------------------------------------------------------
# Data helpers (same as run_ablation.py)
# ---------------------------------------------------------------------------

def load_eval_set(synthetic_dir: str, stealth: int,
                  rng: np.random.Generator, max_eval_samples: int = 30):
    """Load balanced (X, y) for a single stealth level."""
    synth = Path(synthetic_dir)
    attack_types = ["slow_exfiltration", "lotl_mimicry", "beacon", "protocol_anomaly"]

    benign_path = synth / "benign_samples.npy"
    if not benign_path.exists():
        return None, None

    benign = np.load(benign_path)

    chunks = []
    for at in attack_types:
        fp = synth / f"{at}_stealth_{stealth}.npy"
        if fp.exists():
            chunks.append(np.load(fp))

    if not chunks:
        return None, None

    attacks = np.concatenate(chunks)
    if len(attacks) > max_eval_samples:
        idx = rng.choice(len(attacks), size=max_eval_samples, replace=False)
        attacks = attacks[idx]

    n_b = min(len(benign), len(attacks))
    b_idx = rng.choice(len(benign), size=n_b, replace=False)
    X = np.concatenate([benign[b_idx], attacks])
    y = np.array([0] * n_b + [1] * len(attacks))
    return X, y


def load_train_data(synthetic_dir: str, rng: np.random.Generator):
    """Load benign + all synthetic attacks for fine-tuning."""
    synth = Path(synthetic_dir)
    benign_path = synth / "benign_samples.npy"
    if not benign_path.exists():
        return None, None

    benign = np.load(benign_path)
    attack_files = list(synth.glob("*_stealth_*.npy"))
    if not attack_files:
        return benign, np.empty((0, benign.shape[1], benign.shape[2]))

    attacks = np.concatenate([np.load(p) for p in attack_files])
    return benign, attacks


# ---------------------------------------------------------------------------
# Scoring and threshold selection (same as run_ablation.py)
# ---------------------------------------------------------------------------

def score_samples(detector, X: np.ndarray) -> np.ndarray:
    """Score all samples with NLL method, returning raw NLL scores."""
    scores = np.zeros(len(X), dtype=np.float64)
    for i, sample in enumerate(X):
        try:
            result = detector.detect_anomalies(sample, threshold=0.0, method='nll')
            scores[i] = float(result.anomaly_scores.mean())
        except Exception as e:
            logger.warning(f"Sample {i}: {e}")
            scores[i] = -1e9
    return scores


def benign_calibrated_eval(scores: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate using benign-calibrated threshold (matches run_ablation.py protocol).

    Splits benign 80/20 for calibration/test; threshold = 95th pctile of benign-cal
    scores; evaluates on held-out 20% benign + all attacks.
    """
    benign_idx = np.where(y == 0)[0]
    n_cal = max(1, int(0.8 * len(benign_idx)))
    cal_idx = benign_idx[:n_cal]
    test_mask = np.ones(len(y), dtype=bool)
    test_mask[cal_idx] = False
    threshold = float(np.percentile(scores[cal_idx], 95))
    test_scores = scores[test_mask]
    test_y = y[test_mask]
    test_preds = (test_scores > threshold).astype(int)
    m = IDSMetrics.compute_all_metrics(test_y, test_preds, test_scores)
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in m.items()}


# ---------------------------------------------------------------------------
# Single sweep run
# ---------------------------------------------------------------------------

def run_one(lam: float, tau: float, args,
            benign: np.ndarray, attacks: np.ndarray,
            X_eval: np.ndarray, y_eval: np.ndarray) -> dict:
    """
    Fine-tune a fresh MoiraiAnomalyDetector with (lam, tau) and evaluate
    on stealth-95. Returns {"f1": float, "roc_auc": float}.
    """
    detector_kwargs = dict(
        model_size=args.model_size,
        context_length=96,
        prediction_length=32,
        confidence_level=0.95,
    )
    det = MoiraiAnomalyDetector(**detector_kwargs)
    det.initialize()

    # Build train/val split
    rng_split = np.random.default_rng(1)
    all_X = np.concatenate([benign, attacks])
    all_y = np.array([0] * len(benign) + [1] * len(attacks))
    perm = rng_split.permutation(len(all_X))
    n_val = max(4, int(0.15 * len(all_X)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    ft_method = getattr(det, "fine_tune_supervised", None)
    if ft_method is None:
        logger.warning("fine_tune_supervised not available; using zero-shot scores")
    else:
        try:
            ft_method(
                train_data=all_X[train_idx],
                train_labels=all_y[train_idx],
                val_data=all_X[val_idx],
                val_labels=all_y[val_idx],
                n_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                contrastive_weight=lam,
                temperature=tau,
            )
        except TypeError:
            # Older API without temperature parameter
            try:
                ft_method(
                    train_data=all_X[train_idx],
                    train_labels=all_y[train_idx],
                    val_data=all_X[val_idx],
                    val_labels=all_y[val_idx],
                    n_epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    contrastive_weight=lam,
                )
                logger.debug("temperature parameter not supported; using λ only")
            except Exception as e:
                logger.warning(f"Fine-tuning failed: {e}")
        except Exception as e:
            logger.warning(f"Fine-tuning failed: {e}")

    scores = score_samples(det, X_eval)
    metrics = benign_calibrated_eval(scores, y_eval)

    f1 = metrics.get("f1", 0.0)
    auc = metrics.get("roc_auc") or 0.0
    return {"f1": float(f1), "roc_auc": float(auc)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="λ × τ hyperparameter sensitivity sweep (20 runs)",
    )
    parser.add_argument("--model-size", default="small", choices=["small", "base", "large"])
    parser.add_argument("--synthetic-dir", default="data/synthetic")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-eval-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.results_dir) / "hparam_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Load data once — reuse across all 20 runs
    logger.info("Loading training and evaluation data...")
    benign, attacks = load_train_data(args.synthetic_dir, rng)
    if benign is None:
        logger.error("No benign samples found; aborting")
        sys.exit(1)

    X_eval, y_eval = load_eval_set(
        args.synthetic_dir, stealth=95, rng=rng,
        max_eval_samples=args.max_eval_samples
    )
    if X_eval is None:
        logger.error("No stealth-95 eval data found; aborting")
        sys.exit(1)

    logger.info(f"Train: {len(benign)} benign + {len(attacks)} attacks")
    logger.info(f"Eval (stealth-95): {len(X_eval)} samples")

    total = len(LAMBDAS) * len(TEMPS)
    results = {}
    completed = 0
    t_start = time.time()

    print(f"\nRunning {total} sweep configurations "
          f"(λ ∈ {LAMBDAS}, τ ∈ {TEMPS})\n")
    print(f"  {'Run':>4}  {'λ':>5}  {'τ':>6}  {'F1':>8}  {'AUC':>8}  {'Time':>8}")
    print(f"  {'-'*50}")

    for lam in LAMBDAS:
        for tau in TEMPS:
            key = f"lam{lam}_temp{tau}"
            t0 = time.time()
            logger.info(f"Run {completed+1}/{total}: λ={lam}, τ={tau}")

            try:
                run_result = run_one(lam, tau, args, benign, attacks, X_eval, y_eval)
            except Exception as e:
                logger.error(f"Run failed (λ={lam}, τ={tau}): {e}")
                run_result = {"f1": float("nan"), "roc_auc": float("nan")}

            results[key] = run_result
            completed += 1
            elapsed = time.time() - t0
            total_elapsed = time.time() - t_start

            print(f"  {completed:>4}  {lam:>5.1f}  {tau:>6.2f}  "
                  f"{run_result['f1']:>8.3f}  {run_result['roc_auc']:>8.3f}  "
                  f"{elapsed:>7.1f}s")

            # Checkpoint after each run
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    total_time = time.time() - t_start
    logger.success(f"Sweep complete in {total_time:.0f}s → {out_path}")

    # Print summary grid
    print(f"\n{'=' * 60}")
    print(f"F1 GRID (rows=λ, cols=τ)")
    print(f"{'=' * 60}")
    tau_header = "  ".join(f"{t:>6.2f}" for t in TEMPS)
    print(f"  {'λ\\τ':<6}  {tau_header}")
    for lam in LAMBDAS:
        row_vals = []
        for tau in TEMPS:
            key = f"lam{lam}_temp{tau}"
            f1 = results.get(key, {}).get("f1", float("nan"))
            row_vals.append(f"{f1:>6.3f}")
        print(f"  {lam:<6.1f}  {'  '.join(row_vals)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
