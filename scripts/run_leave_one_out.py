#!/usr/bin/env python3
"""
Leave-one-out generalization experiment for the NeurIPS 2026 paper.

Addresses reviewer concern W1/W2: "the paper never demonstrates that DiffIDS
generalises to attack patterns not seen during training."

For each of the 4 attack types, we:
  1. Train DiffIDS (full system, condition D) on the remaining 3 attack types
  2. Evaluate on the held-out type at all 3 stealth levels
  3. Compare against zero-shot Moirai (condition A) as baseline

If the held-out F1 exceeds zero-shot F1, the model has learned transferable
representations, not just memorized training patterns.

Usage:
    python scripts/run_leave_one_out.py --held-out beacon
    python scripts/run_leave_one_out.py --held-out all   # runs all 4 folds
    python scripts/run_leave_one_out.py --held-out all --max-eval-samples 30 --epochs 5
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

ATTACK_TYPES = ["slow_exfiltration", "lotl_mimicry", "beacon", "protocol_anomaly"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data_excluding(synthetic_dir: str, held_out: str,
                        rng: np.random.Generator, max_train_samples: int = 200):
    """
    Load benign + attack data, EXCLUDING held_out attack type from the attack set.

    Returns
    -------
    benign       : np.ndarray (n, T, F)
    train_attacks: np.ndarray attacks from the 3 non-held-out types
    """
    synth = Path(synthetic_dir)
    benign_path = synth / "benign_samples.npy"
    benign = np.load(benign_path) if benign_path.exists() else None

    chunks = []
    for at in ATTACK_TYPES:
        if at == held_out:
            continue
        for stealth in [85, 90, 95]:
            fp = synth / f"{at}_stealth_{stealth}.npy"
            if fp.exists():
                chunks.append(np.load(fp))

    train_attacks = np.concatenate(chunks) if chunks else np.empty((0, 128, 12))
    if len(train_attacks) > max_train_samples:
        idx = rng.choice(len(train_attacks), size=max_train_samples, replace=False)
        train_attacks = train_attacks[idx]

    logger.info(f"Training data (excl. {held_out}): {len(benign) if benign is not None else 0} benign, "
                f"{len(train_attacks)} attacks from {len(ATTACK_TYPES) - 1} types")
    return benign, train_attacks


def load_held_out_eval(synthetic_dir: str, held_out: str,
                       benign: np.ndarray, rng: np.random.Generator,
                       max_eval_samples: int = 30):
    """
    Build eval sets for the held-out attack type at each stealth level.

    Returns dict: {stealth_key → (X, y)}
    """
    synth = Path(synthetic_dir)
    eval_sets = {}
    for stealth in [85, 90, 95]:
        fp = synth / f"{held_out}_stealth_{stealth}.npy"
        if not fp.exists():
            logger.warning(f"Missing: {fp}")
            continue
        attacks = np.load(fp)
        if len(attacks) > max_eval_samples:
            idx = rng.choice(len(attacks), size=max_eval_samples, replace=False)
            attacks = attacks[idx]
        n_b = min(len(benign), len(attacks))
        b_idx = rng.choice(len(benign), size=n_b, replace=False)
        X = np.concatenate([benign[b_idx], attacks])
        y = np.array([0] * n_b + [1] * len(attacks))
        eval_sets[f"stealth_{stealth}"] = (X, y)
    return eval_sets


# ---------------------------------------------------------------------------
# Scoring and evaluation
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


def evaluate_detector(detector, eval_sets: dict) -> dict:
    """Evaluate detector on each eval set using benign-calibrated threshold.

    Calibration protocol (matches run_ablation.py):
      - Split benign samples 80/20 → calibration / test
      - Threshold = 95th percentile of benign-cal NLL scores (target FPR ≤ 0.05)
      - Evaluate on held-out 20% benign + all attacks
    """
    results = {}
    for name, (X, y) in eval_sets.items():
        raw_scores = score_samples(detector, X)
        benign_idx = np.where(y == 0)[0]
        n_cal = max(1, int(0.8 * len(benign_idx)))
        cal_idx = benign_idx[:n_cal]
        test_mask = np.ones(len(y), dtype=bool)
        test_mask[cal_idx] = False
        threshold = float(np.percentile(raw_scores[cal_idx], 95))
        test_scores = raw_scores[test_mask]
        test_y = y[test_mask]
        test_preds = (test_scores > threshold).astype(int)
        m = IDSMetrics.compute_all_metrics(test_y, test_preds, test_scores)
        results[name] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                         for k, v in m.items()}
        logger.info(f"  {name}: F1={m.get('f1', 0):.3f}, "
                    f"FPR={m.get('false_positive_rate', 1):.3f}, "
                    f"AUC={m.get('roc_auc') or 0:.3f} "
                    f"(threshold={threshold:.4f})")
    return results


# ---------------------------------------------------------------------------
# Fine-tuning helper (mirrors run_ablation.py)
# ---------------------------------------------------------------------------

def fine_tune(detector, benign: np.ndarray, train_attacks: np.ndarray,
              epochs: int, batch_size: int, lr: float):
    """Fine-tune with NLL + SupCon on the provided benign + attack arrays."""
    all_X = np.concatenate([benign, train_attacks])
    all_y = np.array([0] * len(benign) + [1] * len(train_attacks))
    rng_split = np.random.default_rng(1)
    perm = rng_split.permutation(len(all_X))
    n_val = max(4, int(0.15 * len(all_X)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    ft = getattr(detector, "fine_tune_supervised", None)
    if ft is None:
        logger.warning("fine_tune_supervised not available")
        return

    try:
        ft(
            train_data=all_X[train_idx],
            train_labels=all_y[train_idx],
            val_data=all_X[val_idx],
            val_labels=all_y[val_idx],
            n_epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            contrastive_weight=0.5,
        )
        logger.info(f"Fine-tuning complete ({epochs} epochs)")
    except Exception as e:
        logger.warning(f"Fine-tuning failed: {e}")


# ---------------------------------------------------------------------------
# Single fold
# ---------------------------------------------------------------------------

def run_fold_single_seed(held_out: str, args, seed: int) -> dict:
    """Run one LOO fold with a single seed. Returns dict with zeroshot/finetuned results."""
    rng = np.random.default_rng(seed)

    benign, train_attacks = load_data_excluding(
        args.synthetic_dir, held_out, rng, max_train_samples=200
    )
    if benign is None:
        logger.error("No benign samples found")
        return {}

    eval_sets = load_held_out_eval(
        args.synthetic_dir, held_out, benign, rng,
        max_eval_samples=args.max_eval_samples
    )
    if not eval_sets:
        logger.error(f"No eval data for held-out type: {held_out}")
        return {}

    detector_kwargs = dict(
        model_size=args.model_size,
        context_length=96,
        prediction_length=32,
        confidence_level=0.95,
    )

    det_zs = MoiraiAnomalyDetector(**detector_kwargs)
    det_zs.initialize()
    zeroshot_results = evaluate_detector(det_zs, eval_sets)

    det_ft = MoiraiAnomalyDetector(**detector_kwargs)
    det_ft.initialize()
    fine_tune(det_ft, benign, train_attacks, args.epochs, args.batch_size, args.lr)
    finetuned_results = evaluate_detector(det_ft, eval_sets)

    return {"zeroshot": zeroshot_results, "finetuned": finetuned_results}


def _mean_std(values: list) -> dict:
    """Compute mean and std over a list of floats."""
    arr = np.array([v for v in values if v is not None], dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def run_fold(held_out: str, args, output_root: Path):
    """Run one leave-one-out fold, averaging over multiple seeds."""
    seeds = [int(s) for s in str(args.seeds).split(",")]
    logger.info(f"[{held_out}] Running {len(seeds)} seed(s): {seeds}")

    seed_results = []
    for seed in seeds:
        logger.info(f"[{held_out}] Seed {seed}")
        r = run_fold_single_seed(held_out, args, seed)
        if r:
            seed_results.append(r)

    if not seed_results:
        logger.error(f"[{held_out}] All seeds failed")
        return

    # Aggregate across seeds
    stealth_keys = list(seed_results[0]["zeroshot"].keys())
    agg_zeroshot, agg_finetuned = {}, {}
    for key in stealth_keys:
        zs_f1s = [r["zeroshot"].get(key, {}).get("f1", float("nan")) for r in seed_results]
        ft_f1s = [r["finetuned"].get(key, {}).get("f1", float("nan")) for r in seed_results]
        agg_zeroshot[key] = _mean_std(zs_f1s)
        agg_finetuned[key] = _mean_std(ft_f1s)

    out_dir = output_root / held_out
    out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "held_out": held_out,
        "trained_on": [at for at in ATTACK_TYPES if at != held_out],
        "seeds": seeds,
        "zeroshot_baseline": agg_zeroshot,
        "results": agg_finetuned,
        "raw_seed_results": seed_results,
    }
    out_file = out_dir / "metrics.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.success(f"Saved → {out_file}")

    # Print comparison
    print(f"\n  Held-out: {held_out}")
    print(f"  {'Eval set':<15} {'Zero-shot F1':>14} {'DiffIDS F1':>12} {'Delta':>8}")
    print(f"  {'-'*55}")
    for key in sorted(stealth_keys):
        zs_f1 = agg_zeroshot.get(key, {}).get("mean", 0)
        ft_f1 = agg_finetuned.get(key, {}).get("mean", 0)
        print(f"  {key:<15} {zs_f1:>14.3f} {ft_f1:>12.3f} {ft_f1 - zs_f1:>+8.3f}")

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Leave-one-out generalization experiment",
    )
    parser.add_argument(
        "--held-out", default="all",
        choices=ATTACK_TYPES + ["all"],
        help="Attack type to hold out from training (default: all = run all 4 folds)",
    )
    parser.add_argument("--model-size", default="small", choices=["small", "base", "large"])
    parser.add_argument("--synthetic-dir", default="data/synthetic")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-eval-samples", type=int, default=30)
    parser.add_argument("--seeds", default="42",
                        help="Comma-separated random seeds, e.g. 42,123,456")
    args = parser.parse_args()

    output_root = Path(args.results_dir) / "leave_one_out"
    output_root.mkdir(parents=True, exist_ok=True)

    folds = ATTACK_TYPES if args.held_out == "all" else [args.held_out]

    all_results = {}
    for held_out in folds:
        print(f"\n{'=' * 70}")
        print(f"  FOLD: held-out = {held_out}")
        print(f"{'=' * 70}")
        t0 = time.time()
        result = run_fold(held_out, args, output_root)
        elapsed = time.time() - t0
        logger.info(f"Fold {held_out} completed in {elapsed:.0f}s")
        if result:
            all_results[held_out] = result

    # Cross-fold summary
    if len(folds) > 1 and all_results:
        print("\n" + "=" * 70)
        print("LEAVE-ONE-OUT SUMMARY  (stealth-95)")
        print("=" * 70)
        print(f"  {'Held-out Type':<25} {'Zero-shot F1':>14} {'DiffIDS F1':>12} {'Delta':>8}")
        print(f"  {'-'*65}")
        zs_all, ft_all = [], []
        for held_out in folds:
            if held_out not in all_results:
                continue
            r = all_results[held_out]
            zs = r["zeroshot_baseline"].get("stealth_95", {}).get("f1", float("nan"))
            ft = r["results"].get("stealth_95", {}).get("f1", float("nan"))
            zs_all.append(zs)
            ft_all.append(ft)
            print(f"  {held_out:<25} {zs:>14.3f} {ft:>12.3f} {ft - zs:>+8.3f}")
        if zs_all and ft_all:
            print(f"  {'-'*65}")
            print(f"  {'Mean':<25} {np.mean(zs_all):>14.3f} {np.mean(ft_all):>12.3f} "
                  f"{np.mean(ft_all) - np.mean(zs_all):>+8.3f}")
        print("=" * 70)


if __name__ == "__main__":
    main()
