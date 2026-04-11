#!/usr/bin/env python3
"""
Run all 9 IDS baselines on a unified train/test split and save results.

Baselines evaluated
-------------------
Traditional (5):
  1. ThresholdIDS       — 95th-percentile statistical thresholding
  2. SignatureIDS       — Pattern matching for known attacks
  3. StatisticalIDS     — Z-score + IQR outlier detection
  4. MLBasedIDS         — Isolation Forest (unsupervised)
  5. CombinedBaseline   — Weighted ensemble of 1-4

Deep Learning (4):
  6. USAD               — Dual autoencoder (KDD 2020)
  7. TranAD             — Bidirectional Transformer (VLDB 2022)
  8. AnomalyTransformer — Association discrepancy (ICLR 2022)
  9. PatchTST-Anomaly   — Masked-patch reconstruction (ICLR 2023)

All methods are fit on the SAME training split (benign only) and evaluated
on the SAME test split (balanced benign + attacks), so results are directly
comparable.

Output: results/all_baselines_evaluation.json
        results/all_baselines_comparison.txt

Usage:
    python scripts/run_all_baselines.py
    python scripts/run_all_baselines.py --synthetic-dir data/synthetic --max-samples 200
    python scripts/run_all_baselines.py --skip-dl   # skip deep-learning baselines (fast)
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

from src.models.baseline import (
    ThresholdIDS, StatisticalIDS, SignatureIDS, MLBasedIDS, CombinedBaselineIDS,
    USADIDS, TranADIDS, AnomalyTransformerIDS, PatchTSTAnomalyIDS,
)
from src.evaluation.metrics import IDSMetrics


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(synthetic_dir: str, max_samples: int = 200, val_frac: float = 0.20):
    """
    Build train/test arrays from pre-generated synthetic .npy files.

    Returns
    -------
    X_train, y_train : training split (benign only; y_train=0 always)
    X_test,  y_test  : test split (balanced benign + attacks)
    eval_sets        : dict of {name: (X, y)} for per-condition breakdown
    """
    synth = Path(synthetic_dir)
    rng = np.random.default_rng(42)

    # Load benign
    benign_path = synth / "benign_samples.npy"
    if not benign_path.exists():
        raise FileNotFoundError(f"benign_samples.npy not found in {synth}")
    benign_all = np.load(benign_path)

    # Load per-condition attacks
    attack_types = ["slow_exfiltration", "lotl_mimicry", "beacon", "protocol_anomaly"]
    eval_sets = {}
    all_attacks = []

    for stealth in [85, 90, 95]:
        stealth_attacks = []
        for at in attack_types:
            fp = synth / f"{at}_stealth_{stealth}.npy"
            if fp.exists():
                arr = np.load(fp)
                # Subsample to max_samples per condition
                if len(arr) > max_samples // 12:
                    idx = rng.choice(len(arr), size=max_samples // 12, replace=False)
                    arr = arr[idx]
                stealth_attacks.append(arr)
                all_attacks.append(arr)
        if stealth_attacks:
            cond_attacks = np.concatenate(stealth_attacks)
            # Build per-condition eval set
            n_b = min(len(benign_all), len(cond_attacks))
            b_idx = rng.choice(len(benign_all), size=n_b, replace=False)
            cond_X = np.concatenate([benign_all[b_idx], cond_attacks])
            cond_y = np.array([0] * n_b + [1] * len(cond_attacks))
            eval_sets[f"synthetic_stealth_{stealth}"] = (cond_X, cond_y)
            logger.info(f"  stealth_{stealth}: {n_b} benign + {len(cond_attacks)} attack")

    if not all_attacks:
        raise RuntimeError("No attack .npy files found in synthetic directory")

    all_attacks_arr = np.concatenate(all_attacks, axis=0)
    if len(all_attacks_arr) > max_samples:
        idx = rng.choice(len(all_attacks_arr), size=max_samples, replace=False)
        all_attacks_arr = all_attacks_arr[idx]

    n_total_benign = min(len(benign_all), max_samples)
    if len(benign_all) > n_total_benign:
        idx = rng.choice(len(benign_all), size=n_total_benign, replace=False)
        benign_all = benign_all[idx]

    # Train/test split (val_frac of benign as test benign, remaining as train)
    n_test_b = max(5, int(n_total_benign * val_frac))
    n_train_b = n_total_benign - n_test_b

    perm = rng.permutation(len(benign_all))
    benign_train = benign_all[perm[:n_train_b]]
    benign_test = benign_all[perm[n_train_b:]]

    # Test: balanced
    n_test_a = min(len(all_attacks_arr), len(benign_test) * 4)
    if len(all_attacks_arr) > n_test_a:
        idx = rng.choice(len(all_attacks_arr), size=n_test_a, replace=False)
        attacks_test = all_attacks_arr[idx]
    else:
        attacks_test = all_attacks_arr

    X_train = benign_train
    y_train = np.zeros(len(X_train), dtype=int)
    X_test = np.concatenate([benign_test, attacks_test])
    y_test = np.array([0] * len(benign_test) + [1] * len(attacks_test))

    logger.info(f"Train: {len(X_train)} benign")
    logger.info(f"Test:  {len(benign_test)} benign + {len(attacks_test)} attack = {len(X_test)} total")
    return X_train, y_train, X_test, y_test, eval_sets


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_method(name: str, ids, X_train, y_train, X_test, y_test, eval_sets):
    """Fit, predict, and collect metrics for one IDS."""
    logger.info(f"  [{name}] Fitting...")
    t0 = time.time()
    try:
        ids.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"  [{name}] fit() failed: {e}")
        return None
    fit_time = time.time() - t0
    logger.info(f"  [{name}] Fit in {fit_time:.1f}s")

    results = {}

    # Main test set
    try:
        y_pred = ids.predict(X_test)
        y_scores = ids.predict_proba(X_test)
        results["main"] = IDSMetrics.compute_all_metrics(y_test, y_pred, y_scores)
    except Exception as e:
        logger.error(f"  [{name}] predict() on main test failed: {e}")

    # Per-condition (stealth level)
    for cname, (X_c, y_c) in eval_sets.items():
        try:
            y_pred_c = ids.predict(X_c)
            y_scores_c = ids.predict_proba(X_c)
            results[cname] = IDSMetrics.compute_all_metrics(y_c, y_pred_c, y_scores_c)
        except Exception as e:
            logger.warning(f"  [{name}] predict() on {cname} failed: {e}")

    logger.info(
        f"  [{name}] Main F1={results.get('main', {}).get('f1', 0):.3f}, "
        f"FPR={results.get('main', {}).get('false_positive_rate', 1):.3f}"
    )
    return {"fit_time_s": fit_time, "results": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all 9 IDS baselines on a unified split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--synthetic-dir", default="data/synthetic")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Max samples to load (controls run time; default 200)")
    parser.add_argument("--skip-dl", action="store_true",
                        help="Skip deep-learning baselines (faster run)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs for DL baselines (default 20)")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    X_train, y_train, X_test, y_test, eval_sets = load_data(
        args.synthetic_dir, max_samples=args.max_samples
    )

    # --- Define all methods ---
    traditional = {
        "Threshold": ThresholdIDS(),
        "Signature": SignatureIDS(),
        "Statistical": StatisticalIDS(),
        "IsolationForest": MLBasedIDS(),
        "Ensemble": CombinedBaselineIDS(),
    }

    dl_baselines = {} if args.skip_dl else {
        "USAD": USADIDS(epochs=args.epochs),
        "TranAD": TranADIDS(epochs=args.epochs),
        "AnomalyTransformer": AnomalyTransformerIDS(epochs=args.epochs),
        "PatchTST-Anomaly": PatchTSTAnomalyIDS(epochs=args.epochs),
    }

    all_methods = {**traditional, **dl_baselines}

    # --- Run evaluation ---
    all_results = {}
    for name, ids in all_methods.items():
        logger.info(f"\nEvaluating: {name}")
        res = evaluate_method(name, ids, X_train, y_train, X_test, y_test, eval_sets)
        if res is not None:
            all_results[name] = res

    # --- Serialize (convert numpy → Python for JSON) ---
    def _to_python(obj):
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_python(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    out_json = output_path / "all_baselines_evaluation.json"
    with open(out_json, "w") as f:
        json.dump(_to_python(all_results), f, indent=2)
    logger.success(f"Saved JSON results to {out_json}")

    # --- Text comparison table ---
    lines = ["=" * 90, "ALL BASELINES — MAIN TEST SET", "=" * 90]
    header = f"{'Method':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'FPR':>6} {'AUC':>6} {'Time':>8}"
    lines.append(header)
    lines.append("-" * 90)
    for name, data in all_results.items():
        m = data["results"].get("main", {})
        lines.append(
            f"{name:<22} "
            f"{m.get('accuracy', 0):>6.3f} "
            f"{m.get('precision', 0):>6.3f} "
            f"{m.get('recall', 0):>6.3f} "
            f"{m.get('f1', 0):>6.3f} "
            f"{m.get('false_positive_rate', 1):>6.3f} "
            f"{m.get('roc_auc') or 0:>6.3f} "
            f"{data.get('fit_time_s', 0):>7.1f}s"
        )
    lines.append("=" * 90)
    comparison = "\n".join(lines)
    print("\n" + comparison)

    out_txt = output_path / "all_baselines_comparison.txt"
    out_txt.write_text(comparison)
    logger.success(f"Saved comparison table to {out_txt}")


if __name__ == "__main__":
    main()
