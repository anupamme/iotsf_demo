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
from typing import Dict, List, Optional

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

def load_data(synthetic_dir: str, max_samples: int = 200, val_frac: float = 0.20,
              seed: int = 42):
    """
    Build train/test arrays from pre-generated synthetic .npy files.

    Returns
    -------
    X_train, y_train : training split (benign only; y_train=0 always)
    X_test,  y_test  : test split (balanced benign + attacks)
    eval_sets        : dict of {name: (X, y)} for per-condition breakdown
    """
    synth = Path(synthetic_dir)
    rng = np.random.default_rng(seed)

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
    """Fit, predict, and collect metrics for one IDS.

    Calibration:
      Traditional methods (ThresholdIDS, StatisticalIDS, MLBasedIDS, etc.) calibrate
      their decision boundary during fit() using benign training data — ids.predict()
      is already benign-calibrated and not circular.
      Deep-learning baselines (USAD, TranAD, etc.) use an arbitrary fixed 0.5 threshold
      after min-max/sigmoid normalisation; for these we apply a benign-calibrated
      percentile threshold: score X_train, set threshold = 95th percentile (5% FPR
      target), apply to X_test.  A degenerate threshold (≥ max test score or ≤ 0) falls
      back to ids.predict() to avoid zeroing out all detections.
    """
    logger.info(f"  [{name}] Fitting...")
    t0 = time.time()
    try:
        ids.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"  [{name}] fit() failed: {e}")
        return None
    fit_time = time.time() - t0
    logger.info(f"  [{name}] Fit in {fit_time:.1f}s")

    # Attempt benign-calibrated threshold for DL methods with arbitrary 0.5 default.
    # Traditional methods' scores saturate at 0 or 1 on training data (by design),
    # so the 95th-percentile approach degenerates for them; those fall through to
    # ids.predict() via the degenerate-threshold check below.
    threshold = None
    try:
        train_scores = ids.predict_proba(X_train)
        test_scores_sample = ids.predict_proba(X_test)
        cal_thresh = float(np.percentile(train_scores, 95))
        max_test = float(test_scores_sample.max())
        # Accept calibrated threshold only if it is non-degenerate
        if 0.0 < cal_thresh < max_test:
            threshold = cal_thresh
            logger.info(f"  [{name}] Calibrated threshold={threshold:.4f} (95th pctile benign train)")
        else:
            logger.info(f"  [{name}] Degenerate cal threshold ({cal_thresh:.4f}); using ids.predict()")
    except Exception as e:
        logger.warning(f"  [{name}] Calibration failed ({e}); using ids.predict()")

    def _eval(X, y):
        sc = ids.predict_proba(X)
        if threshold is not None:
            preds = (sc > threshold).astype(int)
        else:
            preds = ids.predict(X)
        return IDSMetrics.compute_all_metrics(y, preds, sc)

    results = {}

    # Main test set
    try:
        results["main"] = _eval(X_test, y_test)
    except Exception as e:
        logger.error(f"  [{name}] predict() on main test failed: {e}")

    # Per-condition (stealth level)
    for cname, (X_c, y_c) in eval_sets.items():
        try:
            results[cname] = _eval(X_c, y_c)
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

def _aggregate_seeds(all_seed_results: List[dict]) -> dict:
    """
    Aggregate per-seed result dicts (method → {fit_time_s, results}).
    Returns same structure with metric values replaced by {"mean": x, "std": y}.
    """
    if len(all_seed_results) == 1:
        return all_seed_results[0]

    aggregated: dict = {}
    method_names = all_seed_results[0].keys()
    for method in method_names:
        aggregated[method] = {"results": {}, "fit_time_s": []}
        eval_keys = all_seed_results[0][method]["results"].keys()
        for ek in eval_keys:
            aggregated[method]["results"][ek] = {}
            metric_keys = [
                k for k in all_seed_results[0][method]["results"].get(ek, {}).keys()
                if k != "confusion_matrix"
            ]
            for mk in metric_keys:
                vals = [
                    sr[method]["results"].get(ek, {}).get(mk)
                    for sr in all_seed_results
                    if sr.get(method, {}).get("results", {}).get(ek, {}).get(mk) is not None
                ]
                if vals and isinstance(vals[0], (int, float)):
                    aggregated[method]["results"][ek][mk] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                    }
                elif vals:
                    aggregated[method]["results"][ek][mk] = vals[0]
        # Average fit time
        times = [sr[method].get("fit_time_s", 0) for sr in all_seed_results if method in sr]
        aggregated[method]["fit_time_s"] = float(np.mean(times)) if times else 0.0
    return aggregated


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
    parser.add_argument(
        "--seeds", default="42",
        help="Comma-separated random seeds for multi-seed runs (e.g. '42,123,456'). "
             "When multiple seeds given, output includes mean ± std."
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    per_seed_results: List[dict] = []
    for seed in seeds:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Seed {seed} — loading data...")
        logger.info(f"{'=' * 60}")

        X_train, y_train, X_test, y_test, eval_sets = load_data(
            args.synthetic_dir, max_samples=args.max_samples, seed=seed
        )

        # Set torch seed if available
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass

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

        # --- Run evaluation for this seed ---
        seed_results: dict = {}
        for name, ids in all_methods.items():
            logger.info(f"\nEvaluating: {name} (seed={seed})")
            res = evaluate_method(name, ids, X_train, y_train, X_test, y_test, eval_sets)
            if res is not None:
                seed_results[name] = res
        per_seed_results.append(seed_results)

    # Aggregate across seeds
    all_results = _aggregate_seeds(per_seed_results)

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
        json.dump(_to_python({"seeds": seeds, "results": all_results}), f, indent=2)
    logger.success(f"Saved JSON results to {out_json}")

    # --- Text comparison table ---
    multi = len(seeds) > 1

    def _get(m, key, default=0.0):
        val = m.get(key, default)
        return val.get("mean", default) if isinstance(val, dict) else (val or default)

    lines = ["=" * 90, f"ALL BASELINES — MAIN TEST SET  (seeds={seeds})", "=" * 90]
    header = (f"{'Method':<22} {'F1':>10} {'FPR':>6} {'AUC':>10} {'Time':>8}")
    lines.append(header)
    lines.append("-" * 60)
    for name, data in all_results.items():
        m = data["results"].get("main", {})
        f1 = _get(m, "f1")
        fpr = _get(m, "false_positive_rate", 1.0)
        auc = _get(m, "roc_auc")
        fit_t = data.get("fit_time_s", 0)
        if multi:
            f1_s = m.get("f1", {}).get("std", 0) if isinstance(m.get("f1"), dict) else 0
            auc_s = m.get("roc_auc", {}).get("std", 0) if isinstance(m.get("roc_auc"), dict) else 0
            f1_str = f"{f1:.3f}±{f1_s:.3f}"
            auc_str = f"{auc:.3f}±{auc_s:.3f}"
        else:
            f1_str = f"{f1:.3f}"
            auc_str = f"{auc:.3f}"
        lines.append(
            f"{name:<22} {f1_str:>10} {fpr:>6.3f} {auc_str:>10} {fit_t:>7.1f}s"
        )
    lines.append("=" * 90)
    comparison = "\n".join(lines)
    print("\n" + comparison)

    out_txt = output_path / "all_baselines_comparison.txt"
    out_txt.write_text(comparison)
    logger.success(f"Saved comparison table to {out_txt}")


if __name__ == "__main__":
    main()
