#!/usr/bin/env python3
"""
Supervised Baselines for NeurIPS 2026 reviewer response (W7, Q4).

Trains LogReg, XGBoost, and MLP on the same labeled stealth-controlled
negatives used by HNIDS condition D, using 72-dim statistical features
(12 network features × 6 statistics per timestep, flattened).

This isolates the "labeled attack data helps" effect from the foundation-model
contribution: if XGBoost on flat features matches HNIDS D (AUC=0.556), the
foundation model adds nothing; if HNIDS exceeds supervised baselines, temporal
modeling and/or pre-trained features contribute beyond the labeled data.

Usage:
    python scripts/run_supervised_baselines.py
    python scripts/run_supervised_baselines.py --seeds 42,123,456,789,1234
    python scripts/run_supervised_baselines.py --synthetic-dir data/synthetic_1k
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

from src.models.baseline.supervised_baselines import SupervisedLogReg, SupervisedXGBoost, SupervisedMLP
from src.evaluation.metrics import IDSMetrics



def find_best_threshold(benign_scores: np.ndarray, target_fpr: float = 0.05) -> float:
    return float(np.percentile(benign_scores, 100 * (1 - target_fpr)))


def build_train_eval_sets(synthetic_dir: str, rng: np.random.Generator,
                          max_train_samples: int = None, max_eval_samples: int = 50,
                          train_frac: float = 0.8):
    """
    Load benign + stealth-controlled negatives and split into train/eval sets.
    Uses an 80/20 stratified split to avoid data leakage between training and evaluation.
    Returns (X_train, y_train, eval_sets) where eval_sets is compatible with
    evaluate_classifier().
    """
    synth = Path(synthetic_dir)
    benign_all = np.load(synth / "benign_samples.npy")

    attack_types = ["slow_exfiltration", "lotl_mimicry", "beacon", "protocol_anomaly"]
    attacks_by_stealth = {}
    for stealth in [85, 90, 95]:
        chunks = []
        for at in attack_types:
            fp = synth / f"{at}_stealth_{stealth}.npy"
            if fp.exists():
                chunks.append(np.load(fp))
        if chunks:
            attacks_by_stealth[stealth] = np.concatenate(chunks)

    if not attacks_by_stealth:
        raise FileNotFoundError(f"No attack files found in {synthetic_dir}")

    # 80/20 split on benign
    n_benign = len(benign_all)
    n_train_b = int(train_frac * n_benign)
    idx_b = rng.permutation(n_benign)
    benign_train = benign_all[idx_b[:n_train_b]]
    benign_eval = benign_all[idx_b[n_train_b:]]

    # 80/20 split on attacks per stealth level
    attack_train_chunks, eval_sets = [], {}
    all_stealth_attacks, all_stealth_labels = [], []
    for stealth, arr in attacks_by_stealth.items():
        n_train_a = int(train_frac * len(arr))
        idx_a = rng.permutation(len(arr))
        attack_train_chunks.append(arr[idx_a[:n_train_a]])
        attack_eval = arr[idx_a[n_train_a:]]

        # Build eval set for this stealth level
        n_b_eval = min(len(benign_eval), len(attack_eval), max_eval_samples)
        b_idx = rng.choice(len(benign_eval), size=n_b_eval, replace=False)
        a_idx = rng.choice(len(attack_eval), size=n_b_eval, replace=False)
        X_eval = np.concatenate([benign_eval[b_idx], attack_eval[a_idx]])
        y_eval = np.array([0] * n_b_eval + [1] * n_b_eval)
        eval_sets[f"stealth_{stealth}"] = (X_eval, y_eval)
        all_stealth_attacks.append(attack_eval[a_idx])
        all_stealth_labels.extend([1] * n_b_eval)

    # Combined all-stealth eval set
    n_b_all = min(len(benign_eval), max_eval_samples)
    b_idx_all = rng.choice(len(benign_eval), size=n_b_all, replace=False)
    X_all = np.concatenate([benign_eval[b_idx_all]] + all_stealth_attacks)
    y_all = np.array([0] * n_b_all + all_stealth_labels)
    eval_sets["all_stealth"] = (X_all, y_all)

    # Build training set
    attack_train = np.concatenate(attack_train_chunks)
    if max_train_samples:
        n_each = max_train_samples // 2
        benign_train = benign_train[rng.choice(len(benign_train), size=min(n_each, len(benign_train)), replace=False)]
        attack_train = attack_train[rng.choice(len(attack_train), size=min(n_each, len(attack_train)), replace=False)]

    X_train = np.concatenate([benign_train, attack_train])
    y_train = np.array([0] * len(benign_train) + [1] * len(attack_train))
    logger.info(f"Train: {len(benign_train)} benign + {len(attack_train)} attack; "
                f"eval benign pool: {len(benign_eval)}")
    return X_train, y_train, eval_sets


def evaluate_classifier(clf, eval_sets: dict) -> dict:
    """Score a fitted classifier on all eval sets with benign-calibrated threshold."""
    results = {}
    for name, (X, y) in eval_sets.items():
        scores = clf.predict_proba(X)

        benign_idx = np.where(y == 0)[0]
        n_cal = max(1, int(0.8 * len(benign_idx)))
        cal_idx = benign_idx[:n_cal]
        test_mask = np.ones(len(y), dtype=bool)
        test_mask[cal_idx] = False

        threshold = find_best_threshold(scores[cal_idx])
        test_scores = scores[test_mask]
        test_y = y[test_mask]
        test_preds = (test_scores > threshold).astype(int)
        metrics = IDSMetrics.compute_all_metrics(test_y, test_preds, test_scores)
        results[name] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in metrics.items()}
        logger.info(
            f"  {name}: F1={metrics.get('f1', 0):.3f}  "
            f"FPR={metrics.get('false_positive_rate', 1):.3f}  "
            f"AUC={metrics.get('roc_auc', 0):.3f}"
        )
    return results


def aggregate_seeds(all_seed_results):
    if len(all_seed_results) == 1:
        return all_seed_results[0]
    agg = {}
    for ek in all_seed_results[0]:
        agg[ek] = {}
        for mk in all_seed_results[0][ek]:
            if mk in ("confusion_matrix", "best_threshold"):
                continue
            vals = [sr.get(ek, {}).get(mk) for sr in all_seed_results if sr.get(ek, {}).get(mk) is not None]
            if vals and isinstance(vals[0], (int, float)):
                agg[ek][mk] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def main():
    parser = argparse.ArgumentParser(description="Run supervised baselines for NeurIPS reviewer response")
    parser.add_argument("--synthetic-dir", default="data/synthetic")
    parser.add_argument("--results-dir", default="results/supervised_baselines")
    parser.add_argument("--seeds", default="42,123,456,789,1234,1011,2022,3033,4044,5055")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=50)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    output_root = Path(args.results_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    classifiers = {
        "logreg": lambda: SupervisedLogReg(),
        "xgboost": lambda: SupervisedXGBoost(),
        "mlp": lambda: SupervisedMLP(),
    }

    for clf_name, clf_factory in classifiers.items():
        logger.info(f"\n{'='*60}\nRunning {clf_name.upper()} over {len(seeds)} seeds\n{'='*60}")
        t0 = time.time()
        per_seed = []
        for seed in seeds:
            logger.info(f"  Seed {seed}...")
            rng = np.random.default_rng(seed)

            X_train, y_train, eval_sets = build_train_eval_sets(
                args.synthetic_dir, rng, args.max_train_samples, args.max_eval_samples
            )

            clf = clf_factory()
            clf.fit(X_train, y_train)
            seed_results = evaluate_classifier(clf, eval_sets)
            per_seed.append(seed_results)

        elapsed = time.time() - t0
        results = aggregate_seeds(per_seed)

        out = output_root / clf_name
        out.mkdir(exist_ok=True)
        with open(out / "metrics.json", "w") as f:
            json.dump({"classifier": clf_name, "elapsed_s": elapsed,
                       "seeds": seeds, "results": results}, f, indent=2)
        logger.success(f"{clf_name} saved → {out}/metrics.json")

        s95 = results.get("stealth_95", {})
        auc = s95.get("roc_auc", {})
        f1 = s95.get("f1", {})
        auc_str = f"{auc.get('mean', 0):.3f}±{auc.get('std', 0):.3f}" if isinstance(auc, dict) else str(auc)
        f1_str = f"{f1.get('mean', 0):.3f}±{f1.get('std', 0):.3f}" if isinstance(f1, dict) else str(f1)
        logger.info(f"  stealth-95: AUC={auc_str}  F1={f1_str}")


if __name__ == "__main__":
    main()
