#!/usr/bin/env python3
"""
Grid search for deep-learning baseline hyperparameters.

Addresses reviewer MC6: baselines use published defaults while HNIDS has
extensive tuning.

Protocol: train on benign-only, select best config on calibration split
(20% hold-out), report test-set performance of best config with 3 seeds.

Output: results/baseline_tuning/grid_results.json
        results/baseline_tuning/best_configs.json
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.baseline import USADIDS, TranADIDS, AnomalyTransformerIDS, PatchTSTAnomalyIDS
from src.evaluation.metrics import IDSMetrics


def load_data(synthetic_dir, max_samples=200, seed=42):
    """Load benign train + stealth-95 eval set."""
    synth = Path(synthetic_dir)
    rng = np.random.default_rng(seed)

    benign = np.load(synth / "benign_samples.npy")
    if len(benign) > max_samples:
        idx = rng.choice(len(benign), max_samples, replace=False)
        benign = benign[idx]

    # Split benign into train (60%), calibration (20%), test-benign (20%)
    perm = rng.permutation(len(benign))
    n_cal = max(5, int(0.20 * len(benign)))
    n_test_b = max(5, int(0.20 * len(benign)))
    n_train = len(benign) - n_cal - n_test_b

    benign_train = benign[perm[:n_train]]
    benign_cal = benign[perm[n_train:n_train + n_cal]]
    benign_test = benign[perm[n_train + n_cal:]]

    # Load stealth-95 attacks for evaluation
    attack_types = ["slow_exfiltration", "lotl_mimicry", "beacon", "protocol_anomaly"]
    attacks = []
    for at in attack_types:
        fp = synth / f"{at}_stealth_95.npy"
        if fp.exists():
            arr = np.load(fp)
            if len(arr) > max_samples // 4:
                idx = rng.choice(len(arr), max_samples // 4, replace=False)
                arr = arr[idx]
            attacks.append(arr)

    if not attacks:
        raise FileNotFoundError("No stealth-95 attack files found")

    attacks_arr = np.concatenate(attacks)

    # Calibration set: cal-benign + attacks subset
    n_cal_atk = min(len(attacks_arr), len(benign_cal) * 2)
    cal_atk_idx = rng.choice(len(attacks_arr), n_cal_atk, replace=False)
    X_cal = np.concatenate([benign_cal, attacks_arr[cal_atk_idx]])
    y_cal = np.array([0] * len(benign_cal) + [1] * n_cal_atk)

    # Test set: test-benign + remaining attacks
    X_test = np.concatenate([benign_test, attacks_arr])
    y_test = np.array([0] * len(benign_test) + [1] * len(attacks_arr))

    return benign_train, X_cal, y_cal, X_test, y_test


def evaluate_config(ids_cls, params, benign_train, X_cal, y_cal, seed):
    """Train one config, return calibration AUC."""
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    try:
        ids = ids_cls(**params)
        y_train = np.zeros(len(benign_train), dtype=int)
        ids.fit(benign_train, y_train)

        scores = ids.predict_proba(X_cal)
        metrics = IDSMetrics.compute_all_metrics(y_cal, (scores > 0.5).astype(int), scores)
        return metrics.get("roc_auc", 0.0)
    except Exception as e:
        logger.warning(f"Config {params} failed: {e}")
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-dir", default="data/synthetic")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--output-dir", default="results/baseline_tuning")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define grids
    grids = {
        "USAD": {
            "cls": USADIDS,
            "params": list(itertools.product(
                [20, 40, 80],       # z_dim
                [0.3, 0.5, 0.7],    # alpha
                [50, 100],           # epochs
            )),
            "param_names": ["z_dim", "alpha", "epochs"],
        },
        "TranAD": {
            "cls": TranADIDS,
            "params": list(itertools.product(
                [128, 256],    # d_model
                [2, 4],        # n_layers
                [50, 100],     # epochs
            )),
            "param_names": ["d_model", "n_layers", "epochs"],
        },
        "AnomalyTransformer": {
            "cls": AnomalyTransformerIDS,
            "params": list(itertools.product(
                [128, 256, 512],   # d_model
                [50, 100],         # epochs
            )),
            "param_names": ["d_model", "epochs"],
        },
        "PatchTST-Anomaly": {
            "cls": PatchTSTAnomalyIDS,
            "params": list(itertools.product(
                [8, 16, 32],    # patch_size
                [128, 256],     # d_model
            )),
            "param_names": ["patch_size", "d_model"],
        },
    }

    total_configs = sum(len(g["params"]) for g in grids.values())
    logger.info(f"Total configurations to evaluate: {total_configs}")

    all_grid_results = {}
    best_configs = {}

    for method_name, grid in grids.items():
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Tuning {method_name} ({len(grid['params'])} configs)")
        logger.info(f"{'=' * 50}")

        config_results = []
        for param_values in grid["params"]:
            params = dict(zip(grid["param_names"], param_values))
            params_full = {"seq_length": 128, "feature_dim": 12, **params}

            aucs = []
            for seed in seeds:
                benign_train, X_cal, y_cal, X_test, y_test = load_data(
                    args.synthetic_dir, args.max_samples, seed=seed
                )
                auc = evaluate_config(grid["cls"], params_full, benign_train, X_cal, y_cal, seed)
                aucs.append(auc)

            mean_auc = float(np.mean(aucs))
            std_auc = float(np.std(aucs))
            config_results.append({
                "params": params,
                "cal_aucs": aucs,
                "mean_cal_auc": mean_auc,
                "std_cal_auc": std_auc,
            })
            logger.info(f"  {params}: AUC={mean_auc:.4f}±{std_auc:.4f}")

        # Select best
        best_idx = max(range(len(config_results)), key=lambda i: config_results[i]["mean_cal_auc"])
        best = config_results[best_idx]
        logger.info(f"  Best: {best['params']} (AUC={best['mean_cal_auc']:.4f})")

        all_grid_results[method_name] = config_results
        best_configs[method_name] = {
            "params": best["params"],
            "cal_auc": best["mean_cal_auc"],
            "cal_auc_std": best["std_cal_auc"],
        }

    # Now evaluate best configs on test set with all seeds
    logger.info(f"\n{'=' * 50}")
    logger.info("Evaluating best configs on test set")
    logger.info(f"{'=' * 50}")

    test_results = {}
    for method_name, best in best_configs.items():
        params_full = {"seq_length": 128, "feature_dim": 12, **best["params"]}
        grid = grids[method_name]

        test_aucs, test_f1s = [], []
        for seed in seeds:
            benign_train, X_cal, y_cal, X_test, y_test = load_data(
                args.synthetic_dir, args.max_samples, seed=seed
            )
            try:
                import torch
                torch.manual_seed(seed)
            except ImportError:
                pass

            ids = grid["cls"](**params_full)
            ids.fit(benign_train, np.zeros(len(benign_train), dtype=int))

            # Calibrated threshold from benign train
            train_scores = ids.predict_proba(benign_train)
            threshold = float(np.percentile(train_scores, 95))

            test_scores = ids.predict_proba(X_test)
            test_preds = (test_scores > threshold).astype(int)
            metrics = IDSMetrics.compute_all_metrics(y_test, test_preds, test_scores)
            test_aucs.append(metrics.get("roc_auc", 0.0))
            test_f1s.append(metrics.get("f1", 0.0))

        test_results[method_name] = {
            "params": best["params"],
            "test_auc": {"mean": float(np.mean(test_aucs)), "std": float(np.std(test_aucs))},
            "test_f1": {"mean": float(np.mean(test_f1s)), "std": float(np.std(test_f1s))},
        }
        logger.info(f"  {method_name}: AUC={np.mean(test_aucs):.4f}±{np.std(test_aucs):.4f}, "
                     f"F1={np.mean(test_f1s):.4f}±{np.std(test_f1s):.4f}")

    # Save
    (output_dir / "grid_results.json").write_text(json.dumps(all_grid_results, indent=2, default=str))
    (output_dir / "best_configs.json").write_text(json.dumps({
        "seeds": seeds,
        "best_configs": best_configs,
        "test_results": test_results,
    }, indent=2, default=str))
    logger.success(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
