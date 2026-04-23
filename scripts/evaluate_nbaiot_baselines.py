#!/usr/bin/env python3
"""
N-BaIoT Baseline Evaluation: Run IF, OCSVM, and USAD on N-BaIoT benign data
and evaluate on N-BaIoT attacks.

Addresses reviewer W7: compare HNIDS fine-tuned (AUC=0.928) against
traditional unsupervised baselines on the same N-BaIoT data.

All baselines are trained on N-BaIoT benign windows only (no attack labels).

Usage
-----
python scripts/evaluate_nbaiot_baselines.py --data-dir data/nbaiot/
python scripts/evaluate_nbaiot_baselines.py --data-dir data/nbaiot/ --seeds 42,123,456
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.nbaiot_loader import load_nbaiot, PROXY_FEATURE_NAMES
from src.models.baseline.ml_based import MLBasedIDS
from src.models.baseline.ocsvm import OCSVMIDS
from src.models.baseline.usad import USADIDS


def evaluate_baseline(
    model,
    model_name: str,
    X_benign_val: np.ndarray,
    X_attack: np.ndarray,
    y_attack_names: np.ndarray,
    attack_types: List[str],
) -> Dict:
    """Evaluate a fitted baseline on N-BaIoT data. Returns per-attack and overall AUC."""
    logger.info(f"  [{model_name}] Scoring {len(X_benign_val)} benign + {len(X_attack)} attack...")

    scores_benign = model.predict_proba(X_benign_val)
    scores_attack = model.predict_proba(X_attack)

    all_scores = np.concatenate([scores_benign, scores_attack])
    all_labels = np.concatenate([
        np.zeros(len(X_benign_val), dtype=int),
        np.ones(len(X_attack), dtype=int),
    ])

    overall_auc = float(roc_auc_score(all_labels, all_scores))
    logger.info(f"  [{model_name}] Overall AUC = {overall_auc:.3f}")

    per_attack: Dict[str, Dict] = {}
    for atk_type in attack_types:
        mask = (y_attack_names == atk_type)
        if not mask.any():
            continue
        atk_scores = scores_attack[mask]
        eval_scores = np.concatenate([scores_benign, atk_scores])
        eval_labels = np.concatenate([
            np.zeros(len(X_benign_val), dtype=int),
            np.ones(mask.sum(), dtype=int),
        ])
        try:
            auc = float(roc_auc_score(eval_labels, eval_scores))
        except ValueError:
            auc = float("nan")
        per_attack[atk_type] = {
            "roc_auc": auc,
            "n_attack_windows": int(mask.sum()),
        }
        logger.info(f"    {atk_type}: AUC={auc:.3f} ({mask.sum()} windows)")

    return {"overall_auc": overall_auc, "per_attack": per_attack}


def run_seed(seed: int, data_dir: str, device: str, max_samples: int) -> Dict:
    """Run all baselines for one seed."""
    logger.info(f"\n--- Seed {seed} ---")

    X_train, X_val, X_atk, y_names, scaler, atk_types = load_nbaiot(
        data_dir=data_dir,
        device=device,
        max_samples_per_class=max_samples,
        seq_length=128,
        seed=seed,
    )

    results = {}

    # Isolation Forest
    logger.info(f"  Training Isolation Forest...")
    t0 = time.time()
    iforest = MLBasedIDS(seq_length=128, feature_dim=12, contamination=0.05,
                         n_estimators=100, random_state=seed)
    iforest.fit(X_train)
    results["isolation_forest"] = evaluate_baseline(
        iforest, "IF", X_val, X_atk, y_names, atk_types)
    results["isolation_forest"]["train_time_s"] = time.time() - t0

    # One-Class SVM
    logger.info(f"  Training One-Class SVM...")
    t0 = time.time()
    ocsvm = OCSVMIDS(seq_length=128, feature_dim=12, nu=0.05,
                     kernel="rbf", gamma="scale", random_state=seed)
    ocsvm.fit(X_train)
    results["ocsvm"] = evaluate_baseline(
        ocsvm, "OCSVM", X_val, X_atk, y_names, atk_types)
    results["ocsvm"]["train_time_s"] = time.time() - t0

    # USAD (autoencoder)
    logger.info(f"  Training USAD autoencoder...")
    t0 = time.time()
    usad = USADIDS(seq_length=128, feature_dim=12, z_dim=40,
                   epochs=50, batch_size=64, lr=1e-3, alpha=0.5)
    usad.fit(X_train)
    results["usad"] = evaluate_baseline(
        usad, "USAD", X_val, X_atk, y_names, atk_types)
    results["usad"]["train_time_s"] = time.time() - t0

    return results


def aggregate_seeds(all_seed_results: List[Dict]) -> Dict:
    """Average per-seed metrics across seeds."""
    methods = list(all_seed_results[0].keys())
    agg = {}

    for method in methods:
        aucs = [sr[method]["overall_auc"] for sr in all_seed_results]
        agg[method] = {
            "overall_auc": {
                "mean": float(np.mean(aucs)),
                "std": float(np.std(aucs)),
            },
            "per_attack": {},
        }
        all_atk_types = set()
        for sr in all_seed_results:
            all_atk_types.update(sr[method]["per_attack"].keys())

        for atk in sorted(all_atk_types):
            atk_aucs = [
                sr[method]["per_attack"][atk]["roc_auc"]
                for sr in all_seed_results
                if atk in sr[method]["per_attack"]
            ]
            agg[method]["per_attack"][atk] = {
                "roc_auc": {
                    "mean": float(np.nanmean(atk_aucs)),
                    "std": float(np.nanstd(atk_aucs)),
                },
            }

    return agg


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IF/OCSVM/USAD baselines on N-BaIoT"
    )
    parser.add_argument("--data-dir", default="data/nbaiot/")
    parser.add_argument("--device", default="danmini_doorbell")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--max-samples", type=int, default=49548)
    parser.add_argument("--output", default="results/nbaiot_baselines/")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_dir)
    if not data_path.exists():
        logger.error(f"N-BaIoT data directory not found: {data_path.resolve()}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("N-BaIoT Baseline Evaluation (IF / OCSVM / USAD)")
    logger.info(f"Device  : {args.device}")
    logger.info(f"Seeds   : {seeds}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info("=" * 60)

    all_seed_results = []
    for seed in seeds:
        seed_result = run_seed(seed, args.data_dir, args.device, args.max_samples)
        all_seed_results.append(seed_result)

    agg = aggregate_seeds(all_seed_results)
    agg["meta"] = {
        "device": args.device,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "baselines": ["isolation_forest", "ocsvm", "usad"],
        "feature_proxy": PROXY_FEATURE_NAMES,
    }

    out_path = output_dir / "metrics.json"
    out_path.write_text(json.dumps(agg, indent=2))
    logger.success(f"Results saved to {out_path}")

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for method in ["isolation_forest", "ocsvm", "usad"]:
        m = agg[method]["overall_auc"]
        logger.info(f"  {method:20s}  AUC = {m['mean']:.3f} +/- {m['std']:.3f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
