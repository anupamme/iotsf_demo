#!/usr/bin/env python3
"""Aggregate multi-seed results and compute forgetting summary statistics.

Reads individual result JSONs and produces a summary table with
mean +/- std across seeds for each condition/horizon.

Usage:
    python scripts/05_compute_forgetting.py --results-dir runs/
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_results


def aggregate_and_print(results_dir: str):
    """Load results and print aggregated forgetting table."""
    all_results = load_results(results_dir)
    if not all_results:
        print(f"No results found in {results_dir}")
        return

    # Group by (condition, horizon)
    groups = {}
    for r in all_results:
        key = (r.get("condition", "?"), r.get("horizon", 0))
        groups.setdefault(key, []).append(r)

    horizons = sorted(set(h for _, h in groups.keys()))
    conditions = sorted(set(c for c, _ in groups.keys()))

    for horizon in horizons:
        print(f"\n{'='*80}")
        print(f"  HORIZON = {horizon}")
        print(f"{'='*80}")
        print(
            f"{'Cond':<6} {'N':>3} {'ZS MSE':>12} {'Final MSE':>14} "
            f"{'Forgetting%':>14} {'CKA':>10} {'Drift':>10}"
        )
        print("-" * 80)

        for cond in conditions:
            key = (cond, horizon)
            if key not in groups:
                continue
            runs = groups[key]
            n = len(runs)

            zs = [r["zeroshot_mse"] for r in runs if "zeroshot_mse" in r]
            final = [r["final_val_mse"] for r in runs if "final_val_mse" in r]
            forg = [r["forgetting_pct"] for r in runs if "forgetting_pct" in r]
            cka_vals = [r["final_cka"] for r in runs if "final_cka" in r]
            drift = [r["final_weight_drift"] for r in runs if "final_weight_drift" in r]

            if not zs:
                continue

            line = f"  {cond:<4} {n:>3} "
            line += f"{np.mean(zs):>8.4f}+/-{np.std(zs):.4f} "
            if final:
                line += f"{np.mean(final):>8.4f}+/-{np.std(final):.4f} "
            else:
                line += f"{'N/A':>14} "
            if forg:
                line += f"{np.mean(forg):>+8.1f}+/-{np.std(forg):.1f}% "
            else:
                line += f"{'N/A':>14} "
            if cka_vals:
                line += f"{np.mean(cka_vals):>8.3f}+/-{np.std(cka_vals):.3f} "
            if drift:
                line += f"{np.mean(drift):>8.2f}+/-{np.std(drift):.2f}"
            print(line)

    # Per-seed detail
    for horizon in horizons:
        print(f"\n--- Per-seed detail (h={horizon}) ---")
        print(f"{'Cond':<6} {'Seed':>6} {'ZS MSE':>10} {'Final MSE':>10} "
              f"{'Forg%':>8} {'CKA':>8} {'Drift':>8}")
        print("-" * 60)
        for cond in conditions:
            key = (cond, horizon)
            if key not in groups:
                continue
            for r in sorted(groups[key], key=lambda x: x.get("seed", 0)):
                print(
                    f"  {cond:<4} {r.get('seed', '?'):>6} "
                    f"{r.get('zeroshot_mse', 0):>10.4f} "
                    f"{r.get('final_val_mse', 0):>10.4f} "
                    f"{r.get('forgetting_pct', 0):>+7.1f}% "
                    f"{r.get('final_cka', 0):>8.3f} "
                    f"{r.get('final_weight_drift', 0):>8.2f}"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="runs/")
    args = parser.parse_args()
    aggregate_and_print(args.results_dir)


if __name__ == "__main__":
    main()
