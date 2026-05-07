#!/usr/bin/env python3
"""
Aggregate multi-seed forgetting results and compute summary statistics.

Reads individual result JSON files and produces a summary table with
mean ± std across seeds for each condition/horizon.

Usage:
    python3 scripts/aggregate_forgetting_results.py \
        --results-dir results/forecasting_finetune_20ep
"""

import argparse
import json
from pathlib import Path
import numpy as np


def load_results(results_dir: Path) -> dict:
    """Load all condition_X_hY_sZ.json files."""
    results = {}
    for f in sorted(results_dir.glob("condition_*.json")):
        with open(f) as fp:
            d = json.load(fp)
        key = (d['condition'], d['horizon'])
        if key not in results:
            results[key] = []
        results[key].append(d)
    return results


def print_summary(results: dict):
    """Print aggregated summary table."""
    # Group by horizon
    horizons = sorted(set(h for _, h in results.keys()))
    conditions = sorted(set(c for c, _ in results.keys()))

    for horizon in horizons:
        print(f"\n{'='*80}")
        print(f"  HORIZON = {horizon}")
        print(f"{'='*80}")
        print(f"{'Cond':<6} {'N':>3} {'ZS MSE':>10} {'Final MSE':>14} "
              f"{'Forgetting%':>14} {'CKA':>10} {'Drift':>10}")
        print("-" * 80)

        for cond in conditions:
            key = (cond, horizon)
            if key not in results:
                continue
            runs = results[key]
            n = len(runs)

            zs = [r['zeroshot_mse'] for r in runs]
            final = [r['final_val_mse'] for r in runs]
            forg = [r['forgetting_pct'] for r in runs]
            cka_vals = [r['final_cka'] for r in runs]
            drift = [r['final_weight_drift'] for r in runs]

            print(f"  {cond:<4} {n:>3} "
                  f"{np.mean(zs):>8.4f}±{np.std(zs):.4f} "
                  f"{np.mean(final):>8.4f}±{np.std(final):.4f} "
                  f"{np.mean(forg):>+8.1f}±{np.std(forg):.1f}% "
                  f"{np.mean(cka_vals):>8.3f}±{np.std(cka_vals):.3f} "
                  f"{np.mean(drift):>8.2f}±{np.std(drift):.2f}")

        print()

    # Per-seed detail table
    for horizon in horizons:
        print(f"\n--- Per-seed detail (h={horizon}) ---")
        print(f"{'Cond':<6} {'Seed':>6} {'ZS MSE':>10} {'Final MSE':>10} "
              f"{'Forg%':>8} {'CKA':>8} {'Drift':>8}")
        print("-" * 60)
        for cond in conditions:
            key = (cond, horizon)
            if key not in results:
                continue
            for r in sorted(results[key], key=lambda x: x['seed']):
                print(f"  {cond:<4} {r['seed']:>6} "
                      f"{r['zeroshot_mse']:>10.4f} "
                      f"{r['final_val_mse']:>10.4f} "
                      f"{r['forgetting_pct']:>+7.1f}% "
                      f"{r['final_cka']:>8.3f} "
                      f"{r['final_weight_drift']:>8.2f}")
        print()


def export_latex_table(results: dict, output_path: Path):
    """Export a LaTeX-formatted results table."""
    horizons = sorted(set(h for _, h in results.keys()))
    conditions = sorted(set(c for c, _ in results.keys()))
    cond_names = {'A': 'Zero-shot', 'B': 'NLL-only', 'C': 'NLL+SupCon', 'D': 'Frozen'}

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Catastrophic forgetting diagnosis on ETTh2 forecasting (mean $\pm$ std across 5 seeds).}")
    lines.append(r"\label{tab:forecasting_forgetting}")
    lines.append(r"\begin{tabular}{ll" + "cccc" * len(horizons) + "}")
    lines.append(r"\toprule")

    # Header
    header = r"& "
    for h in horizons:
        header += rf"& \multicolumn{{4}}{{c}}{{$h={h}$}} "
    lines.append(header + r"\\")

    subheader = r"Condition & "
    for h in horizons:
        subheader += r"& MSE & Forg.\% & CKA & Drift "
    lines.append(subheader + r"\\")
    lines.append(r"\midrule")

    for cond in conditions:
        name = cond_names.get(cond, cond)
        row = f"{name} "
        for h in horizons:
            key = (cond, h)
            if key not in results:
                row += "& --- & --- & --- & --- "
                continue
            runs = results[key]
            final = [r['final_val_mse'] for r in runs]
            forg = [r['forgetting_pct'] for r in runs]
            cka_vals = [r['final_cka'] for r in runs]
            drift = [r['final_weight_drift'] for r in runs]

            row += (f"& {np.mean(final):.4f}$\\pm${np.std(final):.4f} "
                    f"& {np.mean(forg):+.1f}$\\pm${np.std(forg):.1f} "
                    f"& {np.mean(cka_vals):.3f}$\\pm${np.std(cka_vals):.3f} "
                    f"& {np.mean(drift):.2f}$\\pm${np.std(drift):.2f} ")
        lines.append(row + r"\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nLaTeX table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate forgetting results")
    parser.add_argument('--results-dir', default='results/forecasting_finetune_20ep')
    parser.add_argument('--latex', default=None, help="Export LaTeX table to file")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}")
        return

    print_summary(results)

    if args.latex:
        export_latex_table(results, Path(args.latex))

    # Save aggregated JSON
    agg = {}
    for (cond, horizon), runs in results.items():
        key = f"{cond}_h{horizon}"
        agg[key] = {
            'condition': cond,
            'horizon': horizon,
            'n_seeds': len(runs),
            'seeds': [r['seed'] for r in runs],
            'zeroshot_mse_mean': float(np.mean([r['zeroshot_mse'] for r in runs])),
            'final_val_mse_mean': float(np.mean([r['final_val_mse'] for r in runs])),
            'final_val_mse_std': float(np.std([r['final_val_mse'] for r in runs])),
            'forgetting_pct_mean': float(np.mean([r['forgetting_pct'] for r in runs])),
            'forgetting_pct_std': float(np.std([r['forgetting_pct'] for r in runs])),
            'cka_mean': float(np.mean([r['final_cka'] for r in runs])),
            'cka_std': float(np.std([r['final_cka'] for r in runs])),
            'drift_mean': float(np.mean([r['final_weight_drift'] for r in runs])),
            'drift_std': float(np.std([r['final_weight_drift'] for r in runs])),
        }

    agg_path = results_dir / 'aggregated.json'
    with open(agg_path, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"\nAggregated JSON saved to {agg_path}")


if __name__ == '__main__':
    main()
