#!/usr/bin/env python3
"""
Aggregate V5 experiment results into summary tables for the paper.
Reads JSON result files from results/v5_* directories and produces:
1. Multi-dataset forgetting table (ETTh1, ETTh2, ETTm2)
2. Mitigation spectrum table (LoRA, L2-SP, EWC, Frozen)
3. Moirai-Base comparison
4. Sample size sweep
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_results(results_dir):
    """Load all JSON result files from a directory tree."""
    results = []
    for f in Path(results_dir).rglob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            data["_file"] = str(f)
            results.append(data)
        except (json.JSONDecodeError, IOError):
            pass
    return results


def group_by(results, key):
    """Group results by a key."""
    groups = defaultdict(list)
    for r in results:
        groups[r.get(key, "unknown")].append(r)
    return groups


def summarize_metric(results, metric):
    """Compute mean ± std for a metric across results."""
    vals = [r[metric] for r in results if metric in r]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def fmt(mean, std, decimals=3):
    """Format mean±std."""
    if mean is None:
        return "—"
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_forgetting_table(dataset_name, results_dir, horizons=[96, 192]):
    """Print forgetting diagnosis table for a dataset."""
    results = load_results(results_dir)
    if not results:
        print(f"  No results found in {results_dir}")
        return

    for h in horizons:
        h_results = [r for r in results if r.get("horizon") == h]
        if not h_results:
            continue

        # Get zero-shot MSE from condition A
        a_results = [r for r in h_results if r.get("condition") == "A"]
        zs_mse, _ = summarize_metric(a_results, "zeroshot_mse")

        zs_str = f"{zs_mse:.3f}" if zs_mse else "?"
        print(f"\n  {dataset_name} h={h} (Zero-shot MSE: {zs_str})")
        print(f"  {'Condition':<16} {'MSE':<16} {'Forg.%':<16} {'CKA':<16} {'Drift':<12}")
        print(f"  {'-'*72}")

        for cond in ["B", "C", "D"]:
            c_results = [r for r in h_results if r.get("condition") == cond]
            if not c_results:
                continue

            mse_mean, mse_std = summarize_metric(c_results, "final_val_mse")
            cka_mean, cka_std = summarize_metric(c_results, "final_cka")
            drift_mean, drift_std = summarize_metric(c_results, "final_weight_drift")

            # Compute forgetting %
            if zs_mse and mse_mean:
                forg_vals = [(r["final_val_mse"] - zs_mse) / zs_mse * 100
                             for r in c_results if "final_val_mse" in r]
                forg_mean = float(np.mean(forg_vals))
                forg_std = float(np.std(forg_vals))
                forg_str = f"{forg_mean:+.1f}±{forg_std:.1f}"
            else:
                forg_str = "—"

            print(f"  {cond:<16} {fmt(mse_mean, mse_std):<16} {forg_str:<16} "
                  f"{fmt(cka_mean, cka_std):<16} {fmt(drift_mean, drift_std, 2):<12}")


def print_mitigation_table(horizons=[96, 192]):
    """Print mitigation spectrum table."""
    # Load ETTh2 baseline results (conditions A, B, C, D from original experiments)
    etth2_base = load_results(ROOT / "results" / "v5_etth2_base")  # or existing ETTh2 results

    # Load mitigation results
    lora = load_results(ROOT / "results" / "v5_mitigation" / "lora")
    l2sp_001 = load_results(ROOT / "results" / "v5_mitigation" / "l2sp_0.01")
    l2sp_01 = load_results(ROOT / "results" / "v5_mitigation" / "l2sp_0.1")
    ewc_100 = load_results(ROOT / "results" / "v5_mitigation" / "ewc_100")
    ewc_1000 = load_results(ROOT / "results" / "v5_mitigation" / "ewc_1000")

    for h in horizons:
        print(f"\n  Mitigation Spectrum — ETTh2 h={h}")
        print(f"  {'Method':<20} {'MSE':<16} {'CKA':<16} {'Drift':<12}")
        print(f"  {'-'*60}")

        for label, data in [
            ("E: LoRA (r=8)", lora),
            ("F: L2-SP (λ=0.01)", l2sp_001),
            ("F: L2-SP (λ=0.1)", l2sp_01),
            ("G: EWC (λ=100)", ewc_100),
            ("G: EWC (λ=1000)", ewc_1000),
        ]:
            h_results = [r for r in data if r.get("horizon") == h]
            if not h_results:
                print(f"  {label:<20} [no results yet]")
                continue

            mse_mean, mse_std = summarize_metric(h_results, "final_val_mse")
            cka_mean, cka_std = summarize_metric(h_results, "final_cka")
            drift_mean, drift_std = summarize_metric(h_results, "final_weight_drift")

            print(f"  {label:<20} {fmt(mse_mean, mse_std):<16} {fmt(cka_mean, cka_std):<16} "
                  f"{fmt(drift_mean, drift_std, 2):<12}")


def print_sample_sweep():
    """Print sample size sweep results."""
    results = load_results(ROOT / "results" / "v5_etth2_sweep")
    if not results:
        print("  No results found")
        return

    print(f"\n  {'Samples':<10} {'MSE':<16} {'Forg.%':<16} {'CKA':<16}")
    print(f"  {'-'*54}")

    # Group by max_train_samples
    for n in [200, 500, 1000, 2000]:
        n_results = [r for r in results
                     if r.get("max_train_samples") == n or
                     (f"n{n}" in r.get("_file", ""))]
        if not n_results:
            print(f"  {n:<10} [no results yet]")
            continue

        mse_mean, mse_std = summarize_metric(n_results, "final_val_mse")
        cka_mean, cka_std = summarize_metric(n_results, "final_cka")

        # Get zero-shot for forgetting calc
        zs_vals = [r.get("zeroshot_mse") for r in n_results if "zeroshot_mse" in r]
        forg_vals = [(r["final_val_mse"] - r["zeroshot_mse"]) / r["zeroshot_mse"] * 100
                     for r in n_results if "final_val_mse" in r and "zeroshot_mse" in r]
        if forg_vals:
            forg_str = f"{np.mean(forg_vals):+.1f}±{np.std(forg_vals):.1f}%"
        else:
            forg_str = "—"

        print(f"  {n:<10} {fmt(mse_mean, mse_std):<16} {forg_str:<16} {fmt(cka_mean, cka_std):<16}")


def main():
    print_section("Multi-Dataset Forgetting Diagnosis")

    # ETTh2 (original — check if v5 results exist, otherwise use existing)
    etth2_dir = ROOT / "results" / "v5_etth2"
    if not etth2_dir.exists() or not list(etth2_dir.rglob("*.json")):
        # Fall back to original forecasting results
        etth2_dir = ROOT / "results" / "forecasting"
    print_forgetting_table("ETTh2", etth2_dir)

    print_forgetting_table("ETTh1", ROOT / "results" / "v5_etth1")
    print_forgetting_table("ETTm2", ROOT / "results" / "v5_ettm2")

    print_section("Moirai-Base on ETTh2")
    print_forgetting_table("ETTh2 (Base)", ROOT / "results" / "v5_etth2_base")

    print_section("Mitigation Spectrum (ETTh2)")
    print_mitigation_table()

    print_section("Sample Size Sweep (ETTh2 h=96, Condition B)")
    print_sample_sweep()

    # Summary counts
    total = 0
    for d in ["v5_etth1", "v5_ettm2", "v5_etth2_base", "v5_etth2_sweep", "v5_mitigation"]:
        n = len(list((ROOT / "results" / d).rglob("*.json"))) if (ROOT / "results" / d).exists() else 0
        total += n
        print(f"\n  {d}: {n} result files")
    print(f"\n  Total new V5 results: {total}")


if __name__ == "__main__":
    main()
