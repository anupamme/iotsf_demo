#!/usr/bin/env python3
"""Generate paper tables as CSVs from experiment results.

Produces:
  - table2_etth2_sample_sweep.csv
  - table3_task_native_probe.csv
  - table4_diagnostic_summary.csv
  - table5_drift_diagnosis.csv

Usage:
    python scripts/06_make_tables.py --input runs/ --output expected_outputs/
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_results


def make_table2_sample_sweep(results: list, output_dir: Path):
    """Table 2: Sample-size sweep on ETTh2 (condition B, h=96)."""
    # Group by max_train_samples
    groups = {}
    for r in results:
        if r.get("condition") != "B" or r.get("horizon") != 96:
            continue
        n = r.get("max_train_samples", 0)
        groups.setdefault(n, []).append(r)

    rows = []
    for n_samples in sorted(groups.keys()):
        runs = groups[n_samples]
        mses = [r["final_val_mse"] for r in runs]
        forgs = [r["forgetting_pct"] for r in runs]
        ckas = [r["final_cka"] for r in runs]
        rows.append({
            "n_samples": n_samples,
            "condition": "B",
            "horizon": 96,
            "n_seeds": len(runs),
            "mean_mse": f"{np.mean(mses):.4f}",
            "std_mse": f"{np.std(mses):.4f}",
            "mean_forgetting_pct": f"{np.mean(forgs):.1f}",
            "std_forgetting_pct": f"{np.std(forgs):.1f}",
            "mean_cka": f"{np.mean(ckas):.4f}",
            "std_cka": f"{np.std(ckas):.4f}",
        })

    path = output_dir / "table2_etth2_sample_sweep.csv"
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"  Written: {path} ({len(rows)} rows)")
    else:
        print(f"  Skipped: {path} (no matching results)")


def make_table3_task_native_probe(results: list, output_dir: Path):
    """Table 3: Task-native probe R^2_task at key operating points."""
    rows = []
    for r in results:
        pre = r.get("probe_r2_pre")
        post = r.get("probe_r2_post")
        if pre is None or post is None:
            continue
        if not isinstance(pre, (int, float)):
            continue
        rows.append({
            "dataset": r.get("dataset", "ETTh2"),
            "condition": r.get("condition"),
            "seed": r.get("seed"),
            "horizon": r.get("horizon"),
            "n_samples": r.get("max_train_samples"),
            "r2_pre": f"{pre:.4f}",
            "r2_post": f"{post:.4f}",
            "delta_r2": f"{post - pre:.4f}",
        })

    path = output_dir / "table3_task_native_probe.csv"
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"  Written: {path} ({len(rows)} rows)")
    else:
        print(f"  Skipped: {path} (no probe data)")


def make_table4_diagnostic_summary(results: list, output_dir: Path):
    """Table 4: Cross-domain diagnostic summary."""
    # Group by (condition, horizon, n)
    groups = {}
    for r in results:
        key = (r.get("condition"), r.get("horizon"), r.get("max_train_samples"))
        groups.setdefault(key, []).append(r)

    rows = []
    for (cond, horizon, n), runs in sorted(groups.items()):
        if cond == "A":
            continue
        forgs = [r["forgetting_pct"] for r in runs if "forgetting_pct" in r]
        ckas = [r["final_cka"] for r in runs if "final_cka" in r]
        if not forgs:
            continue

        mean_forg = np.mean(forgs)
        mean_cka = np.mean(ckas) if ckas else None

        if mean_forg > 5:
            verdict = "harmful_drift"
        elif mean_forg < -5:
            verdict = "beneficial_restructuring"
        else:
            verdict = "stable"

        rows.append({
            "condition": cond,
            "horizon": horizon,
            "n_samples": n,
            "n_seeds": len(runs),
            "mean_forgetting_pct": f"{mean_forg:.1f}",
            "mean_cka": f"{mean_cka:.4f}" if mean_cka else "N/A",
            "verdict": verdict,
        })

    path = output_dir / "table4_diagnostic_summary.csv"
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"  Written: {path} ({len(rows)} rows)")
    else:
        print(f"  Skipped: {path} (no data)")


def make_table5_drift_diagnosis(results: list, output_dir: Path):
    """Table 5: Drift diagnosis combining CKA + probe delta."""
    rows = []
    for r in results:
        cka = r.get("final_cka")
        delta = r.get("probe_delta_r2")
        if cka is None or delta is None:
            continue

        if cka > 0.95 and abs(delta) < 0.05:
            diagnosis = "stable"
        elif cka < 0.90 and delta < -0.05:
            diagnosis = "harmful_forgetting"
        elif cka < 0.95 and delta > 0.05:
            diagnosis = "beneficial_restructuring"
        else:
            diagnosis = "mixed"

        rows.append({
            "condition": r.get("condition"),
            "seed": r.get("seed"),
            "horizon": r.get("horizon"),
            "n_samples": r.get("max_train_samples"),
            "cka": f"{cka:.4f}",
            "probe_delta_r2": f"{delta:.4f}",
            "forgetting_pct": f"{r.get('forgetting_pct', 0):.1f}",
            "diagnosis": diagnosis,
        })

    path = output_dir / "table5_drift_diagnosis.csv"
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"  Written: {path} ({len(rows)} rows)")
    else:
        print(f"  Skipped: {path} (no probe+CKA data)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/", help="Results directory")
    parser.add_argument("--output", default="expected_outputs/", help="Output CSV directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.input)
    print(f"Loaded {len(results)} result files from {args.input}\n")

    make_table2_sample_sweep(results, output_dir)
    make_table3_task_native_probe(results, output_dir)
    make_table4_diagnostic_summary(results, output_dir)
    make_table5_drift_diagnosis(results, output_dir)

    print("\nDone. Compare outputs against paper tables.")


if __name__ == "__main__":
    main()
