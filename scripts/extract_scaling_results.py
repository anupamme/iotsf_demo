#!/usr/bin/env python3
"""Extract scaling experiment results and format for LaTeX table.

Usage:
    python scripts/extract_scaling_results.py

Reads from results directories and prints formatted summary for updating
tables/scaling.tex and paper text.
"""

import json
import sys
from pathlib import Path

RESULTS_ROOT = Path("results")

# Map of (scale_label, results_dir, conditions)
SCALES = [
    ("200/5 (10-seed)", "ablation_10seed", ["b", "c", "cprime", "d"]),
    ("500/10 (5-seed)", "ablation_500_es", ["b", "c", "cprime", "d"]),
    ("500/10 (10-seed)", "ablation_500_es_10seed", ["b", "c", "d"]),
    ("1000/20 (5-seed)", "ablation_scaled_es", ["b", "c", "d"]),
    ("1000/20 (10-seed)", "ablation_scaled_es_10seed", ["b", "c", "d"]),
    ("Frozen 1000/20 (5-seed)", "ablation_frozen_es", ["b", "c", "d"]),
    ("Frozen 1000/20 (10-seed)", "ablation_frozen_es_10seed", ["b", "c", "d"]),
    ("L2-SP 1000/20", "ablation_l2sp_1k", ["b", "c", "d"]),
]


def load_metrics(results_dir: str, condition: str):
    path = RESULTS_ROOT / results_dir / "ablation" / condition / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt(mean, std):
    """Format mean±std as LaTeX."""
    return f"${mean:.3f}_{{\\pm{std:.3f}}}$"


def main():
    print("=" * 100)
    print("SCALING RESULTS SUMMARY")
    print("=" * 100)

    for scale_label, results_dir, conditions in SCALES:
        dir_path = RESULTS_ROOT / results_dir
        if not dir_path.exists():
            print(f"\n--- {scale_label} ({results_dir}): DIRECTORY NOT FOUND ---")
            continue

        print(f"\n--- {scale_label} ({results_dir}) ---")
        print(f"{'Cond':<8} {'Seeds':<6} {'S95 AUC':<20} {'S95 F1':<20} {'S95 FPR':<20} {'Comb AUC':<20} {'Comb F1':<20}")
        print("-" * 100)

        for cond in conditions:
            m = load_metrics(results_dir, cond)
            if m is None:
                print(f"{cond:<8} ---")
                continue

            n_seeds = len(m.get("seeds", []))

            for stealth_key, label in [("stealth_95", "S95"), ("combined", "Comb")]:
                r = m["results"].get(stealth_key, {})
                if not r:
                    continue

            s95 = m["results"].get("stealth_95", {})
            comb = m["results"].get("combined", {})

            def safe_fmt(d, key):
                v = d.get(key, {})
                if isinstance(v, dict) and "mean" in v:
                    return f"{v['mean']:.3f}±{v['std']:.3f}"
                return "---"

            s95_auc = safe_fmt(s95, "roc_auc")
            s95_f1 = safe_fmt(s95, "f1")
            s95_fpr = safe_fmt(s95, "false_positive_rate")
            comb_auc = safe_fmt(comb, "roc_auc") if comb else "---"
            comb_f1 = safe_fmt(comb, "f1") if comb else "---"

            print(f"{cond:<8} {n_seeds:<6} {s95_auc:<18} {s95_f1:<18} {s95_fpr:<18} {comb_auc:<18} {comb_f1:<18}")

    # LaTeX-ready output for scaling table
    print("\n" + "=" * 100)
    print("LATEX-READY VALUES (stealth-95 only, for scaling table)")
    print("=" * 100)

    for scale_label, results_dir, conditions in SCALES:
        dir_path = RESULTS_ROOT / results_dir
        if not dir_path.exists():
            continue

        has_results = False
        for cond in conditions:
            m = load_metrics(results_dir, cond)
            if m is not None:
                has_results = True
                break
        if not has_results:
            continue

        print(f"\n% {scale_label}")
        for cond in conditions:
            m = load_metrics(results_dir, cond)
            if m is None:
                continue
            s95 = m["results"].get("stealth_95", {})
            auc = s95.get("roc_auc", {})
            f1 = s95.get("f1", {})
            if auc and f1:
                latex_auc = fmt(auc["mean"], auc["std"])
                latex_f1 = fmt(f1["mean"], f1["std"])
                print(f"% {cond}: AUC={latex_auc}  F1={latex_f1}")


if __name__ == "__main__":
    main()
