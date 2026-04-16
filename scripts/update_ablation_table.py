#!/usr/bin/env python3
"""
Update paper/tables/ablation.tex from results/ablation/*/metrics.json.

Usage:
    python scripts/update_ablation_table.py

Reads the per-condition JSON files written by run_ablation.py (fixed-calibration
200-sample run) and regenerates the LaTeX table body with updated numbers.
"""

import json
import re
from pathlib import Path

RESULTS_DIR = Path("results/ablation")
TABLE_FILE  = Path("paper/tables/ablation.tex")

# Map condition key → table row description (matches existing table)
CONDITION_DESCRIPTIONS = {
    "a": r"Moirai zero-shot",
    "b": r"$+$ Fine-tune, NLL only",
    "c": r"$+$ SupCon (real attacks)",
    "d": r"$+$ Hard negatives (\ours{}$^\dagger$)",
    "e": r"D w/o constraints, w/o retry",
    "eprime": r"D w/o constraints, \emph{with} retry$^\ddagger$",
}

CONDITION_LABELS = {
    "a": "A",
    "b": "B",
    "c": "C",
    "d": "D",
    "e": "E",
    "eprime": r"E$'$",
}


def fmt(mean: float, std: float) -> str:
    """Format as $0.704_{\pm.003}$ — omits leading zero in std if < 0.1."""
    mean_str = f"{mean:.3f}"
    std_str = f"{std:.3f}"[1:]  # strip leading zero: ".003"
    return fr"${mean_str}_{{\pm{std_str}}}$"


def fmt_fpr(mean: float) -> str:
    """FPR has no std in the table (single value)."""
    return f"${mean:.3f}$"


def load_condition(cond_key: str):
    """Load metrics.json for a condition; return None if missing."""
    p = RESULTS_DIR / cond_key / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def build_row(cond_key: str, data: dict) -> str:
    """Build a single LaTeX table row."""
    label = CONDITION_LABELS[cond_key]
    desc  = CONDITION_DESCRIPTIONS[cond_key]
    r     = data["results"]

    s95 = r.get("stealth_95", {})
    all_s = r.get("all_stealth", {})

    def get(d, key):
        v = d.get(key, {})
        if isinstance(v, dict):
            return v.get("mean", 0.0), v.get("std", 0.0)
        return float(v), 0.0

    f1_95, f1_95_std   = get(s95, "f1")
    fpr_95, _          = get(s95, "false_positive_rate")
    auc_95, auc_95_std = get(s95, "roc_auc")
    f1_all, f1_all_std = get(all_s, "f1")
    auc_all, auc_all_std = get(all_s, "roc_auc")

    seeds = data.get("seeds", [])
    n_seeds = len(seeds) if isinstance(seeds, list) else seeds

    return (
        f"{label:<6} & {desc:<50} & "
        f"{fmt(f1_95, f1_95_std)} & "
        f"{fmt_fpr(fpr_95)} & "
        f"{fmt(auc_95, auc_95_std)} & "
        f"{fmt(f1_all, f1_all_std)} & "
        f"{fmt(auc_all, auc_all_std)} \\\\"
    )


def main():
    rows = {}
    for cond_key in ["a", "b", "c", "d", "e", "eprime"]:
        data = load_condition(cond_key)
        if data is None:
            print(f"  SKIP {cond_key}: no metrics.json")
            continue
        rows[cond_key] = build_row(cond_key, data)
        print(f"  OK   {cond_key}")

    # Read existing table
    tex = TABLE_FILE.read_text()

    # Replace midrule…bottomrule block
    midrule_start = tex.index(r"\midrule") + len(r"\midrule")
    bottomrule_pos = tex.index(r"\bottomrule")
    old_body = tex[midrule_start:bottomrule_pos]

    new_lines = ["\n"]
    for cond_key in ["a", "b", "c", "d", "e", "eprime"]:
        if cond_key in rows:
            new_lines.append(rows[cond_key] + "\n")
        else:
            # Preserve existing row if result not yet available
            label = CONDITION_LABELS[cond_key]
            match = re.search(
                rf"^{re.escape(label)}.*$",
                old_body,
                re.MULTILINE,
            )
            if match:
                new_lines.append(match.group(0) + "\n")

    # E'' row is always analytic ≡ D — preserve it
    edoubleprime_match = re.search(
        r"E\$''\$.*?\\\\\n",
        old_body,
        re.DOTALL,
    )
    if edoubleprime_match:
        new_lines.append(edoubleprime_match.group(0))
    else:
        new_lines.append(
            r"E$''$ & D \emph{with} constraints, w/o retry$^\S$"
            r"     & \multicolumn{5}{c}{$\equiv$ D (analytically)} \\" + "\n"
        )

    new_body = "".join(new_lines)
    new_tex = tex[:midrule_start] + new_body + tex[bottomrule_pos:]

    # Update header comment to reflect new run parameters
    seeds_data = None
    for ck in ["a", "b", "c", "d"]:
        d = load_condition(ck)
        if d:
            seeds_data = d
            break

    if seeds_data:
        seeds = seeds_data.get("seeds", [42, 123, 456])
        seed_str = ",".join(str(s) for s in seeds)
        n = len(seeds) if isinstance(seeds, list) else seeds
        new_tex = re.sub(
            r"% Results from run_ablation\.py.*",
            f"% Results from run_ablation.py --condition all --seeds {seed_str} "
            f"--epochs 5 --max-eval-samples 200 (fixed benign-calibrated threshold)",
            new_tex,
        )
        new_tex = re.sub(
            r"% All conditions run with \d+ random seeds.*",
            f"% All conditions run with {n} random seeds (seeds {seed_str}) using the "
            "full neural pipeline (uni2ts/Moirai).",
            new_tex,
        )

    TABLE_FILE.write_text(new_tex)
    print(f"\nWrote updated table to {TABLE_FILE}")


if __name__ == "__main__":
    main()
