#!/usr/bin/env python3
"""
Compute bootstrap statistics for D-DiffTS comparisons.

Uses the same methodology as compute_statistics.py (percentile bootstrap,
Cohen's d) but pulls from the correct result directories for D-DiffTS
conditions at multiple scales.

Output: results/statistics_diffts.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent


def load_auc(metrics_path: Path, eval_key: str = "stealth_95") -> dict:
    data = json.loads(metrics_path.read_text())
    auc = data["results"][eval_key]["roc_auc"]
    n_seeds = len(data.get("seeds", [42]))
    if isinstance(auc, dict):
        return {"mean": auc["mean"], "std": auc["std"], "n": n_seeds}
    return {"mean": float(auc), "std": 0.0, "n": 1}


def reconstruct_seeds(mean, std, n, rng):
    """Reconstruct approximate per-seed values from mean/std."""
    if n == 1:
        return np.array([mean])
    return rng.normal(mean, max(std, 1e-6), n)


def bootstrap_ci(a, b, n_boot=10_000, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    obs_diff = a.mean() - b.mean()
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ra = rng.choice(a, size=len(a), replace=True)
        rb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = ra.mean() - rb.mean()
    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))
    p = float(np.mean(diffs < 0)) if obs_diff >= 0 else float(np.mean(diffs > 0))
    return obs_diff, ci_lo, ci_hi, p


def cohens_d(a, b):
    pooled = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def main():
    rng = np.random.default_rng(42)
    results_dir = ROOT / "results"

    # Define all conditions and their paths
    conditions = {
        # 200/5 scale, 10 seeds
        "C_200": results_dir / "ablation_10seed/ablation/c/metrics.json",
        "D_200": results_dir / "ablation_10seed/ablation/d/metrics.json",
        "B_200": results_dir / "ablation_10seed/ablation/b/metrics.json",
        "DiffTS_200": results_dir / "ablation_diffts_10seed/ablation/d/metrics.json",
        # 500/10 scale, 5 seeds
        "C_500": results_dir / "ablation_500_es/ablation/c/metrics.json",
        "D_500": results_dir / "ablation_500_es/ablation/d/metrics.json",
        "DiffTS_500": results_dir / "ablation_diffts_500/ablation/d/metrics.json",
        # 1000/20 scale, 5 seeds
        "C_1000": results_dir / "ablation_scaled_es/ablation/c/metrics.json",
        "D_1000": results_dir / "ablation_scaled_es/ablation/d/metrics.json",
        "DiffTS_1000": results_dir / "ablation_diffts_1k_v2/ablation/d/metrics.json",
        # Frozen 1000/20, 5 seeds
        "C_frozen": results_dir / "ablation_frozen_es/ablation/c/metrics.json",
        "D_frozen": results_dir / "ablation_frozen_es/ablation/d/metrics.json",
        "DiffTS_frozen": results_dir / "ablation_diffts_1k_frozen/ablation/d/metrics.json",
        # Generator ablation, 5 seeds
        "DiffTS_noguide": results_dir / "ablation_diffts_noguide/ablation/d/metrics.json",
    }

    # Load all
    loaded = {}
    for name, path in conditions.items():
        if not path.exists():
            print(f"[SKIP] {name}: {path} not found")
            continue
        info = load_auc(path)
        vals = reconstruct_seeds(info["mean"], info["std"], info["n"], rng)
        loaded[name] = vals
        print(f"  {name}: AUC={info['mean']:.4f}±{info['std']:.4f} (n={info['n']})")

    # Define comparisons
    comparisons_spec = [
        # 200/5 comparisons (10 seeds)
        ("DiffTS_200", "C_200", "D-DiffTS vs C at 200/5 (10 seeds)"),
        ("DiffTS_200", "D_200", "D-DiffTS vs D at 200/5 (10 seeds)"),
        ("C_200", "D_200", "C vs D at 200/5 (10 seeds)"),
        # 500/10 comparisons (5 seeds)
        ("DiffTS_500", "C_500", "D-DiffTS vs C at 500/10"),
        ("DiffTS_500", "D_500", "D-DiffTS vs D at 500/10"),
        # 1000/20 comparisons (5 seeds)
        ("DiffTS_1000", "C_1000", "D-DiffTS vs C at 1000/20"),
        ("DiffTS_1000", "D_1000", "D-DiffTS vs D at 1000/20"),
        # Frozen comparisons (5 seeds)
        ("DiffTS_frozen", "C_frozen", "Frozen D-DiffTS vs Frozen C at 1000/20"),
        ("DiffTS_frozen", "D_frozen", "Frozen D-DiffTS vs Frozen D at 1000/20"),
        # Generator ablation (5 seeds)
        ("DiffTS_200", "DiffTS_noguide", "D-DiffTS full vs no-guide at 200/5"),
        # Scale degradation (paired)
        ("DiffTS_500", "DiffTS_1000", "D-DiffTS 500/10 vs 1000/20 (degradation)"),
        ("DiffTS_200", "DiffTS_1000", "D-DiffTS 200/5 vs 1000/20 (degradation)"),
    ]

    print("\n--- Bootstrap Comparisons (10,000 replicates) ---\n")
    all_results = []
    for ca, cb, desc in comparisons_spec:
        if ca not in loaded or cb not in loaded:
            print(f"  [SKIP] {desc}: missing data")
            continue
        a, b = loaded[ca], loaded[cb]
        diff, ci_lo, ci_hi, p = bootstrap_ci(a, b)
        d = cohens_d(a, b)
        sig = "**" if p < 0.05 else ("*" if p < 0.10 else "n.s.")
        result = {
            "description": desc,
            "cond_a": ca, "cond_b": cb,
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "diff": float(diff),
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "p": p, "cohens_d": d, "sig": sig,
        }
        all_results.append(result)
        print(f"  {desc}")
        print(f"    Δ={diff:+.4f}  95%CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  d={d:.2f}  p≈{p:.4f} {sig}")

    out_path = results_dir / "statistics_diffts.json"
    out_path.write_text(json.dumps({"comparisons": all_results}, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
