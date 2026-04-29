#!/usr/bin/env python3
"""
Compute Maximum Mean Discrepancy (MMD) between two sets of synthetic
attack distributions (e.g., analytical D vs Diffusion-TS D-DiffTS).

Uses an RBF kernel with the median heuristic for bandwidth selection.
Reports per-attack-type, per-stealth-level, and aggregate MMD values
with permutation-test p-values.

Usage:
    python scripts/compute_mmd.py \
        --dir-a data/synthetic \
        --dir-b data/synthetic_diffts \
        --output results/mmd_d_vs_diffts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Compute RBF (Gaussian) kernel matrix between X and Y."""
    # X: (n, d), Y: (m, d) -> (n, m)
    XX = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)  # (m, 1)
    dists = XX + YY.T - 2.0 * X @ Y.T           # (n, m)
    return np.exp(-dists / (2.0 * sigma ** 2))


def compute_mmd2(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Compute unbiased MMD^2 estimate with RBF kernel."""
    n = len(X)
    m = len(Y)
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)

    # Unbiased estimator: exclude diagonal for Kxx and Kyy
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    mmd2 = (Kxx.sum() / (n * (n - 1))
            + Kyy.sum() / (m * (m - 1))
            - 2.0 * Kxy.mean())
    return float(mmd2)


def median_heuristic(X: np.ndarray, Y: np.ndarray, n_subsample: int = 1000) -> float:
    """Estimate RBF bandwidth via median of pairwise distances."""
    rng = np.random.default_rng(42)
    combined = np.concatenate([X, Y])
    if len(combined) > n_subsample:
        idx = rng.choice(len(combined), size=n_subsample, replace=False)
        combined = combined[idx]
    dists = np.sqrt(np.sum((combined[:, None] - combined[None, :]) ** 2, axis=-1))
    # Take median of off-diagonal distances
    mask = ~np.eye(len(combined), dtype=bool)
    sigma = float(np.median(dists[mask]))
    return max(sigma, 1e-6)


def permutation_test(X: np.ndarray, Y: np.ndarray, sigma: float,
                     n_perms: int = 500, seed: int = 42) -> float:
    """Permutation test p-value for MMD^2 > 0."""
    rng = np.random.default_rng(seed)
    observed = compute_mmd2(X, Y, sigma)
    combined = np.concatenate([X, Y])
    n = len(X)
    count = 0
    for _ in range(n_perms):
        perm = rng.permutation(len(combined))
        X_perm = combined[perm[:n]]
        Y_perm = combined[perm[n:]]
        if compute_mmd2(X_perm, Y_perm, sigma) >= observed:
            count += 1
    return (count + 1) / (n_perms + 1)


def main():
    parser = argparse.ArgumentParser(description="Compute MMD between two synthetic distributions")
    parser.add_argument("--dir-a", required=True, help="First synthetic directory (e.g., data/synthetic)")
    parser.add_argument("--dir-b", required=True, help="Second synthetic directory (e.g., data/synthetic_diffts)")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--n-perms", type=int, default=500, help="Number of permutations for p-value")
    args = parser.parse_args()

    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)

    # Discover matching attack files
    attack_types = ["beacon", "lotl_mimicry", "protocol_anomaly", "slow_exfiltration"]
    stealth_levels = [85, 90, 95]

    results = {}
    all_a_flat = []
    all_b_flat = []

    for attack in attack_types:
        results[attack] = {}
        for stealth in stealth_levels:
            fname = f"{attack}_stealth_{stealth}.npy"
            path_a = dir_a / fname
            path_b = dir_b / fname

            if not path_a.exists() or not path_b.exists():
                print(f"  SKIP {fname}: missing in one or both dirs")
                continue

            X = np.load(path_a).astype(np.float32)
            Y = np.load(path_b).astype(np.float32)

            # Flatten from (N, seq_len, feat_dim) to (N, seq_len*feat_dim)
            X_flat = X.reshape(len(X), -1)
            Y_flat = Y.reshape(len(Y), -1)

            all_a_flat.append(X_flat)
            all_b_flat.append(Y_flat)

            sigma = median_heuristic(X_flat, Y_flat)
            mmd2 = compute_mmd2(X_flat, Y_flat, sigma)
            pval = permutation_test(X_flat, Y_flat, sigma, n_perms=args.n_perms)

            key = f"stealth_{stealth}"
            results[attack][key] = {
                "mmd2": round(mmd2, 6),
                "mmd": round(max(mmd2, 0) ** 0.5, 6),
                "sigma": round(sigma, 4),
                "n_a": len(X),
                "n_b": len(Y),
                "p_value": round(pval, 4),
            }
            print(f"  {attack}/stealth-{stealth}: MMD={results[attack][key]['mmd']:.4f}, "
                  f"p={pval:.4f}, sigma={sigma:.2f}")

    # Aggregate across all attack types and stealth levels
    if all_a_flat and all_b_flat:
        X_all = np.concatenate(all_a_flat)
        Y_all = np.concatenate(all_b_flat)
        sigma_all = median_heuristic(X_all, Y_all)
        mmd2_all = compute_mmd2(X_all, Y_all, sigma_all)
        pval_all = permutation_test(X_all, Y_all, sigma_all, n_perms=args.n_perms)
        results["aggregate"] = {
            "mmd2": round(mmd2_all, 6),
            "mmd": round(max(mmd2_all, 0) ** 0.5, 6),
            "sigma": round(sigma_all, 4),
            "n_a": len(X_all),
            "n_b": len(Y_all),
            "p_value": round(pval_all, 4),
        }
        print(f"\n  AGGREGATE: MMD={results['aggregate']['mmd']:.4f}, "
              f"p={pval_all:.4f}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
