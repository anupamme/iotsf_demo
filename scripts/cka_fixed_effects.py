#!/usr/bin/env python3
"""
Does CKA order the intervention once backbone identity is absorbed?

WHY THIS EXISTS
---------------
The paper reports a strong POOLED rank correlation between CKA and held-out B-D
(rho=+0.741, CI excludes 0) and then declines to license it, on the grounds that CKA is not
comparable across architectures: Chronos sits at 0.09-0.23, Moirai at 0.40-0.97, TimesFM at
0.25-0.56, so "CKA" and "which backbone" are nearly the same variable and the pooled statistic is
mostly reading backbone identity. A reviewer can fairly reply that the within-backbone restriction
is doing the work -- that we adopted the split that removes the significant relationship.

This module answers that formally instead of by argument. It fits

    additive     B-D = b0 + b1*CKA + backbone fixed effects
    interaction  B-D = b0 + b1*CKA + backbone FE + CKA x backbone

so backbone identity is absorbed as a nuisance parameter rather than by subsetting, and b1 is the
partial slope of B-D on CKA *within* backbones, estimated on all cells at once.

INFERENCE. Cells are not independent: datasets and checkpoints recur across horizons and sizes.
A naive OLS standard error would therefore be far too small. We bootstrap by resampling CLUSTERS,
where a cluster is a (backbone, dataset) pair -- the level at which the shared pretraining data,
the shared series and the shared checkpoint induce correlation. Clusters are resampled with
replacement; a draw that leaves fewer than three distinct backbones or is rank-deficient is
discarded, which is why `kept` is reported alongside every interval.

WHAT THIS CAN AND CANNOT SHOW. A CI on b1 that includes zero does not prove CKA carries no
information; at these cell counts it shows the data cannot pin the within-backbone slope down. That
is the paper's claim, and it is deliberately weaker than "CKA is uninformative".

Run:  python3 scripts/cka_fixed_effects.py
      python3 scripts/cka_fixed_effects.py --boot 20000 --seed 7
"""
import argparse
import contextlib
import io
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import cell_matrix  # noqa: E402


def backbone_of(cell):
    if cell.startswith("TimesFM"):
        return "TimesFM"
    if cell.startswith("Chronos"):
        return "Chronos"
    return "Moirai"


def dataset_of(cell):
    """Cluster key: the series a cell is fitted on, however the cell is spelled."""
    m = re.search(r"(ETTh1|ETTh2|ETTm2|Weather|Electricity7|Electricity|ILI|M4)", cell,
                  re.IGNORECASE)
    return m.group(1).lower() if m else "other"


def load():
    with contextlib.redirect_stdout(io.StringIO()):
        rows = cell_matrix.build_rows()
    rows = [r for r in rows if r.get("bd_test") is not None and r.get("cka") is not None]
    y = np.array([r["bd_test"] for r in rows], float)
    x = np.array([r["cka"] for r in rows], float)
    bb = np.array([backbone_of(r["cell"]) for r in rows])
    cl = np.array([f"{backbone_of(r['cell'])}|{dataset_of(r['cell'])}" for r in rows])
    return rows, y, x, bb, cl


def design(x, bb, levels, interaction):
    """Backbone FE with the first level as reference, optionally x by backbone."""
    cols = [np.ones_like(x), x]
    for lv in levels[1:]:
        cols.append((bb == lv).astype(float))
    if interaction:
        for lv in levels[1:]:
            cols.append(x * (bb == lv))
    return np.column_stack(cols)


def fit(M, y):
    beta, *_ = np.linalg.lstsq(M, y, rcond=None)
    return beta


def cluster_bootstrap(y, x, bb, cl, levels, interaction, n_boot, rng):
    """Resample clusters with replacement; b1 is always index 1 of the design."""
    uniq = np.unique(cl)
    idx_by_cluster = {c: np.flatnonzero(cl == c) for c in uniq}
    out = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in draw])
        yb, xb, bbb = y[idx], x[idx], bb[idx]
        if len(np.unique(bbb)) < len(levels):
            continue                      # FE not identified in this draw
        Mb = design(xb, bbb, levels, interaction)
        if np.linalg.matrix_rank(Mb) < Mb.shape[1]:
            continue
        out.append(fit(Mb, yb)[1])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    rows, y, x, bb, cl = load()
    levels = ["Moirai", "Chronos", "TimesFM"]          # reference = Moirai, the largest arm
    levels = [lv for lv in levels if (bb == lv).any()]

    print("=" * 92)
    print("DOES CKA ORDER B-D ONCE BACKBONE IDENTITY IS ABSORBED?")
    print("=" * 92)
    print(f"  cells {len(y)}   clusters (backbone x dataset) {len(np.unique(cl))}")
    for lv in levels:
        m = bb == lv
        print(f"    {lv:8s} n={m.sum():2d}  CKA {x[m].min():.2f}-{x[m].max():.2f}"
              f"   B-D {y[m].min():+7.1f} to {y[m].max():+7.1f}")

    # unadjusted slope, for contrast with the pooled rank statistic the paper already prints
    b_un = fit(np.column_stack([np.ones_like(x), x]), y)[1]
    ci_un = cluster_bootstrap(y, x, bb, cl, ["_all"], False, a.boot, rng) if False else None
    print(f"\n  unadjusted slope (no backbone term)      b1 = {b_un:+8.2f} pp per unit CKA")

    for interaction in (False, True):
        name = "interaction" if interaction else "additive   "
        M = design(x, bb, levels, interaction)
        b1 = fit(M, y)[1]
        boots = cluster_bootstrap(y, x, bb, cl, levels, interaction, a.boot, rng)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        verdict = "EXCLUDES 0" if lo > 0 or hi < 0 else "includes 0"
        print(f"  {name}  b1 = {b1:+8.2f}  95% CI [{lo:+8.2f}, {hi:+8.2f}]  {verdict}"
              f"   (kept {len(boots)}/{a.boot} draws)")

    print("\n  b1 is the within-backbone slope of held-out B-D on CKA, in percentage points per")
    print("  unit of CKA, with backbone identity absorbed as a fixed effect rather than by")
    print("  subsetting. A CI including 0 means these cells cannot pin the slope down; it is not")
    print("  a claim that CKA is uninformative.")


if __name__ == "__main__":
    main()
