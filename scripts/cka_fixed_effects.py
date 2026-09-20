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

--value adds a third fit,

    value        B-D = b0 + b1*CKA + backbone FE + b2*value + b3*(CKA x value)

where `value` is the graded pre-trained-value score from the baseline ladder (R2_task against the
strongest admissible rung, read from results/gate_baselines.json). b3 answers the one question the
binary gate cannot: does drift predict the outcome more strongly on cells that have something to
lose? That is the reading under which this paper's negative result would be an artefact of weak
benchmarks, so it is worth estimating even though 6-15 clusters cannot resolve a moderate b3.

Run:  python3 scripts/cka_fixed_effects.py
      python3 scripts/cka_fixed_effects.py --boot 20000 --seed 7
      python3 scripts/cka_fixed_effects.py --value
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import cell_matrix  # noqa: E402
# The cluster definition is shared with cell_matrix's rank correlations. Keeping one definition is
# what lets both analyses claim to cluster at "(backbone, dataset)" and mean the same thing.
from cluster_keys import backbone_of, cluster_of, dataset_of  # noqa: E402,F401


def load():
    with contextlib.redirect_stdout(io.StringIO()):
        rows = cell_matrix.build_rows()
    rows = [r for r in rows if r.get("bd_test") is not None and r.get("cka") is not None]
    y = np.array([r["bd_test"] for r in rows], float)
    x = np.array([r["cka"] for r in rows], float)
    bb = np.array([backbone_of(r["cell"]) for r in rows])
    cl = np.array([cluster_of(r["cell"]) for r in rows])
    return rows, y, x, bb, cl


def load_value(rows):
    """The graded pre-trained-value score per cell, read from results/gate_baselines.json.

    Read rather than recomputed: `graded` is written by gate_baseline_sensitivity.py as the minimum
    R2_task over ADMISSIBLE rungs of the baseline ladder -- the value that survives the strongest
    rival, not the most flattering one -- and a second implementation here could disagree with the
    figure and the body about what a cell's value is. Returns (kept_rows, v, dropped_cell_names).
    """
    import json
    p = ROOT / "results/gate_baselines.json"
    if not p.exists():
        sys.exit("results/gate_baselines.json not found; run gate_baseline_sensitivity.py first")
    gb = json.load(open(p))
    if "graded" not in gb:
        sys.exit("results/gate_baselines.json predates the graded ladder; re-run "
                 "gate_baseline_sensitivity.py to add the `graded` block")
    graded = gb["graded"]["cells"]
    keep, v, dropped = [], [], []
    for r in rows:
        g = graded.get(r.get("ref"))
        if g is None:
            dropped.append(r["cell"])      # no admissible rung: absent, not zero value
            continue
        keep.append(r)
        v.append(g["r2_best"])
    return keep, np.array(v, float), dropped


def design(x, bb, levels, interaction, v=None):
    """Backbone FE with the first level as reference, optionally x by backbone.

    `v` adds a graded pre-trained-value covariate and the x-by-value interaction, appended LAST so
    that b1 stays at column index 1 for every variant and the interaction is always column -1. That
    ordering is load-bearing: cluster_bootstrap reads a coefficient by position, and a design whose
    column order depended on the options would silently return a different quantity.
    """
    cols = [np.ones_like(x), x]
    for lv in levels[1:]:
        cols.append((bb == lv).astype(float))
    if interaction:
        for lv in levels[1:]:
            cols.append(x * (bb == lv))
    if v is not None:
        cols += [v, x * v]
    return np.column_stack(cols)


def fit(M, y):
    beta, *_ = np.linalg.lstsq(M, y, rcond=None)
    return beta


def cluster_bootstrap(y, x, bb, cl, levels, interaction, n_boot, rng, v=None, coef=1):
    """Resample clusters with replacement; `coef` selects the column (1 = b1, -1 = x by value)."""
    uniq = np.unique(cl)
    idx_by_cluster = {c: np.flatnonzero(cl == c) for c in uniq}
    out = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in draw])
        yb, xb, bbb = y[idx], x[idx], bb[idx]
        vb = None if v is None else v[idx]
        if len(np.unique(bbb)) < len(levels):
            continue                      # FE not identified in this draw
        Mb = design(xb, bbb, levels, interaction, vb)
        if np.linalg.matrix_rank(Mb) < Mb.shape[1]:
            continue
        out.append(fit(Mb, yb)[coef])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--value", action="store_true",
                    help="also fit the CKA x graded-pre-trained-value interaction")
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

    if not a.value:
        return

    # ---- Does the CKA slope depend on how much pre-trained value a cell has? ----------------
    # Fitted as an INTERACTION on all cells at once rather than by splitting into high- and
    # low-value subsets: 15 value-cells from 7 clusters cannot support a split, and two subset
    # slopes with overlapping intervals is not a test of whether they differ. Both regressors are
    # centred so that b1 reads as the CKA slope AT MEAN VALUE rather than at the meaningless point
    # value = 0; the interaction coefficient itself is unaffected by centring.
    rows_v, v_raw, dropped = load_value(rows)
    yv = np.array([r["bd_test"] for r in rows_v], float)
    xv = np.array([r["cka"] for r in rows_v], float)
    bbv = np.array([backbone_of(r["cell"]) for r in rows_v])
    clv = np.array([cluster_of(r["cell"]) for r in rows_v])
    lv_v = [lv for lv in levels if (bbv == lv).any()]
    xc, vc = xv - xv.mean(), v_raw - v_raw.mean()

    print("\n" + "=" * 92)
    print("DOES THE CKA SLOPE DEPEND ON HOW MUCH PRE-TRAINED VALUE THERE IS?")
    print("=" * 92)
    if dropped:
        print(f"  {len(dropped)} cell(s) dropped for having no admissible rung: {', '.join(dropped)}")
    print(f"  cells {len(yv)}   clusters {len(np.unique(clv))}   "
          f"value R2_task(best) {v_raw.min():+.3f} to {v_raw.max():+.3f} (mean {v_raw.mean():+.3f})")

    M = design(xc, bbv, lv_v, False, vc)
    beta = fit(M, yv)
    b_x, b_xv = beta[1], beta[-1]
    boots_x = cluster_bootstrap(yv, xc, bbv, clv, lv_v, False, a.boot, rng, vc, coef=1)
    boots_xv = cluster_bootstrap(yv, xc, bbv, clv, lv_v, False, a.boot, rng, vc, coef=-1)
    for nm, b, bo in (("CKA slope at mean value ", b_x, boots_x),
                      ("CKA x value interaction ", b_xv, boots_xv)):
        lo, hi = np.percentile(bo, [2.5, 97.5])
        print(f"  {nm} {b:+9.2f}  95% CI [{lo:+9.2f}, {hi:+9.2f}]  "
              f"{'EXCLUDES 0' if lo > 0 or hi < 0 else 'includes 0'}   (kept {len(bo)}/{a.boot})")

    print("\n  The interaction is the quantity at issue: a POSITIVE value would mean drift predicts")
    print("  the outcome more strongly where there is more pre-trained value to lose, which is the")
    print("  reading under which our negative result would be an artefact of weak benchmarks. An")
    print("  interval spanning zero at 6-15 clusters does not refute that reading -- it says these")
    print("  cells cannot separate it from no dependence at all, and only an interaction far larger")
    print("  than the CKA slope itself would have been visible at this n.")


if __name__ == "__main__":
    main()
