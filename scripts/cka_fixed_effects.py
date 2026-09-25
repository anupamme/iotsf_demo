#!/usr/bin/env python3
"""
Does CKA order the intervention once backbone identity is absorbed?

WHY THIS EXISTS
---------------
The paper reports a strong POOLED rank correlation between CKA and held-out B-D
(rho=+0.567, CI excludes 0) and then declines to license it, on the grounds that CKA is not
comparable across architectures: over the 31 scored cells Chronos sits at 0.092-0.232, Moirai at
0.362-0.970 and TimesFM at 0.246-0.561 (the body's narrower 0.092-0.169 is Chronos condition B at
h=24 only), so "CKA" and "which backbone" are nearly the same variable and the pooled statistic is
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

--loco answers the decision-relevant version of the same question OUT OF SAMPLE: if you already know
which backbone you are on, the horizon and the training-set size, does measuring CKA improve your
prediction of the value of encoder adaptation? It compares a nested ladder M0-M4 under
leave-one-cluster-out validation. The specification, the deciding statistic dR2 = R2_LOCO(M3) -
R2_LOCO(M2) and the sentence the paper prints under each outcome were all frozen first, in
results/preregister_loco.json -- see that file for what the guarantee is and is not.

Run:  python3 scripts/cka_fixed_effects.py
      python3 scripts/cka_fixed_effects.py --boot 20000 --seed 7
      python3 scripts/cka_fixed_effects.py --value
      python3 scripts/cka_fixed_effects.py --loco --json --latex
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import cell_matrix  # noqa: E402
# The cluster definition is shared with cell_matrix's rank correlations. Keeping one definition is
# what lets both analyses claim to cluster at "(backbone, dataset)" and mean the same thing.
from cluster_keys import backbone_of, cluster_of, dataset_of  # noqa: E402,F401
# The horizon/n parser lives with the pre-registration, imported rather than copied: it is the file
# that FROZE how covariates are built, so a second implementation here could quietly fit a different
# design than the one registered. It closes two traps -- h(\d+) reads ETTh2 as horizon 2, and n_train
# is None for exactly the cells whose label carries an n token.
from preregister_loco import covariates_of  # noqa: E402

LOCO_JSON = ROOT / "results/cka_loco.json"
LOCO_TEX = ROOT / "paper_8/tables/cka_loco.tex"
# Fixed here rather than taken from --boot/--seed so that --latex is reproducible from one documented
# command, matching value_axis.py's TEX_BOOT/TEX_SEED convention. 10,000 draws costs ~9 s and is what
# the reported figures were checked at: over seeds 0/1/7/13 the b1 interval's bounds move by under
# 5 pp, which is why the table rounds them to whole percentage points rather than tenths.
TEX_BOOT, TEX_SEED = 10_000, 0


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


# ---------------------------------------------------------------------------------------------
# The nested ladder, out of sample. Pre-registered in results/preregister_loco.json.
# ---------------------------------------------------------------------------------------------
# Kept separate from design() above rather than generalising it: design()'s column ORDER is
# load-bearing (cluster_bootstrap reads b1 by position), so extending it to carry horizon, log n and
# dataset FE would put a nuisance covariate between b0 and b1 and silently change what every existing
# interval in this file reports.

SPECS = ("M0", "M1", "M2", "M3", "M4")
SPEC_LABEL = {
    "M0": "intercept only",
    "M1": "backbone FE",
    "M2": "backbone FE + horizon + log n",
    "M3": "M2 + CKA",
    "M4": "M2 + CKA x backbone",
}


def loco_frame():
    """One row per cell: outcome, CKA, backbone, dataset, cluster, horizon, log n."""
    with contextlib.redirect_stdout(io.StringIO()):
        rows = cell_matrix.build_rows()
    rows = [r for r in rows if r.get("bd_test") is not None and r.get("cka") is not None]
    f = dict(cell=[], y=[], cka=[], bb=[], ds=[], cl=[], h=[], logn=[])
    for r in rows:
        h, n = covariates_of(r["cell"], r.get("n_train"))
        f["cell"].append(r["cell"])
        f["y"].append(float(r["bd_test"]))
        f["cka"].append(float(r["cka"]))
        f["bb"].append(backbone_of(r["cell"]))
        f["ds"].append(dataset_of(r["cell"]))
        f["cl"].append(cluster_of(r["cell"]))
        f["h"].append(float(h))
        f["logn"].append(float(np.log(n)))
    for k in ("y", "cka", "h", "logn"):
        f[k] = np.asarray(f[k], float)
    for k in ("cell", "bb", "ds", "cl"):
        f[k] = np.asarray(f[k])
    return f


def ladder_design(spec, f, idx, bb_levels, ds_levels=None):
    """Design matrix for one rung of the ladder on rows `idx`.

    Fixed-effect LEVELS ARE PASSED IN, taken from the full data rather than from `idx`. That is what
    makes a held-out fold scorable: if the levels were derived per fold, a training fold missing a
    backbone would produce a narrower matrix than the held-out rows need and the two could not be
    multiplied. When a level is absent from the training fold its column is all-zero there, lstsq
    returns the minimum-norm solution, and the coefficient is 0 -- no information, no effect, which is
    the honest behaviour for an unidentified term.
    """
    cols = [np.ones(len(idx))]
    if spec != "M0":
        for lv in bb_levels[1:]:
            cols.append((f["bb"][idx] == lv).astype(float))
    if spec in ("M2", "M3", "M4"):
        cols += [f["h"][idx], f["logn"][idx]]
    if spec in ("M3", "M4"):
        cols.append(f["cka"][idx])
    if spec == "M4":
        for lv in bb_levels[1:]:
            cols.append(f["cka"][idx] * (f["bb"][idx] == lv))
    if ds_levels is not None:
        for lv in ds_levels[1:]:
            cols.append((f["ds"][idx] == lv).astype(float))
    return np.column_stack(cols)


def loco_scores(f, spec, fold_of, bb_levels, ds_levels=None):
    """Out-of-sample predictions over the folds in `fold_of`, plus R2 and per-fold MAE.

    R2 IS AGAINST THE TRAINING-FOLD MEAN, not the held-out fold's own mean. Scored against its own
    mean, a fold whose cells happen to sit close together looks catastrophic however good the model
    is, and one with wide internal spread looks good; aggregating those reports fold composition
    rather than model quality. The training-fold mean is also exactly the prediction M0 would make,
    so R2 > 0 means "beats knowing nothing but the other clusters' average".
    """
    n = len(f["y"])
    pred = np.full(n, np.nan)
    null = np.full(n, np.nan)
    per_fold = {}
    for fid in sorted(set(fold_of)):
        te = np.flatnonzero(np.asarray(fold_of) == fid)
        tr = np.flatnonzero(np.asarray(fold_of) != fid)
        if len(tr) < 2:
            continue
        Mtr = ladder_design(spec, f, tr, bb_levels, ds_levels)
        Mte = ladder_design(spec, f, te, bb_levels, ds_levels)
        beta = fit(Mtr, f["y"][tr])
        pred[te] = Mte @ beta
        null[te] = f["y"][tr].mean()
        per_fold[fid] = float(np.abs(f["y"][te] - pred[te]).mean())
    ok = ~np.isnan(pred)
    ss_res = float(((f["y"][ok] - pred[ok]) ** 2).sum())
    ss_tot = float(((f["y"][ok] - null[ok]) ** 2).sum())
    return dict(spec=spec, r2=1.0 - ss_res / ss_tot, mae=float(np.abs(f["y"][ok] - pred[ok]).mean()),
                n_scored=int(ok.sum()), per_fold=per_fold)


def loco_all(f, fold_of, bb_levels):
    return {s: loco_scores(f, s, fold_of, bb_levels) for s in SPECS}


def boot_cka_coef(f, n_boot, rng, bb_levels):
    """Cluster-bootstrap CI for the CKA coefficient in M3 (backbone FE + horizon + log n + CKA).

    This is the one interval in this analysis that is a LINEAR coefficient rather than a ratio of sums
    of squares, so its bootstrap distribution is well behaved and its percentiles are reproducible
    across seeds. It is also the quantity the review asked for by name. In M3's design CKA is the last
    column, so the coefficient is read at index -1.
    """
    uniq = np.unique(f["cl"])
    idx_by_cluster = {c: np.flatnonzero(f["cl"] == c) for c in uniq}
    out = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in draw])
        sub = {k: v[idx] for k, v in f.items()}
        if len(np.unique(sub["bb"])) < len(bb_levels):
            continue                       # backbone FE not identified in this draw
        M = ladder_design("M3", sub, np.arange(len(idx)), bb_levels)
        if np.linalg.matrix_rank(M) < M.shape[1]:
            continue
        out.append(fit(M, sub["y"])[-1])
    return np.array(out)


def boot_dr2(f, n_boot, rng):
    """Cluster bootstrap of dR2 = R2_LOCO(M3) - R2_LOCO(M2), and of dMAE alongside it.

    Clusters are resampled with replacement and the WHOLE LOCO is re-run inside each draw. Duplicate
    copies of a cluster share one fold id, so a cell can never be in both the training and held-out
    side of the same fold -- getting that wrong would leak the answer into the training set and drive
    both R2 values up, making the comparison look more precise than it is.

    WHY dMAE IS BOOTSTRAPPED TOO, AND WHY IT IS THE ONE WE QUOTE AN INTERVAL FOR. An out-of-sample R2
    is unbounded below, so its bootstrap distribution has a heavy left tail: a draw that duplicates
    clusters and leaves a one-cell fold can score an arbitrarily large negative R2. The 2.5th
    percentile of dR2 is therefore NOT reproducible -- at 2,000 draws it came out at -3.33, -1.74 and
    -32.47 on seeds 0, 1 and 7, while the point estimate, the median and the share of negative draws
    moved in the third decimal. Quoting that percentile would put a number in the paper that a reader
    re-running the script with another seed cannot reproduce. dMAE is a difference of mean absolute
    errors in percentage points, bounded and well behaved, so it carries the 95% interval; dR2 keeps
    the point estimate, the median and the interquartile range, which are stable.
    """
    uniq = np.unique(f["cl"])
    idx_by_cluster = {c: np.flatnonzero(f["cl"] == c) for c in uniq}
    bb_levels_full = [lv for lv in ("Moirai", "Chronos", "TimesFM") if (f["bb"] == lv).any()]
    out = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx, fold = [], []
        for c in np.unique(draw):
            reps = int((draw == c).sum())
            for _ in range(reps):
                idx.append(idx_by_cluster[c])
                fold.append(np.full(len(idx_by_cluster[c]), c))   # duplicates share a fold id
        idx = np.concatenate(idx)
        fold = np.concatenate(fold)
        sub = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in f.items()}
        if len(np.unique(sub["cl"])) < 3:
            continue
        try:
            s2 = loco_scores(sub, "M2", fold, bb_levels_full)
            s3 = loco_scores(sub, "M3", fold, bb_levels_full)
        except (np.linalg.LinAlgError, ZeroDivisionError, ValueError):
            continue
        if not all(np.isfinite(v) for v in (s2["r2"], s3["r2"], s2["mae"], s3["mae"])):
            continue
        out.append((s3["r2"] - s2["r2"], s3["mae"] - s2["mae"]))
    return np.array(out)


def run_loco(a, rng):
    f = loco_frame()
    bb_levels = [lv for lv in ("Moirai", "Chronos", "TimesFM") if (f["bb"] == lv).any()]
    ds_levels = sorted(set(f["ds"]))
    fold_of = list(f["cl"])

    print("\n" + "=" * 92)
    print("DOES CKA ADD OUT-OF-SAMPLE PREDICTIVE INFORMATION BEYOND BACKBONE IDENTITY?")
    print("=" * 92)
    print(f"  {len(f['y'])} cells, {len(set(fold_of))} leave-one-cluster-out folds "
          f"(cluster = backbone x dataset)")
    print(f"  pre-registered: results/preregister_loco.json  "
          f"(prediction dR2 <= 0)")
    print(f"  R2 is out-of-sample against the TRAINING-fold mean; folds hold 1-"
          f"{max(np.bincount(np.unique(f['cl'], return_inverse=True)[1]))} cells\n")

    sc = loco_all(f, fold_of, bb_levels)
    print(f"  {'model':5s} {'predictors':34s} {'R2_LOCO':>9s} {'MAE':>7s}")
    for s in SPECS:
        print(f"  {s:5s} {SPEC_LABEL[s]:34s} {sc[s]['r2']:+9.3f} {sc[s]['mae']:7.2f}")

    dr2 = sc["M3"]["r2"] - sc["M2"]["r2"]
    dmae = sc["M3"]["mae"] - sc["M2"]["mae"]
    boots = boot_dr2(f, a.boot_r2, rng)
    b_r2, b_mae = boots[:, 0], boots[:, 1]
    q25, med, q75 = (float(v) for v in np.percentile(b_r2, [25, 50, 75]))
    frac_le0 = float((b_r2 <= 0).mean())
    mq25, mq75 = (float(v) for v in np.percentile(b_mae, [25, 75]))
    mlo, mhi = (float(v) for v in np.percentile(b_mae, [2.5, 97.5]))
    print(f"\n  dR2  = R2(M3) - R2(M2)   = {dr2:+.3f}   median draw {med:+.3f}"
          f"   IQR [{q25:+.3f}, {q75:+.3f}]")
    print(f"  dMAE = MAE(M3) - MAE(M2) = {dmae:+.2f} pp  median {float(np.median(b_mae)):+.2f}"
          f"   IQR [{mq25:+.2f}, {mq75:+.2f}]")
    print(f"         share of draws in which adding CKA does not help: {frac_le0:.0%}"
          f"   (kept {len(b_r2)}/{a.boot_r2})")
    print( "  MEDIAN AND IQR, NOT A 95% INTERVAL, for both: an out-of-sample R2 is unbounded below and")
    print( "  a degenerate bootstrap fold can blow up either statistic, so the extreme percentiles are")
    print(f"  not reproducible across seeds (dR2's 2.5th pct moved -1.7 to -32.5; dMAE's 97.5th moved")
    print(f"  +11 to +19). For reference only, dMAE's 95% range here is [{mlo:+.2f}, {mhi:+.2f}].")

    # b1 in M3: the interval the review asked for by name, and the only well-behaved one here --
    # a linear coefficient rather than a ratio of sums of squares, so its percentiles reproduce.
    M3 = ladder_design("M3", f, np.arange(len(f["y"])), bb_levels)
    b1_cka = float(fit(M3, f["y"])[-1])
    b_coef = boot_cka_coef(f, a.boot_r2, rng, bb_levels)
    clo, chi = (float(v) for v in np.percentile(b_coef, [2.5, 97.5]))
    b1_excl = bool(clo > 0 or chi < 0)
    print(f"\n  b1(CKA) in M3 = {b1_cka:+.2f} pp per unit CKA   95% cluster CI "
          f"[{clo:+.2f}, {chi:+.2f}]   {'EXCLUDES 0' if b1_excl else 'includes 0'}"
          f"   (kept {len(b_coef)}/{a.boot_r2})")

    # The pre-registered branch is keyed on the SIGN of dR2, and only if that sign is positive on
    # whether dR2's interval covers zero. That second test is read off the percentiles boot_dr2
    # documents as irreproducible, so it is consulted only in a branch that does not fire here; if a
    # future run makes dR2 positive, replace it with a bounded statistic before reporting the branch.
    r2lo, r2hi = (float(v) for v in np.percentile(b_r2, [2.5, 97.5]))
    branch = ("dR2 <= 0" if dr2 <= 0 else
              "dR2 > 0, bootstrap CI excludes 0" if (r2lo > 0 or r2hi < 0) else
              "dR2 > 0, bootstrap CI covers 0")
    print(f"\n  PRE-REGISTERED BRANCH THAT FIRES: {branch}")

    # ---- the dataset-FE variant, reported for what it is ------------------------------------
    # Leave-one-SERIES-out with dataset FE cannot be out-of-sample: the held-out series' own fixed
    # effect is identified by no other fold, so its column is all-zero in training and the prediction
    # falls back on the remaining terms. Reported as an in-sample fit plus per-series residuals, and
    # labelled as such, because calling it validation would be the error the pre-registration names.
    M2d = ladder_design("M2", f, np.arange(len(f["y"])), bb_levels, ds_levels)
    M3d = ladder_design("M3", f, np.arange(len(f["y"])), bb_levels, ds_levels)
    r2_in = {}
    for nm, M in (("M2+dataset FE", M2d), ("M3+dataset FE", M3d)):
        beta = fit(M, f["y"])
        res = f["y"] - M @ beta
        r2_in[nm] = 1.0 - float((res ** 2).sum()) / float(((f["y"] - f["y"].mean()) ** 2).sum())
    print(f"\n  IN-SAMPLE with dataset FE (not validation -- the held-out series' FE is")
    print(f"  unidentified by construction, so there is no out-of-sample version of this):")
    for nm, v in r2_in.items():
        print(f"    {nm:16s} R2_in = {v:+.3f}")
    print(f"    incremental CKA R2_in = {r2_in['M3+dataset FE'] - r2_in['M2+dataset FE']:+.3f}"
          f"  on {len(ds_levels)} series and {len(f['y']) } cells with "
          f"{M3d.shape[1]} parameters")

    payload = dict(
        n_cells=int(len(f["y"])), n_folds=len(set(fold_of)),
        preregistration="results/preregister_loco.json",
        models={s: SPEC_LABEL[s] for s in SPECS},
        r2_loco={s: sc[s]["r2"] for s in SPECS},
        mae_loco={s: sc[s]["mae"] for s in SPECS},
        per_fold_mae={s: sc[s]["per_fold"] for s in SPECS},
        dr2=dr2, dr2_median=med, dr2_iqr=[q25, q75], dr2_frac_not_helping=frac_le0,
        dmae=dmae, dmae_iqr=[mq25, mq75], dmae_ci_not_reported=[mlo, mhi],
        dispersion_note=("dR2 and dMAE carry median and IQR, NOT a 95% interval; the 95% interval is "
                         "carried by b1(CKA), the one linear coefficient here. An out-of-sample R2 "
                         "is unbounded below, so dR2's 2.5th percentile moved between -1.7 and -32.5 "
                         "across bootstrap seeds, and dMAE's 97.5th moved between +11 and +19, while "
                         "the point estimates, medians and sign share moved in the third decimal. "
                         "dmae_ci_not_reported is kept for the record and is deliberately NOT quoted "
                         "in the paper. The pre-registered DECIDING STATISTIC (the sign of dR2) is "
                         "unchanged; only the dispersion measure reported beside it differs from the "
                         "registration."),
        dr2_boot_kept=int(len(b_r2)), dr2_boot_requested=int(a.boot_r2),
        cka_coef_m3_insample=b1_cka,
        cka_coef_ci=[clo, chi], cka_coef_excludes_zero=b1_excl,
        cka_coef_boot_kept=int(len(b_coef)),
        branch=branch,
        r2_insample_dataset_fe=r2_in,
        n_series=len(ds_levels),
    )
    if a.json:
        LOCO_JSON.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\n  wrote {LOCO_JSON.relative_to(ROOT)}")
    if a.latex:
        emit_loco_tex(payload)
        print(f"  wrote {LOCO_TEX.relative_to(ROOT)}")
    return payload


def tex_label(lab):
    """SPEC_LABEL is written for the console; render its two ASCII idioms as maths for the table."""
    return lab.replace(" x ", " $\\times$ ").replace("log n", "$\\log n$")


def emit_loco_tex(p):
    rows = "\n".join(
        f"{s} & {tex_label(p['models'][s])} & "
        f"${p['r2_loco'][s]:+.3f}$ & ${p['mae_loco'][s]:.2f}$ \\\\"
        for s in SPECS)
    q25, q75 = p["dr2_iqr"]
    mq25, mq75 = p["dmae_iqr"]
    clo, chi = p["cka_coef_ci"]
    verdict = ("excludes zero" if p["cka_coef_excludes_zero"] else "covers zero")
    LOCO_TEX.write_text(rf"""% GENERATED by scripts/cka_fixed_effects.py --loco --json --latex -- do not edit by hand.
\begin{{table}}[t]
\centering
\caption{{\textbf{{Does CKA add out-of-sample predictive information about
$\denc$ beyond backbone identity?}} Nested models over all
{p['n_cells']} scored cells, validated by leave-one-cluster-out across the
{p['n_folds']} (backbone, dataset) clusters. $R^2_\text{{LOCO}}$ is scored
\textbf{{against the training-fold mean}}, never the held-out fold's own mean, so
a positive value means the model beats predicting the other clusters' average;
MAE is in percentage points of zero-shot loss.
\textbf{{The deciding statistic and this table's interpretation were fixed before
the models were fitted}} (\texttt{{results/preregister\_loco.json}}):
$\Delta R^2 = R^2_\text{{LOCO}}(\text{{M3}}) - R^2_\text{{LOCO}}(\text{{M2}}) =
{p['dr2']:+.3f}$, median bootstrap draw ${p['dr2_median']:+.3f}$, IQR
$[{q25:+.3f}, {q75:+.3f}]$, and adding CKA fails to help in
{p['dr2_frac_not_helping'] * 100:.0f}\% of draws; $\Delta\text{{MAE}} = {p['dmae']:+.2f}$~pp,
IQR $[{mq25:+.2f}, {mq75:+.2f}]$.
\textbf{{Interquartile ranges, not 95\% intervals, are quoted for both}}: an
out-of-sample $R^2$ is unbounded below and a degenerate bootstrap fold can blow up
either statistic, so their extreme percentiles are not reproducible: $\Delta R^2$'s
2.5th percentile moved between $-1.7$ and $-32.5$ across bootstrap seeds, and
$\Delta$MAE's 97.5th between $+11$ and $+19$, while the point estimates, medians and
sign share moved in the third decimal. \textbf{{The 95\% interval is carried instead
by the one linear coefficient here}}, CKA's slope in M3:
${p['cka_coef_m3_insample']:+.0f}$~pp per unit CKA, cluster-bootstrap CI
$[{clo:+.0f}, {chi:+.0f}]$, which {verdict}, so the slope's sign is not even
determined, let alone its size. That interval is quoted to whole percentage points
because its bounds move by under 5~pp across bootstrap seeds.
The pre-registered deciding statistic
(the \emph{{sign}} of $\Delta R^2$) is unchanged; only the dispersion measure beside
it differs from the registration, which says so.
Dataset identity is deliberately \emph{{absent}} from the ladder: a cluster is a
(backbone, dataset) pair, so holding one out removes the only cells that identify
that dataset's fixed effect and the fold would silently score a model nobody
specified: three of the {p['n_series']} series are covered by a single backbone,
so no other fold could identify them even in principle. The in-sample fit
\emph{{with}} dataset fixed effects is reported in the text and is not validation.}}
\label{{tab:cka_loco}}
\small
\begin{{tabular}}{{llrr}}
\toprule
Model & Predictors & $R^2_\text{{LOCO}}$ & MAE \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--value", action="store_true",
                    help="also fit the CKA x graded-pre-trained-value interaction")
    ap.add_argument("--loco", action="store_true",
                    help="the pre-registered nested ladder under leave-one-cluster-out validation")
    ap.add_argument("--boot-r2", type=int, default=TEX_BOOT,
                    help="cluster-bootstrap draws for dR2 (each draw re-runs the whole LOCO)")
    ap.add_argument("--json", action="store_true", help="write results/cka_loco.json")
    ap.add_argument("--latex", action="store_true", help="write paper_8/tables/cka_loco.tex")
    a = ap.parse_args()
    # --latex must be reproducible from one documented command, so it pins the bootstrap settings
    # regardless of what --boot-r2/--seed say, exactly as value_axis.py does.
    if a.latex:
        a.boot_r2 = TEX_BOOT
    rng = np.random.default_rng(TEX_SEED if a.latex else a.seed)

    if a.loco:
        run_loco(a, rng)
        return

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
