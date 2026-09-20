#!/usr/bin/env python3
"""
Every cross-cell interval in the paper, recomputed at the level the dependence actually lives at.

WHY THIS EXISTS. Three tables in Appendix C/D were hand-authored: the pooled CKA / l2-drift table,
the backbone-stratified CKA table, and the Moirai predictor table. In this project every table with
an emitting script has been exact and every defect has been in a table with none, so the first job
here is to give those three an emitter. The second job is the substantive one: all three reported
bootstrap intervals over CELLS and p-values computed from cell-level resampling, and cells are not
independent. Twenty-two Moirai cells come from six series; the Small, Base and Large runs on ETTh2 at
h=96 and h=192 share the series, the chronological split, the normalisation constants and, per size,
the checkpoint. A cell-level bootstrap treats those as six independent facts when they are closer to
one, so its interval is too narrow and a p-value derived from it is not conventional inferential
evidence.

WHAT CHANGES, AND THE ONE CLAIM THAT DOES NOT SURVIVE. Both intervals are reported side by side, so
the cost of the wrong level is a number rather than an argument. On the two claims the body leads
with, the reading is unchanged: forg_B still orders B-D with a clustered interval far from zero, and
CKA still fails to order it within Moirai. On one claim it changes: the gate's negative correlation
with held-out B-D, published as rho=-0.518 with a cell-level CI of [-0.841,-0.026] that excludes
zero, has a CLUSTERED interval that includes zero. The rho is unchanged -- it is a point estimate and
does not depend on the resampling scheme -- but the "excludes zero" clause does not survive, and the
paper must say so. The direction of the correction is against us: it weakens an anti-gate finding
this paper argues for.

WHY NO P-VALUES ARE EMITTED. A p-value here would need a null distribution for a rank correlation
among dependent observations at an effective sample size of roughly the cluster count, and the paper
already describes this analysis as descriptive. Printing p=0.0136 beside "every cross-cell number in
this appendix is descriptive" was a contradiction; the columns are dropped rather than recomputed.

Run:  .venv12/bin/python scripts/clustered_inference.py [--boot 10000] [--seed 0] [--latex]
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

import cell_matrix                                                    # noqa: E402
import cka_fixed_effects as cfe                                       # noqa: E402
import cluster_keys as ck                                             # noqa: E402

OUT = ROOT / "results/clustered_inference.json"
TAB_POOLED = ROOT / "paper_8/tables/clustered_pooled.tex"
TAB_BACKBONE = ROOT / "paper_8/tables/clustered_backbone.tex"
TAB_PRED = ROOT / "paper_8/tables/clustered_predictors.tex"
TAB_FE = ROOT / "paper_8/tables/clustered_fixedeffects.tex"

PREDICTORS = [
    ("forg_b", r"forg$_\text{B}$ (condition~B only)"),
    ("forg_d", r"forg$_\text{D}$"),
    ("gate", r"Gate $\Vb{\text{ridge}}$"),
    ("cka", r"CKA"),
    ("drift", r"$\ell_2$ weight drift"),
]


def rows():
    with contextlib.redirect_stdout(io.StringIO()):
        rs = cell_matrix.build_rows()
    return [r for r in rs if r.get("bd_test") is not None]


def both(x, y, cells, n_boot, seed):
    """Cell-level and cluster-level intervals for one predictor, on a common point estimate.

    Each bootstrap gets its OWN generator seeded identically rather than sharing one stream. Sharing
    made an interval depend on how many predictors happened to be computed before it, which is why
    the published cell-level bounds cannot be reproduced by running one predictor in isolation.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 4:
        return None
    cl = ck.clusters_for([c for c, k in zip(cells, ok) if k])
    rho, lo, hi, kept = ck.cell_bootstrap_spearman(
        x[ok], y[ok], n_boot=n_boot, rng=np.random.default_rng(seed))
    crho, clo, chi, ckept, ncl = ck.cluster_bootstrap_spearman(
        x[ok], y[ok], cl, n_boot=n_boot, rng=np.random.default_rng(seed))
    assert abs(crho - rho) < 1e-12, "point estimate must not depend on the resampling scheme"
    return dict(n=int(ok.sum()), rho=float(rho), cell_ci=[lo, hi], cell_kept=kept,
                cluster_ci=[clo, chi], cluster_kept=ckept, n_clusters=ncl)


def _ci(lo, hi, dec=3):
    """`dec=2` for the fixed-effects panel: b1 is in percentage points, where a third decimal is
    spurious precision on a quantity whose interval is 100 points wide."""
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "n/a"
    return f"$[{lo:+.{dec}f}, {hi:+.{dec}f}]$"


def _verdict(lo, hi, short=False):
    """`short=True` for the two seven-column tables, which overflow the text block otherwise.

    The wording is the only slack in those rows: the columns are numbers and the widest row sets the
    width, so trading "excludes 0" for "excl.~0" is what keeps the table inside the margins without
    shrinking the font below the surrounding footnotesize.
    """
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "n/a"
    if lo > 0 or hi < 0:
        return "excl.~0" if short else "excludes 0"
    return "incl.~0" if short else "includes 0"


# ---------------------------------------------------------------------------------------------
# Panel 1: the two observational diagnostics, pooled over all 31 cells
# ---------------------------------------------------------------------------------------------
def pooled(rs, n_boot, seed):
    y = [r["bd_test"] for r in rs]
    cells = [r["cell"] for r in rs]
    out = {}
    for key, label in (("cka", "CKA"), ("drift", r"$\ell_2$ weight drift")):
        res = both([r.get(key) if r.get(key) is not None else np.nan for r in rs],
                   y, cells, n_boot, seed)
        if res:
            out[key] = dict(label=label, **res)
    return out


# ---------------------------------------------------------------------------------------------
# Panel 2: CKA stratified by backbone
# ---------------------------------------------------------------------------------------------
BACKBONE_GROUPS = [
    ("Pooled (against our rule)", None),
    ("Moirai only", "Moirai"),
    ("Chronos only", "Chronos"),
    ("TimesFM only", "TimesFM"),
]


def by_backbone(rs, n_boot, seed):
    out = []
    for label, bb in BACKBONE_GROUPS:
        g = [r for r in rs if bb is None or ck.backbone_of(r["cell"]) == bb]
        g = [r for r in g if r.get("cka") is not None]
        if len(g) < 4:
            out.append(dict(label=label, n=len(g), skipped="too few cells"))
            continue
        x = [r["cka"] for r in g]
        res = both(x, [r["bd_test"] for r in g], [r["cell"] for r in g], n_boot, seed)
        out.append(dict(label=label, cka_lo=float(min(x)), cka_hi=float(max(x)), **res))
    return out


# ---------------------------------------------------------------------------------------------
# Panel 3: every candidate predictor, Moirai only (the only arm with cells enough to ask)
# ---------------------------------------------------------------------------------------------
def predictors(rs, n_boot, seed, backbone="Moirai"):
    g = [r for r in rs if ck.backbone_of(r["cell"]) == backbone and r.get("forg_b") is not None]
    out = []
    for key, label in PREDICTORS:
        res = both([r.get(key) if r.get(key) is not None else np.nan for r in g],
                   [r["bd_test"] for r in g], [r["cell"] for r in g], n_boot, seed)
        if res:
            out.append(dict(key=key, label=label, **res))
    return out


# ---------------------------------------------------------------------------------------------
# Panel 4: the backbone fixed-effects slope, from cka_fixed_effects
# ---------------------------------------------------------------------------------------------
def fixed_effects(n_boot, seed):
    _rs, y, x, bb, cl = cfe.load()
    levels = [lv for lv in ["Moirai", "Chronos", "TimesFM"] if (bb == lv).any()]
    out = dict(n=int(len(y)), n_clusters=int(len(np.unique(cl))), specs=[])
    out["unadjusted_b1"] = float(cfe.fit(np.column_stack([np.ones_like(x), x]), y)[1])
    for interaction in (False, True):
        rng = np.random.default_rng(seed)
        b1 = float(cfe.fit(cfe.design(x, bb, levels, interaction), y)[1])
        boots = cfe.cluster_bootstrap(y, x, bb, cl, levels, interaction, n_boot, rng)
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) else (np.nan, np.nan))
        out["specs"].append(dict(spec="interaction" if interaction else "additive",
                                 b1=b1, ci=[float(lo), float(hi)], kept=int(len(boots))))
    return out


# ---------------------------------------------------------------------------------------------
# Emitters. Each table states its own cluster count so a reader can see the effective sample size.
# ---------------------------------------------------------------------------------------------
HEAD = "% GENERATED by scripts/clustered_inference.py --latex -- do not edit by hand."


def _wrap(body, cols, bare=False, tabcolsep=4):
    """`bare=True` omits the center/small wrapper.

    Two of these four tables are \\input inside a float that already sets \\centering and a font
    size; nesting a `center` environment there adds stray vertical space above the rules. The other
    two are dropped straight into running text and need the wrapper.
    """
    head = [] if bare else [r"\begin{center}\small"]
    tail = [] if bare else [r"\end{center}"]
    return ([HEAD] + head + [rf"\setlength{{\tabcolsep}}{{{tabcolsep}pt}}",
             rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule"] + body
            + [r"\bottomrule", r"\end{tabular}"] + tail)


def emit_pooled(res, path=TAB_POOLED):
    body = [r"Predictor & $n$ & Clusters & $\rho$ & 95\% CI over cells "
            r"& 95\% CI over clusters \\", r"\midrule"]
    for key in ("cka", "drift"):
        d = res.get(key)
        if not d:
            continue
        body.append(f"{d['label']} & {d['n']} & {d['n_clusters']} & ${d['rho']:+.3f}$ & "
                    f"{_ci(*d['cell_ci'])} {_verdict(*d['cell_ci'])} & "
                    f"{_ci(*d['cluster_ci'])} {_verdict(*d['cluster_ci'])} \\\\")
    path.write_text("\n".join(_wrap(body, "lcccll")) + "\n")
    return path


def emit_backbone(res, path=TAB_BACKBONE):
    body = [r"Grouping & $n$ & Cl. & CKA range & $\rho$ & 95\% CI over cells "
            r"& 95\% CI over clusters \\", r"\midrule"]
    vd = lambda ci: _verdict(*ci, short=True)
    for i, d in enumerate(res):
        if d.get("skipped"):
            body.append(f"{d['label']} & {d['n']} & --- & --- & --- & --- & "
                        f"{d['skipped']} \\\\")
            continue
        if i == 1:
            body.append(r"\midrule")
        body.append(
            f"{d['label']} & {d['n']} & {d['n_clusters']} & "
            f"${d['cka_lo']:.2f}$--${d['cka_hi']:.2f}$ & ${d['rho']:+.3f}$ & "
            f"{_ci(*d['cell_ci'])} {vd(d['cell_ci'])} & "
            f"{_ci(*d['cluster_ci'])} {vd(d['cluster_ci'])} \\\\")
    path.write_text("\n".join(_wrap(body, "lccccll", bare=True, tabcolsep=2)) + "\n")
    return path


def emit_pred(res, path=TAB_PRED):
    body = [r"Predictor (Moirai only) & $n$ & Cl. & $\rho$ & 95\% CI over cells "
            r"& 95\% CI over clusters \\", r"\midrule"]
    for d in res:
        body.append(f"{d['label']} & {d['n']} & {d['n_clusters']} & ${d['rho']:+.3f}$ & "
                    f"{_ci(*d['cell_ci'])} {_verdict(*d['cell_ci'])} & "
                    f"{_ci(*d['cluster_ci'])} {_verdict(*d['cluster_ci'])} \\\\")
    path.write_text("\n".join(_wrap(body, "lcccll")) + "\n")
    return path


def emit_fe(res, path=TAB_FE):
    body = [r"Specification & $b_1$ (pp per unit CKA) & 95\% CI over clusters "
            r"& Draws kept \\", r"\midrule",
            rf"No backbone term & ${res['unadjusted_b1']:+.2f}$ & --- & --- \\",
            r"\midrule"]
    for s in res["specs"]:
        name = {"additive": r"CKA $+$ backbone FE",
                "interaction": r"CKA $+$ backbone FE $+$ CKA${\times}$backbone"}[s["spec"]]
        body.append(f"{name} & ${s['b1']:+.2f}$ & {_ci(*s['ci'], dec=2)} "
                    f"{_verdict(*s['ci'])} & {s['kept']} \\\\")
    path.write_text("\n".join(_wrap(body, "lccc", bare=True, tabcolsep=3)) + "\n")
    return path


def console(res):
    print("=" * 104)
    print("CROSS-CELL INTERVALS: OVER CELLS (as published) vs OVER (BACKBONE, DATASET) CLUSTERS")
    print("=" * 104)

    def line(label, d):
        flip = ""
        cell_e = _verdict(*d["cell_ci"]) == "excludes 0"
        cl_e = _verdict(*d["cluster_ci"]) == "excludes 0"
        if cell_e != cl_e:
            flip = "   <-- VERDICT CHANGES under clustering"
        print(f"  {label:34s} n={d['n']:2d} cl={d['n_clusters']:2d} rho={d['rho']:+.3f}  "
              f"cells [{d['cell_ci'][0]:+.3f},{d['cell_ci'][1]:+.3f}] "
              f"{'excl' if cell_e else 'incl':4s}  "
              f"clusters [{d['cluster_ci'][0]:+.3f},{d['cluster_ci'][1]:+.3f}] "
              f"{'excl' if cl_e else 'incl':4s}{flip}")

    print("\n  Pooled over all cells:")
    for k, d in res["pooled"].items():
        line(k, d)
    print("\n  CKA stratified by backbone:")
    for d in res["by_backbone"]:
        if d.get("skipped"):
            print(f"  {d['label']:34s} n={d['n']:2d}  {d['skipped']}")
        else:
            line(d["label"], d)
    print("\n  Candidate predictors, Moirai only:")
    for d in res["predictors"]:
        line(d["key"], d)
    fe = res["fixed_effects"]
    print(f"\n  Backbone fixed effects (n={fe['n']}, {fe['n_clusters']} clusters):")
    print(f"    {'unadjusted':34s} b1={fe['unadjusted_b1']:+8.2f}")
    for s in fe["specs"]:
        print(f"    {s['spec']:34s} b1={s['b1']:+8.2f}  cluster CI "
              f"[{s['ci'][0]:+8.2f},{s['ci'][1]:+8.2f}]  {_verdict(*s['ci'])}  "
              f"(kept {s['kept']})")

    changed = [k for k, d in list(res["pooled"].items())
               + [(d["key"], d) for d in res["predictors"]]
               + [(d["label"], d) for d in res["by_backbone"] if not d.get("skipped")]
               if (_verdict(*d["cell_ci"]) == "excludes 0")
               != (_verdict(*d["cluster_ci"]) == "excludes 0")]
    print(f"\n  verdict changes under clustering: {len(changed)}"
          + (f"  ({', '.join(changed)})" if changed else ""))
    print("  No p-values are computed: this analysis is descriptive, and a p-value from a\n"
          "  resampling scheme that mis-states the dependence would not be inferential evidence.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()

    rs = rows()
    res = dict(n_cells=len(rs), n_boot=a.boot, seed=a.seed,
               cluster_definition="(backbone, dataset) -- see scripts/cluster_keys.py",
               pooled=pooled(rs, a.boot, a.seed),
               by_backbone=by_backbone(rs, a.boot, a.seed),
               predictors=predictors(rs, a.boot, a.seed),
               fixed_effects=fixed_effects(a.boot, a.seed))
    console(res)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    if a.latex:
        for p in (emit_pooled(res["pooled"]), emit_backbone(res["by_backbone"]),
                  emit_pred(res["predictors"]), emit_fe(res["fixed_effects"])):
            print(f"wrote {p.relative_to(ROOT)}")
