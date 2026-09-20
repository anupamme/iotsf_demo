#!/usr/bin/env python3
"""Does the drift-to-outcome relation depend on how much pre-trained value a cell has?

THE QUESTION, AND WHY IT IS NEW HERE. The paper has treated pre-trained value as a BINARY screen:
a cell either clears R2_task = 0.20 against one fitted ridge map or it does not, and only the seven
of 31 that clear it carry the degradation test. A fair objection is that this throws away the only
interesting covariate. Cells do not divide into "has value" and "has none"; they differ in how far up
a ladder of baselines their advantage survives. Turning that into a continuous score lets us ask the
question the binary screen cannot: IS THE DISSOCIATION AN ARTEFACT OF LOOKING AT WORTHLESS CELLS?
If drift predicted harm wherever there was something to harm, the CKA-to-outcome relation would
strengthen with value, and the paper's negative result would be a statement about weak benchmarks
rather than about the diagnostic.

THE X AXIS IS R2_task(best), A MINIMUM, NOT A MAXIMUM. R2_task = 1 - MSE_zs/MSE_base, so a WEAKER
rival inflates it. Value measured against the strongest admissible rival is therefore the SMALLEST
R2_task across admissible rungs, which is what gate_baseline_sensitivity.graded_value computes and
stores under `graded` in results/gate_baselines.json. This script reads that stored score rather than
recomputing it, so the figure, the correlations and the console table cannot disagree about what a
cell's value is.

WHAT THIS CAN AND CANNOT SETTLE, stated before the numbers. There are 15 value-cells across 7
clusters, and a cluster bootstrap on 7 clusters cannot resolve a moderate interaction: the interval
will be wide whatever the point estimate. So the output is an interval and a direction, not a verdict,
and the honest summary is "these cells cannot pin it down" rather than "there is no dependence".
Reporting it is still worth doing, because a LARGE positive interaction would have been visible even
at this n, and its absence is informative in a way a null p-value is not.

Dispersion and clustering conventions are the project's: clusters are (backbone, dataset), and the
cluster bootstrap and the cluster definition are imported from cluster_keys rather than reimplemented
-- the same rule that keeps the body, Figure 1 and the appendix quoting one interval.
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
import cluster_keys  # noqa: E402

# The SELECTION-split ladder: the value axis has to be measured on windows disjoint from the held-out
# ones `bd_test` is scored on, or the x axis and the y axis share a split and the whole question is
# circular. The test-side ladder is still reachable with --gate, as the retrospective variant.
GATE_JSON = ROOT / "results/gate_baselines_val.json"
OUT_JSON = ROOT / "results/value_axis.json"
OUT_TEX = ROOT / "paper_8/tables/value_axis.tex"

# The emitted table's bootstrap settings. Fixed here rather than taken from --boot/--seed so that
# `--latex` is deterministic: the console default may be raised for a closer look, but the table the
# paper prints must be reproducible from one documented command.
TEX_BOOT, TEX_SEED = 10_000, 0


def load(gate_json=GATE_JSON):
    """Rows joined to their stored graded value score, keyed on `ref`.

    A cell with no graded score is dropped and counted, never silently treated as zero value: a
    missing denominator and a denominator of zero are different facts, and the second would place the
    cell at the origin of the very axis this script is about.

    `gate_json` selects WHICH SPLIT the value axis is measured on. It defaults to the test-side
    ladder, so every existing invocation is unchanged; pass results/gate_baselines_val.json for the
    selection-split ladder the paper now scopes its analyses by. The outcome column (`bd_test`) is
    held-out on both paths -- only the admission axis moves.
    """
    gate_json = Path(gate_json)
    if not gate_json.exists():
        sys.exit(f"{gate_json} not found; run gate_baseline_sensitivity.py first")
    gb = json.load(open(gate_json))
    if "graded" not in gb:
        sys.exit(f"{gate_json.name} has no `graded` block; re-run "
                 "gate_baseline_sensitivity.py (--from-json is enough) to add it")
    graded = gb["graded"]["cells"]
    with contextlib.redirect_stdout(io.StringIO()):
        rows = cell_matrix.build_rows()
    rows = [r for r in rows if r.get("bd_test") is not None and r.get("cka") is not None]
    kept, dropped = [], []
    for r in rows:
        g = graded.get(r.get("ref"))
        (kept if g else dropped).append(r if g else r["cell"])
        if g:
            r["_value"] = g["r2_best"]
            r["_best_baseline"] = g["best_baseline"]
            r["_clears_all"] = g["clears_all_admissible"]
            r["_clears_any"] = g["clears_any_admissible"]
    return kept, dropped, gb


def rank_ci(x, y, cells, label):
    cl = cluster_keys.clusters_for(cells)
    rho, lo, hi, kept, n_cl = cluster_keys.cluster_bootstrap_spearman(x, y, cl)
    print(f"  {label:34s} n={len(x):2d}  clusters={n_cl}  rho={rho:+.3f}  "
          f"CI [{lo:+.3f}, {hi:+.3f}]  {'EXCLUDES 0' if lo > 0 or hi < 0 else 'includes 0'}"
          f"   (kept {kept} draws)")
    return dict(n=len(x), clusters=n_cl, rho=rho, lo=lo, hi=hi, kept=kept)


def interaction_fit(rows, n_boot=TEX_BOOT, seed=TEX_SEED):
    """The slope-scale answer: does the CKA-to-outcome slope depend on how much value a cell has?

    Delegated to cka_fixed_effects rather than refitted here. The design-matrix column order is
    load-bearing there (b1 at index 1, the interaction always last) and a second implementation would
    eventually disagree about which coefficient it was reporting. A rank correlation cannot answer
    this question at all: a difference of two Spearman rhos on 15 and 16 cells has no usable sampling
    distribution, whereas on the slope scale the interaction IS a single estimable coefficient.
    """
    import cka_fixed_effects as cfe

    rng = np.random.default_rng(seed)
    y = np.array([r["bd_test"] for r in rows], float)
    x = np.array([r["cka"] for r in rows], float)
    v = np.array([r["_value"] for r in rows], float)
    bb = np.array([cluster_keys.backbone_of(r["cell"]) for r in rows])
    cl = np.array([cluster_keys.cluster_of(r["cell"]) for r in rows])
    levels = [lv for lv in ("Moirai", "Chronos", "TimesFM") if (bb == lv).any()]
    # Centred, so b1 reads as the slope AT MEAN VALUE rather than at the meaningless point value=0.
    # Centring does not move the interaction coefficient itself.
    xc, vc = x - x.mean(), v - v.mean()
    beta = cfe.fit(cfe.design(xc, bb, levels, False, vc), y)

    out = {}
    for key, coef, b in (("cka_slope_at_mean_value", 1, beta[1]), ("cka_x_value", -1, beta[-1])):
        bo = cfe.cluster_bootstrap(y, xc, bb, cl, levels, False, n_boot, rng, vc, coef=coef)
        lo, hi = np.percentile(bo, [2.5, 97.5])
        out[key] = dict(b=float(b), lo=float(lo), hi=float(hi), kept=int(len(bo)),
                        n_boot=n_boot, n=len(y), clusters=int(len(np.unique(cl))))
        print(f"  {key:26s} {b:+9.2f}  95% CI [{lo:+9.2f}, {hi:+9.2f}]  "
              f"{'EXCLUDES 0' if lo > 0 or hi < 0 else 'includes 0'}   (kept {len(bo)}/{n_boot})")
    return out


# Row labels for the emitted table, in the order the appendix argues them: the two CKA rows the
# reviewer's objection turns on first, then the value rows, then the slope-scale interaction.
TEX_ROWS = [
    ("cka_vs_denc_valuecells", r"CKA vs $\denc$, the value-cells"),
    ("cka_vs_denc_lowvalue",   r"CKA vs $\denc$, cells with no admissible value"),
    ("cka_vs_denc_moirai",     r"CKA vs $\denc$, within Moirai"),
    ("cka_vs_denc_all",        r"CKA vs $\denc$, pooled (not a licensed reading)"),
    ("value_vs_denc_valuecells", r"$\Vbest$ vs $\denc$, the value-cells"),
    ("value_vs_denc_all",      r"$\Vbest$ vs $\denc$, all cells"),
]


def emit_latex(out, fit, n_scored, n_value, out_tex=OUT_TEX):
    def ci(lo, hi, dp=3):
        # dp differs by block: a rank correlation lives in [-1, 1] and wants three decimals, a slope
        # in percentage points wants one. Printing a pp interval to three decimals implies a
        # precision the 10,000 draws do not have.
        return (f"$[{lo:+.{dp}f}, {hi:+.{dp}f}]$ "
                + ("excludes 0" if lo > 0 or hi < 0 else "includes 0"))

    L = ["% GENERATED by scripts/value_axis.py --latex -- do not edit by hand.",
         "\\setlength{\\tabcolsep}{4pt}", "\\footnotesize",
         "\\begin{tabular}{@{}lrrrl@{}}", "\\toprule",
         "Relation & $n$ & clusters & $\\rho$ or $b$ & 95\\% CI over clusters \\\\", "\\midrule"]
    for key, label in TEX_ROWS:
        c = out.get(key)
        if c is None:
            continue
        L.append(f"{label} & {c['n']} & {c['clusters']} & ${c['rho']:+.3f}$ & {ci(c['lo'], c['hi'])} \\\\")
    L += ["\\midrule",
          "\\multicolumn{5}{@{}l}{\\emph{Slope scale, all cells at once: "
          "$\\denc = b_0 + b_1\\mathrm{CKA} + \\text{backbone FE} + b_2 v + b_3(\\mathrm{CKA}"
          "{\\times}v)$, $v={}$value}} \\\\"]
    for key, label in (("cka_slope_at_mean_value", "$b_1$, CKA slope at mean value (pp per unit CKA)"),
                       ("cka_x_value", "$b_3$, CKA${\\times}$value interaction")):
        f = fit[key]
        L.append(f"{label} & {f['n']} & {f['clusters']} & ${f['b']:+.1f}$ & "
                 f"{ci(f['lo'], f['hi'], dp=1)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    out_tex = Path(out_tex)
    out_tex.write_text("\n".join(L) + "\n")
    print(f"\nwrote {out_tex}  ({n_scored} cells, {n_value} value-cells, "
          f"{TEX_BOOT} cluster-bootstrap draws, seed {TEX_SEED})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help=f"write {OUT_JSON.name}")
    ap.add_argument("--latex", action="store_true", help=f"write {OUT_TEX.name}")
    ap.add_argument("--gate", default=str(GATE_JSON),
                    help="Ladder file supplying the value axis. Default is the test-side ladder; "
                         "results/gate_baselines_val.json is the selection-split one.")
    ap.add_argument("--out-json", default=str(OUT_JSON),
                    help="Where --json writes. Point this elsewhere when --gate is not the default, "
                         "so the two splits cannot overwrite each other's record.")
    ap.add_argument("--out-tex", default=str(OUT_TEX), help="Where --latex writes.")
    a = ap.parse_args()

    rows, dropped, gb = load(a.gate)
    val = [r for r in rows if r["_clears_any"]]
    print(f"  value axis from {Path(a.gate).name} (split={gb.get('split', 'test')})")

    print("=" * 104)
    print("THE GRADED VALUE AXIS: does pre-trained value modify the drift-to-outcome relation?")
    print("=" * 104)
    if dropped:
        print(f"  {len(dropped)} cell(s) dropped for having no graded score: {', '.join(dropped)}")
    print(f"  {len(rows)} scored cells; {len(val)} clear {gb['gate_threshold']} against at least one\n"
          f"  admissible rung; {sum(r['_clears_all'] for r in rows)} clear it against all of them.")

    print("\n  Value score = R2_task against the STRONGEST admissible rival (a minimum over rungs),\n"
          "  so a cell high on this axis has an advantage that no baseline in the ladder erases.")
    print(f"\n  {'cell':30s}{'value':>8s}{'strongest rival':>18s}{'CKA':>7s}{'D_enc':>9s}")
    print("  " + "-" * 72)
    for r in sorted(rows, key=lambda r: -r["_value"]):
        print(f"  {r['cell'][:30]:30s}{r['_value']:+8.3f}{r['_best_baseline']:>18s}"
              f"{r['cka']:7.3f}{r['bd_test']:+9.1f}"
              + ("  <- clears all" if r["_clears_all"] else ""))

    print("\n  RANK CORRELATIONS, clustered on (backbone, dataset):")
    out = {}
    out["value_vs_denc_all"] = rank_ci([r["_value"] for r in rows], [r["bd_test"] for r in rows],
                                       [r["cell"] for r in rows], "value vs D_enc, all cells")
    if len(val) >= 4:
        out["value_vs_denc_valuecells"] = rank_ci(
            [r["_value"] for r in val], [r["bd_test"] for r in val],
            [r["cell"] for r in val], "value vs D_enc, value-cells")
    out["cka_vs_denc_all"] = rank_ci([r["cka"] for r in rows], [r["bd_test"] for r in rows],
                                     [r["cell"] for r in rows], "CKA vs D_enc, all cells")
    # Within Moirai as well as pooled, because the pooled figure is the one the body calls a backbone
    # artefact. A figure that annotated only the pooled rho would contradict its own caption, and this
    # script is the only place the two are computed side by side from one set of rows.
    moirai = [r for r in rows if not r["cell"].startswith(("Chronos", "TimesFM"))]
    out["cka_vs_denc_moirai"] = rank_ci([r["cka"] for r in moirai], [r["bd_test"] for r in moirai],
                                        [r["cell"] for r in moirai], "CKA vs D_enc, within Moirai")
    if len(val) >= 4:
        out["cka_vs_denc_valuecells"] = rank_ci(
            [r["cka"] for r in val], [r["bd_test"] for r in val],
            [r["cell"] for r in val], "CKA vs D_enc, value-cells")

    # The comparison the reviewer actually asks for: the CKA-to-outcome relation restricted to cells
    # WITH value against the same relation on cells without. Reported as two intervals rather than a
    # difference test, because a difference of two rank statistics on 13 and 18 cells from 6 and 9
    # clusters has no usable sampling distribution at this n. cka_fixed_effects.py --value fits the
    # interaction properly, on the slope scale, where a difference IS estimable.
    low = [r for r in rows if not r["_clears_any"]]
    if len(low) >= 4:
        out["cka_vs_denc_lowvalue"] = rank_ci(
            [r["cka"] for r in low], [r["bd_test"] for r in low],
            [r["cell"] for r in low], "CKA vs D_enc, no-value cells")

    print("\n  Read the two CKA rows together: if drift only failed to predict the outcome on cells\n"
          "  with nothing to lose, the value-cell row would be the stronger of the two. Neither row\n"
          "  excluding zero is the result; which is larger at this n is not.")

    print("\n  THE SAME QUESTION ON THE SLOPE SCALE, where the dependence is one coefficient:")
    fit = interaction_fit(rows)
    print("  A positive b3 would mean drift predicts the outcome more strongly where there is\n"
          "  more value to lose -- the reading under which this paper's negative result would be an\n"
          "  artefact of weak benchmarks. The point estimate is not positive and the interval spans\n"
          "  zero, so these cells cannot separate that reading from no dependence at all.")

    if a.latex:
        emit_latex(out, fit, len(rows), len(val), a.out_tex)

    if a.json:
        out_json = Path(a.out_json)
        out_json.write_text(json.dumps(dict(
            correlations=out, interaction=fit,
            n_scored=len(rows), n_value_cells=len(val),
            n_clearing_all=sum(r["_clears_all"] for r in rows),
            dropped=dropped, gate_threshold=gb["gate_threshold"],
            baselines=gb["baselines"],
            cells={r["ref"]: dict(cell=r["cell"], value=r["_value"],
                                  best_baseline=r["_best_baseline"], cka=r["cka"],
                                  d_enc=r["bd_test"], clears_any=r["_clears_any"],
                                  clears_all=r["_clears_all"]) for r in rows},
        ), indent=2))
        print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
