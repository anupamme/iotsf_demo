#!/usr/bin/env python3
"""
Paired seed-level inference on the two interventions, per cell and in aggregate.

WHY THIS EXISTS. The paper decides whether a cell "degrades" with a three-clause definition whose
second and third clauses are SIGN COUNTS over seeds: every seed positive, every seed negative. A
reviewer is entitled to object that a unanimity count is not inference -- it reports no interval, no
effect size, and no p-value, so a cell with a large consistent effect and a cell with a tiny
consistent effect are recorded identically. This script reports the conventional quantities BESIDE
the counts, on the same runs, so the definition can be read against them rather than instead of them.

WHICH CONTRASTS, AND WHY THESE. CKA is observational: it is measured after the fact and nothing in
this design manipulates it. The thing the protocol actually manipulates is whether the encoder is
allowed to adapt. So the estimand is

    Delta_encoder = L_full - L_frozen      (condition B - condition D)

and the causal claim the runs support is about encoder adaptation, not about representation drift.
Two further contrasts are reported because they answer different questions:

    Delta_FT      = L_full   - L_zeroshot  (is fine-tuning at all an improvement?)
    Delta_frozen  = L_frozen - L_zeroshot  (is the frozen-encoder arm an improvement?)

All three are in the published units -- percent of the cell's zero-shot reference, so a positive
value means WORSE -- and are read off cell_matrix's per-seed lists, which reproduce bd_test, forg_b
and forg_d exactly. Nothing here recomputes a denominator.

WHAT THE INFERENCE CAN AND CANNOT DO, stated because it bounds every number below. The permutation
test is an exact sign-flip test, enumerated over all 2^N assignments. That makes its p-value
distribution-free, and it also makes its resolution a function of N alone: the smallest attainable
two-sided p is 2/2^N, which is 0.25 at N=3 and 0.0625 at N=5. **No 3-seed cell in this matrix can
reach p < 0.05 no matter how large or how consistent its effect is**, and the paired-N histogram is
{3: 24, 5: 4, 10: 3} -- so only 3 of the 31 cells can resolve anything at alpha=0.05 without a
distributional assumption. That is a real limit on what this matrix can establish, and it is a better
answer to "give me conventional inference" than an interval that looks precise at N=3. Cells are
flagged accordingly. The histogram is printed by main() and the caption's count is computed from the
same N the table's column prints, because a hand-typed seed count in this project has been wrong
before (an earlier draft of this docstring said 21).

Paired-ness is not uniform and is recorded per cell. Delta_encoder is genuinely paired everywhere:
B and D share a seed, an initialisation and a window sample. Delta_FT and Delta_frozen are paired on
the Chronos, TimesFM and ILI arms, whose run files carry each seed's own zero-shot test MSE, but NOT
on the 21 Moirai cells, which divide by a per-cell mean over condition-A seeds. Where they are not
paired, the reference's own measurement error is not propagated into the interval; the zs_sem column
reports how large that omitted term is rather than leaving it implicit.

Regenerate with:  .venv12/bin/python scripts/paired_inference.py
"""
import argparse
import collections
import itertools
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cell_matrix as cm                                          # noqa: E402
import gate_baseline_sensitivity as gbs                           # noqa: E402
from cluster_keys import cluster_of                               # noqa: E402

OUT_JSON = ROOT / "results/paired_inference.json"
OUT_TEX = ROOT / "paper_8/tables/paired_inference.tex"
# The SELECTION-split ladder, because that is the split the paper's admission decisions are made on:
# a value-cell is admitted by windows disjoint from the held-out ones the aggregate below is computed
# from. results/gate_baselines.json is the retrospective test-side ladder and is reachable with
# --gate; it must not be the default, or this table's value-cell marks would disagree with S4's count.
GATE_BASELINES = ROOT / "results/gate_baselines_val.json"

CONTRASTS = (("d_enc", r"$\Delta_\text{enc}$"),
             ("d_ft", r"$\Delta_\text{FT}$"),
             ("d_frozen", r"$\Delta_\text{frz}$"))

# 2^N enumeration is exact and cheap at these sizes; the largest cell runs 10 seeds (1,024 flips).
MAX_EXACT = 20


def exact_sign_flip_p(x):
    """Two-sided p from enumerating all 2^N sign assignments of the paired differences.

    The null is that each paired difference is equally likely to have come out with either sign,
    which is the exchangeability the pairing buys. Returns (p, n_arrangements, p_floor) where
    p_floor = 2/2^N is the smallest p this N can produce -- reported so a non-significant result at
    N=3 is not mistaken for evidence of no effect.
    """
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return float("nan"), 0, float("nan")
    obs = abs(x.mean())
    if n > MAX_EXACT:
        raise ValueError(f"{n} seeds exceeds exact enumeration cap {MAX_EXACT}")
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    means = np.abs(signs @ x) / n
    # >= with a tolerance: the identity arrangement must count itself, and floating-point means of
    # the same multiset must not be split by rounding.
    p = float((means >= obs - 1e-12).mean())
    return p, len(signs), 2.0 / 2 ** n


def paired_bootstrap_ci(x, n_boot=10000, seed=0):
    """Percentile CI for the mean of the paired differences, resampling SEEDS with replacement.

    NOT the interval this table reports, and the reason is a construction artifact worth naming.
    Every bootstrap resample mean lies inside [min(x), max(x)], so at N=3 with all three observations
    on one side of zero the percentile interval EXCLUDES zero no matter how small the effect --
    "CI excludes zero" is then a restatement of "all seeds agreed in sign", which is the sign count
    the interval was supposed to improve on. Kept in the JSON for the cells with enough seeds for it
    to mean something, and reported beside the t-interval so the gap is visible rather than assumed
    away.
    """
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def t_interval(x):
    """Paired t interval and p-value on the differences: (lo, hi, p).

    This is the interval the table reports. It buys resolution at N=3, where the exact test cannot
    reach p<0.05 and the percentile bootstrap cannot reach zero, and it pays for it with a normality
    assumption on three points -- which is not checkable. So it is reported BESIDE the exact p rather
    than instead of it: agreement means the reading does not rest on the assumption, and the cells
    where they disagree are exactly the cells whose evidence is assumption-dependent.
    """
    from scipy import stats

    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    r = stats.ttest_1samp(np.asarray(x, float), 0.0)
    ci = r.confidence_interval(0.95)
    return float(ci.low), float(ci.high), float(r.pvalue)


def cohens_d(x):
    """Paired Cohen's d: mean difference over its sample sd (ddof=1). None when N < 2 or sd == 0."""
    if len(x) < 2:
        return None
    sd = st.stdev(x)
    return st.mean(x) / sd if sd > 0 else None


def summarise(x, n_boot, seed):
    boot_lo, boot_hi = paired_bootstrap_ci(x, n_boot, seed)
    t_lo, t_hi, t_p = t_interval(x)
    p, arr, p_floor = exact_sign_flip_p(x)
    return dict(n=len(x), mean=st.mean(x) if x else None,
                sd=cm._sd(x), sem=cm._sem(x),
                lo=t_lo, hi=t_hi, t_p=t_p,
                boot_lo=boot_lo, boot_hi=boot_hi,
                p=p, arrangements=arr, p_floor=p_floor, d=cohens_d(x),
                pos=sum(v > 0 for v in x), neg=sum(v < 0 for v in x))


def value_cells(rows, gate_json=GATE_BASELINES):
    """Cell refs clearing the gate threshold under at least one ADMISSIBLE baseline.

    Reuses gate_baseline_sensitivity.passes/admissible rather than re-deriving the rule, so "value
    cell" means the same thing here as in the baseline-ladder appendix. A baseline whose denominator
    is worse than the training-mean floor cannot admit a cell: it would be measuring value against a
    predictor that is itself beaten by a constant.
    """
    gb = json.load(open(gate_json))
    results, baselines = gb["cells"], gb["baselines"]
    out = {}
    for r in rows:
        ref = r["ref"]
        if ref not in results:
            continue
        clearing = [b for b in baselines
                    if b != gbs.FLOOR and gbs.passes(results, ref, b, require_admissible=True)]
        if clearing:
            out[ref] = clearing
    return out


def analyse(rows, n_boot=10000, seed=0, gate_json=GATE_BASELINES):
    vc = value_cells(rows, gate_json)
    cells = []
    for r in rows:
        ps = r["per_seed"]
        rec = dict(cell=r["cell"], ref=r["ref"], cluster=cluster_of(r["cell"]),
                   seeds=r["seeds"], gate=r["gate"], cka=r["cka"], n_train=r.get("n_train"),
                   is_value_cell=r["ref"] in vc, clearing=vc.get(r["ref"], []),
                   ft_paired=ps["ft_paired"], zs_seeds=ps["zs_seeds"], zs_sem=ps["zs_sem"],
                   forg_confounded=bool(r.get("forg_confounded")))
        for key, _ in CONTRASTS:
            rec[key] = summarise(ps[key], n_boot, seed) if ps[key] else None
        cells.append(rec)
    return cells, vc


def cluster_aggregate(cells, key, subset, n_boot=10000, seed=0):
    """Mean of the per-cell means over `subset`, with a CI from resampling (backbone, dataset).

    Cells are not independent: the 31 reuse 3 backbones and 7 series, so a cell-level interval is too
    narrow. Clusters are resampled instead, which is the same level cluster_keys.py defines for every
    other cross-cell interval in the paper.
    """
    sel = [c for c in cells if c["ref"] in subset and c.get(key)]
    if not sel:
        return None
    vals = np.array([c[key]["mean"] for c in sel], float)
    cl = np.array([c["cluster"] for c in sel])
    uniq = np.unique(cl)
    idx_by = {c: np.flatnonzero(cl == c) for c in uniq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        boots.append(vals[np.concatenate([idx_by[c] for c in draw])].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n_cells=len(sel), n_clusters=len(uniq), mean=float(vals.mean()),
                lo=float(lo), hi=float(hi),
                pos=int((vals > 0).sum()), neg=int((vals < 0).sum()))


def fmt(v, places=1, signed=True):
    if v is None or (isinstance(v, float) and v != v):
        return "--"
    s = f"{v:+.{places}f}" if signed else f"{v:.{places}f}"
    return s.replace("+", "$+$").replace("-", "$-$")


def emit_tex(cells, agg, path=OUT_TEX):
    """One row per cell: Delta_enc with its interval, exact p, effect size and sign count."""
    vcells = [c for c in cells if c["is_value_cell"]]
    # Counted off the SAME N the table's column prints, not off row["seeds"]: they agree today, but
    # they are different quantities (seeds run vs seeds surviving the B/D intersection), and a caption
    # that counts one while the column shows the other is the kind of claim that silently goes stale.
    n3 = sum(1 for c in cells if c.get("d_enc") and c["d_enc"]["n"] == 3)
    L = ["% GENERATED by scripts/paired_inference.py -- do not edit by hand.",
         r"\begin{table}[t]", r"\centering",
         r"\caption{\textbf{Paired seed-level inference on encoder adaptation.}",
         r"$\Delta_\text{enc}={L_\text{full}}-{L_\text{frozen}}$ per seed, in percent of the cell's",
         r"zero-shot reference, so a positive value means allowing the encoder to adapt made the cell",
         r"\emph{worse}.  B and D share a seed, an initialisation and a window sample, so this",
         r"contrast is paired by construction.",
         r"$d$ is paired Cohen's $d$; dispersion is SEM, matching the body convention",
         r"(\S\ref{sec:method}) and not the population std the stored JSON fields use.",
         r"Each row names its own $n$, which is not constant down the column",
         r"(Table~\ref{tab:heldout_all}).",
         r"\textbf{Two tests, because neither alone is adequate at these seed counts.}",
         r"$p_\text{exact}$ enumerates all $2^N$ sign assignments, so it assumes nothing---but its",
         rf"resolution is fixed by $N$: the smallest two-sided value an $N$-seed cell can produce is",
         rf"$2/2^N$, which is $0.25$ at $N{{=}}3$ and $0.0625$ at $N{{=}}5$.  The {n3} cells at",
         rf"$N{{=}}3$ therefore \emph{{cannot}} reach $p<0.05$ however large or consistent their",
         r"effect, and a non-significant exact $p$ on those cells is a statement about seed budget",
         r"rather than about the effect.  The CI and $p_t$ are the paired $t$ versions, which do",
         r"resolve at $N{=}3$ and pay for it with a normality assumption on three points that cannot",
         r"be checked.  Where the two agree the reading does not rest on that assumption; where they",
         r"disagree, the evidence is assumption-dependent and is treated as such in the text.",
         r"We do \emph{not} report a percentile bootstrap interval here: every resample mean lies",
         r"within the range of the observations, so at $N{=}3$ with all seeds on one side of zero such",
         r"an interval excludes zero by construction, restating the sign count it was meant to improve",
         r"on.  It is kept in \texttt{results/paired\_inference.json} for the larger cells.",
         r"In the two summary rows the last column is \emph{not} an effect size: it gives the number",
         r"of cells whose mean $\Delta_\text{enc}$ is positive, out of the cells in that row.",
         r"$\dagger$ marks cells clearing the gate threshold under at least one admissible baseline",
         r"of the ladder (Appendix~\ref{app:baselines}).",
         r"$\ddagger$ marks the Chronos cells, whose per-condition forgetting is confounded by the",
         r"head/decoder mismatch; their $\Delta_\text{enc}$ is unaffected.}",
         # \footnotesize and a tightened tabcolsep, not \small: eight columns with a bracketed CI
         # among them overruns the 5.5in ICLR textwidth by 19pt at \small, and an overfull hbox in a
         # generated table is a defect the emitter should not be able to reintroduce.
         r"\label{tab:paired_inference}", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}l r rr r rr r@{}}", r"\toprule",
         r"Cell & $N$ & $\Delta_\text{enc}$ & SEM & $95\%$ CI ($t$) & $p_t$ & "
         r"$p_\text{exact}$ & $d$ \\",
         r"\midrule"]
    for c in sorted(cells, key=lambda c: (not c["is_value_cell"], c["cell"])):
        e = c["d_enc"]
        if not e:
            continue
        mark = (r"\textsuperscript{$\dagger$}" if c["is_value_cell"] else "")
        mark += (r"\textsuperscript{$\ddagger$}" if c["forg_confounded"] else "")
        ci = f"[{fmt(e['lo'])}, {fmt(e['hi'])}]"
        d = fmt(e["d"], 2) if e["d"] is not None else "--"
        # cm._display, not c["cell"]: the internal keys carry raw underscores ("Moirai-base_ETTm2_h192")
        # which LaTeX rejects, and reusing the one display helper keeps cell names identical across
        # every generated table rather than inventing a second spelling here.
        if c["seeds"] != e["n"]:
            sys.exit(f"{c['cell']}: {c['seeds']} seeds run but {e['n']} paired differences; the N "
                     f"column must report the paired count, and the caption's tally with it")
        L.append(f"{cm._display(c['cell'], c.get('n_train'))}{mark} & {e['n']} & {fmt(e['mean'])} & "
                 f"{fmt(e['sem'], 1, signed=False)} & {ci} & {e['t_p']:.3f} & "
                 f"{e['p']:.3f} & {d} \\\\")
    L.append(r"\midrule")
    a = agg["value"]["d_enc"]
    L.append(r"\multicolumn{8}{l}{\emph{Mean of cell means, $95\%$ CI from resampling "
             r"(backbone, dataset) clusters:}} \\")
    L.append(rf"\quad {a['n_cells']} value-cells / {a['n_clusters']} clusters & & {fmt(a['mean'])} "
             rf"& & [{fmt(a['lo'])}, {fmt(a['hi'])}] & & & {a['pos']}/{a['n_cells']} \\")
    b = agg["all"]["d_enc"]
    L.append(rf"\quad all {b['n_cells']} cells / {b['n_clusters']} clusters & & {fmt(b['mean'])} & & "
             rf"[{fmt(b['lo'])}, {fmt(b['hi'])}] & & & {b['pos']}/{b['n_cells']} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")
    return len(vcells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-tex", action="store_true")
    ap.add_argument("--gate", default=str(GATE_BASELINES),
                    help="Ladder file defining a value-cell. Default is the selection-split "
                         "ladder; results/gate_baselines.json is the retrospective one.")
    a = ap.parse_args()

    rows = cm.build_rows()
    cells, vc = analyse(rows, a.boot, a.seed, a.gate)
    all_refs = {c["ref"] for c in cells}
    agg = {"value": {k: cluster_aggregate(cells, k, set(vc), a.boot, a.seed) for k, _ in CONTRASTS},
           "all": {k: cluster_aggregate(cells, k, all_refs, a.boot, a.seed) for k, _ in CONTRASTS}}

    OUT_JSON.write_text(json.dumps(dict(cells=cells, aggregate=agg, value_cells=vc,
                                        n_boot=a.boot, seed=a.seed), indent=1) + "\n")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}  ({len(cells)} cells, {len(vc)} value-cells)")

    print(f"\n{'cell':34s} {'N':>2s} {'D_enc':>8s} {'SEM':>6s} {'95% t-CI':>18s} "
          f"{'p_t':>6s} {'p_ex':>6s} {'floor':>6s} {'d':>6s} sign")
    for c in sorted(cells, key=lambda c: (not c["is_value_cell"], c["cell"])):
        e = c["d_enc"]
        if not e:
            continue
        flag = "*" if c["is_value_cell"] else " "
        print(f"{flag}{c['cell']:33s} {c['seeds']:2d} {e['mean']:+8.2f} {e['sem']:6.2f} "
              f"[{e['lo']:+7.2f},{e['hi']:+7.2f}] {e['t_p']:6.3f} {e['p']:6.3f} "
              f"{e['p_floor']:6.3f} "
              f"{(e['d'] if e['d'] is not None else float('nan')):+6.2f} {e['pos']}/{e['n']}")

    hist = collections.Counter(c["d_enc"]["n"] for c in cells if c["d_enc"])
    print(f"\nPaired-N histogram for Delta_enc: {dict(sorted(hist.items()))} "
          f"(total {sum(hist.values())} cells)")

    print("\nWhat can the ASSUMPTION-FREE test resolve at alpha=0.05?")
    resolvable = [c for c in cells if c["d_enc"] and c["d_enc"]["p_floor"] <= 0.05]
    sig = [c for c in resolvable if c["d_enc"]["p"] < 0.05]
    print(f"  {len(resolvable)} of {len(cells)} cells have enough seeds (p_floor <= 0.05); "
          f"{len(sig)} of those reach p < 0.05")
    for c in sig:
        print(f"    {c['cell']:34s} D_enc={c['d_enc']['mean']:+.2f}  p={c['d_enc']['p']:.4f}")

    print("\nWhere do the two tests DISAGREE at alpha=0.05?  (t rejects, exact cannot)")
    dis = [c for c in cells if c["d_enc"] and c["d_enc"]["t_p"] < 0.05 and c["d_enc"]["p"] >= 0.05]
    print(f"  {len(dis)} cells. On these the reading rests on the normality assumption:")
    for c in dis:
        e = c["d_enc"]
        print(f"    {c['cell']:34s} N={c['seeds']} D_enc={e['mean']:+7.2f} "
              f"p_t={e['t_p']:.4f} p_exact={e['p']:.3f} (floor {e['p_floor']:.3f})")
    both = [c for c in cells if c["d_enc"] and c["d_enc"]["t_p"] < 0.05 and c["d_enc"]["p"] < 0.05]
    print(f"  {len(both)} cells are significant under BOTH: "
          f"{', '.join(c['cell'] for c in both) or 'none'}")

    print("\nAggregates (mean of cell means, clustered on (backbone, dataset)):")
    for scope in ("value", "all"):
        for k, lab in CONTRASTS:
            g = agg[scope][k]
            if g:
                print(f"  {scope:6s} {k:9s} n={g['n_cells']:2d} clusters={g['n_clusters']:2d} "
                      f"mean={g['mean']:+7.2f}  CI [{g['lo']:+7.2f},{g['hi']:+7.2f}]  "
                      f"pos={g['pos']} neg={g['neg']}")

    unpaired = [c for c in cells if not c["ft_paired"]]
    print(f"\nDelta_FT/Delta_frozen are NOT seed-paired on {len(unpaired)} cells (shared zero-shot "
          f"reference). Unpropagated reference SEM, in the same units:")
    sems = [c["zs_sem"] for c in unpaired if c["zs_sem"] is not None]
    if sems:
        print(f"  median {st.median(sems):.2f}, max {max(sems):.2f} "
              f"(reference measured on {min(c['zs_seeds'] or 0 for c in unpaired)}"
              f"-{max(c['zs_seeds'] or 0 for c in unpaired)} condition-A seeds)")

    if not a.no_tex:
        n = emit_tex(cells, agg)
        print(f"\nwrote {OUT_TEX.relative_to(ROOT)}  ({n} value-cells marked)")


if __name__ == "__main__":
    main()
