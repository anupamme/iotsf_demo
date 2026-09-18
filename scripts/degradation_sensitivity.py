#!/usr/bin/env python3
"""
How many degradation cells exist under DEFINITIONS OTHER THAN OURS.

WHY THIS EXISTS. The paper's headline is that no cell meets its definition of damaging a demonstrated
pre-trained capability. A reader's first objection is that the definition does the work: it requires
forg_B > 0 in EVERY seed and forg_D < 0 in EVERY seed, and unanimity over ten seeds is a demanding
bar. If relaxing it to a simple majority produced five degradation cells, the negative result would
be an artefact of the rule rather than a property of the runs. That is a fair objection and it is
answerable from the numbers already stored, with no new runs: cell_matrix.build_rows() already
carries forg_b_pos, forg_d_neg, forg_b_sd, forg_d_sd and seeds for every cell.

This script therefore walks a ladder of definitions from the weakest defensible one (the means alone,
signs unchecked) up to the paper's (unanimity), crossed with both gate thresholds, and reports which
cells each admits BY NAME. A count alone invites "which ones?"; the names let a reader check.

WHAT IS HELD FIXED, AND WHY IT IS NOT A THUMB ON THE SCALE.
Two exclusions apply identically at every rung of the ladder, exactly as degradation_cells() applies
them:

  * forg_confounded cells are excluded. The Chronos-T5 arm fine-tunes with a decoder/head mismatch
    that makes its per-condition forgetting numbers non-comparable to Moirai's, so its forg_B and
    forg_D do not measure what the definition needs. If these were admitted at the relaxed rungs the
    table would appear to show that loosening the rule finds degradation cells, when what it found is
    a measurement already known to be invalid. The excluded set is printed and named in the caption.
  * Cells with no zero-shot test reference (forg_b is None) cannot be scored at all.

Both are properties of the measurement, not of the outcome, so applying them uniformly is not a
selection on the result.

The 'significant' rung is a one-sided t-test on forg_B against zero, computed from the stored SD and
seed count. Note that the stored *_sd fields from _sd() are the sample SD; the SEM is SD/sqrt(k) and
the t statistic is mean/SEM on k-1 degrees of freedom. This rung is included because "the mean is
positive" and "the mean is reliably positive" are different claims and the reviewer asked for both.

Usage:  python3 scripts/degradation_sensitivity.py [--latex]
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cell_matrix import ROOT, build_rows, _display  # noqa: E402

try:
    from scipy import stats as sps
except ImportError:                                   # keep the script runnable without scipy
    sps = None


# ---------------------------------------------------------------------------------------------
# The ladder. Each entry is (key, LaTeX label, plain label, predicate(row) -> bool).
# Every predicate assumes the row already passed the gate and the two uniform exclusions.
# ---------------------------------------------------------------------------------------------
def _sem(r, field):
    return r[f"{field}_sd"] / max(r["seeds"], 1) ** 0.5


def _t_positive(r):
    """One-sided t-test that forg_B > 0, from the stored SD and seed count."""
    k = r["seeds"]
    sem = _sem(r, "forg_b")
    if k < 2 or sem == 0:
        return None
    t = r["forg_b"] / sem
    if sps is None:
        return None
    return 1.0 - sps.t.cdf(t, df=k - 1)


def _frac(r, need):
    """Both sign clauses hold in at least `need` of the seeds (need given as a fraction)."""
    k = r["seeds"]
    thresh = math.ceil(need * k - 1e-9)
    return r["forg_b_pos"] >= thresh and r["forg_d_neg"] >= thresh


def _means(r):
    return r["forg_b"] > 0 and r["forg_d"] < 0


# Rungs are (key, LaTeX label, plain label, predicate). The predicate is the WHOLE definition:
# whether the means clause applies is part of the rung, not imposed on top of it. That distinction
# is not cosmetic. Moirai-S/ETTh2 h96 has forg_D mean +6.91 while 9 of its 10 seeds are negative --
# a single outlier drags the mean across zero -- so a means-based rule and a seed-count rule
# disagree in DIRECTION on that cell. A ladder that silently required the means clause at every
# rung would never show that, and "majority seed-consistency" is what the objection actually
# proposes as the REPLACEMENT for unanimity, not as an extra hurdle on top of it. Both families are
# reported so the disagreement is visible rather than absorbed.
LADDER = [
    ("mean", r"Means only (seed signs unchecked)", "means only",
     _means),
    ("half", r"Means, $+\ge$ half the seeds agree", "means + majority of seeds",
     lambda r: _means(r) and _frac(r, 0.5)),
    ("p70", r"Means, $+\ge 70\%$ of seeds agree", "means + 70% of seeds",
     lambda r: _means(r) and _frac(r, 0.7)),
    ("p80", r"Means, $+\ge 80\%$ of seeds agree", "means + 80% of seeds",
     lambda r: _means(r) and _frac(r, 0.8)),
    ("unan", r"\textbf{Means, $+$ every seed agrees (ours)}", "means + every seed (OURS)",
     lambda r: _means(r) and r["forg_b_pos"] == r["seeds"] and r["forg_d_neg"] == r["seeds"]),
    ("sig", r"Means, $+$ one-sided $t$ on forg$_\text{B}$ at $\alpha{=}0.05$",
     "means + one-sided t",
     lambda r: _means(r) and (_t_positive(r) is not None) and _t_positive(r) < 0.05),
    ("half_only", r"Seed counts only, $\ge$ half agree", "seed counts only, majority",
     lambda r: _frac(r, 0.5)),
    ("unan_only", r"Seed counts only, every seed agrees", "seed counts only, unanimous",
     lambda r: r["forg_b_pos"] == r["seeds"] and r["forg_d_neg"] == r["seeds"]),
]

GATES = (0.10, 0.20)


def scorable(rows):
    """The uniform exclusions, applied once and reported."""
    keep, no_ref, confounded = [], [], []
    for r in rows:
        if r.get("forg_b") is None or r.get("gate") is None:
            no_ref.append(r)
        elif r.get("forg_confounded"):
            confounded.append(r)
        else:
            keep.append(r)
    return keep, no_ref, confounded


def admits(rows, rung, gate_threshold):
    """Cells admitted as degradation cells under one rung of the ladder at one gate threshold."""
    _key, _tex, _plain, pred = rung
    out = []
    for r in rows:
        if r["gate"] < gate_threshold:
            continue
        if pred(r):          # the rung's predicate IS the definition; nothing is added on top
            out.append(r)
    return out


def report(rows):
    keep, no_ref, confounded = scorable(rows)
    print("=" * 100)
    print("DEGRADATION-CELL COUNT UNDER ALTERNATIVE DEFINITIONS")
    print("=" * 100)
    print(f"  cells scorable                : {len(keep)}")
    print(f"  excluded, no zero-shot ref    : {len(no_ref)}"
          + (f"  ({', '.join(r['cell'] for r in no_ref)})" if no_ref else ""))
    print(f"  excluded, forgetting confounded: {len(confounded)}"
          + (f"  ({', '.join(r['cell'] for r in confounded)})" if confounded else ""))
    if sps is None:
        print("  WARNING: scipy unavailable -- the 'means + one-sided t' rung is reported as n/a")

    table = {}
    for gt in GATES:
        passing = [r for r in keep if r["gate"] >= gt]
        print(f"\n  gate >= {gt:.2f}: {len(passing)} gate-passing cells"
              f"  ({', '.join(_display(r['cell']) for r in passing)})")
        for rung in LADDER:
            hits = admits(keep, rung, gt)
            table[(rung[0], gt)] = hits
            names = ", ".join(_display(r["cell"]) for r in hits) or "---"
            print(f"      {rung[2]:24s} -> {len(hits)}   {names}")

    # The near-miss detail a reader will want: which cells fail unanimity and by how much.
    print("\n  seed agreement on every gate-passing cell (gate >= 0.20), "
          "regardless of whether the means clause holds:")
    for r in sorted((r for r in keep if r["gate"] >= 0.20), key=lambda r: -r["gate"]):
        t = _t_positive(r)
        print(f"      {_display(r['cell']):30s} gate={r['gate']:+.3f}  "
              f"forg_B={r['forg_b']:+7.2f}+-{_sem(r, 'forg_b'):5.2f}SEM "
              f"({r['forg_b_pos']:2d}/{r['seeds']:2d} pos)  "
              f"forg_D={r['forg_d']:+7.2f}+-{_sem(r, 'forg_d'):5.2f}SEM "
              f"({r['forg_d_neg']:2d}/{r['seeds']:2d} neg)  "
              f"p(forg_B>0)={'n/a' if t is None else f'{t:.3f}'}")
    return table, keep, no_ref, confounded


def _short(cell):
    """Compact cell label. The full _display() names are far too wide for four such columns."""
    c = cell.replace("Moirai-", "")
    size, rest = c.split("/", 1)
    ds, h, *_ = rest.split()
    return f"{size[0].upper()}/{ds} {h.replace('h', '$h$')}"


def emit_latex(table, keep, confounded,
               path=ROOT / "paper_8/tables/degradation_sensitivity.tex"):
    n_gate = {gt: sum(1 for r in keep if r["gate"] >= gt) for gt in GATES}
    lines = [
        "% GENERATED by scripts/degradation_sensitivity.py --latex -- do not edit by hand.",
        "\\begin{center}", "\\small", "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{@{}lccl@{}}", "\\toprule",
        "& \\multicolumn{2}{c}{Cells admitted} & \\\\",
        "\\cmidrule(lr){2-3}",
        "Definition of a degradation cell & Gate $\\ge 0.10$ & Gate $\\ge 0.20$ "
        "& Which cells \\\\",
        "\\midrule",
    ]
    for key, tex, _plain, _pred in LADDER:
        wide, narrow = table[(key, 0.10)], table[(key, 0.20)]
        # The gate-0.10 set always contains the gate-0.20 set, so listing the wider one and
        # daggering what only survives the looser gate names every cell exactly once.
        narrow_refs = {r["ref"] for r in narrow}
        names = ", ".join(
            _short(r["cell"]) + ("" if r["ref"] in narrow_refs else "$^\\dagger$")
            for r in wide) or "---"
        lines.append(f"{tex} & {len(wide)} & {len(narrow)} & {names} \\\\")
    lines += [
        "\\midrule",
        f"\\emph{{Gate-passing cells screened}} & {n_gate[0.10]} & {n_gate[0.20]} & \\\\",
        "\\bottomrule", "\\end{tabular}", "\\end{center}",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {path.relative_to(ROOT)}")
    print("  $^\\dagger$ = admitted at gate >= 0.10 only, not at the paper's 0.20.")
    if confounded:
        print("  caption must name the excluded confounded cells: "
              + ", ".join(r["cell"] for r in confounded))


if __name__ == "__main__":
    rows = [r for r in build_rows() if r.get("bd_test") is not None]
    table, keep, no_ref, confounded = report(rows)
    if "--latex" in sys.argv:
        emit_latex(table, keep, confounded)
