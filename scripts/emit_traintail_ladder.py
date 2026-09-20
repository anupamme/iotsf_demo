#!/usr/bin/env python3
"""The eight-rung ladder on THREE window sets, side by side, on the cells all three share.

WHY A THIRD SET. The primary gate is scored on the selection windows, which answers the reviewer's
protocol (train -> validation gate -> untouched test outcome) but not its residual: validation also
drove early stopping and checkpoint choice, so the outcome depends on those windows indirectly. The
train tail removes even that. It is the last `--frac` of the training region, held back BY INDEX
(scripts/gate_thirdsplit.py); every rung's denominator is fitted on the earlier part of train only and
the numerator is a zero-shot pass, so neither arm of the gate has seen these windows. The precise
claim is "used for neither checkpoint selection nor outcome scoring" -- NOT "never seen", because the
condition-B/D runs subsampled their training windows from the whole train region including the tail.
That does not touch the gate, whose numerator is the untrained checkpoint, but the paper must not
overstate it.

WHAT THE TABLE IS COUNTED OVER. The 21 Moirai cells the train-tail arm covers, which is the
intersection of the three splits -- the Chronos and TimesFM arms have no train-tail pass. Counting the
selection split over its own 31 would compare 19-of-21 against 20-of-31 and the comparison would be
about coverage rather than about windows, so every column here is out of the same 21 and the caption
says so.

ADMISSIBLE, not merely clearing. A rung counts for a cell only if its denominator is at least as good
as the training-mean floor on that cell (gate_baseline_sensitivity.admissible), which is why the
per-rung counts here are smaller than the raw "clears 0.20" tallies in the run logs: on the train tail
boosted trees clear the threshold on 19 cells but are beaten by the floor on 10 of them.

Run:  .venv12/bin/python scripts/emit_traintail_ladder.py [--latex]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import gate_baseline_sensitivity as gbs  # noqa: E402

SPLITS = [("Selection", "results/gate_baselines_val.json"),
          ("Held-out", "results/gate_baselines.json"),
          ("Train tail", "results/gate_baselines_traintail.json")]
OUT = ROOT / "paper_8/tables/gate_splitcompare.tex"

LABEL = {"fitted": "Fitted ridge (ours)", "constant": "Training mean (floor)",
         "persistence": "Persistence", "seasonal_naive": "Seasonal naive",
         "ar": "Per-feature AR", "dlinear": "DLinear (shared)",
         "ridge_tuned": "Tuned ridge", "mlp": "MLP (128)", "gbm": "Boosted trees"}


def load():
    out = {}
    for name, rel in SPLITS:
        d = json.loads((ROOT / rel).read_text())
        out[name] = d["cells"]
    return out


def analyse():
    cells = load()
    common = sorted(set.intersection(*(set(c) for c in cells.values())))
    rungs = ["fitted"] + [b for b in gbs.BASELINES if b != "fitted"]
    A = {"n_common": len(common), "rungs": rungs, "splits": [s for s, _ in SPLITS]}
    A["per_rung"] = {s: {b: sum(1 for k in common if gbs.passes(cells[s], k, b)) for b in rungs}
                     for s in cells}
    # The two headline tallies, recomputed over `common` rather than read from each file's own
    # `graded` block -- those are computed over each split's own cell set (31, 32, 21).
    A["n_any"], A["n_all"], A["fitted_set"] = {}, {}, {}
    for s, cc in cells.items():
        g = {k: gbs.graded_value(cc, k, rungs) for k in common}
        A["n_any"][s] = sum(1 for v in g.values() if v and v["clears_any_admissible"])
        A["n_all"][s] = sum(1 for v in g.values() if v and v["clears_all_admissible"])
        A["fitted_set"][s] = sorted(k for k in common if gbs.passes(cc, k, "fitted"))
    sel, tail = A["fitted_set"]["Selection"], A["fitted_set"]["Train tail"]
    A["fitted_overlap"] = sorted(set(sel) & set(tail))
    A["fitted_sel_only"] = sorted(set(sel) - set(tail))
    A["fitted_tail_only"] = sorted(set(tail) - set(sel))
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    A = analyse()

    print(f"three window sets, {A['n_common']} Moirai cells common to all of them")
    print(f"{'rung':24s}" + "".join(f"{s:>12s}" for s in A["splits"]))
    for b in A["rungs"]:
        print(f"  {LABEL[b]:22s}" + "".join(f"{A['per_rung'][s][b]:>12d}" for s in A["splits"]))
    print(f"{'clears >=1 admissible':24s}" + "".join(f"{A['n_any'][s]:>12d}" for s in A["splits"]))
    print(f"{'clears ALL admissible':24s}" + "".join(f"{A['n_all'][s]:>12d}" for s in A["splits"]))
    print(f"\nfitted rung, selection: {A['fitted_set']['Selection']}")
    print(f"fitted rung, train tail: {A['fitted_set']['Train tail']}")
    print(f"  in both: {A['fitted_overlap']}")
    print(f"  selection only: {A['fitted_sel_only']}")
    print(f"  train tail only: {A['fitted_tail_only']}")

    if a.latex:
        n = A["n_common"]
        L = ["% GENERATED by scripts/emit_traintail_ladder.py --latex -- do not edit by hand.",
             r"\begin{center}", r"\footnotesize",
             r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Rung (clause~(i)'s denominator) & Selection & Held-out & Train tail \\",
             r"\midrule"]
        for b in A["rungs"]:
            row = " & ".join(str(A["per_rung"][s][b]) for s in A["splits"])
            L.append(f"{LABEL[b]} & {row} \\\\")
        L += [r"\midrule",
              "Clears $0.20$ against \\emph{at least one} admissible rung & "
              + " & ".join(str(A["n_any"][s]) for s in A["splits"]) + r" \\",
              "Clears $0.20$ against \\emph{every} admissible rung & "
              + " & ".join(f"\\textbf{{{A['n_all'][s]}}}" for s in A["splits"]) + r" \\",
              r"\bottomrule", r"\end{tabular}", r"\end{center}", ""]
        OUT.write_text("\n".join(L))
        print(f"\nwrote {OUT.relative_to(ROOT)} (each column out of {n})")


if __name__ == "__main__":
    main()
