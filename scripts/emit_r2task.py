#!/usr/bin/env python3
"""Emit paper_8/tables/r2task.tex from results/gate_val_side.json and results/gate_test_side.json.

WHY THIS EXISTS. This was the last table the paper \\input{}s that carried hand-typed numbers -- 21
of them -- and its own header said so. Seven of the ten hand-authored tables in this project have
turned out to be defective at least once, so "hand-maintained" is not a neutral property: it means
`git status --porcelain paper_8/tables/` after a regeneration sweep was NOT a complete staleness
check, because this file could not move. With this emitter it can, which is also what makes
scripts/rederive_all.sh able to claim it re-derives every reported number rather than nearly all of
them.

WHAT IT DOES NOT CHANGE. The caption prose is reproduced verbatim from the hand-written version: it
documents the estimator, the 300 matched windows, the train-normalisation, the retraction of the
linear-probe axis and the Electricity7 protocol deviation, and none of that is derivable from the
JSON. Only the NUMBERS and the counts in the Reading paragraph are generated. The bold-face rule and
the "5 of 21 and all five are ETTh2" sentence are computed, not asserted, so a change in the run
records changes the sentence too instead of leaving it stale beside new numbers.

BOTH SPLITS, SIDE BY SIDE. The paper's primary gate is scored on the SELECTION windows, which are
disjoint from the held-out windows the intervention outcome is scored on; scoring it on the held-out
windows makes cell inclusion a retrospective judgement, and the paper reports that variant as a
sensitivity analysis rather than dropping it. This table prints both, because the difference between
them is itself a result: seven cells clear on selection and five on held-out, and the two extra are
ETTm2 rather than ETTh2. Bold marks the PRIMARY (selection) pass, so the bold set is the one every
count in the body refers to. Both columns come from baseline `fitted`, which is what the gate is
defined against; the superseded trend-baseline values stay in results/gate_*_side_trend.json.

Run:  .venv12/bin/python scripts/emit_r2task.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# (split, path). The selection split is first everywhere in this file because it is the primary.
SRCS = [("val", ROOT / "results/gate_val_side.json"),
        ("test", ROOT / "results/gate_test_side.json")]
OUT = ROOT / "paper_8/tables/r2task.tex"
THRESHOLD = 0.20

# (heading, size-prefix, datasets in display order, horizons in display order). Moirai only: the
# non-Moirai arms are scored on 200 windows rather than 300 and live in tables/crossbackbone.tex,
# and mixing the two window counts in one table would make the column headers a lie.
ARMS = [
    ("Moirai-Small", "small", ["ETTh1", "ETTh2", "ETTm2", "Weather", "Electricity7"]),
    ("Moirai-Base", "base", ["ETTh1", "ETTh2", "ETTm2", "Weather"]),
    ("Moirai-Large", "large", ["ETTh1", "ETTh2", "Weather"]),
]
HORIZONS = [96, 192]
# Datasets whose label is set in typewriter because the name is a protocol deviation, not a
# standard benchmark; the caption explains what the restriction is.
TT = {"Electricity7"}


def fmt(v, bold):
    """Signed, three decimals, in the hand-written table's exact two forms.

    Non-bold entries put the sign in its own math group (`$-$0.243`) so the minus renders as a math
    minus rather than a hyphen and the decimal points still align in the column; bold entries carry
    the sign inside \\mathbf. `bold` is passed rather than derived from `v >= THRESHOLD` because only
    the PRIMARY (selection) column is bolded: bolding both would make the held-out column look like a
    second set of survivors when it is the retrospective variant of the same decision.
    """
    if bold:
        return r"$\mathbf{" + f"{v:+.3f}" + "}$"
    return ("$-$" if v < 0 else "$+$") + f"{abs(v):.3f}"


def main():
    G = {}
    for split, src in SRCS:
        if not src.exists():
            sys.exit(f"{src.relative_to(ROOT)} not found; "
                     f"run gate_all_cells.py --baseline fitted --split {split} first")
        G[split] = json.load(open(src))

    vals, rows, n_win = {}, [], {s: set() for s, _ in SRCS}
    for heading, pref, datasets in ARMS:
        rows.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{heading}}}}} \\")
        for ds in datasets:
            cells = []
            for split, _ in SRCS:
                for h in HORIZONS:
                    key = f"{pref}_{ds}_h{h}"
                    e = G[split].get(key)
                    if e is None:
                        cells.append("---")
                        continue
                    if e["baseline"] != "fitted" or e["split"] != split:
                        sys.exit(f"{key} is {e['baseline']}/{e['split']}, not fitted/{split}")
                    vals.setdefault(key, {})[split] = e["r2_task"]
                    n_win[split].add(e.get("n_windows", e.get("n", None)))
                    # bold only where the PRIMARY split passes, in both columns of that row-pair,
                    # so the bold set reads as "these cells are in" rather than "this number is big"
                    prim = G["val"].get(key)
                    cells.append(fmt(e["r2_task"],
                                     prim is not None and prim["r2_task"] >= THRESHOLD))
            label = rf"\texttt{{{ds}}}" if ds in TT else ds
            rows.append(rf" & {label:<22s} & " + " & ".join(cells) + r" \\")
        if heading != ARMS[-1][0]:
            rows.append(r"\midrule")

    nw = {}
    for split, _ in SRCS:
        n_win[split].discard(None)
        if len(n_win[split]) != 1:
            sys.exit(f"{split} cells disagree on window count {sorted(n_win[split])}; "
                     f"the caption states one number per split")
        nw[split] = n_win[split].pop()

    # The Reading paragraph is prose, and the paper spells small counts as words there while using
    # digits inside tables. Generated text has to follow the same rule or the emitter introduces a
    # typographic regression on every regeneration.
    def word(n):
        return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten"][n] if n <= 10 else str(n)

    passing = {s: sorted(k for k, v in vals.items() if v.get(s, -9) >= THRESHOLD)
               for s, _ in SRCS}
    datasets_passing = sorted({k.split("_")[1] for k in passing["val"]})
    # Whether the primary survivors are confined to one dataset is computed, not asserted: if a run
    # record ever breaks it the sentence changes instead of silently misdescribing the table. The
    # same goes for the two splits' sets -- whether one contains the other is checked, because the
    # body says the selection split ADDS cells and that would be wrong if any test-side cell dropped.
    one_dataset = len(datasets_passing) == 1
    superset = set(passing["test"]) <= set(passing["val"])
    added = sorted(set(passing["val"]) - set(passing["test"]))
    reading_pass = (
        rf"{word(len(passing['val']))} of these {len(vals)} cells clear the ${THRESHOLD:.2f}$ "
        rf"threshold on the selection windows and {word(len(passing['test']))} on the held-out "
        "windows.\n"
        + (rf"The selection set contains the held-out set and adds "
           rf"{word(len(added))} ({', '.join(d.split('_')[1] for d in added)}), "
           "so the retrospective variant is the stricter of the two here.\n"
           if superset and added else
           "The two sets are not nested, so neither split's count bounds the other.\n")
        + (rf"Every primary survivor is {datasets_passing[0]}; on every other dataset the released "
           "checkpoint does\nnot beat a supervised lookback-96 linear regression, at any capacity."
           if one_dataset else
           rf"The primary survivors span {word(len(datasets_passing))} datasets: "
           rf"{', '.join(datasets_passing)}.")
    )

    tex = "\n".join([
        "% GENERATED by scripts/emit_r2task.py from "
        + " and ".join(str(s.relative_to(ROOT)) for _, s in SRCS) + " -- do not hand-edit.",
        "% The corrected fitted-baseline gate: scripts/gate_all_cells.py --baseline fitted, on both",
        f"% splits ({nw['val']} matched selection windows, {nw['test']} matched held-out windows).",
        "% Superseded trend-baseline values are preserved in",
        "% results/gate_test_side_trend.json / results/gate_val_side_trend.json and are not read here.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Baseline-relative pre-trained advantage $\Vb{\text{ridge}} {=} 1 {-}",
        r"\text{MSE}_{\text{ZS}}/\text{MSE}_{\text{Linear}}$ for every Moirai cell, on both",
        r"splits. The \emph{selection} columns are the paper's primary gate: they are disjoint from",
        r"the held-out windows every intervention outcome is scored on, so cell inclusion is not a",
        r"judgement made after seeing the outcome. The \emph{held-out} columns are the retrospective",
        r"variant, reported as a sensitivity analysis. Bold marks a pass on the \emph{primary} split",
        r"in both columns of that row, so the bold set is the one the body's counts refer to.",
        r"\textbf{Not a linear probe}: this is a ratio of forecast MSEs computed through",
        r"Moirai's own forecasting head, and is unaffected by the retraction of the",
        r"linear-probe $\Delta R^2$ axis.",
        r"\textbf{Estimator, exactly}: $\text{MSE}_{\text{ZS}}$ is the median of 20",
        r"Moirai forecast samples at extended lookback (context $+$ horizon);",
        r"$\text{MSE}_{\text{Linear}}$ is a single lookback-96 ridge-OLS linear map",
        r"$\mathbb{R}^{96 \times D} {\to} \mathbb{R}^{h \times D}$ \emph{fit on the",
        r"cell's training windows} and applied unchanged out of sample; both are computed",
        rf"on the same matched windows within a split ({nw['val']} selection, {nw['test']}",
        r"held-out), train-normalised.",
        r"The gate involves no fine-tuning, so a cell has one value at every $n$: the",
        r"$n{=}500$ and $n{=}10$k (ES) operating points reported in",
        r"\S\ref{sec:forecasting} share the entry shown here.",
        rf"Bold ${{=}}$ gate-pass at the ${THRESHOLD:.2f}$ threshold on the primary split."
        r" ``---'' ${=}$ not run.}",
        r"\label{tab:r2task}",
        r"\small",
        r"\begin{tabular}{ll cc cc}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{Selection (primary)} & \multicolumn{2}{c}{Held-out (retrosp.)} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Backbone & Dataset & $h{=}96$ & $h{=}192$ & $h{=}96$ & $h{=}192$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1pt}",
        "",
        r"{\footnotesize",
        r"\textbf{Reading:} " + reading_pass,
        r"$\Vb{\text{ridge}}{<}0$ means the linear baseline wins outright.",
        r"These values supersede the ones printed in earlier versions of this table,",
        r"which were computed against a per-window trend extrapolation rather than the",
        r"fitted regression the gate is defined with; the substitution and its",
        r"consequences are documented in Appendix~\ref{app:gatecorrection}.",
        r"\texttt{Electricity7} is Electricity restricted to 7 of its 370 series (six",
        r"\texttt{MT} series plus \texttt{OT}), a deliberate protocol deviation that",
        r"keeps the cell dimensionality-matched to ETT/Weather.}",
        r"\end{table}",
        "",
    ])
    OUT.write_text(tex)
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(vals)} cells)")
    for split, _ in SRCS:
        print(f"  {split:4s}: {len(passing[split])} gate-pass: {', '.join(passing[split])}")
    if not superset:
        print("  NOTE: the held-out pass set is NOT contained in the selection pass set -- the body "
              "says the selection split ADDS cells, which is now wrong")
    if not one_dataset:
        print(f"  NOTE: primary gate-pass cells span {datasets_passing} -- check every body sentence "
              "that says the survivors are all one dataset")


if __name__ == "__main__":
    main()
