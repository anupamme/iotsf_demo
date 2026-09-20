#!/usr/bin/env python3
"""Emit paper_8/tables/r2task.tex from results/gate_test_side.json.

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

Source is the TEST-side gate at baseline `fitted`, which is what the gate is defined against; the
superseded trend-baseline values stay in results/gate_test_side_trend.json and are not read here.

Run:  .venv12/bin/python scripts/emit_r2task.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/gate_test_side.json"
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


def fmt(v):
    """Signed, three decimals, in the hand-written table's exact two forms.

    Non-bold entries put the sign in its own math group (`$-$0.243`) so the minus renders as a math
    minus rather than a hyphen and the decimal points still align in the column; bold entries carry
    the sign inside \\mathbf. A gate-pass is by definition >= +0.20, so the bold branch is never
    negative and does not need the $-$ treatment.
    """
    if v >= THRESHOLD:
        return r"$\mathbf{" + f"{v:+.3f}" + "}$"
    return ("$-$" if v < 0 else "$+$") + f"{abs(v):.3f}"


def main():
    if not SRC.exists():
        sys.exit(f"{SRC.relative_to(ROOT)} not found; run gate_all_cells.py --baseline fitted first")
    g = json.load(open(SRC))

    vals, rows, n_win = {}, [], set()
    for heading, pref, datasets in ARMS:
        rows.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{heading}}}}} \\")
        for ds in datasets:
            cells = []
            for h in HORIZONS:
                e = g.get(f"{pref}_{ds}_h{h}")
                if e is None:
                    cells.append("---")
                    continue
                if e["baseline"] != "fitted" or e["split"] != "test":
                    sys.exit(f"{pref}_{ds}_h{h} is {e['baseline']}/{e['split']}, not fitted/test")
                vals[f"{pref}_{ds}_h{h}"] = e["r2_task"]
                n_win.add(e.get("n_windows", e.get("n", None)))
                cells.append(fmt(e["r2_task"]))
            label = rf"\texttt{{{ds}}}" if ds in TT else ds
            rows.append(rf" & {label:<22s} & {cells[0]} & {cells[1]} \\")
        if heading != ARMS[-1][0]:
            rows.append(r"\midrule")

    n_win.discard(None)
    if len(n_win) != 1:
        sys.exit(f"cells disagree on window count {sorted(n_win)}; the caption states one number")
    nw = n_win.pop()

    # The Reading paragraph is prose, and the paper spells small counts as words there while using
    # digits inside tables. Generated text has to follow the same rule or the emitter introduces a
    # typographic regression on every regeneration.
    def word(n):
        return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten"][n] if n <= 10 else str(n)

    passing = sorted(k for k, v in vals.items() if v >= THRESHOLD)
    datasets_passing = sorted({k.split("_")[1] for k in passing})
    # The Reading paragraph's claim is that every passing cell is ETTh2. Computed, so that if a new
    # run record ever breaks it the sentence changes instead of silently misdescribing the table.
    all_etth2 = datasets_passing == ["ETTh2"]
    reading_pass = (
        rf"{word(len(passing))} of these {len(vals)} cells clear the ${THRESHOLD:.2f}$ threshold and"
        "\n"
        rf"all {word(len(passing))} are ETTh2; on every other dataset the released checkpoint does"
        "\nnot beat a supervised lookback-96 linear regression out of sample, at any capacity."
        if all_etth2 else
        rf"{word(len(passing))} of these {len(vals)} cells clear the ${THRESHOLD:.2f}$ threshold, on"
        "\n"
        rf"{word(len(datasets_passing))} dataset(s): {', '.join(datasets_passing)}."
    )

    tex = "\n".join([
        f"% GENERATED by scripts/emit_r2task.py from {SRC.relative_to(ROOT)} -- do not hand-edit.",
        "% The corrected fitted-baseline gate: scripts/gate_all_cells.py --baseline fitted, test",
        f"% split, {nw} matched windows. Superseded trend-baseline values are preserved in",
        "% results/gate_test_side_trend.json / results/gate_val_side_trend.json and are not read here.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Task-native gate score $R^2_{\text{task}}(\text{PT}) {=} 1 {-}",
        r"\text{MSE}_{\text{ZS}}/\text{MSE}_{\text{Linear}}$ for every Moirai cell, on",
        r"held-out (test) windows.",
        r"\textbf{Not a linear probe}: this is a ratio of forecast MSEs computed through",
        r"Moirai's own forecasting head, and is unaffected by the retraction of the",
        r"linear-probe $\Delta R^2$ axis.",
        r"\textbf{Estimator, exactly}: $\text{MSE}_{\text{ZS}}$ is the median of 20",
        r"Moirai forecast samples at extended lookback (context $+$ horizon);",
        r"$\text{MSE}_{\text{Linear}}$ is a single lookback-96 ridge-OLS linear map",
        r"$\mathbb{R}^{96 \times D} {\to} \mathbb{R}^{h \times D}$ \emph{fit on the",
        r"cell's training windows} and applied unchanged out of sample; both are computed",
        rf"on the same {nw} matched test windows, train-normalised.",
        r"The gate involves no fine-tuning, so a cell has one value at every $n$: the",
        r"$n{=}500$ and $n{=}10$k (ES) operating points reported in",
        r"\S\ref{sec:forecasting} share the entry shown here.",
        rf"Bold ${{=}}$ gate-pass at the ${THRESHOLD:.2f}$ threshold. ``---'' ${{=}}$ not run.}}",
        r"\label{tab:r2task}",
        r"\small",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Backbone & Dataset & $h{=}96$ & $h{=}192$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1pt}",
        "",
        r"{\footnotesize",
        r"\textbf{Reading:} " + reading_pass,
        r"$R^2_{\text{task}}(\text{PT}){<}0$ means the linear baseline wins outright.",
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
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(vals)} cells, {len(passing)} gate-pass: "
          f"{', '.join(passing)})")
    if not all_etth2:
        print("  NOTE: the gate-pass cells are no longer all ETTh2 -- check every body sentence "
              "that says 'all five are ETTh2'")


if __name__ == "__main__":
    main()
