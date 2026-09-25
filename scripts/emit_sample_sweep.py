#!/usr/bin/env python3
"""Emit paper_8/tables/sample_sweep.tex from the run records.

WHY THIS EXISTS. Until now this was the one table the paper \\input{}s that no emitter wrote, which
made the Reproducibility Statement's "no table in this paper is assembled by hand" false and left
`git status --porcelain paper_8/tables/` an incomplete staleness check: a number here could drift for
any number of rounds without the sweep noticing. One number had. The hand-typed n=10,000 MSE
dispersion read +/-.008 where the records give +/-.0075, and the body quoted the n=10,000 forgetting
spread as +/-6.2, which is neither the table's ddof=1 standard deviation (6.56) nor the body's
declared SEM (2.07) but the population standard deviation stored in the JSON.

DISPERSION, AND WHY THIS TABLE DEPARTS FROM THE BODY'S CONVENTION. Every other generated table prints
the standard error of the mean, as sec:method declares. This one prints the SAMPLE STANDARD DEVIATION
(ddof=1) and says so in its caption, because the table's whole point is how the SPREAD ACROSS SEEDS
moves with n -- the crossover at n=5,000 is an increase in seed variance, and SEM would hide it
behind the seed count (7 seeds at n=5,000 against 3 at n=1,000, so SEM falls where the variance
rises). A table whose caption claims elevated variance must print a column in which that is visible.

THE RETRACTED PROBE COLUMNS ARE NOT HERE. The earlier hand-made version of this table carried
R2(FT) and two Delta-R2 columns from the withdrawn linear-probe analysis. Those readings are
retracted (sections/076_retraction.tex), the per-n probe records are scattered across half a dozen
result directories with 1-3 seeds each, and reconstructing them would mean an emitter whose main
output is numbers the paper tells the reader not to read. They are retained as a record in the
retraction section instead, which is where a withdrawn computation belongs.

Run:  .venv12/bin/python scripts/emit_sample_sweep.py
"""
import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper_8/tables/sample_sweep.tex"

# One cell, five training-set sizes: Moirai-Small / ETTh2, h=96, condition B. The n=5,000 row pools
# two result directories because the seed extension (v9) was run separately from the original sweep
# (v8); the n=10,000 row is the CUDA arm, which is the only one with early stopping and the only one
# that ran ten seeds. Directories are listed rather than globbed loosely so that a new result
# directory cannot silently join a row.
GROUPS = [
    (500,    ["results/v8_final/n500/*.json"],                                     ""),
    (1000,   ["results/v8_final/n1000/*.json"],                                    ""),
    (2000,   ["results/v8_final/n2000/*.json"],                                    ""),
    (5000,   ["results/v8_final/n5000/*.json", "results/v9_n5k_seeds/*.json"],     ""),
    (10000,  ["results/v19_cuda_etth2_n10k/seed*/*.json"],                   "cuda"),
]


def load(pats):
    rows = []
    for p in pats:
        for f in sorted(glob.glob(str(ROOT / p))):
            rows.append(json.load(open(f)))
    return rows


def ms(rows, key, dp):
    """mean +/- sample standard deviation over seeds, in the table's .xxx{\\pm}.xxx form."""
    v = [r[key] for r in rows if key in r]
    if not v:
        return "n/a"
    m, s = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
    # Leading zero stripped on the MSE and CKA columns only, matching the rest of the paper's tables;
    # a signed percentage keeps its digit.
    def f(x):
        t = f"{x:.{dp}f}"
        return t[1:] if t.startswith("0.") else ("-" + t[2:] if t.startswith("-0.") else t)
    return f"{f(m)}{{\\tiny$\\pm${f(s)}}}"


def pct(rows, key):
    v = [r[key] for r in rows if key in r]
    m, s = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
    sign = "+" if m >= 0 else "$-$"
    return f"{sign}{abs(m):.1f}{{\\tiny$\\pm${s:.1f}}}"


def wilson(k, n, z=1.96):
    """95% Wilson interval on the fraction of seeds with negative forgetting. The caption quotes a
    sign count, and a sign count over 10 seeds without an interval is the kind of claim this paper
    spends an appendix criticising."""
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, c - h), min(1.0, c + h)


def protocol(rows, n):
    """Assert the row is protocol-homogeneous and return (epochs, early-stopping-enabled).

    A sweep row silently mixing two protocols is this project's recurring defect: the n=10,000 arm
    runs 10 epochs WITH early stopping while every smaller n runs 20 epochs WITHOUT it, and that
    difference alone reverses the sign of forgetting (sec:forecasting:replication). So the emitter
    refuses to print a row whose records disagree, rather than averaging across the difference.
    """
    got = {(r.get("epochs"), bool(isinstance(r.get("early_stopping"), dict)
                                  and r["early_stopping"].get("enabled")),
            r.get("max_train_samples")) for r in rows}
    if len(got) != 1:
        raise SystemExit(f"n={n}: records disagree on (epochs, early_stopping, n_train): {got}")
    ep, es, ntr = got.pop()
    if ntr != n:
        raise SystemExit(f"n={n}: records say max_train_samples={ntr}")
    return ep, es


def main():
    data = [(n, load(pats), tag) for n, pats, tag in GROUPS]
    missing = [n for n, rows, _ in data if not rows]
    if missing:
        raise SystemExit(f"no run records for n={missing}; check the directories in GROUPS")

    neg = {n: (sum(r["forgetting_pct"] < 0 for r in rows), len(rows)) for n, rows, _ in data}
    proto = {n: protocol(rows, n) for n, rows, _ in data}
    small = sorted({proto[n] for n in (500, 1000, 2000, 5000)})
    if len(small) != 1:
        raise SystemExit(f"the n<=5,000 rows are not protocol-matched: {small}")
    n10, rows10, _ = data[-1]
    k10, kk10 = neg[10000]
    lo, hi = wilson(k10, kk10)

    L = ["% GENERATED by scripts/emit_sample_sweep.py -- do not edit by hand.",
         "\\begin{table}[t]", "\\centering",
         "\\caption{\\textbf{Sample-size sweep on Moirai-Small/ETTh2} ($h{=}96$, condition~B).",
         "Drift grows monotonically (CKA$\\downarrow$ as $n\\uparrow$) while forgetting is",
         "\\emph{non-monotonic} and reverses sign: positive at $n{\\leq}1{,}000$, negative at",
         "$n{=}2{,}000$, positive again at $n{=}5{,}000$ and negative at $n{=}10{,}000$",
         f"(${k10}/{kk10}$ seeds negative, Wilson CI $[{lo:.2f}, {hi:.2f}]$).",
         "\\textbf{If drift magnitude predicted harm the CKA and Forg.\\ columns would",
         "track; they do not}: the dissociation of",
         "\\S\\ref{sec:forecasting:replication} measured inside a single cell.",
         "\\textbf{Dispersion here is the sample standard deviation over seeds}",
         "($\\text{ddof}{=}1$), not the standard error the rest of the paper reports",
         "(\\S\\ref{sec:method}): what this table is about is how the spread across seeds",
         "moves with $n$ (the $n{=}5{,}000$ crossover \\emph{is} a rise in seed",
         "variance), and SEM would hide that behind $k$, which rises at the same place.}",
         "\\label{tab:sample_sweep}", "\\small",
         "\\begin{tabular}{@{}rrccc@{}}", "\\toprule",
         "Samples & $k$ & MSE & Forg.\\% & CKA \\\\", "\\midrule"]
    for n, rows, tag in data:
        label = f"{n:,}".replace(",", "{,}") + ("$^{\\mathrm{cuda}}$" if tag == "cuda" else "")
        L.append(f"{label} & {len(rows)} & {ms(rows, 'final_val_mse', 3)} & "
                 f"{pct(rows, 'forgetting_pct')} & {ms(rows, 'final_cka', 3)} \\\\")
    best = [r["early_stopping"]["best_epoch"] for r in rows10]
    L += ["\\bottomrule", "\\end{tabular}", "", "\\smallskip", "{\\footnotesize",
          f"$^{{\\mathrm{{cuda}}}}$The $n{{=}}10{{,}}000$ row is the CUDA arm (A10G): "
          f"{len(rows10)} deterministic seeds, {proto[10000][0]} epochs \\emph{{with}} early",
          f"stopping (mean best epoch ${st.mean(best):.1f}{{\\pm}}{st.stdev(best):.1f}$).  Every",
          f"smaller $n$ is MPS, {small[0][0]} epochs \\emph{{without}} early stopping.  That is the",
          "protocol dependence \\S\\ref{sec:forecasting:replication} discusses;",
          "Appendix~\\ref{app:limitations}~(2) gives the $n{=}10{,}000$ seeds and the",
          "matched MPS comparisons, and Appendix~\\ref{app:n5k_trajectories} the",
          f"trajectories of the {len(data[3][1])} $n{{=}}5{{,}}000$ seeds.",
          "Rows read \\texttt{final\\_val\\_mse}, \\texttt{forgetting\\_pct} and",
          "\\texttt{final\\_cka} from the per-run records in",
          "\\texttt{results/v8\\_final/n\\{500,1000,2000,5000\\}},",
          "\\texttt{results/v9\\_n5k\\_seeds} and",
          "\\texttt{results/v19\\_cuda\\_etth2\\_n10k}.  The three withdrawn linear-probe columns",
          "this table used to carry are retained as a record in \\S\\ref{sec:retraction}.}",
          "\\end{table}"]
    OUT.write_text("\n".join(L) + "\n")

    print("SAMPLE-SIZE SWEEP, from the run records")
    print(f"  {'n':>7} {'k':>3} {'MSE':>22} {'Forg.%':>24} {'CKA':>22}  neg/k  protocol")
    for n, rows, _ in data:
        ep, es = proto[n]
        print(f"  {n:>7} {len(rows):>3} {ms(rows, 'final_val_mse', 3):>22} "
              f"{pct(rows, 'forgetting_pct'):>24} {ms(rows, 'final_cka', 3):>22}  "
              f"{neg[n][0]}/{neg[n][1]}    {ep} ep, es={es}")
    print(f"  n=10,000 sign count {k10}/{kk10}, Wilson CI [{lo:.3f}, {hi:.3f}]; "
          f"best epoch {st.mean(best):.2f}+-{st.stdev(best):.2f}")
    # All three conventions, printed every run, because the body and this table disagreed for ten
    # rounds on which one the sweep's dispersions were.
    for key in ("final_val_mse", "forgetting_pct", "final_cka"):
        v = [r[key] for r in rows10]
        print(f"  n=10,000 {key:>15}: mean {st.mean(v):+.4f}  sd1 {st.stdev(v):.4f}  "
              f"sd0 {st.pstdev(v):.4f}  sem {st.stdev(v) / len(v) ** 0.5:.4f}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
