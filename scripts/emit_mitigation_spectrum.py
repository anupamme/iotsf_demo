#!/usr/bin/env python3
"""
Emit the mitigation-spectrum table (Moirai-Small/ETTh2, conditions B-G).

This table was hand-authored until 2026-09-18 and carried four defects that a
hand check found and an emitter would have prevented: three rows overstated
their seed count (n=5 printed, 3 on disk), the Forgetting% column mixed two
denominators (B/C/D against each run's own zero-shot, the anchoring rows
against a rounded shared value) while the caption asserted a single one, and
the caption's "matched at n=1,000" comparison quoted a full-fine-tuning figure
with no run record behind it.

One convention, applied to every row: Forgetting% is the mean over seeds of
each run's own (fine-tuned - zero-shot) / zero-shot, so a row is always scored
against the checkpoint it started from. Dispersion is population std (ddof=0),
which is what the previous table printed and what the stored JSON *_std fields
use; the body's SEM convention (04_methodology.tex) does not apply to this
table and the caption says so.

The two condition groups differ in training-set size (B/C/D at n=500, E/F/G at
n=1,000) because that is how they were run. The table states the mismatch
rather than papering over it, and claims no ordering across it.

Regenerate with:  .venv12/bin/python scripts/emit_mitigation_spectrum.py
"""
import glob
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper_8/tables/mitigation_spectrum.tex"

# (label, glob pattern with %s for the horizon, n_train)
# B/C/D are the 10-seed spectrum arms; E/F/G the anchoring and LoRA arms.
ARMS = [
    ("B: Full fine-tune",            "results/forecasting_finetune_20ep/condition_B_%s_s*.json", 500),
    ("C: NLL+SupCon",                "results/forecasting_finetune_20ep/condition_C_%s_s*.json", 500),
    ("D: Frozen encoder",            "results/forecasting_finetune_20ep/condition_D_%s_s*.json", 500),
    (r"F: L2-SP ($\lambda{=}0.01$)", "results/v5_mitigation/l2sp_0.01/%s/*.json",               1000),
    (r"F: L2-SP ($\lambda{=}0.1$)",  "results/v5_mitigation/l2sp_0.1/%s/*.json",                1000),
    (r"G: EWC ($\lambda{=}1000$)",   "results/v5_mitigation/ewc_1000/%s/*.json",                1000),
    (r"E: LoRA ($r{=}8$)",           "results/v5_mitigation/lora/%s/*.json",                    1000),
]
HORIZONS = ["h96", "h192"]


def sd0(v):
    """Population std, matching the stored JSON *_std convention."""
    return st.pstdev(v) if len(v) > 1 else 0.0


def read(pattern):
    files = sorted(glob.glob(str(ROOT / pattern)))
    if not files:
        return None
    mse, zs, cka, drift = [], [], [], []
    for f in files:
        d = json.load(open(f))
        mse.append(d["final_val_mse"])
        zs.append(d["zeroshot_mse"])
        cka.append(d["final_cka"])
        drift.append(d["final_weight_drift"])
    # per-seed forgetting against that seed's own zero-shot, then averaged
    forg = [100.0 * (m - z) / z for m, z in zip(mse, zs)]
    return {
        "n": len(files),
        "mse": st.mean(mse), "mse_sd": sd0(mse), "zs": st.mean(zs),
        "forg": st.mean(forg),
        "cka": st.mean(cka), "cka_sd": sd0(cka),
        "drift": st.mean(drift), "drift_sd": sd0(drift),
    }


def num(x, places):
    """Format like the paper: drop the leading zero on values below 1."""
    s = f"{x:.{places}f}"
    return s[1:] if s.startswith("0.") else s


def cell(a, best_mse=False, best_forg=False):
    places = 5 if a["cka"] > 0.999 else 3
    cka = num(a["cka"], places)
    cka_sd = num(a["cka_sd"], places)
    mse = num(a["mse"], 3)
    forg = f"{a['forg']:+.1f}".replace("+", "$+$").replace("-", "$-$")
    if best_mse:
        mse = rf"\textbf{{{mse}}}"
    if best_forg:
        forg = rf"\textbf{{{forg}}}"
    return (f"{mse}{{\\tiny$\\pm${num(a['mse_sd'], 3)}}} & {forg} & "
            f"{cka}{{\\tiny$\\pm${cka_sd}}} & {a['drift']:.2f}{{\\tiny$\\pm${num(a['drift_sd'], 2)}}}")


def lora_scope():
    """
    The three facts the caption needs about the eight-cell LoRA arm.

    Read from cell_matrix rather than typed into the caption. The caption's job here is to stop a
    reader generalising this cell's CKA${\\approx}0.98$, so if the wider arm ever moves, the sentence
    that scopes it has to move with it -- and a hand-typed "$0.55$ to $0.97$" would not.
    """
    import cell_matrix as cm
    lo = {k: v for k, v in cm.lora_value_cells().items() if v["be_seeds"] >= cm.MIN_SEEDS}
    if not lo:
        sys.exit("MISSING condition-E value-cell runs; the caption's scoping sentence needs them")
    cka = [v["cka_e"] for v in lo.values()]
    return {"n": len(lo), "cka_lo": min(cka), "cka_hi": max(cka),
            "be_neg": sum(1 for v in lo.values() if v["be_test"] < 0)}


def main():
    lora = lora_scope()
    data = {}
    for label, pat, n_train in ARMS:
        for h in HORIZONS:
            a = read(pat % h)
            if a is None:
                sys.exit(f"MISSING run records for {label} {h}: {pat % h}")
            data[(label, h)] = a
        if data[(label, "h96")]["n"] != data[(label, "h192")]["n"]:
            sys.exit(f"{label}: seed counts differ across horizons")

    # order by representation preservation at h=96, as the caption states
    order = sorted(ARMS, key=lambda t: data[(t[0], "h96")]["cka"])

    zs = {h: {n: None for n in (500, 1000)} for h in HORIZONS}
    for label, pat, n_train in ARMS:
        for h in HORIZONS:
            zs[h][n_train] = data[(label, h)]["zs"]

    L = ["% GENERATED by scripts/emit_mitigation_spectrum.py -- do not edit by hand.",
         r"\begin{table}[t]", r"\centering",
         r"\caption{\textbf{Mitigation spectrum on Moirai-Small/ETTh2.}",
         r"Rows ordered by representation preservation (CKA at $h{=}96$).",
         r"On \emph{this} cell LoRA preserves representations (CKA${\approx}0.98$) while",
         r"\emph{improving} on zero-shot, and weight anchoring reduces drift without buying",
         r"utility.  \textbf{Neither reading generalises, and we checked rather than assumed:}",
         (rf"extending condition~E to {lora['n']} further value-cells"),
         # Plain formatting, not num(): num() drops the leading zero, which is the body's convention
         # for MSE columns but not how this caption or the LoRA appendix writes a CKA.
         (r"(Appendix~\ref{app:loravaluecells}) puts CKA$_\text{E}$ anywhere from "
          rf"${lora['cka_lo']:.2f}$ to"),
         (rf"${lora['cka_hi']:.2f}$ and leaves full fine-tuning ahead of LoRA on "
          rf"{lora['be_neg']} of the {lora['n']}."),
         r"\textbf{Every row is scored against its own zero-shot.} Forgetting\% is the",
         r"mean over seeds of each run's $(\text{fine-tuned}-\text{zero-shot})/",
         r"\text{zero-shot}$, so no row borrows another's denominator; earlier versions of",
         r"this table divided the anchoring and LoRA rows by a single shared value and read",
         r"up to $3.4$~pp away as a result.  MSE, CKA and $\ell_2$ drift are means over",
         r"seeds with population std (ddof${=}0$); the body's SEM convention",
         r"(\S\ref{sec:method}) does not apply here.",
         r"\textbf{The rows are not matched on training-set size.}",
         (r"Conditions B, C and D use $n{=}500$ training samples (zero-shot MSE "
          rf"${num(zs['h96'][500], 4)}$ at $h{{=}}96$, ${num(zs['h192'][500], 4)}$ at $h{{=}}192$);"),
         (r"E, F and G use $n{=}1{,}000$ (zero-shot MSE "
          rf"${num(zs['h96'][1000], 4)}$ and ${num(zs['h192'][1000], 4)}$); the two rightmost"),
         r"columns give the seed count and the training-set size separately.",
         r"We therefore claim no ordering \emph{between} the two",
         r"groups: there is no full-fine-tuning arm at $n{=}1{,}000$ on this cell with more",
         r"than a single seed, so ``anchoring ties full fine-tuning'' is not a comparison",
         r"these runs support.  The dissociation the table is cited for (CKA not predicting",
         r"utility) is within-group and unaffected.",
         r"Condition~D reads CKA $0.99996$ rather than $1$: it leaves \texttt{in\_proj} and",
         r"\texttt{mask\_encoding} trainable, so it is a frozen-\emph{encoder} control and not",
         r"a strict freeze.  Only condition~H (Appendix~\ref{app:strictfreeze}) is $1.0000$ by",
         r"construction.  Earlier versions of this table printed D as $1.000$, which rounded",
         r"that distinction away.}",
         r"\label{tab:mitigation_spectrum}", r"\small",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l cccc cccc cc}", r"\toprule",
         r"& \multicolumn{4}{c}{$h=96$} & \multicolumn{4}{c}{$h=192$} & & \\",
         r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
         r"Method & MSE & Forg.\% & CKA & Drift & MSE & Forg.\% & CKA & Drift & seeds & $n$ \\",
         r"\midrule"]

    best = {h: min(ARMS, key=lambda t: data[(t[0], h)]["forg"])[0] for h in HORIZONS}
    for label, pat, n_train in order:
        if label.startswith("D:"):
            L.append(r"\midrule")
        a96, a192 = data[(label, "h96")], data[(label, "h192")]
        dag = r"\textsuperscript{$\dagger$}" if label.startswith("E:") else ""
        c96 = cell(a96, best_mse=best["h96"] == label, best_forg=best["h96"] == label)
        c192 = cell(a192, best_mse=best["h192"] == label, best_forg=best["h192"] == label)
        L.append(f"{label}{dag} & {c96} & {c192} & {a96['n']} & "
                 f"{n_train if n_train != 1000 else '1{,}000'} \\\\")

    L += [r"\bottomrule", r"\end{tabular}%", r"}", "",
          r"\smallskip",
          r"{\footnotesize $\dagger$ LoRA ($\alpha{=}16$) adds low-rank updates that do not",
          r"modify base encoder parameters; $\ell_2$ drift is exactly zero by construction.}",
          r"\end{table}"]

    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for label, pat, n_train in order:
        a = data[(label, "h96")]
        print(f"  {label:32s} n={a['n']} n_train={n_train:5d} "
              f"mse={a['mse']:.4f} forg={a['forg']:+6.2f} cka={a['cka']:.5f}")


if __name__ == "__main__":
    main()
