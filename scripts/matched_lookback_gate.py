#!/usr/bin/env python3
"""
Does the gate's verdict survive giving the BASELINE the same history the model gets?

WHY THIS EXISTS. The gate is defined against a lookback-96 regression, but Moirai is evaluated with
`extended_lookback = 96 + h` steps of context (Appendix's window-layout figure). A reviewer asked
whether that asymmetry is what produces the negative result. The paper has always argued the
asymmetry runs the OTHER way -- more context lowers MSE_ZS, raises R2_task, and so admits MORE cells,
against our own headline -- but an argument is not a measurement. This script measures it: the same
fitted ridge, same windows, same fitting region, same normalisation, same lam, given 96+h inputs
instead of the last 96.

WHAT IS HELD IDENTICAL. Everything except the integer handed to fit_linear_map. The numerator
MSE_zeroshot is read from the stored gate record and never recomputed, so `fitted` and
`fitted_matched` differ in the baseline's INFORMATION SET and in nothing else. Before reporting, the
`fitted` column is checked against the published denominator through the same code path the ladder
uses -- if window construction had drifted, the matched column would differ for the wrong reason.

WHY IT IS NOT A RUNG OF gate_baseline_sensitivity's LADDER. That ladder varies the estimator FAMILY at
fixed lookback, and its rungs feed graded_value()'s minimum-over-admissible value score, which
fig_value_axis.py and cka_fixed_effects.py both read. Adding a different axis to that minimum would
silently move the value score of every cell in the paper, so the column is reported here instead --
and the last section below states exactly what folding it in WOULD do, per cell, so nothing is hidden
by the separation.

WHICH DIRECTION IS BAD NEWS FOR US. A matched baseline is given strictly more information, so if it is
also strictly better the denominator falls, R2_task falls, and FEWER cells clear the gate. That would
strengthen the paper's negative claim while shrinking the survivor set the intervention analysis is
conditioned on, and both halves have to be reported. More parameters is not automatically stronger,
though: at h=192 the matched map estimates (96+h)*D inputs from the same training windows, so it can
also lose. The measurement decides.

Usage:  .venv12/bin/python scripts/matched_lookback_gate.py [--split test|val] [--latex]
        .venv12/bin/python scripts/matched_lookback_gate.py --from-json --latex   # no CSVs needed
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gate_all_cells as gac                                           # noqa: E402
import gate_baseline_sensitivity as gbs                                # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

NOMINAL = "fitted"           # the gate as the paper defines it: last 96 steps
MATCHED = "fitted_matched"   # the same map, given all 96+h available steps
FLOOR = gbs.FLOOR            # the training-mean admissibility floor, shared with the ladder
BASELINES = (NOMINAL, MATCHED, FLOOR)

OUT_BY_SPLIT = {"test": ROOT / "results/matched_lookback.json",
                "val": ROOT / "results/matched_lookback_val.json"}
TABLE_BY_SPLIT = {"test": ROOT / "paper_8/tables/matched_lookback.tex",
                  "val": ROOT / "paper_8/tables/matched_lookback_val.tex"}
# The ladder's record for the same split, read only to answer "what if this were a rung".
LADDER_BY_SPLIT = {"test": ROOT / "results/gate_baselines.json",
                   "val": ROOT / "results/gate_baselines_val.json"}


def summarise(res):
    """Per-cell comparison plus the two counts the paper states."""
    per = {}
    for key in sorted(res):
        v_n, v_m = res[key].get(NOMINAL), res[key].get(MATCHED)
        if v_n is None or v_m is None:
            continue
        per[key] = dict(
            r2_nominal=v_n["r2_task"], r2_matched=v_m["r2_task"],
            mse_nominal=v_n["linear_test"], mse_matched=v_m["linear_test"],
            zs=v_n["zs_test"],
            adm_nominal=gbs.admissible(res, key, NOMINAL),
            adm_matched=gbs.admissible(res, key, MATCHED),
            pass_nominal=gbs.passes(res, key, NOMINAL),
            pass_matched=gbs.passes(res, key, MATCHED),
            lookback_used=v_m["info"].get("lookback_used"),
            n_features=v_m["info"].get("n_features"),
            n_fit_windows=v_m["info"].get("n_fit_windows"),
        )
        p = per[key]
        # "stronger" = smaller MSE. Recorded per cell rather than asserted globally, because the
        # matched map estimates more coefficients from the same windows and can lose at h=192.
        p["matched_stronger"] = p["mse_matched"] < p["mse_nominal"]
        p["flips_to_fail"] = p["pass_nominal"] and not p["pass_matched"]
        p["flips_to_pass"] = p["pass_matched"] and not p["pass_nominal"]
    n = len(per)
    return dict(
        cells=per, n_scored=n,
        n_pass_nominal=sum(1 for v in per.values() if v["pass_nominal"]),
        n_pass_matched=sum(1 for v in per.values() if v["pass_matched"]),
        n_matched_stronger=sum(1 for v in per.values() if v["matched_stronger"]),
        n_flips_to_fail=sum(1 for v in per.values() if v["flips_to_fail"]),
        n_flips_to_pass=sum(1 for v in per.values() if v["flips_to_pass"]),
        flips_to_fail=[k for k, v in per.items() if v["flips_to_fail"]],
        flips_to_pass=[k for k, v in per.items() if v["flips_to_pass"]],
        threshold=gac.GATE_THRESHOLD,
    )


def as_ladder_rung(res, split):
    """What the graded value score would do if this column WERE a rung, per cell.

    Kept as a disclosure rather than as the reported quantity: the point of separating the axes is not
    to protect the count. Returns None when the ladder record for this split is missing, so an absent
    file reads as "not computed" rather than as "no effect".
    """
    path = LADDER_BY_SPLIT[split]
    if not path.exists():
        return None
    ladder = json.load(open(path))
    merged, out = {}, {}
    for key, cols in ladder["cells"].items():
        if key not in res or MATCHED not in res[key]:
            continue
        merged[key] = dict(cols)
        merged[key][MATCHED] = res[key][MATCHED]
    rungs = [b for b in ladder["baselines"] if b != FLOOR]
    for key in merged:
        before = gbs.graded_value(merged, key, [b for b in ladder["baselines"]])
        after = gbs.graded_value(merged, key, list(ladder["baselines"]) + [MATCHED])
        if before is None or after is None:
            continue
        out[key] = dict(
            r2_best_before=before["r2_best"], best_before=before["best_baseline"],
            r2_best_after=after["r2_best"], best_after=after["best_baseline"],
            clears_before=before["clears_all_admissible"],
            clears_after=after["clears_all_admissible"],
            matched_becomes_strongest=after["best_baseline"] == MATCHED,
        )
    return dict(
        per_cell=out, ladder_rungs=rungs,
        n_clearing_before=sum(1 for v in out.values() if v["clears_before"]),
        n_clearing_after=sum(1 for v in out.values() if v["clears_after"]),
        n_matched_strongest=sum(1 for v in out.values() if v["matched_becomes_strongest"]),
    )


def report(s, rung, split):
    print(f"\n{'=' * 104}\nMATCHED-LOOKBACK GATE ({split} windows): the SAME fitted ridge, given "
          f"96+h steps instead of 96\n{'=' * 104}")
    hdr = ("cell".ljust(30) + "R2(96)".rjust(9) + "R2(96+h)".rjust(10) + "delta".rjust(9)
           + "MSE(96)".rjust(10) + "MSE(96+h)".rjust(11) + "  lb".rjust(6) + "  verdict")
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(s["cells"], key=lambda k: -s["cells"][k]["r2_nominal"]):
        v = s["cells"][key]
        verdict = ("PASS -> fail" if v["flips_to_fail"] else
                   "fail -> PASS" if v["flips_to_pass"] else
                   "PASS (both)" if v["pass_nominal"] else "fail (both)")
        flag = "" if v["adm_matched"] is not False else "   [matched denom below the floor]"
        print(key.ljust(30) + f"{v['r2_nominal']:+.3f}".rjust(9)
              + f"{v['r2_matched']:+.3f}".rjust(10)
              + f"{v['r2_matched'] - v['r2_nominal']:+.3f}".rjust(9)
              + f"{v['mse_nominal']:.4f}".rjust(10) + f"{v['mse_matched']:.4f}".rjust(11)
              + f"{v['lookback_used']}".rjust(6) + "  " + verdict + flag)
    print("-" * len(hdr))
    print(f"  clearing {s['threshold']} against the lookback-96 ridge (the gate as defined): "
          f"{s['n_pass_nominal']}/{s['n_scored']}")
    print(f"  clearing {s['threshold']} against the matched 96+h ridge:                      "
          f"{s['n_pass_matched']}/{s['n_scored']}")
    print(f"  matched denominator is the STRONGER of the two on {s['n_matched_stronger']}"
          f"/{s['n_scored']} cells")
    print(f"  cells that flip PASS -> fail: {s['n_flips_to_fail']}"
          + (f"   {', '.join(s['flips_to_fail'])}" if s["flips_to_fail"] else ""))
    print(f"  cells that flip fail -> PASS: {s['n_flips_to_pass']}"
          + (f"   {', '.join(s['flips_to_pass'])}" if s["flips_to_pass"] else ""))
    print("\n  Read the direction carefully: R2_task = 1 - MSE_zs/MSE_base, so a STRONGER baseline\n"
          "  (smaller MSE_base) gives a SMALLER R2_task. Matching the lookback can therefore only\n"
          "  hurt the pre-trained model's score where it helps the baseline -- which is why the\n"
          "  asymmetry in the published gate is generous to the checkpoint rather than to us.")
    if rung is None:
        print("\n  WHAT IF IT WERE A LADDER RUNG: not computed -- no ladder record for this split.")
        return
    print(f"\n{'=' * 104}\nDISCLOSURE: what folding this column into the ladder's graded minimum "
          f"would do\n{'=' * 104}")
    print("  The graded value score is a MINIMUM over admissible rungs, so adding any rung can only\n"
          "  lower it. This is reported because separating the axes must not be a way of protecting\n"
          f"  a count. Ladder rungs: {', '.join(rung['ladder_rungs'])}.")
    print(f"  clears 0.20 against ALL admissible rungs, ladder as published: "
          f"{rung['n_clearing_before']}/{len(rung['per_cell'])}")
    print(f"  ... with the matched rung added:                               "
          f"{rung['n_clearing_after']}/{len(rung['per_cell'])}")
    print(f"  cells where the matched rung becomes the STRONGEST rival: {rung['n_matched_strongest']}")
    changed = [k for k, v in rung["per_cell"].items() if v["clears_before"] != v["clears_after"]]
    print(f"  cells whose graded verdict would change: {len(changed)}"
          + (f"   {', '.join(changed)}" if changed else ""))


def emit_latex(s, rung, path):
    lines = [
        "% GENERATED by scripts/matched_lookback_gate.py --latex -- do not edit by hand.",
        "\\begin{center}", "\\footnotesize",
        "\\begin{tabular}{@{}lrrrrc@{}}", "\\toprule",
        # $\Vb{...}$, not R^2_task: the paper renamed the statistic in round 5 and a table heading is
        # exactly where the old name survives longest.
        r"Cell & $\Vb{\text{ridge}}$ (96) & $\Vb{\text{ridge}}$ ($96{+}h$) & $\Delta$ "
        r"& MSE ratio & Verdict \\",
        "\\midrule",
    ]
    for key in sorted(s["cells"], key=lambda k: -s["cells"][k]["r2_nominal"]):
        v = s["cells"][key]
        verdict = (r"pass $\rightarrow$ \textbf{fail}" if v["flips_to_fail"] else
                   r"fail $\rightarrow$ \textbf{pass}" if v["flips_to_pass"] else
                   "pass (both)" if v["pass_nominal"] else "fail (both)")
        # The MSE ratio rather than two raw denominators: the question is whether the extra history
        # makes the baseline stronger, and a ratio below 1 says so in one glance at any scale.
        ratio = v["mse_matched"] / v["mse_nominal"]
        n1 = f"{v['r2_nominal']:+.3f}"
        n2 = f"{v['r2_matched']:+.3f}"
        lines.append(
            f"{gbs.display(key)} & "
            + (f"$\\mathbf{{{n1}}}$" if v["pass_nominal"] else f"${n1}$") + " & "
            + (f"$\\mathbf{{{n2}}}$" if v["pass_matched"] else f"${n2}$") + " & "
            + f"${v['r2_matched'] - v['r2_nominal']:+.3f}$ & ${ratio:.3f}$ & {verdict} \\\\")
    lines.append("\\midrule")
    lines.append(r"\emph{Clearing } $0.20$ & "
                 f"{s['n_pass_nominal']}/{s['n_scored']} & "
                 f"{s['n_pass_matched']}/{s['n_scored']} & & "
                 f"{s['n_matched_stronger']}/{s['n_scored']} stronger & "
                 f"{s['n_flips_to_fail']} lost, {s['n_flips_to_pass']} gained \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--from-json", action="store_true",
                    help="re-report and re-emit from the committed results/matched_lookback*.json "
                         "without touching the benchmark CSVs, so the TABLE is re-derivable from a "
                         "clean clone even though the DENOMINATORS need TIER B data")
    a = ap.parse_args()
    OUT, TABLE = OUT_BY_SPLIT[a.split], TABLE_BY_SPLIT[a.split]

    if a.from_json:
        prev = json.load(open(OUT))
        res, ok = prev["cells"], prev["selfcheck_passed"]
        print(f"re-reporting from {OUT.relative_to(ROOT)} "
              f"(self-check as recorded there: {'passed' if ok else 'FAILED'})")
    else:
        # The Moirai arm only: the extended-lookback asymmetry is a property of how Moirai is
        # evaluated. The Chronos and TimesFM arms build lookback-96 evaluation windows, so their
        # model and baseline already see the same history and a matched column would be the nominal
        # one recomputed.
        res = gbs.run(["moirai"], list(BASELINES), a.split)
        ok = gbs.selfcheck(res, a.split)
        if not ok:
            sys.exit("`fitted` does not reproduce the published denominator; the matched column "
                     "would differ for the wrong reason")

    s = summarise(res)
    rung = as_ladder_rung(res, a.split)
    report(s, rung, a.split)
    OUT.write_text(json.dumps(dict(split=a.split, baselines=list(BASELINES),
                                   nominal_lookback=96, gate_threshold=gac.GATE_THRESHOLD,
                                   summary={k: v for k, v in s.items() if k != "cells"},
                                   as_ladder_rung=rung, selfcheck_passed=ok,
                                   per_cell=s["cells"], cells=res), indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")
    if a.latex:
        emit_latex(s, rung, TABLE)
