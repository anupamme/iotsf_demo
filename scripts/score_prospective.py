#!/usr/bin/env python3
"""
Score the pre-registered predictions against what the prospective cells actually did.

The predictions were frozen in results/v47_prospective/preregistration.json and committed to git
BEFORE any condition B or D run existed (commit 69d3379). This script only reads that file and the
finished runs; it decides nothing.

Two rules were registered:
  gate rule     gate_val >= 0.20 -> at risk of degradation.  The paper's own criterion.
  dataset rule  dataset in {ETTh1, Weather} -> degradation.  POST HOC pattern from the published
                cells, registered as a competitor so the gate's performance has a reference point.

Outcome is degradation_cells()' definition applied unchanged: gate-passing AND forg_B > 0 in every
seed AND forg_D < 0 in every seed. Cells with fewer than cell_matrix.MIN_SEEDS seeds are still
running and are reported as pending, never scored -- a one-seed cell passes the "every seed" clause
trivially.

WHICH GATE CLAUSE (i) READS, AND WHY BOTH. The frozen predictor's gate_val column is the UNCORRECTED
trend-denominator score as it stood on 2026-08-26: that is the predictor that was registered, so it
is reported unchanged even though the paper's gate has since been corrected -- rewriting it would
turn a prospective test into a retrospective one. Clause (i) of the OUTCOME is a different matter:
the registration says "exactly as cell_matrix.degradation_cells() applies it", and cell_matrix has
since moved its primary to the selection split. So the outcome is scored under BOTH corrected gates
and the two readings are compared. They agree here (no cell degrades on either), which is worth
printing: it is what rules out the suspicion that the split choice is what emptied the positive class.

Run:  python3 scripts/score_prospective.py
      python3 scripts/score_prospective.py --latex
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import cell_matrix  # noqa: E402

PRE = ROOT / "results/v47_prospective/preregistration.json"


def load():
    pre = json.loads(PRE.read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        rows = {r["ref"]: r for r in cell_matrix.build_rows()}
    # Both corrected gates. `deg_primary` is the selection-split reading, which is what
    # cell_matrix.degradation_cells() now applies and therefore what the registration's wording
    # points at; `deg_retro` is the held-out one the earlier draft scored. Reported side by side.
    def gates_from(name):
        return {k: v["r2_task"] for k, v in
                json.loads((ROOT / "results" / name).read_text()).items()}
    g_val, g_test = gates_from("gate_val_side.json"), gates_from("gate_test_side.json")
    out = []
    for c in pre["cells"]:
        r = rows.get(c["cell"])
        if r is None or r.get("forg_b") is None:
            out.append(dict(c, status="pending")); continue
        # Clauses (ii) and (iii) are identical across the two readings; only clause (i) moves.
        clauses_ii_iii = (r["forg_b"] > 0 and r["forg_d"] < 0
                          and r["forg_b_pos"] == r["seeds"] and r["forg_d_neg"] == r["seeds"])
        deg = g_val.get(c["cell"], -1) >= pre["gate_threshold"] and clauses_ii_iii
        deg_retro = g_test.get(c["cell"], -1) >= pre["gate_threshold"] and clauses_ii_iii
        sem = (r["bd_test_sd"] or 0) / max(r["seeds"], 1) ** 0.5
        reading = ("directional" if abs(r["bd_test"]) < 2 * sem
                   else ("freezing better" if r["bd_test"] > 0 else "adaptation helps"))
        out.append(dict(c, status="scored", degraded=deg, degraded_retro=deg_retro,
                        reading=reading, sem=sem,
                        gate_corrected=g_val.get(c["cell"]), gate_test=g_test.get(c["cell"]),
                        seeds=r["seeds"],
                        forg_b=r["forg_b"], forg_b_pos=r["forg_b_pos"],
                        forg_d=r["forg_d"], bd_test=r["bd_test"]))
    return pre, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    pre, cells = load()
    done = [c for c in cells if c["status"] == "scored"]

    print("=" * 100)
    print("PROSPECTIVE TEST -- predictions frozen in git before any B/D run existed")
    print(f"registered {pre['written_utc']} at {pre['git_head'][:12]}")
    print("=" * 100)
    hdr = (f"{'cell':24s} {'gate_pre':>8s} {'gate_val':>8s} {'gate_test':>9s} {'forg_B':>14s} "
           f"{'B-D':>15s} {'degraded':>9s}")
    print(hdr + "  gate  data")
    for c in cells:
        if c["status"] == "pending":
            print(f"{c['cell']:24s} {c['gate_val']:+8.3f}   ...still running"); continue
        print(f"{c['cell']:24s} {c['gate_val']:+8.3f} {c['gate_corrected']:+8.3f} "
              f"{c['gate_test']:+9.3f} "
              f"{c['forg_b']:+7.2f}({c['forg_b_pos']}/{c['seeds']}) "
              f"{c['bd_test']:+7.2f}±{c['sem']:5.2f} {'YES' if c['degraded'] else 'no':>9s}  "
              f"{'OK' if c['predict_gate_rule'] == c['degraded'] else ' X':>4s}"
              f"{'OK' if c['predict_dataset_rule'] == c['degraded'] else ' X':>6s}")

    ng = sum(c["predict_gate_rule"] == c["degraded"] for c in done)
    nd = sum(c["predict_dataset_rule"] == c["degraded"] for c in done)
    ndeg = sum(c["degraded"] for c in done)
    ndeg_retro = sum(c["degraded_retro"] for c in done)
    print(f"\n  scored {len(done)}/{len(cells)}   degradation cells among them: {ndeg}")
    # Printed whether or not they agree. If a future run makes them disagree, the disagreement is the
    # finding and belongs in the paper -- it must not be discoverable only by reading this source.
    print(f"  clause (i) on the selection-split gate: {ndeg}; on the held-out gate: {ndeg_retro}"
          f"  -- {'AGREE' if ndeg == ndeg_retro else 'THEY DISAGREE, report this'}")
    print(f"  gate rule    {ng}/{len(done)} correct")
    print(f"  dataset rule {nd}/{len(done)} correct")
    # the gate fires on everything, so state its confusion matrix rather than just accuracy
    tp = sum(c["predict_gate_rule"] and c["degraded"] for c in done)
    fp = sum(c["predict_gate_rule"] and not c["degraded"] for c in done)
    fn = sum((not c["predict_gate_rule"]) and c["degraded"] for c in done)
    print(f"  gate rule confusion: TP {tp}  FP {fp}  FN {fn}  "
          f"(precision {tp/(tp+fp):.2f} at recall {tp/(tp+fn) if tp+fn else float('nan'):.2f})")

    if a.latex:
        out = ROOT / "paper_8/tables/prospective.tex"
        L = ["% GENERATED by scripts/score_prospective.py --latex -- do not edit by hand.",
             r"\begin{center}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             # gate_pre is the FROZEN predictor's score, computed with the pre-correction trend
             # denominator; gate_val is the corrected selection-split gate the body reports. Two
             # different quantities, so they get two different names: an earlier version printed the
             # frozen one under the body's own label, which reads as a contradiction of S4.
             r"Cell (new) & gate$_\text{pre}$ & gate$_\text{val}$ & forg$_\text{B}$ "
             r"& B$-$D held-out & Degradation? & Predicted \\", r"\midrule"]
        for c in done:
            pg = "gate\\,$\\times$" if c["predict_gate_rule"] != c["degraded"] else "gate\\,$\\checkmark$"
            pd_ = "data\\,$\\times$" if c["predict_dataset_rule"] != c["degraded"] else "data\\,$\\checkmark$"
            name = c["cell"].replace("_", " ").replace("h96", "($h{=}96$)").replace("h192", "($h{=}192$)")
            L.append(f"{name} & ${c['gate_val']:+.2f}$ & ${c['gate_corrected']:+.2f}$ "
                     f"& ${c['forg_b']:+.1f}$ "
                     f"({c['forg_b_pos']}/{c['seeds']}) & ${c['bd_test']:+.1f}{{\\pm}}{c['sem']:.1f}$ & "
                     f"{'\\textbf{yes}' if c['degraded'] else 'no'} & {pg}, {pd_} \\\\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{center}", ""]
        out.write_text("\n".join(L))
        print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
