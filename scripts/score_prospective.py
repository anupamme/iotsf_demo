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

BATCH 3 IS SCORED HERE TOO, AND THE TWO ARE NEVER POOLED. results/v57_prospective3 registers a
DIFFERENT predictor (the corrected ridge gate, not the trend-denominator one batch 1 froze) and a
DIFFERENT outcome (paired_inference.degrades_ci_ungated, an interval rule with the screen clause
removed, not three-clause unanimity with it included). A precision computed over the union of the two
batches would be a number about no single hypothesis, so the two sections below print their own
confusion matrices and no combined one is computed anywhere in this file.

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
import paired_inference as pi  # noqa: E402

PRE = ROOT / "results/v47_prospective/preregistration.json"

# --- batch 3 -----------------------------------------------------------------------------------
PRE3 = ROOT / "results/v57_prospective3/preregistration_v3.json"
B3_ROOT = "results/v57_prospective3"
# The batch-3 Moirai cells' condition-A records live in three directories; see new_moirai_cells'
# extra_a_dirs. Seed 42 predates the registration (disclosed in it), seeds 123 and 456 were run into
# the batch directory before any outcome existed.
B3_ZS_DIRS = ("results/v48_prospective2", "results/v56_pool3")
POOL = ROOT / "results/v56_pool3/pool_gate_val.json"
GATE_CACHE = ROOT / "results/gate_val_side.json"
# The batch-3 Moirai cells' independent recomputation. gate_all_cells.py scores them with the same
# moirai_gates() call the matrix uses, over the same condition_A records, but writes them here rather
# than into GATE_CACHE -- see the comment at its write site: that cache's LENGTH is a published
# denominator, so verification must not add keys to it.
GATE_CACHE_P3 = ROOT / "results/gate_val_side_prospective3.json"


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


def batch3_rows(pre3):
    """The batch-3 cells as cell_matrix rows, with the REGISTERED gate attached as their predictor.

    Both readers are the published ones: new_moirai_cells (the function that reads v43_moirai_matrix
    and batch 1) and timesfm_cells (the function that reads the published h=24 TimesFM cells), called
    with this batch's directory. Nothing about a batch-3 cell is computed by code written for batch 3,
    which is the property that makes "the gate was tested prospectively" checkable rather than
    asserted.

    The gate attached is the value FROZEN in the registration, not a recomputation: recomputing the
    predictor after the outcomes exist is the failure batch 1 is an example of. check_gate_agreement()
    below separately verifies that the paper's own cache reproduces those frozen values once the cells
    are added to it.
    """
    rows = (cell_matrix.new_moirai_cells(dirs=(B3_ROOT,), extra_a_dirs=B3_ZS_DIRS)
            + cell_matrix.timesfm_cells(root=B3_ROOT, horizon=48, ref_suffix="_h48"))
    gate = {c["cell"]: c["gate_val"] for c in pre3["cells"]}
    rows = [r for r in rows if r["ref"] in gate]          # nothing unregistered can enter the batch
    for r in rows:
        r["gate"] = gate[r["ref"]]
    return rows


def check_gate_agreement(pre3):
    """The paper's gate cache must reproduce the registered gate values exactly, once it has them.

    The pool screen wrote results/v56_pool3/pool_gate_val.json by calling the paper's own gate
    functions with the pool substituted for the cell list, so the two must agree to the bit. An absent
    key is not agreement, so a cell the cache does not carry is reported rather than passed over.

    TWO FILES, ONE CHECK. The Moirai cells are recomputed into GATE_CACHE_P3 by
    gate_all_cells.MOIRAI_CELLS_PROSPECTIVE3 -- a genuine second pass over the condition_A records
    through the paper's own estimator, which is what makes this an audit rather than a tautology. The
    five TimesFM h=48 cells are NOT recomputed anywhere: their registered numerator was measured by
    loading the checkpoint, and the only other route to it (each run's stored zeroshot_mse at the
    seed-42 windows) reproduces it to ~6e-7 relative -- real confirmation, but ~3e-8 in R2_task, which
    would trip the 1e-9 tolerance below over float32 batching noise and read as a voided claim. So
    they stay absent by design and are reported as such; gate_all_cells.timesfm_gates records why.
    """
    if not GATE_CACHE.exists():
        return ["no results/gate_val_side.json; cannot check the registered gates against the cache"]
    cache = {k: v["r2_task"] for k, v in json.loads(GATE_CACHE.read_text()).items()}
    if GATE_CACHE_P3.exists():
        for k, v in json.loads(GATE_CACHE_P3.read_text()).items():
            assert k not in cache, (
                f"{k} is in both gate caches; the prospective recomputation must not shadow or "
                f"duplicate a retrospective cell")
            cache[k] = v["r2_task"]
    pool = {k: v["r2_task"] for k, v in json.loads(POOL.read_text()).items()}
    msgs = []
    for c in pre3["cells"]:
        ref, g = c["cell"], c["gate_val"]
        assert abs(pool[ref] - g) < 1e-12, (
            f"{ref}: the registration says {g} but the pool file it was read from says {pool[ref]}")
        if ref not in cache:
            msgs.append(f"  {ref:26s} not in the paper's cache yet (cells enter it at integration)")
        elif abs(cache[ref] - g) > 1e-9:
            raise AssertionError(
                f"{ref}: the paper's gate cache says {cache[ref]:+.6f} but the registered predictor "
                f"is {g:+.6f}. One of the two was recomputed with different inputs; the prospective "
                f"claim is void until this is explained.")
        else:
            msgs.append(f"  {ref:26s} cache reproduces the registered gate ({g:+.3f})")
    return msgs


def _decisive(s):
    """Does this contrast's ZS-propagated interval exclude zero?"""
    if not s:
        return False
    lo, hi = s.get("lo_zs", s["lo"]), s.get("hi_zs", s["hi"])
    return bool(lo == lo and hi == hi and (lo > 0 or hi < 0))


def load3():
    """(registration, scored cells, batch-3 paired-inference records). Decides nothing."""
    pre3 = json.loads(PRE3.read_text())
    rows = batch3_rows(pre3)
    # BH across the batch-3 cells ONLY, as registered: pooling them into the published 31's family
    # would change the published cells' q values, and a prospective test may not move its own
    # comparison set.
    recs = {}
    if rows:
        # A cell can only widen its B-ZS/D-ZS interval if its zero-shot reference is unpaired AND ran
        # more than one seed. That is true of the five Moirai cells and of none of the TimesFM cells,
        # whose reference is per-seed, so the global "something must widen" check is required only once
        # a Moirai cell has finished -- otherwise scoring the batch mid-flight would abort on an
        # invariant that is vacuous rather than violated.
        can_widen = any(not r["per_seed"]["ft_paired"] and (r["per_seed"]["zs_sem"] or 0) > 0
                        for r in rows)
        with contextlib.redirect_stdout(io.StringIO()):
            cells, _ = pi.analyse(rows, require_widening=can_widen)
        recs = {c["ref"]: c for c in cells}
    out = []
    for c in pre3["cells"]:
        r = recs.get(c["cell"])
        if r is None:
            out.append(dict(c, status="pending")); continue
        out.append(dict(c, status="scored", seeds_run=r["seeds"],
                        degraded=r["degrades_ci_ungated"],
                        degraded_published=r["degrades_unanimous"],
                        decisive=_decisive(r.get("d_ft")) and _decisive(r.get("d_frozen")),
                        call=r["call"], d_enc=r["d_enc"], d_ft=r["d_ft"], d_frozen=r["d_frozen"],
                        zs_seeds=r["zs_seeds"], zs_sem=r["zs_sem"], ft_paired=r["ft_paired"]))
    return pre3, out, recs


def matrix(done, rule_key, outcome_key="degraded"):
    tp = sum(c[rule_key] and c[outcome_key] for c in done)
    fp = sum(c[rule_key] and not c[outcome_key] for c in done)
    fn = sum((not c[rule_key]) and c[outcome_key] for c in done)
    tn = sum((not c[rule_key]) and not c[outcome_key] for c in done)
    return tp, fp, fn, tn


def _rate(num, den):
    return f"{num/den:.2f}" if den else "n/a (0 cells in this class)"


def report3():
    pre3, cells, _ = load3()
    done = [c for c in cells if c["status"] == "scored"]
    print()
    print("=" * 100)
    print("BATCH 3 -- the CORRECTED ridge gate, registered before any B/D run of these cells existed")
    print(f"registered {pre3['written_utc']} at {pre3['git_head'][:12]}")
    print("NOT POOLED with batch 1: different predictor, different outcome rule.")
    print("=" * 100)
    for m in check_gate_agreement(pre3):
        print(m)
    print(f"\n{'cell':26s} {'gate':>7s} {'pred':>5s} {'n':>2s} {'B-ZS (95% CI)':>22s} "
          f"{'D-ZS (95% CI)':>22s} {'B-D':>16s} {'deg?':>5s} {'call':>12s}")
    for c in cells:
        if c["status"] == "pending":
            print(f"{c['cell']:26s} {c['gate_val']:+7.3f} {'RISK' if c['predict_gate_rule'] else '-':>5s}"
                  f"   ...no complete B/D pair yet")
            continue
        ft, fz, en = c["d_ft"], c["d_frozen"], c["d_enc"]
        def iv(s):
            lo, hi = s.get("lo_zs", s["lo"]), s.get("hi_zs", s["hi"])
            return f"{s['mean']:+7.2f} [{lo:+6.1f},{hi:+6.1f}]"
        print(f"{c['cell']:26s} {c['gate_val']:+7.3f} {'RISK' if c['predict_gate_rule'] else '-':>5s} "
              f"{c['seeds_run']:2d} {iv(ft):>22s} {iv(fz):>22s} "
              f"{en['mean']:+7.2f} q={en['q']:.3f} {'YES' if c['degraded'] else 'no':>5s} "
              f"{c['call']:>12s}")

    print(f"\n  scored {len(done)}/{len(cells)} cells; "
          f"{sum(c['degraded'] for c in done)} degrade under the registered ungated interval rule, "
          f"{sum(c['degraded_published'] for c in done)} under the published unanimity definition")
    dec = [c for c in done if c["decisive"]]
    print(f"  both intervals decisive on {len(dec)}/{len(done)} cells; the primary matrix counts an "
          f"inconclusive cell as NOT degrading, as registered")

    for rule, label in (("predict_gate_rule", "gate rule (R2_task >= 0.20)"),
                        ("predict_dataset_rule", "dataset rule (ETTh1/Weather)")):
        tp, fp, fn, tn = matrix(done, rule)
        print(f"\n  {label}")
        print(f"    TP {tp}  FP {fp}  FN {fn}  TN {tn}")
        print(f"    precision   {_rate(tp, tp+fp)}     specificity {_rate(tn, tn+fp)}")
        print(f"    sensitivity {_rate(tp, tp+fn)}"
              + ("   -- UNESTIMABLE: no cell degraded, so the positive class is empty"
                 if tp + fn == 0 else ""))
        if dec:
            tp2, fp2, fn2, tn2 = matrix(dec, rule)
            print(f"    decisive-only (n={len(dec)}): TP {tp2}  FP {fp2}  FN {fn2}  TN {tn2}")

    # Registered in advance and therefore printed whether or not it is flattering: the secondary
    # outcome is the three-way call on the encoder contrast, BH-adjusted within this batch.
    import collections
    k = collections.Counter(c["call"] for c in done)
    print(f"\n  secondary (Delta_enc, BH within batch 3): freeze-decisive {k['freeze']}  "
          f"adapt-decisive {k['adapt']}  inconclusive {k['inconclusive']}")
    for c in done:
        if c["call"] == "freeze":
            print(f"    FREEZE decisive: {c['cell']} gate {c['gate_val']:+.3f} "
                  f"({'PASS' if c['predict_gate_rule'] else 'gate-FAILING'})")

    # Two disclosures that belong beside the numbers rather than in a source comment.
    one_seed = [c for c in done if not c["ft_paired"] and (c["zs_seeds"] or 0) < 2]
    if one_seed:
        print(f"\n  WARNING: the zero-shot reference ran one seed on {len(one_seed)} cell(s) "
              f"({', '.join(c['cell'] for c in one_seed)}), so its error is unestimated and their "
              f"B-ZS/D-ZS intervals are NOT widened -- which makes degradation easier to declare "
              f"there than on the other cells")
    print("  disclosure: the five TimesFM cells' registered gate was computed on the seed-42 "
          "selection-window draw, while gate_all_cells.timesfm_gates(split='val') averages the draws "
          "of the three run seeds. The frozen value stands; the multi-seed recomputation is a "
          "sensitivity, not the predictor.")
    return pre3, cells, done


def emit3_latex(cells, path=ROOT / "paper_8/tables/prospective3.tex"):
    done = [c for c in cells if c["status"] == "scored"]
    # \scriptsize and 3pt, not \footnotesize and 4pt: eight columns carrying two intervals each ran
    # 57.5pt over \linewidth at footnotesize, and one TimesFM cell's B-ZS interval is
    # [-345.3, +543.7], so the width is set by real data rather than by a label that could be shortened.
    L = ["% GENERATED by scripts/score_prospective.py --latex -- do not edit by hand.",
         r"\begin{center}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}lccrrrcc@{}}", r"\toprule",
         r"Cell (batch 3) & gate$_\text{val}$ & pred. & $n$ & $\Delta_\text{FT}$ "
         r"& $\Delta_\text{frozen}$ & Degr.? & $\Delta_\text{enc}$ call \\", r"\midrule"]
    # The dataset token arrives lower-cased on the TimesFM refs and capitalised on the Moirai ones,
    # so casing is restored from a map rather than by .title(), which would render ETTh1 as "Etth1".
    DS = {"etth1": "ETTh1", "etth2": "ETTh2", "ettm2": "ETTm2",
          "weather": "Weather", "electricity": "Electricity"}
    for c in done:
        name = c["cell"].replace("_", " ")
        if name.startswith("timesfm "):
            ds, h = name.split()[1], name.split()[2]
            name = f"TimesFM-2.5 / {DS.get(ds, ds)} {h}"
        for _h in ("h48", "h96", "h192"):
            name = name.replace(_h, f"($h{{=}}{_h[1:]}$)")
        def iv(s):
            lo, hi = s.get("lo_zs", s["lo"]), s.get("hi_zs", s["hi"])
            return f"${s['mean']:+.1f}$ [{lo:+.1f}, {hi:+.1f}]"
        L.append(f"{name} & ${c['gate_val']:+.2f}$ & "
                 f"{'risk' if c['predict_gate_rule'] else '--'} & {c['seeds_run']} & "
                 f"{iv(c['d_ft'])} & {iv(c['d_frozen'])} & "
                 f"{'\\textbf{yes}' if c['degraded'] else 'no'} & {c['call']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{center}", ""]
    path.write_text("\n".join(L))
    return path


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

    if PRE3.exists():
        _, cells3, done3 = report3()
        if a.latex and done3:
            print(f"\nwrote {emit3_latex(cells3).relative_to(ROOT)}")
        elif a.latex:
            print("\nno batch-3 cell has a complete B/D pair yet; prospective3.tex not written")


if __name__ == "__main__":
    main()
