#!/usr/bin/env python3
"""
Does the value gate's verdict survive changing the BASELINE ESTIMATOR?

WHY THIS EXISTS. The paper reports that on 27 of 32 screened cells the released checkpoint shows no
demonstrated advantage over a lookback-96 regression fit on the target data. The paper's own gate
appendix then shows the verdict barely moves when the 0.20 THRESHOLD is swept but moves for 17 of 21
Moirai cells when the BASELINE is swapped. So the threshold sensitivity analysis already in the paper
tests the parameter that does not matter, and the parameter that does matter has been varied exactly
once -- between the correct fitted map and the erroneous trend extrapolation it replaced. This script
varies it across a family.

WHAT IS HELD IDENTICAL, AND WHY THAT MATTERS. R2_task = 1 - MSE_zeroshot / MSE_baseline. Only the
DENOMINATOR depends on the baseline. The numerator, MSE_zeroshot, is a property of the checkpoint and
the window set, so it is read from results/gate_test_side.json and never recomputed -- which means
every column here differs from the published column in exactly one respect. Recomputing the numerator
would also require loading TimesFM, turning a CPU-only re-analysis into a GPU job for no gain.

THE SELF-CHECK IS THE POINT. Before reporting anything, the script recomputes the `fitted` baseline
through the same code path and compares it to the stored `linear_test`. If the window construction,
normalisation or split had drifted, that comparison fails and the run aborts. Without it, an
alternative-baseline column could differ from the published one because of a window off-by-one rather
than because of the estimator, and the table would silently be measuring the wrong thing.

Usage:  .venv12/bin/python scripts/gate_baseline_sensitivity.py [--arms moirai,chronos,timesfm,ili]
                                                               [--baselines seasonal_naive,ar,...]
                                                               [--latex]
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gate_all_cells as gac                                          # noqa: E402
import gate_baselines                                                 # noqa: E402
from cell_matrix import _display                                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STORED = ROOT / "results/gate_test_side.json"
OUT = ROOT / "results/gate_baselines.json"
# NOT gate_baselines.tex: the appendix already has a `tab:gate_baselines`, the two-cell table that
# recorded the trend-to-fitted correction. That one is a historical audit of one swap; this one is the
# estimator family across every cell. Sharing a name would eventually get one \input in place of the
# other.
TABLE = ROOT / "paper_8/tables/gate_baseline_family.tex"

BASELINES = ("fitted",) + gate_baselines.NAMES
SELFCHECK_RTOL = 1e-6

# The training-mean predictor is not one of the competing estimators; it is the floor each of them has
# to clear to be usable as a denominator at all (see gate_baselines.constant_mean). A column whose MSE
# exceeds it is inadmissible, and its R2_task is reported struck through rather than tallied as though
# it meant something.
FLOOR = "constant"


def stored():
    d = json.load(open(STORED))
    return d.get("cells", d)


# ---------------------------------------------------------------------------------------------
# Denominators per arm. Each returns {cell_key: (linear_mse, info)} or, for one arm, a third element.
# ---------------------------------------------------------------------------------------------
def _unpack(v):
    """Arm returns are (linear_mse, info) or (linear_mse, info, r2_task).

    The third element exists for one arm and one reason. The ILI cell aggregates R2_task as the mean
    over FEATURES of per-feature ratios, because finetune_ili averages percentages across features and
    gate_all_cells.ili_gate matches that deliberately. Its stored zs_test and linear_test are
    themselves feature means, so 1 - zs/lin computed on them gives -1.309 where the paper's ILI gate
    is -1.625. Recomputing the ratio here would have silently re-aggregated one row of this table
    under a convention no other number in the paper uses -- and the `fitted` self-check would not have
    caught it, because it checks the denominator and this is the aggregation.
    """
    return v[0], v[1], (v[2] if len(v) > 2 else None)



def moirai_denoms(baseline, refs):
    return {k: (v["linear_test"], v.get("baseline_info", {}))
            for k, v in gac.moirai_gates(refs, split="test", baseline=baseline).items()}


def chronos_denoms(baseline):
    return {k: (v["linear_test"], v.get("baseline_info", {}))
            for k, v in gac.chronos_gates(split="test", baseline=baseline).items()}


def ili_denoms(baseline):
    """The ILI cell, carrying gate_all_cells' own per-feature R2 aggregation through -- see _unpack."""
    return {k: (v["linear_test"], v.get("baseline_info", {}), v["r2_task"])
            for k, v in gac.ili_gate(split="test", baseline=baseline).items()}


def timesfm_denoms(baseline, lookback=96, horizon=24):
    """The TimesFM denominator only.

    gac.timesfm_gates() would LOAD TIMESFM to recompute its numerator, which this analysis does not
    need -- the numerator is stored. So the denominator is rebuilt here from the same helpers
    timesfm_gates uses, on the same window set (build_windows(test_s, ..., max_windows=200, seed=0),
    documented there as shared with the Chronos arm). The `fitted` self-check confirms the
    reconstruction matches the published number rather than merely resembling it.
    """
    from chronos_mse_finetune import build_windows, load_series

    out = {}
    for ds in gac.TIMESFM_DATASETS:
        train_s, _, test_s = load_series(ds)
        ctx, tgt = build_windows(test_s, lookback, horizon, max_windows=200, seed=0)
        tr_w = (build_windows(train_s, lookback, horizon)
                if baseline not in ("trend", "seasonal_naive") else None)
        info = {}
        lin = gac._window_linear_mse(ctx, tgt, lookback, horizon, train=tr_w,
                                     baseline=baseline, dataset=ds, info_out=info)
        out[f"timesfm_{ds.lower()}"] = (lin, info)
    return out


ARMS = {"moirai": moirai_denoms, "chronos": chronos_denoms,
        "timesfm": timesfm_denoms, "ili": ili_denoms}


def run(arms, baselines):
    cells = stored()
    refs = gac._zs_test_refs()
    results = {}                        # cell -> baseline -> dict
    for baseline in baselines:
        print(f"\n{'=' * 96}\nBASELINE: {baseline}\n{'=' * 96}")
        denoms = {}
        for arm in arms:
            fn = ARMS[arm]
            denoms.update(fn(baseline, refs) if arm == "moirai" else fn(baseline))
        for key, v in denoms.items():
            lin, info, r2_arm = _unpack(v)
            if key not in cells:
                print(f"  {key:24s} SKIPPED -- not in {STORED.name}")
                continue
            zs = cells[key]["zs_test"]
            r2 = 1 - zs / lin if r2_arm is None else r2_arm
            results.setdefault(key, {})[baseline] = dict(
                r2_task=r2, zs_test=zs, linear_test=lin, info=info,
                r2_from_arm=r2_arm is not None)
            if baseline == "fitted":
                # Two checks, not one. The denominator check catches drift in window construction,
                # normalisation or the split. The R2 check catches drift in the AGGREGATION -- the way
                # the ILI cell's per-feature mean differs from every other cell's ratio of means. The
                # first cannot see the second: ILI's denominator matched to 1e-6 while its R2_task was
                # off by 0.32 for exactly as long as this check was missing.
                pub, pub_r2 = cells[key]["linear_test"], cells[key].get("r2_task")
                ok = abs(lin - pub) <= SELFCHECK_RTOL * max(abs(pub), 1e-12)
                ok_r2 = (pub_r2 is None
                         or abs(r2 - pub_r2) <= SELFCHECK_RTOL * max(abs(pub_r2), 1e-12))
                results[key][baseline].update(selfcheck_published=pub, selfcheck_published_r2=pub_r2,
                                              selfcheck_ok=bool(ok and ok_r2))
                if not ok:
                    print(f"  {key:24s} SELF-CHECK FAIL  denominator {lin:.10f} "
                          f"vs published {pub:.10f}")
                if not ok_r2:
                    print(f"  {key:24s} SELF-CHECK FAIL  R2_task {r2:+.10f} "
                          f"vs published {pub_r2:+.10f}")
    return results


def selfcheck(results):
    bad = [k for k, v in results.items()
           if "fitted" in v and not v["fitted"].get("selfcheck_ok", True)]
    checked = [k for k, v in results.items() if "fitted" in v]
    print(f"\nSELF-CHECK: {len(checked) - len(bad)}/{len(checked)} recomputed `fitted` denominators "
          f"match the published gate_test_side.json to rtol={SELFCHECK_RTOL:g}")
    if bad:
        print("  MISMATCHED (window construction has drifted -- do NOT report the other columns):")
        for k in bad:
            print(f"    {k}: {results[k]['fitted']['linear_test']:.10f} vs "
                  f"{results[k]['fitted']['selfcheck_published']:.10f}")
    return not bad


PRETTY_DS = {"etth1": "ETTh1", "etth2": "ETTh2", "ettm2": "ETTm2", "weather": "Weather",
             "electricity": "Electricity", "electricity7": "Electricity7", "ili": "ILI"}


def display(key):
    """LaTeX row label for a gate cache key.

    cell_matrix._display expects its own cell strings ("Moirai-small/ETTh1 h96 n1000") and raises on
    the flat non-Moirai gate keys ("timesfm_etth1"), which have no horizon in them at all. Rather than
    loosening that function -- it is what labels four other tables -- the four arm shapes are named
    here, with the two screening horizons written out because they are not recoverable from the key.
    """
    arm, _, rest = key.partition("_")
    if arm in ("chronos", "timesfm"):
        name = "Chronos-T5-S" if arm == "chronos" else "TimesFM-2.5"
        return f"{name} / {PRETTY_DS.get(rest, rest)} ($h{{=}}24$)"
    if arm == "ili":                                   # the cache key is bare "ili", no arm prefix
        return r"Moirai-S / ILI ($h{=}24$)"
    return _display(key)


def admissible(results, key, baseline):
    """Is `baseline`'s denominator on this cell at least as good as the training-mean predictor?

    Returns None when the floor was not computed for the cell, so a missing floor is never silently
    read as a pass. `FLOOR` itself is admissible by definition -- it is the reference, not a candidate.
    """
    v, f = results[key].get(baseline), results[key].get(FLOOR)
    if v is None or f is None:
        return None
    if baseline == FLOOR:
        return True
    return v["linear_test"] <= f["linear_test"]


def passes(results, key, baseline, require_admissible=True):
    v = results[key].get(baseline)
    if v is None or v["r2_task"] < gac.GATE_THRESHOLD:
        return False
    return not (require_admissible and admissible(results, key, baseline) is False)


def degradation_counts(results, baselines, threshold=None):
    """Cells meeting the paper's degradation definition when clause~(i) uses each baseline.

    A table of gate columns is only half an answer. Clause (i) *is* the gate, so what the reviewer's
    question really asks is whether some other estimator would have admitted a degradation cell that
    the ridge map screened out -- and the possibility is concrete, not hypothetical: all eight
    degradation cells this paper retracted were ETTh1 or Weather, and ETTh1 is exactly where the
    seasonal-naive gate admits a cell the ridge map rejects. Clauses (ii) and (iii) are untouched
    here; only the inclusion decision moves. Inadmissible denominators are not allowed to admit a
    cell, for the same reason the trend baseline is not allowed to.
    """
    import contextlib
    import io

    import cell_matrix                                                # noqa: PLC0415
    threshold = gac.GATE_THRESHOLD if threshold is None else threshold
    with contextlib.redirect_stdout(io.StringIO()):
        rows = cell_matrix.build_rows()
    out = {}
    for b in baselines:
        hits, near = [], []
        for r in rows:
            g = (results.get(r["ref"], {}) or {}).get(b, {}).get("r2_task")
            if g is None or r.get("forg_b") is None or r.get("forg_confounded"):
                continue
            if admissible(results, r["ref"], b) is False:
                continue
            if not (g >= threshold and r["forg_b"] > 0 and r["forg_d"] < 0):
                continue
            unan = r["forg_b_pos"] == r["seeds"] and r["forg_d_neg"] == r["seeds"]
            (hits if unan else near).append(r["cell"])
        out[b] = dict(n=len(hits), cells=hits, near_misses=near,
                      n_scored=sum(1 for r in rows
                                   if r["ref"] in results and b in results[r["ref"]]))
    return out


def report(results, baselines):
    print(f"\n{'=' * 110}\nGATE R2_task BY BASELINE ESTIMATOR "
          f"(numerator identical throughout; only the denominator changes)\n{'=' * 110}")
    hdr = "cell".ljust(26) + "".join(b[:13].rjust(15) for b in baselines)
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(results, key=lambda k: -(results[k].get("fitted", {}).get("r2_task") or -9)):
        line = key.ljust(26)
        for b in baselines:
            v = results[key].get(b)
            if v is None:
                line += "".rjust(15)
                continue
            mark = "!" if admissible(results, key, b) is False else (
                "*" if v["r2_task"] >= gac.GATE_THRESHOLD else " ")
            line += f"{v['r2_task']:+.3f}{mark}".rjust(15)
        print(line)
    print("-" * len(hdr))
    for title, fn in (("cells clearing 0.20", lambda k, b: passes(results, k, b, False)),
                      ("  ... and admissible", lambda k, b: passes(results, k, b, True)),
                      ("denom worse than mean", lambda k, b: admissible(results, k, b) is False)):
        row = title.ljust(26)
        for b in baselines:
            tot = sum(1 for k in results if b in results[k])
            row += f"{sum(1 for k in results if fn(k, b))}/{tot}".rjust(15)
        print(row)
    print("\n  * = clears the 0.20 threshold.  ! = denominator is worse than the training-mean\n"
          "      predictor, so this cell's R2_task is not evidence of a pre-trained advantage.")
    deg = degradation_counts(results, baselines)
    print("\n  Cells meeting the paper's degradation definition, with clause (i) taken from each\n"
          "  baseline (clauses (ii) and (iii) unchanged; inadmissible denominators cannot admit):")
    for b in baselines:
        d = deg[b]
        print(f"    {b:16s} {d['n']} of {d['n_scored']} scored"
              + (f"   {', '.join(d['cells'])}" if d["cells"] else "")
              + (f"   [near-miss on seeds: {', '.join(d['near_misses'])}]"
                 if d["near_misses"] else ""))
    for b in baselines:
        infos = [(k, results[k][b]["info"]) for k in results
                 if b in results[k] and results[k][b]["info"]]
        if infos:
            print(f"\n  {b} disclosures:")
            for k, i in infos:
                print(f"    {k:24s} {i}")


def emit_latex(results, baselines, path=TABLE):
    label = {"fitted": r"Fitted ridge \\ (ours)", "constant": r"Training \\ mean",
             "seasonal_naive": r"Seasonal \\ naive", "ar": r"Per-feature \\ AR",
             "ridge_tuned": r"Tuned \\ ridge", "gbm": r"Boosted \\ trees"}
    cols = "l" + "c" * len(baselines)
    lines = [
        "% GENERATED by scripts/gate_baseline_sensitivity.py --latex -- do not edit by hand.",
        "\\begin{center}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}", "\\toprule",
        "Cell & " + " & ".join(
            f"\\makecell{{{label.get(b, b)}}}" for b in baselines) + " \\\\",
        "\\midrule",
    ]
    for key in sorted(results, key=lambda k: -(results[k].get("fitted", {}).get("r2_task") or -9)):
        cells = []
        for b in baselines:
            v = results[key].get(b)
            if v is None:
                cells.append("---")
                continue
            # \mathbf inside the math, not \textbf around it: \textbf{$x$} leaves the math upright and
            # unbolded, so the emphasis that marks a gate-pass would have been silently invisible. And
            # the dagger goes inside the same $...$ rather than in a second group, which would set two
            # math atoms side by side with a spurious italic correction between them. A dagger rather
            # than \st{} because striking out needs ulem, and no table is worth a new package.
            n = f"{v['r2_task']:+.3f}"
            if admissible(results, key, b) is False:
                cells.append(f"${n}^\\dagger$")
            elif v["r2_task"] >= gac.GATE_THRESHOLD:
                cells.append(f"$\\mathbf{{{n}}}$")
            else:
                cells.append(f"${n}$")
        lines.append(f"{display(key)} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    for title, fn in ((r"\emph{Clearing $0.20$}", lambda k, b: passes(results, k, b, False)),
                      (r"\emph{\quad and admissible}", lambda k, b: passes(results, k, b, True)),
                      (r"\emph{Below the floor}",
                       lambda k, b: admissible(results, k, b) is False)):
        tally = [f"{sum(1 for k in results if fn(k, b))}/"
                 f"{sum(1 for k in results if b in results[k])}" for b in baselines]
        lines.append(title + " & " + " & ".join(tally) + r" \\")
    deg = degradation_counts(results, baselines)
    lines.append(r"\emph{Degradation cells} & "
                 + " & ".join(f"{deg[b]['n']}/{deg[b]['n_scored']}" for b in baselines) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="moirai,chronos,timesfm,ili")
    ap.add_argument("--baselines", default=",".join(BASELINES))
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--from-json", action="store_true",
                    help="re-report and re-emit from results/gate_baselines.json without "
                         "recomputing any denominator. The boosted-tree column is a ten-minute "
                         "fit and changing a table caption or a tally is not a reason to pay it "
                         "again; the self-check verdict is read back from the file that produced it.")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    baselines = [x for x in a.baselines.split(",") if x]

    if a.from_json:
        prev = json.load(open(OUT))
        res, baselines, arms = prev["cells"], prev["baselines"], prev["arms"]
        ok = prev["selfcheck_passed"]
        print(f"re-reporting from {OUT.relative_to(ROOT)} "
              f"(self-check as recorded there: {'passed' if ok else 'FAILED'})")
    else:
        res = run(arms, baselines)
        ok = selfcheck(res)
    report(res, baselines)
    OUT.write_text(json.dumps(dict(baselines=baselines, arms=arms,
                                   gbm_max_rows=gate_baselines.GBM_MAX_ROWS,
                                   season_of=gate_baselines.SEASON_OF,
                                   floor=FLOOR, gate_threshold=gac.GATE_THRESHOLD,
                                   degradation=degradation_counts(res, baselines),
                                   selfcheck_passed=ok, cells=res), indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")
    if a.latex:
        if not ok:
            sys.exit("self-check failed; refusing to emit a table from denominators that do not "
                     "reproduce the published `fitted` column")
        emit_latex(res, baselines)
