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

WHICH WINDOWS. `--split test` reproduces the published analysis: the gate's numerator and denominator
are both scored on the same held-out windows the B-D outcome is scored on, which makes cell INCLUSION a
retrospective judgement. `--split val` scores the gate on the selection windows instead -- disjoint from
the outcome by construction (forecasting_loader.get_splits slices one frame into non-overlapping
chronological ranges, and windows are built inside each range only). The val side needs no GPU on any
arm: every numerator is already stored (`zeroshot_mse` / `zs_mse`), so only the denominators are
recomputed, and gate_all_cells.timesfm_gates(split="val") returns without loading TimesFM at all.
Each split writes its own JSON and its own table; neither overwrites the other.

Usage:  .venv12/bin/python scripts/gate_baseline_sensitivity.py [--split test|val]
                                                               [--arms moirai,chronos,timesfm,ili]
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
# Per split: the published gate to self-check against, where the recomputation lands, and the table.
# Keyed rather than reassigned so a val-side run can never silently overwrite the test-side record the
# paper's existing appendix table is emitted from.
# NOT gate_baselines.tex: the appendix already has a `tab:gate_baselines`, the two-cell table that
# recorded the trend-to-fitted correction. That one is a historical audit of one swap; this one is the
# estimator family across every cell. Sharing a name would eventually get one \input in place of the
# other.
STORED_BY_SPLIT = {"test": ROOT / "results/gate_test_side.json",
                   "val": ROOT / "results/gate_val_side.json"}
OUT_BY_SPLIT = {"test": ROOT / "results/gate_baselines.json",
                "val": ROOT / "results/gate_baselines_val.json"}
TABLE_BY_SPLIT = {"test": ROOT / "paper_8/tables/gate_baseline_family.tex",
                  "val": ROOT / "paper_8/tables/gate_baseline_family_val.tex"}

BASELINES = ("fitted",) + gate_baselines.NAMES
SELFCHECK_RTOL = 1e-6

# The training-mean predictor is not one of the competing estimators; it is the floor each of them has
# to clear to be usable as a denominator at all (see gate_baselines.constant_mean). A column whose MSE
# exceeds it is inadmissible, and its R2_task is reported struck through rather than tallied as though
# it meant something.
FLOOR = "constant"


def stored(split):
    d = json.load(open(STORED_BY_SPLIT[split]))
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



def moirai_denoms(baseline, split, refs):
    return {k: (v["linear_test"], v.get("baseline_info", {}))
            for k, v in gac.moirai_gates(refs, split=split, baseline=baseline).items()}


def chronos_denoms(baseline, split):
    return {k: (v["linear_test"], v.get("baseline_info", {}))
            for k, v in gac.chronos_gates(split=split, baseline=baseline).items()}


def ili_denoms(baseline, split):
    """The ILI cell, carrying gate_all_cells' own per-feature R2 aggregation through -- see _unpack."""
    return {k: (v["linear_test"], v.get("baseline_info", {}), v["r2_task"])
            for k, v in gac.ili_gate(split=split, baseline=baseline).items()}


def timesfm_denoms(baseline, split, lookback=96, horizon=24):
    """The TimesFM denominator only.

    On the TEST side, gac.timesfm_gates() would LOAD TIMESFM to recompute its numerator, which this
    analysis does not need -- the numerator is stored. So the denominator is rebuilt here from the same
    helpers timesfm_gates uses, on the same window set (build_windows(test_s, ..., max_windows=200,
    seed=0), documented there as shared with the Chronos arm). The `fitted` self-check confirms the
    reconstruction matches the published number rather than merely resembling it.

    On the VAL side there is nothing to reconstruct: timesfm_gates(split="val") reads the stored
    `zeroshot_mse` and never touches the model, so it is called directly. Duplicating its per-seed
    window averaging here would be a second implementation of the aggregation with no self-check
    covering it -- the ILI cell already showed that a denominator can match to 1e-6 while the
    aggregation around it is wrong.
    """
    if split == "val":
        return {k: (v["linear_test"], v.get("baseline_info", {}))
                for k, v in gac.timesfm_gates(split="val", baseline=baseline,
                                              lookback=lookback, horizon=horizon).items()}

    from chronos_mse_finetune import build_windows, load_series

    out = {}
    for ds in gac.TIMESFM_DATASETS:
        train_s, _, test_s = load_series(ds)
        ctx, tgt = build_windows(test_s, lookback, horizon, max_windows=200, seed=0)
        tr_w = (build_windows(train_s, lookback, horizon)
                if baseline not in gate_baselines.NO_TRAINING else None)
        info = {}
        lin = gac._window_linear_mse(ctx, tgt, lookback, horizon, train=tr_w,
                                     baseline=baseline, dataset=ds, info_out=info)
        out[f"timesfm_{ds.lower()}"] = (lin, info)
    return out


ARMS = {"moirai": moirai_denoms, "chronos": chronos_denoms,
        "timesfm": timesfm_denoms, "ili": ili_denoms}


def run(arms, baselines, split="test"):
    cells = stored(split)
    refs = gac._zs_val_refs() if split == "val" else gac._zs_test_refs()
    results = {}                        # cell -> baseline -> dict
    for baseline in baselines:
        print(f"\n{'=' * 96}\nBASELINE: {baseline}\n{'=' * 96}")
        denoms = {}
        for arm in arms:
            fn = ARMS[arm]
            denoms.update(fn(baseline, split, refs) if arm == "moirai" else fn(baseline, split))
        for key, v in denoms.items():
            lin, info, r2_arm = _unpack(v)
            if key not in cells:
                print(f"  {key:24s} SKIPPED -- not in {STORED_BY_SPLIT[split].name}")
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


def selfcheck(results, split="test"):
    bad = [k for k, v in results.items()
           if "fitted" in v and not v["fitted"].get("selfcheck_ok", True)]
    checked = [k for k, v in results.items() if "fitted" in v]
    print(f"\nSELF-CHECK: {len(checked) - len(bad)}/{len(checked)} recomputed `fitted` denominators "
          f"match the published {STORED_BY_SPLIT[split].name} to rtol={SELFCHECK_RTOL:g}")
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


def graded_value(results, key, baselines=None):
    """The cell's pre-trained value as a GRADED quantity rather than a pass/fail verdict.

    WHY GRADED. The gate as published is a binary test against one estimator, and a reviewer's fair
    objection is that this throws away the only interesting axis: cells do not divide into "has value"
    and "has none", they differ in how far up the ladder their advantage survives. Reporting the
    strongest rival each checkpoint still beats turns the screen into a continuous covariate, which is
    what the drift-versus-outcome question actually needs -- see fig_value_axis.py.

    WHICH DIRECTION IS "BEST", because it is the opposite of the intuitive reading. R2_task =
    1 - MSE_zs/MSE_base, so a WEAKER rival (larger MSE_base) yields a LARGER R2_task. The strongest
    admissible rival is therefore the one giving the SMALLEST R2_task, and `r2_best` is a MINIMUM over
    admissible rungs. Getting this backwards would report each cell's most flattering column and call
    it the strongest baseline -- and since the ladder now includes `persistence`, the most flattering
    column is often the one nothing should be measured against.

    Consequences of that, both worth stating in the paper:
      * `r2_best >= 0.20` means the advantage survives EVERY admissible rung. This is the strong
        claim, and it is monotone in ladder length: adding a rung can only remove cells from it.
      * `r2_any >= 0.20` means it survives at least one, i.e. the weakest. Adding rungs can only add
        cells here, so a large count is not evidence -- which is why it is reported beside the other
        and never alone.

    Inadmissible rungs (denominator worse than the training mean) are excluded from both, because a
    rung that loses to a constant cannot establish anything about a checkpoint either way.
    """
    baselines = list(results[key]) if baselines is None else [b for b in baselines
                                                             if b in results[key]]
    adm = [b for b in baselines if b != FLOOR and admissible(results, key, b) is not False]
    if not adm:
        return None
    r2 = {b: results[key][b]["r2_task"] for b in adm}
    best = min(r2, key=r2.get)                      # strongest rival == least flattering column
    weakest = max(r2, key=r2.get)
    return dict(r2_best=r2[best], best_baseline=best,
                r2_any=r2[weakest], weakest_baseline=weakest,
                n_admissible=len(adm), admissible=adm,
                inadmissible=[b for b in baselines
                              if b != FLOOR and admissible(results, key, b) is False],
                clears_all_admissible=r2[best] >= gac.GATE_THRESHOLD,
                clears_any_admissible=r2[weakest] >= gac.GATE_THRESHOLD)


def graded_table(results, baselines):
    """{cell: graded_value(...)} for every scored cell, plus the two headline tallies."""
    per = {k: graded_value(results, k, baselines) for k in results}
    per = {k: v for k, v in per.items() if v}
    return dict(cells=per,
                n_scored=len(per),
                n_clearing_all_admissible=sum(1 for v in per.values()
                                              if v["clears_all_admissible"]),
                n_clearing_any_admissible=sum(1 for v in per.values()
                                              if v["clears_any_admissible"]),
                threshold=gac.GATE_THRESHOLD)


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
    g = graded_table(results, baselines)
    print(f"\n{'=' * 110}\nTHE LADDER AS A GRADED AXIS: value measured against the STRONGEST "
          f"admissible rival\n{'=' * 110}")
    print("  R2_task(best) is a MINIMUM over admissible rungs, because a weaker rival inflates\n"
          "  R2_task. It is the honest per-cell value score, and it is what fig_value_axis.py uses\n"
          "  as its x-axis. `any` is the maximum, i.e. the most flattering rung, printed only so the\n"
          "  gap between the two is visible.")
    col = "cell".ljust(26) + "R2(best)".rjust(10) + "  strongest rival".ljust(20) \
        + "R2(any)".rjust(9) + "  weakest rival".ljust(20) + "adm".rjust(5)
    print(f"\n{col}\n{'-' * len(col)}")
    for key in sorted(g["cells"], key=lambda k: -g["cells"][k]["r2_best"]):
        v = g["cells"][key]
        print(key.ljust(26) + f"{v['r2_best']:+.3f}".rjust(10)
              + f"  {v['best_baseline']}".ljust(20) + f"{v['r2_any']:+.3f}".rjust(9)
              + f"  {v['weakest_baseline']}".ljust(20)
              + f"{v['n_admissible']}".rjust(5)
              + ("   <- clears against ALL admissible" if v["clears_all_admissible"] else ""))
    print(f"{'-' * len(col)}")
    print(f"  clears {gac.GATE_THRESHOLD} against ALL admissible rungs: "
          f"{g['n_clearing_all_admissible']}/{g['n_scored']}   "
          f"(monotone DOWN as rungs are added -- the conservative headline)")
    print(f"  clears {gac.GATE_THRESHOLD} against AT LEAST ONE:         "
          f"{g['n_clearing_any_admissible']}/{g['n_scored']}   "
          f"(monotone UP as rungs are added -- not evidence on its own)")
    for b in baselines:
        infos = [(k, results[k][b]["info"]) for k in results
                 if b in results[k] and results[k][b]["info"]]
        if infos:
            print(f"\n  {b} disclosures:")
            for k, i in infos:
                print(f"    {k:24s} {i}")


def emit_latex(results, baselines, path):
    label = {"fitted": r"Fitted ridge \\ (ours)", "constant": r"Training \\ mean",
             "persistence": r"Persist- \\ ence",
             "seasonal_naive": r"Seasonal \\ naive", "ar": r"Per-feature \\ AR",
             "dlinear": r"DLinear \\ (shared)",
             "ridge_tuned": r"Tuned \\ ridge", "mlp": r"MLP \\ (128)",
             "gbm": r"Boosted \\ trees"}
    missing = [b for b in baselines if b not in label]
    if missing:
        # A rung with no label would silently print its bare snake_case key as a column header.
        sys.exit(f"no column label for baseline(s) {missing}; add them to `label` in emit_latex")
    cols = "l" + "c" * len(baselines)
    lines = [
        "% GENERATED by scripts/gate_baseline_sensitivity.py --latex -- do not edit by hand.",
        "\\begin{center}", "\\footnotesize", "\\setlength{\\tabcolsep}{2pt}",
        # \resizebox rather than a hand-picked font size: the ladder has grown from five rungs to
        # eight once already, and every rung adds a column. At footnotesize/3pt the eight-rung version
        # ran 85pt past \textwidth, which LaTeX reports as an Overfull hbox and then prints anyway --
        # i.e. into the margin. Scaling to \linewidth keeps the table legal whatever the rung count,
        # at the cost of an effective font size smaller than the surrounding footnotesize.
        "\\resizebox{\\linewidth}{!}{%",
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
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{center}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="test = the gate as published, scored on the same windows as the outcome; "
                         "val = scored on the selection windows, disjoint from the outcome")
    ap.add_argument("--arms", default="moirai,chronos,timesfm,ili")
    ap.add_argument("--baselines", default=",".join(BASELINES))
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--from-json", action="store_true",
                    help="re-report and re-emit from this split's results/gate_baselines*.json "
                         "without recomputing any denominator. The boosted-tree column is a ten-minute "
                         "fit and changing a table caption or a tally is not a reason to pay it "
                         "again; the self-check verdict is read back from the file that produced it.")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    baselines = [x for x in a.baselines.split(",") if x]
    OUT, TABLE = OUT_BY_SPLIT[a.split], TABLE_BY_SPLIT[a.split]
    print(f"gate split: {a.split}   (numerator from {STORED_BY_SPLIT[a.split].name}, "
          f"denominators recomputed)")

    if a.from_json:
        prev = json.load(open(OUT))
        res, baselines, arms = prev["cells"], prev["baselines"], prev["arms"]
        ok = prev["selfcheck_passed"]
        print(f"re-reporting from {OUT.relative_to(ROOT)} "
              f"(self-check as recorded there: {'passed' if ok else 'FAILED'})")
    else:
        res = run(arms, baselines, a.split)
        ok = selfcheck(res, a.split)
    report(res, baselines)
    OUT.write_text(json.dumps(dict(split=a.split, baselines=baselines, arms=arms,
                                   ladder_order=list(gate_baselines.NAMES),
                                   channel_independent=list(gate_baselines.CHANNEL_INDEPENDENT),
                                   no_training=list(gate_baselines.NO_TRAINING),
                                   gbm_max_rows=gate_baselines.GBM_MAX_ROWS,
                                   mlp_hidden=list(gate_baselines.MLP_HIDDEN),
                                   mlp_max_iter=gate_baselines.MLP_MAX_ITER,
                                   mlp_alpha_grid=list(gate_baselines.MLP_ALPHA_GRID),
                                   mlp_max_windows=gate_baselines.MLP_MAX_WINDOWS,
                                   dlinear_kernel=gate_baselines.DLINEAR_KERNEL,
                                   season_of=gate_baselines.SEASON_OF,
                                   floor=FLOOR, gate_threshold=gac.GATE_THRESHOLD,
                                   degradation=degradation_counts(res, baselines),
                                   # The graded axis, stored so fig_value_axis.py and
                                   # cka_fixed_effects.py read the same per-cell value score the
                                   # console printed rather than each deriving its own from `cells`.
                                   graded=graded_table(res, baselines),
                                   selfcheck_passed=ok, cells=res), indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")
    if a.latex:
        if not ok:
            sys.exit("self-check failed; refusing to emit a table from denominators that do not "
                     "reproduce the published `fitted` column")
        emit_latex(res, baselines, TABLE)
