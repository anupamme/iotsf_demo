#!/usr/bin/env python3
"""
Decompose each val->test sign reversal into its two arms, to bound what the held-out re-scoring
can and cannot establish.

The paper names the reversals `selection-window optimism'. A reviewer objected -- correctly -- that
naming a direction is not the same as identifying a mechanism, and that the obvious competitor is a
validation-to-test distribution shift that happens to hurt the frozen arm. This script answers that
from the run records already on disk. No new compute, and nothing here is re-measured: every reader
below is imported from cell_matrix, so a reversal is scored exactly as its published row is.

For each cell and seed, each arm's error is normalized by the zero-shot error MEASURED ON THAT ARM'S
OWN SPLIT, so whatever both arms share divides out:

    relB_val = B.val / zs_val      relB_test = B.test / zs_test      dB = relB_test - relB_val
    relD_val = D.val / zs_val      relD_test = D.test / zs_test      dD = relD_test - relD_val

which is an identity, not a model:  (B-D)_test - (B-D)_val == dB - dD. That identity is asserted per
cell, and (B-D)_val / (B-D)_test are asserted against cell_matrix.build_rows(), so this file cannot
disagree with the tables.

Two things come out, and both LIMIT the paper's claim rather than extend it:

  1. The four reversals do not share one per-arm story. Note first what is NOT evidence: dD > dB holds
     in all four by construction, since a reversal from positive to negative IS dB - dD < 0. What is
     not entailed is the SIGN. On Chronos/ETTh1 and Moirai-B/ETTh2 both arms come out worse than their
     own zero-shot on test (dB, dD > 0) and the frozen arm collapses -- the selection story. On
     Moirai-L/ETTh2 and TimesFM/ETTh2 both arms come out BETTER (dB, dD < 0): nothing degraded, the
     adapted arm simply gained more out of sample. Half the reversals are therefore not the frozen arm
     failing at all, which is a different phenomenon wearing the same sign change.
  2. zs_test/zs_val -- the ratio for the UNTRAINED model, which fits nothing and selects nothing --
     is far from 1 and wildly heterogeneous across the matrix. Where an untrained model's own error
     moves by several-fold between two window sets, a selection bias and a distribution shift are not
     separable in principle, let alone from four reversals.

Scope: 30 of the 31 intervention cells decompose. ILI is out because finetune_ili.py stores aggregate
percentages rather than the per-split MSEs this needs; its published B-D is unaffected.

CAVEAT, and it is why the paper says `window sets' and not `distribution shift': this ratio mixes
genuine shift with window-set construction. The two sets differ in count and in how they are built,
and for the legacy Moirai cells zs_test is a dataset-level average from _zs_test_refs() while zs_val
is per-run. It bounds exchangeability; it does not measure a shift.
"""
import contextlib
import glob
import io
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import cell_matrix as cm  # noqa: E402

TOL = 0.05  # pp; guards against a reader drifting from cell_matrix, not against float noise


def _mean_ratios(acc):
    """acc: per-seed (relB_val, relD_val, relB_test, relD_test, zs_test/zs_val) -> the five means."""
    return [st.mean(x[i] for x in acc) for i in range(5)]


def _row(cell, acc):
    m = _mean_ratios(acc)
    return dict(cell=cell, seeds=len(acc),
                bd_val=(m[0] - m[1]) * 100, bd_test=(m[2] - m[3]) * 100,
                d_b=(m[2] - m[0]) * 100, d_d=(m[3] - m[1]) * 100, zs_ratio=m[4])


def legacy_moirai():
    """Moirai cells read by cell_matrix.moirai_cells(), whose zs_test is a dataset-level average."""
    refs, out = cm._zs_test_refs(), []
    for (rel, size, h, n), seeds in sorted(cm.moirai_cells().items(), key=str):
        ds = cm.DATASET_OF.get(rel)
        paired = {s: v for s, v in seeds.items() if {"B", "D"} <= set(v)}
        zs_t = refs.get(f"{size}_{ds}_h{h}")
        if not (ds and paired and zs_t):
            continue
        acc = [(v["B"]["final_val_mse"] / v["B"]["zeroshot_mse"],
                v["D"]["final_val_mse"] / v["B"]["zeroshot_mse"],
                v["B"]["test_mse"] / zs_t, v["D"]["test_mse"] / zs_t,
                zs_t / v["B"]["zeroshot_mse"]) for v in paired.values()]
        out.append(_row(f"Moirai-{size}/{ds} h{h} n{n}", acc))
    return out


def new_moirai():
    """The v43/v47 layout, which carries its own condition A."""
    out = []
    for d in sorted(glob.glob(str(ROOT / "results/v43_moirai_matrix/*"))
                    + glob.glob(str(ROOT / "results/v47_prospective/*"))):
        if not Path(d).is_dir():
            continue
        A = [json.load(open(f)) for f in glob.glob(d + "/condition_A/*.json")]
        A = [x for x in A if "zeroshot_test_mse" in x]
        B = {json.load(open(f))["seed"]: json.load(open(f))
             for f in glob.glob(d + "/condition_B/*.json")}
        D = {json.load(open(f))["seed"]: json.load(open(f))
             for f in glob.glob(d + "/condition_D/*.json")}
        seeds = sorted(set(B) & set(D))
        if not (A and seeds):
            continue
        zs_t = st.mean(x["zeroshot_test_mse"] for x in A)
        acc = [(B[s]["final_val_mse"] / B[s]["zeroshot_mse"],
                D[s]["final_val_mse"] / B[s]["zeroshot_mse"],
                B[s]["test_mse"] / zs_t, D[s]["test_mse"] / zs_t,
                zs_t / B[s]["zeroshot_mse"]) for s in seeds]
        out.append(_row(f"Moirai-{Path(d).name}", acc))
    return out


def chronos(root="results/v44_chronos_guarded", horizon=24):
    """Chronos. The condition-D estimator is whichever one validation preferred, as in cell_matrix."""
    out = []
    for ds in ("etth1", "etth2", "weather", "ettm2", "electricity"):
        B = {json.load(open(f))["seed"]: json.load(open(f)) for f in
             glob.glob(str(ROOT / root / f"cond_B/mse_{ds}/seed*/condition_B_s*.json"))}
        D = {json.load(open(f))["seed"]: json.load(open(f)) for f in
             glob.glob(str(ROOT / root / f"cond_D/mse_{ds}/seed*/condition_D_s*.json"))}
        seeds = sorted(set(B) & set(D))
        if not seeds:
            continue
        acc = []
        for s in seeds:
            b, d = B[s], D[s]
            ridge_ok = d["ridge_optimum"]["best_val_loss"] <= d["best_val_loss"] * 1.05
            dv = (d["best_val_loss_ols"] if ridge_ok else d["best_val_loss"]) / horizon
            dt = (d["test_mse_per_element_ols"] if ridge_ok
                  else d["test_mse_per_element_adamw"])
            zs_t = (d["zs_mse_test"] + b["zs_mse_test"]) / 2
            zs_v = d["zs_mse"]
            acc.append((b["best_val_loss"] / horizon / zs_v, dv / zs_v,
                        b["test_mse_per_element"] / zs_t, dt / zs_t, zs_t / zs_v))
        out.append(_row(f"Chronos/{ds} h{horizon}", acc))
    return out


def timesfm(root="results/v46_timesfm", horizon=24):
    out = []
    for ds in ("ETTh1", "Weather", "ETTm2", "ETTh2", "Electricity"):
        base = ROOT / root / f"{ds}_h{horizon}"
        B = {json.load(open(f))["seed"]: json.load(open(f))
             for f in glob.glob(str(base / f"condition_B/condition_B_h{horizon}_s*.json"))}
        D = {json.load(open(f))["seed"]: json.load(open(f))
             for f in glob.glob(str(base / f"condition_D/condition_D_h{horizon}_s*.json"))}
        seeds = sorted(set(B) & set(D))
        if not seeds:
            continue
        acc = [(B[s]["final_val_mse"] / B[s]["zeroshot_mse"],
                D[s]["final_val_mse"] / B[s]["zeroshot_mse"],
                B[s]["test_mse"] / B[s]["zeroshot_test_mse"],
                D[s]["test_mse"] / B[s]["zeroshot_test_mse"],
                B[s]["zeroshot_test_mse"] / B[s]["zeroshot_mse"]) for s in seeds]
        out.append(_row(f"TimesFM/{ds} h{horizon}", acc))
    return out


def build():
    with contextlib.redirect_stdout(io.StringIO()):
        published = {r["cell"]: r for r in cm.build_rows()}
    rows = legacy_moirai() + new_moirai() + chronos() + timesfm()

    for r in rows:
        # The decomposition is an identity, so this can only fail if a reader was edited wrongly.
        assert abs((r["bd_test"] - r["bd_val"]) - (r["d_b"] - r["d_d"])) < 1e-6, \
            f"{r['cell']}: dB - dD does not close the gap"
        p = published.get(r["cell"])
        assert p is not None, f"{r['cell']} is not a published row -- reader keys have drifted"
        for k in ("bd_val", "bd_test"):
            assert abs(r[k] - p[k]) < TOL, \
                f"{r['cell']} {k}: recomputed {r[k]:+.2f} vs published {p[k]:+.2f}"
    return rows


def emit_latex(rev, path=ROOT / "paper_8/tables/heldout_decomposition.tex"):
    """Ordered by held-out B$-$D, matching cell_matrix.emit_latex()'s table."""
    def f(x):
        return f"${x:+.1f}$"
    lines = [
        "% GENERATED by scripts/heldout_decomposition.py -- do not edit by hand.",
        # 4pt, not 5: at 5pt the six columns overrun \textwidth by ~4.5pt.
        r"\begin{center}", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
        r"Cell & B$-$D val & B$-$D held-out & $\Delta$ adapted (B) & $\Delta$ frozen (D)"
        r" & $\text{zs}_\text{test}/\text{zs}_\text{val}$ \\",
        r"\midrule",
    ]
    for r in sorted(rev, key=lambda x: x["bd_test"]):
        lines.append(f"{cm._display(r['cell'])} & {f(r['bd_val'])} & {f(r['bd_test'])} & "
                     f"{f(r['d_b'])} & {f(r['d_d'])} & ${r['zs_ratio']:.2f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    path.write_text("\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    rows = build()
    rev = [r for r in rows if (r["bd_val"] > 0) != (r["bd_test"] > 0)]
    assert len(rev) == 4, f"expected 4 sign reversals, got {len(rev)}"
    assert all(r["bd_val"] > 0 > r["bd_test"] for r in rev), \
        "reversals are no longer all in the direction that flatters the frozen encoder"

    ratios = [r["zs_ratio"] for r in rows]
    lo, hi = min(ratios), max(ratios)

    print(f"{len(rows)} cells reproduce cell_matrix.build_rows() within {TOL} pp\n")
    print(f"{'cell':32s} {'B-D val':>8s} {'test':>8s} {'dB':>8s} {'dD':>8s}   zs_t/zs_v")
    for r in sorted(rev, key=lambda x: x["bd_test"]):
        print(f"{r['cell']:32s} {r['bd_val']:+8.1f} {r['bd_test']:+8.1f} "
              f"{r['d_b']:+8.1f} {r['d_d']:+8.1f}   {r['zs_ratio']:.2f}")

    # dD > dB is entailed by the reversal, so it is asserted as a sanity check and NOT reported as a
    # finding. The reportable split is the SIGN: did either arm actually get worse out of sample?
    assert all(r["d_d"] > r["d_b"] for r in rev), "dD > dB is implied by a positive->negative reversal"
    worse = [r for r in rev if r["d_b"] > 0 and r["d_d"] > 0]
    better = [r for r in rev if r["d_b"] < 0 and r["d_d"] < 0]
    assert len(worse) + len(better) == len(rev), "a reversal has arms moving in opposite directions"
    print(f"\nboth arms worse than own zero-shot on test: {len(worse)}/{len(rev)} "
          f"({', '.join(r['cell'] for r in worse)})")
    print(f"both arms better, adapted gaining more:      {len(better)}/{len(rev)} "
          f"({', '.join(r['cell'] for r in better)})")
    print(f"zs_test/zs_val over all {len(rows)} cells: {lo:.2f} to {hi:.2f}  "
          f"(untrained model, no fitting or selection)")
    print(f"\nwrote {emit_latex(rev)}")
