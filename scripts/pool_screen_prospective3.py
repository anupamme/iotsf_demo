#!/usr/bin/env python3
"""Batch-3 candidate pool: the corrected ridge gate on cells that have NEVER been fine-tuned.

WHY A SEPARATE FILE FROM results/gate_val_side.json.  This screen exists to CHOOSE the batch-3
cells, so it must run before any of them is registered and long before any of them has a B/D run.
Writing it into the paper's gate cache would have two effects we do not want yet: emit_r2task.py
would fill three currently-empty Moirai-Large slots and its "7 of 21" counts would move, and
score_prospective.py would see cells with no outcome.  The pool therefore lives in
results/v56_pool3/pool_gate_val.json and the paper's cache is untouched until the registered cells
have outcomes.  The two must then agree exactly, which is checkable because both call the same
function on the same inputs -- scripts/score_prospective.py asserts it.

WHAT THE SCREEN IS.  Identical to the paper's primary gate: R2_task = 1 - MSE_zeroshot / MSE_fitted,
where `fitted` is the one ridge map (lookback 96 -> horizon h, lam=1e-4) fitted by least squares on
the cell's TRAIN windows and applied unchanged, scored on the SELECTION split.  Operating point
0.20.  Nothing here is a new estimator: the Moirai arm calls gate_all_cells.moirai_gates with the
pool substituted for MOIRAI_CELLS, and the two non-Moirai arms call the same
gate_all_cells._window_linear_mse the Chronos/TimesFM cells in the paper are scored with.  In
particular this is NOT scripts/timesfm_gate_screen_h24.py, which the plan pointed at for its shape
only: that script's denominator is the per-window trend extrapolation a constant predictor beats,
which is the defect this round corrects, and it hardcodes h=24.

NO OUTCOME IS READ ANYWHERE IN THIS FILE.  It touches condition_A (zero-shot) records and the
benchmark CSVs and nothing else.  That is what makes a pool screened with it admissible as the basis
for a pre-registration.

PROVENANCE OF THE MOIRAI ZERO-SHOT ARMS.  Six of the nine Moirai pool cells take their numerator
from results/v48_prospective2/, an abandoned batch 2 whose condition_A runs were made on
2026-08-27 -- BEFORE the batch-3 registration.  They are zero-shot measurements of a released
checkpoint, not outcomes: no cell's B or D arm was ever run there, and a zero-shot MSE cannot move
with a fine-tuning result.  The registration says so in as many words; the remaining three cells'
condition_A runs are made by scripts/run_pool3_zs.sh into results/v56_pool3/.

TRAFFIC IS DROPPED, NOT FIXED.  data/forecasting/Traffic.csv is a 14-byte HTTP 404 page and the
repository contains no downloader for it, so its provenance could not be recorded even if a copy
were fetched -- and results/data_manifest.json is the thing that makes a cell reproducible here.
Both batch-2 Traffic cells failed for this reason and neither is in the pool.

Run:  .venv-probe/bin/python scripts/pool_screen_prospective3.py --arm moirai
      .venv-probe/bin/python scripts/pool_screen_prospective3.py --arm chronos
      .venv-probe/bin/python scripts/pool_screen_prospective3.py --arm timesfm
"""
import argparse
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gate_all_cells as G                                            # noqa: E402

OUT_PATH = ROOT / "results/v56_pool3/pool_gate_val.json"
LOOKBACK = 96

# ---------------------------------------------------------------------------
# The pool, fixed before any gate value below was computed.
# ---------------------------------------------------------------------------
# Moirai: every (size, dataset, horizon) combination in the paper's display grid that has never
# been fine-tuned in any arm -- the complement of MOIRAI_CELLS within {small, base, large} x
# {ETTh1, ETTh2, ETTm2, Weather, Electricity7} x {96, 192}, restricted to the sizes and datasets
# tables/r2task.tex already prints rows for.  Nine cells; the rule is mechanical and no outcome
# enters it.
POOL_MOIRAI = [
    ("base", "Electricity7", 96), ("base", "Electricity7", 192),
    ("large", "ETTh1", 192),
    ("large", "ETTh2", 192),
    ("large", "ETTm2", 96), ("large", "ETTm2", 192),
    ("large", "Weather", 192),
    ("large", "Electricity7", 96), ("large", "Electricity7", 192),
]

# The two extra backbones at a horizon neither has been trained at.  h=24 is exhausted -- all five
# Chronos cells and four of five TimesFM cells already have B/D -- so the unseen axis is the
# horizon, which the paper already treats as a cell coordinate.  h=48 keeps both backbones inside
# the context/horizon regime they were screened in (lookback 96) and doubles the horizon, which is
# where a checkpoint's zero-shot advantage over a fitted linear map is most likely to move.
NONMOIRAI_H = 48
CHRONOS_DATASETS = ["ETTh1", "ETTh2", "ETTm2", "Weather", "Electricity"]
TIMESFM_DATASETS = ["ETTh1", "ETTh2", "ETTm2", "Weather", "Electricity"]
NONMOIRAI_SEED = 42        # the window-subsample seed; 200 windows, as both arms use at h=24


def pool_zs_refs():
    """cell-key -> zero-shot SELECTION-split MSE for the Moirai pool.

    Same field and same measurement as gate_all_cells._zs_val_refs (`zeroshot_mse`, on
    X_val_eval[:300], deterministic window construction), read from the two directories the pool's
    condition_A runs live in.  Deliberately a separate function rather than an extension of
    _zs_val_refs: that one feeds the paper's cache, and the pool must not change what the paper's
    gate is computed from until the cells it selects have outcomes.
    """
    refs = defaultdict(list)
    for pat in ("results/v48_prospective2/*/condition_A*/*.json",
                "results/v56_pool3/*/condition_A*/*.json"):
        for f in glob.glob(str(ROOT / pat)):
            d = json.load(open(f))
            if isinstance(d.get("zeroshot_mse"), float):
                refs[Path(f).parents[1].name].append(d["zeroshot_mse"])
    return {k: st.mean(v) for k, v in refs.items()}


def moirai_pool(baseline="fitted"):
    """The paper's Moirai gate function, with the pool substituted for the matrix."""
    G.MOIRAI_CELLS = POOL_MOIRAI          # read inside moirai_gates; nothing else consults it here
    return G.moirai_gates(pool_zs_refs(), split="val", lookback=LOOKBACK, baseline=baseline)


def chronos_pool(baseline="fitted", device="cpu"):
    """Chronos-T5-Small zero-shot at h=48 on the selection windows, against the fitted ridge.

    The numerator is computed here rather than read, because no run of these cells exists -- that is
    the point of a pool.  It uses chronos_mse_finetune.chronos_zs_mse, so the scoring convention
    (median of 20 sampled paths, per-window z-score by the context's own mean/sd) is the one the
    paper's five Chronos cells are scored under, and the denominator is the same
    _window_linear_mse those cells use.

    `device` is recorded and not used: chronos_zs_mse builds its batches on the CPU and the pipeline
    is loaded without a device map, so this arm runs on the CPU whatever is passed -- which is also
    why it can run beside a Moirai job holding the MPS device.
    """
    import torch
    from chronos import ChronosPipeline
    from chronos_mse_finetune import MODEL_ID, build_windows, chronos_zs_mse, load_series

    pipe = ChronosPipeline.from_pretrained(MODEL_ID, dtype=torch.float32)
    out = {}
    for ds in CHRONOS_DATASETS:
        torch.manual_seed(NONMOIRAI_SEED)          # pipe.predict draws from the global RNG
        train_s, val_s, _ = load_series(ds)
        ctx, tgt = build_windows(val_s, LOOKBACK, NONMOIRAI_H, max_windows=200,
                                 seed=NONMOIRAI_SEED)
        tr_w = (build_windows(train_s, LOOKBACK, NONMOIRAI_H)
                if baseline not in G.gate_baselines.NO_TRAINING else None)
        zs = chronos_zs_mse(pipe, ctx, tgt, device)
        lin = G._window_linear_mse(ctx, tgt, LOOKBACK, NONMOIRAI_H, train=tr_w, baseline=baseline,
                                   dataset=ds)
        out[f"chronos_{ds.lower()}_h{NONMOIRAI_H}"] = dict(
            r2_task=1 - zs / lin, zs_test=zs, linear_test=lin, n_windows=int(len(ctx)),
            split="val", arm="chronos", horizon=NONMOIRAI_H, baseline=baseline,
            zs_source="computed by this script (no run of this cell exists)",
            window_seed=NONMOIRAI_SEED)
        _report(f"chronos_{ds.lower()}_h{NONMOIRAI_H}", zs, lin)
    return out


def timesfm_pool(baseline="fitted", device="cpu"):
    """TimesFM-2.5 zero-shot at h=48 on the selection windows, against the fitted ridge.

    Same window set and same denominator as the Chronos arm above, which is the invariant the
    paper's h=24 screen also maintains between those two backbones.  `device` defaults to cpu
    because timesfm_common.load_timesfm's MPS path is the one the paper's TimesFM runs avoided.
    """
    from chronos_mse_finetune import build_windows, load_series
    from timesfm_common import assert_no_ar, batched_point_mse, load_timesfm, verify_native_path

    model, _ = load_timesfm(max_context=LOOKBACK, max_horizon=128, device=device)
    assert_no_ar(model)
    out = {}
    for i, ds in enumerate(TIMESFM_DATASETS):
        train_s, val_s, _ = load_series(ds)
        ctx, tgt = build_windows(val_s, LOOKBACK, NONMOIRAI_H, max_windows=200,
                                 seed=NONMOIRAI_SEED)
        if i == 0:
            verify_native_path(model, ctx, NONMOIRAI_H, device)
        zs = batched_point_mse(model, ctx, tgt, NONMOIRAI_H, device)
        tr_w = (build_windows(train_s, LOOKBACK, NONMOIRAI_H)
                if baseline not in G.gate_baselines.NO_TRAINING else None)
        lin = G._window_linear_mse(ctx, tgt, LOOKBACK, NONMOIRAI_H, train=tr_w, baseline=baseline,
                                   dataset=ds)
        out[f"timesfm_{ds.lower()}_h{NONMOIRAI_H}"] = dict(
            r2_task=1 - zs / lin, zs_test=zs, linear_test=lin, n_windows=int(len(ctx)),
            split="val", arm="timesfm", horizon=NONMOIRAI_H, baseline=baseline,
            zs_source="computed by this script (no run of this cell exists)",
            window_seed=NONMOIRAI_SEED)
        _report(f"timesfm_{ds.lower()}_h{NONMOIRAI_H}", zs, lin)
    return out


def _report(key, zs, lin):
    r2 = 1 - zs / lin
    print(f"  {key:28s} ZS {zs:.4f}  fitted {lin:.4f}  R2_task {r2:+.3f}  "
          f"{'PASS' if r2 >= G.GATE_THRESHOLD else 'fail'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="moirai", choices=["moirai", "chronos", "timesfm"])
    ap.add_argument("--baseline", default="fitted")
    ap.add_argument("--device", default=None, help="recorded on the non-Moirai arms; both of them run on the CPU")
    a = ap.parse_args()

    print(f"batch-3 candidate pool, arm={a.arm}, baseline={a.baseline}, "
          f"split=val (selection), threshold {G.GATE_THRESHOLD}")
    if a.arm == "moirai":
        gates = moirai_pool(baseline=a.baseline)
    elif a.arm == "chronos":
        gates = chronos_pool(baseline=a.baseline, device=a.device or "cpu")
    else:
        gates = timesfm_pool(baseline=a.baseline, device=a.device or "cpu")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}
    for k, v in gates.items():
        old = merged.get(k)
        if old and old.get("baseline") != v.get("baseline"):
            print(f"  NOTE {k}: replacing a cached value computed with baseline "
                  f"{old.get('baseline')!r}")
        merged[k] = v
    OUT_PATH.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    n_pass = sum(v["r2_task"] >= G.GATE_THRESHOLD for v in merged.values())
    print(f"wrote {OUT_PATH.relative_to(ROOT)}: {len(merged)} cells, {n_pass} clear "
          f"{G.GATE_THRESHOLD} on the selection split")


if __name__ == "__main__":
    main()
