#!/usr/bin/env python3
"""
The preservation-priority gate, R2_task = 1 - MSE_ZS / MSE_Linear, for every cell in the matrix.

WHY THIS EXISTS
---------------
The "degradation" reading -- gate-passing cell where full fine-tuning ends up worse than the
un-tuned model while freezing improves on it -- is only meaningful if the gate is checked on the
same windows the intervention is scored on. Until now the gate was computed ad hoc per arm:
gate_linear_baseline.py printed Moirai linear baselines but stored nothing, the Chronos runs stored
a `gate_improvement_pct` measured on their SELECTION windows, and the ILI gate (57%) lived in a
docstring with no code behind it. Three provenances for one column invites exactly the kind of
quiet mismatch that produced the preservation-spectrum defect.

This module computes the gate on the HELD-OUT split for all three arms, each matched to that arm's
own evaluation protocol, and caches the result to results/gate_test_side.json so cell_matrix.py can
read one number per cell without recomputing.

  Moirai forecasting: extended lookback 192, first 300 test windows, MSE on the TRAIN-normalised
                      scale -- what evaluate_forecasting uses, so it is comparable to the stored
                      `zeroshot_test_mse`. Reuses linear_forecast() from gate_linear_baseline.py.
  Chronos:            univariate OT, lookback 96, h=24, the same 200-window test subsample the runs
                      used (build_windows(..., max_windows=200, seed=test_seed=0)), MSE on the
                      per-window z-scored scale. Reuses load_series()/build_windows() from
                      chronos_mse_finetune.py so the windows cannot drift from the runs'.
  ILI:                7 features, lookback 104, h=24, every test window, per-window z-score, and the
                      per-feature gate averaged as PERCENTAGES -- matching how finetune_ili.py
                      aggregates its own per-feature numbers.

A positive R2_task means the released checkpoint beats the linear baseline out of sample. Threshold
is 0.20.

Run:  .venv-probe/bin/python scripts/gate_all_cells.py
      .venv-probe/bin/python scripts/gate_all_cells.py --arm moirai   # skip the slow Chronos import
"""
import argparse
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Module level, after the path insert. Only NO_TRAINING and NAMES are read up here; the estimators
# themselves are still reached through the lazy imports inside the two gate functions, which is what
# keeps sklearn and the Moirai/Chronos loaders off the import path of anything that only wants the
# constants. gate_baselines itself imports nothing heavier than numpy at module scope.
import gate_baselines                                                  # noqa: E402

GATE_THRESHOLD = 0.20
OUT_PATH = ROOT / "results/gate_test_side.json"

# (model_size, dataset, horizon) for every Moirai forecasting cell in the matrix.
MOIRAI_CELLS = [
    ("small", "ETTh1", 96), ("small", "ETTh1", 192),
    ("small", "ETTh2", 96), ("small", "ETTh2", 192),
    ("small", "ETTm2", 96), ("small", "ETTm2", 192),
    ("small", "Weather", 96), ("small", "Weather", 192),
    ("base", "ETTh1", 96), ("base", "ETTh1", 192),
    ("base", "ETTh2", 96), ("base", "ETTh2", 192),
    ("large", "ETTh2", 96),
    # --- prospective arm (results/v47_prospective), pre-registered before any B/D run ---
    ("base", "Weather", 96), ("base", "Weather", 192),
    ("base", "ETTm2", 96), ("base", "ETTm2", 192),
    ("small", "Electricity7", 96), ("small", "Electricity7", 192),
    ("large", "ETTh1", 96), ("large", "Weather", 96),
]


def _zs_test_refs():
    """Reuse cell_matrix's denominator discovery so the gate and B-D share one reference."""
    import cell_matrix
    return cell_matrix._zs_test_refs()


def _zs_val_refs():
    """dataset-key -> zero-shot VALIDATION MSE, the denominator a prospective gate would have had.

    finetune_forecasting.py stores this as `zeroshot_mse`, measured on X_val_eval[:300] -- window
    construction there is deterministic (no seed), so every run of a cell carries the same number
    and averaging over whatever runs exist is a no-op that only guards against a missing file.
    """
    refs = defaultdict(list)
    for pat, depth in (("results/v41_zs_test/*/condition_A_*.json", 1),
                       ("results/v43_moirai_matrix/*/condition_*/*.json", 2),
                       ("results/v47_prospective/*/condition_*/*.json", 2)):
        for f in glob.glob(str(ROOT / pat)):
            key = Path(f).parents[depth - 1].name if depth == 1 else Path(f).parents[1].name
            d = json.load(open(f))
            if isinstance(d.get("zeroshot_mse"), float):
                refs[key].append(d["zeroshot_mse"])
    for f in glob.glob(str(ROOT / "results/v39_moirai_zs_test/h96/condition_*/*.json")):
        d = json.load(open(f))
        if isinstance(d.get("zeroshot_mse"), float):
            refs["base_ETTh2_h96"].append(d["zeroshot_mse"])
    return {k: st.mean(v) for k, v in refs.items()}


def _zs_traintail_refs():
    """dataset-key -> zero-shot MSE on the train-tail third split, plus the window bookkeeping.

    Written by `finetune_forecasting.py --condition A --zs-windows traintail`. Returns
    (refs, meta) where meta carries the cut index, window count and extended lookback the
    numerator was measured with, so the denominator here can assert it built the SAME windows
    rather than trusting that two scripts agree.
    """
    refs, meta = {}, {}
    for f in sorted(glob.glob(str(ROOT / "results/v49_thirdsplit/*/condition_A_traintail_*.json"))):
        d = json.load(open(f))
        key = Path(f).parent.name
        refs[key] = d["zeroshot_traintail_mse"]
        meta[key] = {k: d[k] for k in ("traintail_cut_index", "n_traintail_windows",
                                       "n_traintail_eval", "extended_lookback",
                                       "traintail_frac", "n_train_head_steps")}
    return refs, meta


# ---------------------------------------------------------------------------
# Moirai forecasting arm
# ---------------------------------------------------------------------------

def moirai_gates(refs, split="test", lookback=96, max_eval=300, baseline="fitted",
                 traintail_meta=None, traintail_frac=0.2):
    """R2_task per Moirai cell on `split`.

    The extended lookback MUST be the one finetune_forecasting.py evaluates with
    (`extended_lookback = lookback + horizon`, :709), not lookback*2. They coincide at h=96 and
    diverge at h=192, where lookback*2 pairs the stored zero-shot MSE with a linear baseline
    measured over a DIFFERENT set of target windows -- the numerator and denominator of R2_task
    stop describing the same forecasting problem. An earlier revision of this file used
    lookback*2 throughout; see the `gate` column note in the paper's threshold appendix.

    `baseline` selects the denominator's estimator: "fitted" is the lookback-96 regression the
    paper defines, trained on this cell's TRAIN windows; "trend" is the per-window extrapolation
    used through 26 Aug 2026, kept so the correction can be reported. They disagree by a factor of
    ~3 on ETTh1 and ~9 on Weather -- see gate_linear_baseline.fit_linear_map.

    Note the asymmetry the paper's definition imposes: Moirai sees `lookback + h` steps of context
    while the linear baseline sees `lookback`. That is what "lookback-96 linear baseline" means
    here, and it favours Moirai; the gate is a lower bound on the linear model's strength.
    """
    from src.data.forecasting_loader import get_forecasting_loader
    from gate_linear_baseline import apply_linear_map, fit_linear_map, linear_forecast

    out = {}
    linear_cache = {}
    for size, ds, h in MOIRAI_CELLS:
        key = f"{size}_{ds}_h{h}"
        zs = refs.get(key)
        if zs is None:
            print(f"  {key:24s} SKIPPED -- no zero-shot {split} reference")
            continue
        if (ds, h) not in linear_cache:
            loader = get_forecasting_loader(f"data/forecasting/{ds}.csv", lookback_window=lookback,
                                            forecast_horizon=h, features="M")
            train_df, val_df, test_df = loader.get_splits()
            cols = loader.FEATURE_COLUMNS
            tr = train_df[cols].values
            ext_lb = lookback + h                     # matches finetune_forecasting.py:709
            total = ext_lb + h

            # The third split: the gate is scored on the tail of the train region and every rung is
            # fitted on the head alone, with head-only normalisation. Both halves must be exactly
            # the ones the numerator was measured on, so the cut is recomputed by the same rule and
            # then checked against what the condition-A run recorded.
            if split == "traintail":
                cut = int(len(tr) * (1.0 - traintail_frac))
                fit_vals, eval_vals = tr[:cut], tr[cut:]
            else:
                cut = None
                fit_vals = tr
                eval_vals = (val_df if split == "val" else test_df)[cols].values
            mu, sd = fit_vals.mean(axis=0), fit_vals.std(axis=0) + 1e-8

            def windows(vals, cap=None):
                X = np.array([vals[i:i + ext_lb] for i in range(len(vals) - total + 1)])
                y = np.array([vals[i + ext_lb:i + total] for i in range(len(vals) - total + 1)])
                return (X, y) if cap is None else (X[:cap], y[:cap])

            X, y = windows(eval_vals, max_eval)
            binfo = {}
            if baseline == "fitted":
                # Fit on the fitting region only -- the TRAIN split, or its head under the third
                # split -- on the same normalised scale the stored zero-shot numerator is measured
                # on, so numerator and denominator stay comparable.
                Xtr, ytr = windows(fit_vals)
                coef = fit_linear_map((Xtr - mu) / sd, (ytr - mu) / sd, lookback)
                pred_n = apply_linear_map(coef, (X - mu) / sd, lookback, h)
            elif baseline == "fitted_matched":
                # THE MATCHED-LOOKBACK RUNG, added 21 Sep 2026 because a reviewer asked whether the
                # gate's verdict is an artifact of the context asymmetry noted in this docstring:
                # Moirai is given ext_lb = lookback + h steps of history, the baseline the last
                # `lookback` of them. Everything here is byte-identical to the `fitted` branch except
                # the number passed to fit_linear_map/apply_linear_map -- same windows, same fitting
                # region, same normalisation, same lam -- so the two columns differ in the baseline's
                # INFORMATION SET and in nothing else. fit_linear_map already documents that it uses
                # the last `lookback` steps of a (N, >=lookback, D) context, which is why passing
                # ext_lb needs no new window construction.
                #
                # It is deliberately NOT in gate_baselines.NAMES. The ladder there varies the
                # estimator FAMILY at fixed lookback and its rungs feed graded_value()'s
                # min-over-admissible score, which fig_value_axis and cka_fixed_effects read; folding
                # a different axis into that minimum would silently move the value score of every
                # cell in the paper. scripts/matched_lookback_gate.py reports this column separately
                # and states what folding it in would do.
                Xtr, ytr = windows(fit_vals)
                coef = fit_linear_map((Xtr - mu) / sd, (ytr - mu) / sd, ext_lb)
                pred_n = apply_linear_map(coef, (X - mu) / sd, ext_lb, h)
                binfo = dict(lookback_used=int(ext_lb), lookback_nominal=int(lookback),
                             n_features=int(ext_lb * tr.shape[1]),
                             n_fit_windows=int(len(Xtr)))
            elif baseline == "trend":
                pred_n = (linear_forecast(X, h, lookback) - mu) / sd
            else:
                # Alternative baseline families (seasonal naive, per-feature AR, tuned ridge,
                # gradient-boosted trees). Everything is handed over already train-normalised, so
                # the denominator stays on the same scale as the stored zero-shot numerator.
                import gate_baselines
                Xtr, ytr = windows(fit_vals)
                pred_n, binfo = gate_baselines.predict(
                    baseline, (X - mu) / sd, lookback, h,
                    train=((Xtr - mu) / sd, (ytr - mu) / sd), dataset=ds)
            lin = float(np.mean((pred_n - (y - mu) / sd) ** 2))
            geom = dict(cut=cut, n_all=len(windows(eval_vals)[0]), n_scored=len(X), ext_lb=ext_lb)
            linear_cache[(ds, h)] = (lin, len(X), binfo, geom)
        lin, n_win, binfo, geom = linear_cache[(ds, h)]
        # Per cell, not per (dataset, horizon): the cache is shared across model sizes, and a cell
        # whose numerator was measured with a different cut must not borrow a sibling's denominator.
        if split == "traintail":
            m = (traintail_meta or {}).get(key)
            if m is None:
                print(f"  {key:24s} SKIPPED -- no third-split numerator recorded")
                continue
            bad = [(n, a, b) for n, a, b in
                   (("cut index", geom["cut"], m["traintail_cut_index"]),
                    ("window count", geom["n_all"], m["n_traintail_windows"]),
                    ("scored windows", geom["n_scored"], m["n_traintail_eval"]),
                    ("extended lookback", geom["ext_lb"], m["extended_lookback"]))
                   if a != b]
            if bad:
                raise SystemExit(
                    f"{key}: third-split window construction disagrees with the numerator's -- "
                    + "; ".join(f"{n}: denominator {a} vs numerator {b}" for n, a, b in bad))
        out[key] = dict(r2_task=1 - zs / lin, zs_test=zs, linear_test=lin,
                        n_windows=n_win, split=split, arm="moirai", baseline=baseline,
                        **({"baseline_info": binfo} if binfo else {}))
        print(f"  {key:24s} ZS {zs:.4f}  linear {lin:.4f}  R2_task {1 - zs / lin:+.3f}"
              f"  {'PASS' if 1 - zs / lin >= GATE_THRESHOLD else 'fail'}")
    return out


# ---------------------------------------------------------------------------
# Chronos arm
# ---------------------------------------------------------------------------

def _zscore_windows(ctx, tgt):
    """Per-window z-score by the context's own mean/sd -- the scale the Chronos/TimesFM arms score
    on. Returns (ctx_n, tgt_n) as (N, T, 1) so the shared linear-map helpers apply unchanged.
    """
    c = np.asarray(ctx, dtype=np.float64).reshape(len(ctx), -1)
    t = np.asarray(tgt, dtype=np.float64).reshape(len(tgt), -1)
    mu = c.mean(axis=1, keepdims=True)
    sd = c.std(axis=1, keepdims=True) + 1e-8
    return ((c - mu) / sd)[:, :, None], ((t - mu) / sd)[:, :, None]


def _window_linear_mse(ctx, tgt, lookback, horizon, train=None, baseline="fitted",
                       dataset=None, info_out=None):
    """Lookback-`lookback` linear baseline on the per-window z-scored scale, shared by the Chronos
    and TimesFM arms so they cannot drift apart.

    baseline="fitted" is the regression the paper defines: one ridge-OLS map estimated on `train`
    (a (ctx, tgt) window pair from the TRAIN series) and applied unchanged here.
    baseline="trend" is the per-window extrapolation used through 26 Aug 2026
    (chronos_mse_finetune.py:438), retained so the correction can be reported.
    Any other name is one of the alternative families in gate_baselines.py; `dataset` is needed
    there only to look up the seasonal period, and `info_out`, if given, is updated with whatever
    that baseline needs disclosed (a selected penalty, a row cap).
    """
    c_n, t_n = _zscore_windows(ctx, tgt)
    if baseline == "fitted":
        if train is None:
            raise ValueError("baseline='fitted' needs train windows; pass train=(ctx_tr, tgt_tr)")
        from gate_linear_baseline import apply_linear_map, fit_linear_map
        ctr_n, ttr_n = _zscore_windows(*train)
        coef = fit_linear_map(ctr_n, ttr_n, lookback)
        pred = apply_linear_map(coef, c_n, lookback, horizon)
        return float(np.mean((pred - t_n) ** 2))

    if baseline != "trend":
        import gate_baselines
        tr_n = _zscore_windows(*train) if train is not None else None
        pred, info = gate_baselines.predict(baseline, c_n, lookback, horizon,
                                            train=tr_n, dataset=dataset)
        if info_out is not None:
            info_out.update(info)
        return float(np.mean((pred - t_n) ** 2))

    from sklearn.linear_model import LinearRegression
    t_in = np.arange(lookback).reshape(-1, 1)
    t_out = np.arange(lookback, lookback + horizon).reshape(-1, 1)
    errs = []
    for i in range(len(ctx)):
        mu, sd = ctx[i].mean(), ctx[i].std() + 1e-8
        pred = LinearRegression().fit(t_in, ctx[i]).predict(t_out)
        errs.append(np.mean(((pred - mu) / sd - (tgt[i] - mu) / sd) ** 2))
    return float(np.mean(errs))


def chronos_gates(root="results/v44_chronos_guarded", horizon=24, lookback=96, split="test",
                  baseline="fitted"):
    """R2_task per Chronos cell.

    On the validation side both terms are already stored per run: chronos_mse_finetune.py measures
    `zs_mse` and `linear_mse` on eval_ctx (= the 200 validation windows, :391/:436/:437), so the
    prospective gate needs no recomputation and cannot drift from what the runs actually saw.
    Validation windows are drawn with seed=args.seed, so they differ per run; zs and linear are
    averaged separately before the ratio, mirroring the test-side aggregation.
    """
    from chronos_mse_finetune import build_windows, load_series

    out = {}
    for ds_dir, ds_cfg in (("etth1", "ETTh1"), ("etth2", "ETTh2"), ("weather", "Weather"),
                           ("ettm2", "ETTm2"), ("electricity", "Electricity")):
        runs = [json.load(open(f)) for f in
                glob.glob(str(ROOT / root / f"cond_B/mse_{ds_dir}/seed*/condition_B_s*.json"))]
        key = f"chronos_{ds_dir}"
        if split == "val":
            runs = [r for r in runs if "zs_mse" in r and "linear_mse" in r]
            if not runs:
                print(f"  {key:24s} SKIPPED -- no stored zs_mse/linear_mse")
                continue
            zs = st.mean(r["zs_mse"] for r in runs)
            if baseline == "trend":
                # The ONLY rung that may be read from the runs: `linear_mse` is the superseded
                # per-window trend extrapolation, stored so the correction can still be reported.
                lin = st.mean(r["linear_mse"] for r in runs)
            else:
                # Every other rung is recomputed. This branch read `linear_mse` for all
                # non-"fitted" baselines until 2026-09-20, from a time when "trend" was the only
                # alternative -- so a ladder sweep would have returned the trend denominator under
                # all eight rung names and printed eight identical columns for these five cells.
                # Validation windows are drawn with seed=args.seed, so rebuild them per seed.
                train_s, val_s, _ = load_series(ds_cfg)
                tr_w = (build_windows(train_s, lookback, horizon)
                        if baseline not in gate_baselines.NO_TRAINING else None)
                by_seed = {}
                for r in runs:
                    sd_ = r["seed"]
                    if sd_ not in by_seed:
                        c, t = build_windows(val_s, lookback, horizon, max_windows=200, seed=sd_)
                        by_seed[sd_] = _window_linear_mse(c, t, lookback, horizon,
                                                         train=tr_w, baseline=baseline,
                                                         dataset=ds_dir)
                lin = st.mean(by_seed[r["seed"]] for r in runs)
            n_win = 200
        else:
            runs = [r for r in runs if "zs_mse_test" in r]
            if not runs:
                print(f"  {key:24s} SKIPPED -- no stored zs_mse_test")
                continue
            zs = st.mean(r["zs_mse_test"] for r in runs)
            test_seed = runs[0].get("test_seed", 0)
            train_s, _, test_s = load_series(ds_cfg)
            ctx, tgt = build_windows(test_s, lookback, horizon, max_windows=200, seed=test_seed)
            tr_w = build_windows(train_s, lookback, horizon) if baseline not in gate_baselines.NO_TRAINING else None
            lin = _window_linear_mse(ctx, tgt, lookback, horizon, train=tr_w, baseline=baseline,
                                     dataset=ds_dir)
            n_win = int(len(ctx))
        out[key] = dict(r2_task=1 - zs / lin, zs_test=zs, linear_test=lin,
                        n_windows=n_win, split=split, arm="chronos", baseline=baseline)
        print(f"  {key:24s} ZS {zs:.4f}  linear {lin:.4f}  R2_task {1 - zs / lin:+.3f}")
    return out


# ---------------------------------------------------------------------------
# TimesFM arm (third backbone)
# ---------------------------------------------------------------------------

TIMESFM_DATASETS = ["ETTh1", "ETTh2", "ETTm2", "Weather", "Electricity"]


def timesfm_gates(horizon=24, lookback=96, device="cpu", datasets=None, split="test",
                  baseline="fitted"):
    """Screen TimesFM 2.5 at h=24 on the SAME window set the Chronos arm uses.

    This runs BEFORE any TimesFM training, and it decides whether training is licensed at all:
    the paper's own screening criterion says an intervention on a cell is only interpretable if
    the released checkpoint demonstrably beats the linear baseline out of sample there. At h=96
    TimesFM gate-FAILS on ETTh1/ETTh2 (-0.139/-0.138, results/v14_timesfm_etth1.json,
    results/v11_timesfm_etth2.json); h=24 is the shorter horizon where it might not.

    Whatever this prints is reportable either way -- a third backbone with no gate-passing cell is
    a narrowing of the paper's scope, not a failed experiment.
    """
    from chronos_mse_finetune import build_windows, load_series

    out = {}
    if split == "val":
        # No model load: finetune_timesfm.py already stored the validation zero-shot as
        # `zeroshot_mse` (:167), measured on build_windows(val_s, ..., seed=args.seed) (:134).
        # Rebuild those same per-seed windows for the linear denominator and average.
        for ds in (datasets or TIMESFM_DATASETS):
            runs = [(json.load(open(f)), f) for f in
                    glob.glob(str(ROOT / f"results/v46_timesfm/{ds}_h{horizon}/condition_*/*.json"))]
            runs = [(d, f) for d, f in runs if isinstance(d.get("zeroshot_mse"), float)]
            key = f"timesfm_{ds.lower()}"
            if not runs:
                print(f"  {key:24s} SKIPPED -- no stored zeroshot_mse")
                continue
            train_s, val_s, _ = load_series(ds)
            tr_w = build_windows(train_s, lookback, horizon) if baseline not in gate_baselines.NO_TRAINING else None
            zs_by_seed, lin_by_seed = {}, {}
            for d, _f in runs:
                sd_ = d["seed"]
                zs_by_seed[sd_] = d["zeroshot_mse"]
                if sd_ not in lin_by_seed:
                    c, t = build_windows(val_s, lookback, horizon, max_windows=200, seed=sd_)
                    lin_by_seed[sd_] = _window_linear_mse(c, t, lookback, horizon, dataset=ds,
                                                          train=tr_w, baseline=baseline)
            zs = st.mean(zs_by_seed.values())
            lin = st.mean(lin_by_seed[s_] for s_ in zs_by_seed)
            out[key] = dict(r2_task=1 - zs / lin, zs_test=zs, linear_test=lin, n_windows=200,
                            split="val", arm="timesfm", horizon=horizon, baseline=baseline)
            print(f"  {key:24s} ZS {zs:.4f}  linear {lin:.4f}  R2_task {1 - zs / lin:+.3f}"
                  f"  {'PASS' if 1 - zs / lin >= GATE_THRESHOLD else 'fail'}")
        return out

    from timesfm_common import (assert_no_ar, batched_point_mse, load_timesfm,
                                verify_native_path)

    model, _ = load_timesfm(max_context=lookback, max_horizon=128, device=device)
    assert_no_ar(model)

    for ds in (datasets or TIMESFM_DATASETS):
        train_s, _, test_s = load_series(ds)
        # Same univariate-OT series, lookback, horizon and 200-window seed-0 subsample as the
        # Chronos arm, so the zero-shot reference, the gate and (later) B-D share one window set.
        ctx, tgt = build_windows(test_s, lookback, horizon, max_windows=200, seed=0)
        if ds == (datasets or TIMESFM_DATASETS)[0]:
            verify_native_path(model, ctx, horizon, device)
        zs = batched_point_mse(model, ctx, tgt, horizon, device)
        tr_w = build_windows(train_s, lookback, horizon) if baseline not in gate_baselines.NO_TRAINING else None
        lin = _window_linear_mse(ctx, tgt, lookback, horizon, train=tr_w, baseline=baseline,
                                 dataset=ds)
        key = f"timesfm_{ds.lower()}"
        out[key] = dict(r2_task=1 - zs / lin, zs_test=zs, linear_test=lin,
                        n_windows=int(len(ctx)), split="test", arm="timesfm", horizon=horizon,
                        baseline=baseline)
        print(f"  {key:24s} ZS {zs:.4f}  linear {lin:.4f}  R2_task {1 - zs / lin:+.3f}"
              f"  {'PASS' if 1 - zs / lin >= GATE_THRESHOLD else 'fail'}")
    return out


# ---------------------------------------------------------------------------
# ILI arm
# ---------------------------------------------------------------------------

def ili_gate(lookback=104, horizon=24, data_path="data/national_illness.csv", split="test",
             baseline="fitted"):
    from finetune_ili import load_ili_data

    runs = [json.load(open(f)) for f in
            glob.glob(str(ROOT / "results/v40_ili_heldout/condition_B_seed*.json"))]
    if not runs:
        print("  ili                      SKIPPED -- no condition B runs")
        return {}
    train, val, test, columns = load_ili_data(str(ROOT / data_path))
    series = val if split == "val" else test
    zs_key = "zs_mse" if split == "val" else "zs_mse_test"

    def slide(col):
        n = len(col) - lookback - horizon + 1
        c = np.array([col[i:i + lookback] for i in range(n)])
        t = np.array([col[i + lookback:i + lookback + horizon] for i in range(n)])
        return c, t

    # finetune_ili averages PERCENTAGES across features, so the gate must too.
    per_feature = {}
    for r in runs:
        for feat in r["per_feature"]:
            if isinstance(feat.get(zs_key), float) and feat[zs_key] == feat[zs_key]:
                per_feature.setdefault(feat["feature"], []).append(feat[zs_key])
    r2s, zs_all, lin_all = [], [], []
    for name, zs_vals in per_feature.items():
        j = columns.index(name)
        c, t = slide(series[:, j].astype(np.float64))
        tr_w = slide(train[:, j].astype(np.float64)) if baseline not in gate_baselines.NO_TRAINING else None
        lin = _window_linear_mse(c, t, lookback, horizon, train=tr_w, baseline=baseline,
                                 dataset="ILI")
        zs = st.mean(zs_vals)
        r2s.append(1 - zs / lin)
        zs_all.append(zs)
        lin_all.append(lin)
    out = {"ili": dict(r2_task=float(np.mean(r2s)), zs_test=float(np.mean(zs_all)),
                       linear_test=float(np.mean(lin_all)), n_features=len(r2s),
                       split=split, arm="ili", baseline=baseline)}
    print(f"  {'ili':24s} ZS {np.mean(zs_all):.4f}  linear {np.mean(lin_all):.4f}  "
          f"R2_task {np.mean(r2s):+.3f}  (mean over {len(r2s)} features)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all", choices=["all", "moirai", "chronos", "ili", "timesfm"])
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="test = the gate as reported; val = what a PROSPECTIVE gate, fixed before "
                         "any test window was consulted, would have said")
    ap.add_argument("--baseline", default="fitted", choices=["fitted", "trend"],
                    help="fitted = the lookback-96 regression trained on the train split, which is "
                         "the baseline the paper DEFINES; trend = the per-window extrapolation used "
                         "through 26 Aug 2026, retained only to report the correction")
    ap.add_argument("--dry-run", action="store_true", help="print, do not touch the cache")
    a = ap.parse_args()

    suffix = "" if a.baseline == "fitted" else "_trend"
    out_path = ROOT / f"results/gate_{a.split}_side{suffix}.json"
    print("=" * 84)
    print(f"PRESERVATION-PRIORITY GATE ON {a.split.upper()} WINDOWS   "
          f"R2_task = 1 - MSE_ZS / MSE_Linear")
    print(f"baseline: {a.baseline}   gate-pass threshold {GATE_THRESHOLD}")
    print("=" * 84)
    gates = {}
    if a.arm in ("all", "moirai"):
        gates.update(moirai_gates(_zs_val_refs() if a.split == "val" else _zs_test_refs(),
                                  split=a.split, baseline=a.baseline))
    if a.arm in ("all", "chronos"):
        gates.update(chronos_gates(split=a.split, baseline=a.baseline))
    if a.arm in ("all", "ili"):
        gates.update(ili_gate(split=a.split, baseline=a.baseline))
    if a.arm in ("all", "timesfm"):
        gates.update(timesfm_gates(split=a.split, baseline=a.baseline))

    if gates and not a.dry_run:
        # Merge rather than overwrite, so re-running one arm cannot silently drop the others'
        # cells from the cached file cell_matrix.py reads.
        merged = json.loads(out_path.read_text()) if out_path.exists() else {}
        merged.update(gates)
        out_path.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {out_path.relative_to(ROOT)}  "
              f"({len(gates)} cells updated, {len(merged)} total)")
        # A merged cache that mixes estimators would put two different quantities in one column.
        stale = sorted(k for k, v in merged.items() if v.get("baseline", "trend") != a.baseline)
        if stale:
            print(f"\n!! {len(stale)} cell(s) in this cache still carry a different baseline and "
                  f"must be re-run before the file is used:\n   {', '.join(stale)}")
    n_pass = sum(1 for g in gates.values() if g["r2_task"] >= GATE_THRESHOLD)
    print(f"gate-pass: {n_pass}/{len(gates)} cells")


if __name__ == "__main__":
    main()
