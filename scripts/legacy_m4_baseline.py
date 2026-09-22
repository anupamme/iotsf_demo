#!/usr/bin/env python3
"""Reproduce the superseded 84.5% figure for Chronos/M4-Monthly, and measure why it is meaningless.

WHY BOTHER. The corrected gate is already computed, registered and published as a fail
(results/chronos_m4/gate.json, +0.092 on the selection split), so nothing here can change a decision.
What is still missing is a record for the OLD number. "We cannot reproduce it" would have been an
honest but weak statement; "it reproduces, on its own terms, and here is the arithmetic that makes it
an artefact" is the statement this appendix can actually stand on. The three defects are separable and
all three are measured below:

  1. The denominator was a per-series unregularised OLS (finetune_chronos_m4.py:303), one model per
     series fitted on that series' own history, 96 free coefficients per output step against a history
     of 114 to a few hundred points. It is not a baseline that has been fitted badly; it is a baseline
     with about as many parameters as observations.
  2. Its normalisation did not match the numerator's. The zero-shot MSE was z-scored by each WINDOW's
     own context mean/sd (line 237); the linear baseline by each SERIES' training mean/sd (line 316).
     The ratio of two MSEs on different scales is not a ratio of anything.
  3. It averaged over a different set of series than the numerator. Series with fewer than 118
     observations yield fewer than 5 training windows and were SKIPPED in the denominator (line 314),
     while the zero-shot arm kept every series with at least 114. So the two halves of the ratio were
     computed over different populations.

The old selection rule is reproduced exactly, min_len = lookback + horizon = 114 rather than the 228
of scripts/m4_monthly_data.py, because the point is to reproduce that number and not a nearby one.
Every quantity is then recomputed on THOSE windows: the legacy denominator, the training-mean floor,
and the paper's own fitted ridge, against one shared zero-shot numerator.

This was written after the gate was computed and is a diagnostic, not part of the registration. It can
only make the superseded figure look worse, which is the direction that does not need protecting.

Run:  .venv-probe/bin/python scripts/legacy_m4_baseline.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from gate_all_cells import _window_linear_mse                        # noqa: E402
from gate_chronos_m4 import chronos_zs_mse                           # noqa: E402
from m4_monthly_data import HORIZON, LOOKBACK, TEST_CSV, TRAIN_CSV    # noqa: E402

OUT = ROOT / "results/chronos_m4/legacy_baseline.json"
MODEL_ID = "amazon/chronos-t5-small"
NUM_SERIES = 200
LEGACY_MIN_LEN = LOOKBACK + HORIZON      # 114: finetune_chronos_m4.load_m4_series's default
MIN_WINDOWS_FOR_OLS = 5                  # finetune_chronos_m4.py:314, `if len(X_tr) < 5: continue`
ZS_SEED = 42


def legacy_series_pairs():
    """finetune_chronos_m4.load_m4_series, reading the committed CSVs instead of re-downloading."""
    import pandas as pd
    tr = pd.read_csv(ROOT / TRAIN_CSV, index_col=0)
    te = pd.read_csv(ROOT / TEST_CSV, index_col=0)
    pairs, ids = [], []
    for idx in tr.index[:NUM_SERIES * 3]:
        train_vals = tr.loc[idx].dropna().values.astype(np.float64)
        if len(train_vals) < LEGACY_MIN_LEN:
            continue
        test_vals = te.loc[idx].dropna().values.astype(np.float64)
        if len(test_vals) < HORIZON:
            continue
        pairs.append((train_vals, test_vals))
        ids.append(str(idx))
        if len(pairs) >= NUM_SERIES:
            break
    return pairs, ids


def legacy_linear_baseline(pairs):
    """finetune_chronos_m4.linear_baseline_mse, verbatim in behaviour, instrumented.

    The instrumentation is the point: which series it silently drops, and how many training windows
    each surviving per-series OLS had against its 96 free coefficients per output step.
    """
    from sklearn.linear_model import LinearRegression
    mses, n_windows, skipped = [], [], 0
    for train_vals, test_vals in pairs:
        if len(train_vals) < LOOKBACK + HORIZON:
            skipped += 1
            continue
        X = np.array([train_vals[i:i + LOOKBACK]
                      for i in range(len(train_vals) - LOOKBACK - HORIZON + 1)])
        Y = np.array([train_vals[i + LOOKBACK:i + LOOKBACK + HORIZON]
                      for i in range(len(train_vals) - LOOKBACK - HORIZON + 1)])
        if len(X) < MIN_WINDOWS_FOR_OLS:
            skipped += 1
            continue
        mu, sd = train_vals.mean(), train_vals.std() + 1e-8
        pred = LinearRegression().fit(X, Y).predict(train_vals[-LOOKBACK:].reshape(1, -1))[0]
        mses.append(float(np.mean(((pred - mu) / sd - (test_vals[:HORIZON] - mu) / sd) ** 2)))
        n_windows.append(len(X))
    return float(np.mean(mses)), dict(
        n_series_scored=len(mses), n_series_skipped=skipped,
        windows_per_series_min=int(min(n_windows)), windows_per_series_median=int(np.median(n_windows)),
        windows_per_series_max=int(max(n_windows)), free_coefficients_per_output_step=LOOKBACK,
        n_series_with_fewer_windows_than_coefficients=int(sum(w < LOOKBACK for w in n_windows)))


def main():
    pairs, ids = legacy_series_pairs()
    hist = [len(t) for t, _ in pairs]
    ctx = np.array([t[-LOOKBACK:] for t, _ in pairs])
    tgt = np.array([e[:HORIZON] for _, e in pairs])
    # The legacy training pool: sliding windows over the WHOLE history, which is what
    # build_windows_from_series did -- it did not hold back the tail the selection window now uses.
    train_ctx, train_tgt = [], []
    for t, _ in pairs:
        for i in range(len(t) - LOOKBACK - HORIZON + 1):
            train_ctx.append(t[i:i + LOOKBACK])
            train_tgt.append(t[i + LOOKBACK:i + LOOKBACK + HORIZON])
    train = (np.array(train_ctx), np.array(train_tgt))
    print(f"legacy selection: {len(pairs)} series, history {min(hist)}..{max(hist)} "
          f"(min_len {LEGACY_MIN_LEN}), {len(train[0])} training windows")

    torch.manual_seed(ZS_SEED)
    np.random.seed(ZS_SEED)
    from chronos import ChronosPipeline
    pipe = ChronosPipeline.from_pretrained(MODEL_ID, dtype=torch.float32)
    zs = chronos_zs_mse(pipe, ctx, tgt)

    legacy_mse, diag = legacy_linear_baseline(pairs)
    const = _window_linear_mse(ctx, tgt, LOOKBACK, HORIZON, train=train,
                               baseline="constant", dataset="m4_monthly")
    fitted = _window_linear_mse(ctx, tgt, LOOKBACK, HORIZON, train=train,
                                baseline="fitted", dataset="m4_monthly")

    out = dict(
        what=("a diagnostic reproduction of the superseded 84.5% figure for Chronos/M4-Monthly, "
              "computed after the registered gate and unable to change any decision"),
        legacy_selection=dict(min_len=LEGACY_MIN_LEN, n_series=len(pairs),
                              history_min=min(hist), history_max=max(hist),
                              first_series=ids[:5], last_series=ids[-1],
                              n_train_windows=int(len(train[0]))),
        zs_mse=zs, zs_seed=ZS_SEED, zs_normalisation="per-window context mean/sd",
        legacy=dict(baseline_mse=legacy_mse, r2_task=1 - zs / legacy_mse,
                    normalisation="per-series training mean/sd (does NOT match the numerator)",
                    estimator="per-series unregularised OLS on that series' own history", **diag),
        constant_floor=dict(baseline_mse=const, r2_task=1 - zs / const),
        fitted=dict(baseline_mse=fitted, r2_task=1 - zs / fitted),
        legacy_over_floor=legacy_mse / const,
        legacy_admissible=bool(legacy_mse <= const),
    )
    print(f"  zero-shot                       {zs:.4f}")
    print(f"  legacy per-series OLS           {legacy_mse:.4f}  ->  "
          f"{100 * (1 - zs / legacy_mse):.1f}%  (the superseded figure)")
    print(f"  constant floor (training mean)  {const:.4f}  ->  {100 * (1 - zs / const):.1f}%")
    print(f"  paper's fitted ridge            {fitted:.4f}  ->  {100 * (1 - zs / fitted):.1f}%")
    print(f"  legacy denominator is {legacy_mse / const:.1f}x the floor; admissible: "
          f"{out['legacy_admissible']}")
    print(f"  per-series OLS: {diag['n_series_scored']} series scored, "
          f"{diag['n_series_skipped']} skipped for having fewer than {MIN_WINDOWS_FOR_OLS} windows; "
          f"{diag['n_series_with_fewer_windows_than_coefficients']} of the scored series had fewer "
          f"training windows than the {LOOKBACK} coefficients each output step fits "
          f"(median {diag['windows_per_series_median']})")
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
