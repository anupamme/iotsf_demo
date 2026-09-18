#!/usr/bin/env python3
"""
Linear-baseline gate for the Moirai cells: R2_task = 1 - MSE_ZS / MSE_Linear.

WHY THIS EXISTS
---------------
finetune_forecasting.py never computes a linear baseline, so no Moirai result file records the
gate. That gap matters now: a "degradation" claim requires a demonstrated pretrained capability to
degrade. If a cell gate-fails, full fine-tuning being worse than zero-shot is benign replacement of
features that were not valuable; only in a gate-passing cell is it damage.

The baseline is matched to the evaluation the runs use: same eval windows, same extended lookback,
and MSE taken on the train-normalised scale (evaluate_forecasting normalises predictions and
targets by train mean/std before squaring), so the numbers are comparable to the stored
`zeroshot_mse` / `test_mse`.

Run:  python3 scripts/gate_linear_baseline.py --data Weather --horizon 96
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")


def fit_linear_map(ctx_tr, tgt_tr, lookback, lam=1e-4):
    """Fit the lookback-`lookback` linear regression the gate is DEFINED as.

    THIS IS THE BASELINE THE PAPER SPECIFIES and `linear_forecast` below is not: a single ridge-OLS
    map R^(lookback*D) -> R^(horizon*D) estimated ONCE on the training windows, then applied
    unchanged out of sample. `linear_forecast` instead re-fits a slope and intercept inside each
    evaluation window and extrapolates, using no training data at all.

    The distinction is load-bearing, not stylistic. On ETTh1 h=96 (300 matched test windows,
    train-normalised) the two denominators are 0.4145 (fitted) against 1.2207 (trend); predicting
    the training mean scores 0.9937. A per-window trend extrapolation is therefore BEATEN BY A
    CONSTANT on the ETT and Weather cells, so a gate computed against it cannot evidence "the
    pre-trained model has a capability here worth preserving" -- which is the sole premise the
    degradation reading rests on. Correcting the estimator moves 16 of 21 Moirai cells from
    gate-pass to gate-fail. See the paper's threshold appendix.

    Args:
        ctx_tr: (N, >=lookback, D) training contexts; only the LAST `lookback` steps are used, so
                an extended-lookback window set can be passed in unchanged.
        tgt_tr: (N, horizon, D) training targets.
    Returns:
        (W, b) suitable for `apply_linear_map`.
    """
    A = ctx_tr[:, -lookback:, :].reshape(len(ctx_tr), -1)
    B = tgt_tr.reshape(len(tgt_tr), -1)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ B)
    b = B.mean(axis=0) - A.mean(axis=0) @ W
    return W, b


def apply_linear_map(coef, ctx, lookback, horizon):
    """Apply a `fit_linear_map` result to (N, >=lookback, D) contexts -> (N, horizon, D)."""
    W, b = coef
    A = ctx[:, -lookback:, :].reshape(len(ctx), -1)
    return (A @ W + b).reshape(len(ctx), horizon, ctx.shape[2])


def linear_forecast(ctx, horizon, lookback=96):
    """Least-squares linear extrapolation from the last `lookback` steps, per window per feature.

    RETAINED FOR THE AUDIT TRAIL ONLY -- this is the estimator the gate was computed against
    through 26 Aug 2026, and the paper reports the correction. It is NOT the baseline the paper
    defines; use `fit_linear_map` for that, and see its docstring for why the difference matters.
    """
    win = ctx[:, -lookback:, :]                       # (N, lookback, D)
    t = np.arange(lookback)
    tt = np.stack([t, np.ones_like(t)], 1)            # (lookback, 2)
    coef, *_ = np.linalg.lstsq(tt, win.transpose(1, 0, 2).reshape(lookback, -1), rcond=None)
    future = np.stack([np.arange(lookback, lookback + horizon),
                       np.ones(horizon)], 1)          # (horizon, 2)
    pred = (future @ coef).reshape(horizon, win.shape[0], win.shape[2]).transpose(1, 0, 2)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Weather")
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--lookback", type=int, default=96)
    ap.add_argument("--max-eval", type=int, default=300)
    a = ap.parse_args()

    from src.data.forecasting_loader import get_forecasting_loader
    loader = get_forecasting_loader(f"data/forecasting/{a.data}.csv",
                                    lookback_window=a.lookback, forecast_horizon=a.horizon,
                                    features="M")
    train_df, val_df, test_df = loader.get_splits()
    cols = loader.FEATURE_COLUMNS
    tr = train_df[cols].values
    mu, sd = tr.mean(axis=0), tr.std(axis=0) + 1e-8
    ext_lb = a.lookback * 2                            # runs use an extended lookback for inference

    for name, df in (("val", val_df), ("test", test_df)):
        vals = df[cols].values
        total = ext_lb + a.horizon
        X = np.array([vals[i:i + ext_lb] for i in range(len(vals) - total + 1)])[:a.max_eval]
        y = np.array([vals[i + ext_lb:i + total] for i in range(len(vals) - total + 1)])[:a.max_eval]
        if len(X) == 0:
            print(f"  {name}: no windows"); continue
        pred = linear_forecast(X, a.horizon, a.lookback)
        mse = float(np.mean(((pred - mu) / sd - (y - mu) / sd) ** 2))
        print(f"  {a.data} h={a.horizon} {name}: linear MSE = {mse:.4f}  over {len(X)} windows")


if __name__ == "__main__":
    main()
