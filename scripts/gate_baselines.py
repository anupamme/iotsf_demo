#!/usr/bin/env python3
"""
Alternative baselines for the value gate, so the negative result does not rest on one estimator.

WHY THIS EXISTS. The gate is R2_task = 1 - MSE_zeroshot / MSE_baseline, and the paper's own
correction appendix shows the conclusion is far more sensitive to WHICH baseline is used than to
where the threshold sits: swapping a fitted ridge-OLS map for a per-window trend extrapolation moves
17 of 21 Moirai cells across the line. A negative result whose denominator comes from a single
estimator family is therefore under-evidenced -- if the ridge map is simply a strong baseline on
these series, "the checkpoint has no demonstrated advantage" would be a statement about ridge
regression rather than about the checkpoint. This module supplies four alternatives spanning
no-training, univariate-linear, tuned-linear and nonlinear, so the claim can be checked against a
family rather than a point.

AND ONE THING THAT IS NOT AN ALTERNATIVE. `constant` -- the training-mean predictor -- is a FLOOR, not
a rival. Any denominator worse than it makes R2_task meaningless in the direction that flatters the
checkpoint, which is the precise defect the trend baseline had. It is computed for every cell so that
each column can be read as admissible or not instead of being trusted because it was published.

SCALE DISCIPLINE. Every function here takes arrays ALREADY ON THE CALLER'S SCALE and returns
predictions on that same scale. The two arms of the gate normalise differently -- moirai_gates()
works on the train-normalised scale the stored zeroshot_mse is measured on, while the Chronos and
TimesFM arms z-score each window by its own context -- and a baseline computed on the wrong scale
produces a denominator that is not comparable to the numerator. Keeping normalisation entirely in
the callers is what stops that from happening silently.

NO TRAINING DATA LEAKS INTO SELECTION. `ridge_tuned` picks its penalty on a chronological holdout
carved out of the TRAINING windows, never on the evaluation windows. Splitting chronologically
rather than at random matters for series data: a shuffled split lets near-duplicate neighbouring
windows sit on both sides and would select an over-fitted penalty.
"""
import numpy as np

from gate_linear_baseline import apply_linear_map, fit_linear_map

# Seasonal period in TIME STEPS per dataset, hardcoded rather than inferred. Inferring a period from
# an autocorrelation peak would silently pick 1 on the flatter series and quietly turn the seasonal
# baseline into a persistence baseline, which is a different estimator making a different claim.
# ETT hourly and Weather/Electricity hourly -> 24. ETTm2 is 15-minute -> 96 steps per day. ILI is
# weekly -> 52.
SEASON_OF = {
    "etth1": 24, "etth2": 24, "weather": 24, "electricity": 24, "electricity7": 24,
    "ettm2": 96, "ili": 52,
}

NAMES = ("constant", "seasonal_naive", "ar", "ridge_tuned", "gbm")

# HistGradientBoosting is expanded to one row per (window, horizon step), so the design matrix grows
# by a factor of the horizon. This caps total rows; the cap is returned in the info dict and must be
# reported, because a silent cap reads as full coverage.
GBM_MAX_ROWS = 200_000
# Thread count is the caller's job, via OMP_NUM_THREADS -- HistGradientBoosting reads it directly and
# there is no constructor argument for it. This matters in practice: these runs share the machine with
# an MPS fine-tuning job, and an unbounded OpenMP pool takes all ten cores and starves it.


def season_for(dataset):
    """Seasonal period for a dataset name, case-insensitively. Raises rather than guessing."""
    k = str(dataset).lower()
    if k not in SEASON_OF:
        raise KeyError(f"no seasonal period declared for dataset {dataset!r}; "
                       f"add it to SEASON_OF rather than letting it default")
    return SEASON_OF[k]


# ---------------------------------------------------------------------------------------------
# 0. Training-mean constant -- not a competitor, an admissibility floor
# ---------------------------------------------------------------------------------------------
def constant_mean(tgt_tr, n_eval, horizon):
    """Per-feature mean of the TRAINING targets, held flat across the horizon.

    This is not offered as a rival forecaster. It is the floor a denominator must clear before
    R2_task means anything: if MSE_baseline exceeds this, the gate is dividing by the error of an
    estimator worse than answering "the average", and a checkpoint can clear the threshold without
    beating anything at all. That failure is not hypothetical in this project -- it is exactly what
    the per-window trend extrapolation did through 26 Aug 2026, and it is why 17 of 21 Moirai cells
    changed gate status when it was replaced. Carrying the floor as a column turns the check that
    caught that error into a standing criterion rather than a one-off audit.
    """
    mu = tgt_tr.reshape(-1, tgt_tr.shape[2]).mean(axis=0)
    return (np.broadcast_to(mu, (int(n_eval), horizon, len(mu))).copy(),
            dict(train_windows=int(len(tgt_tr))))


# ---------------------------------------------------------------------------------------------
# 1. Seasonal naive -- no training at all
# ---------------------------------------------------------------------------------------------
def seasonal_naive(ctx, lookback, horizon, season):
    """Repeat the last `season` observed steps, tiled forward to `horizon`.

    The strongest baseline that fits no parameters. If the checkpoint cannot beat this, the cell
    carries no advantage that any amount of estimator tuning could be accused of manufacturing.
    """
    win = ctx[:, -lookback:, :]
    s = min(int(season), win.shape[1])                 # clamp: cannot repeat more than we observed
    last = win[:, -s:, :]
    reps = -(-horizon // s)                            # ceil division
    return np.tile(last, (1, reps, 1))[:, :horizon, :], dict(season=s, clamped=s != int(season))


# ---------------------------------------------------------------------------------------------
# 2. Per-feature autoregression -- the univariate sibling of the paper's map
# ---------------------------------------------------------------------------------------------
def fit_ar(ctx_tr, tgt_tr, lookback, lam=1e-4):
    """One direct-multistep AR map per feature, using only that feature's own lags.

    The paper's baseline is a single MULTIVARIATE map: every feature's lags predict every feature's
    future. This strips the cross-feature terms. Comparing the two isolates how much of the gate is
    carried by cross-channel information rather than by linear extrapolation per channel -- if the
    gate survives here but not there, the checkpoint's advantage is a multivariate one.
    """
    coefs = []
    for d in range(ctx_tr.shape[2]):
        A = ctx_tr[:, -lookback:, d]
        B = tgt_tr[:, :, d]
        W = np.linalg.solve(A.T @ A + lam * np.eye(lookback), A.T @ B)
        b = B.mean(axis=0) - A.mean(axis=0) @ W
        coefs.append((W, b))
    return coefs


def apply_ar(coefs, ctx, lookback, horizon):
    return np.stack([ctx[:, -lookback:, d] @ W + b for d, (W, b) in enumerate(coefs)], axis=2)


# ---------------------------------------------------------------------------------------------
# 3. Tuned ridge -- the paper's own map, but with the penalty selected instead of fixed
# ---------------------------------------------------------------------------------------------
LAM_GRID = (1e-6, 1e-4, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)


def fit_ridge_tuned(ctx_tr, tgt_tr, lookback, horizon, lams=LAM_GRID, frac=0.8):
    """fit_linear_map with lam chosen on a chronological holdout of the training windows.

    The paper fixes lam=1e-4. A reviewer can reasonably ask whether the baseline was given a fair
    hyperparameter search before being declared beaten -- or, in the direction that matters here,
    whether a better-tuned baseline would beat the checkpoint on cells it currently loses. This
    answers that without touching the evaluation windows.
    """
    n = len(ctx_tr)
    k = max(1, int(frac * n))
    if n - k < 1:                                      # too few windows to select on; keep default
        return fit_linear_map(ctx_tr, tgt_tr, lookback), dict(lam=1e-4, selected=False, n_sel=0)
    best_lam, best_mse = None, np.inf
    for lam in lams:
        coef = fit_linear_map(ctx_tr[:k], tgt_tr[:k], lookback, lam=lam)
        pred = apply_linear_map(coef, ctx_tr[k:], lookback, horizon)
        mse = float(np.mean((pred - tgt_tr[k:]) ** 2))
        if mse < best_mse:
            best_lam, best_mse = lam, mse
    # Refit on ALL training windows at the selected penalty.
    return (fit_linear_map(ctx_tr, tgt_tr, lookback, lam=best_lam),
            dict(lam=best_lam, selected=True, n_sel=n - k, sel_mse=best_mse))


# ---------------------------------------------------------------------------------------------
# 4. Gradient-boosted trees -- the only nonlinear alternative
# ---------------------------------------------------------------------------------------------
def fit_gbm(ctx_tr, tgt_tr, lookback, horizon, max_rows=GBM_MAX_ROWS, seed=0):
    """One HistGradientBoostingRegressor per feature over (that feature's lags, horizon step).

    One model per (feature, horizon step) would be D*horizon fits -- 672 on a 7-feature h=96 cell,
    which is not worth the wall-clock. Instead the horizon step enters as an input feature and one
    model covers all steps of a feature. Rows are (windows * horizon), so the window count is capped
    at max_rows // horizon; the effective cap is returned and must be reported.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    D = ctx_tr.shape[2]
    cap = max(1, max_rows // horizon)
    n = min(len(ctx_tr), cap)
    step = np.tile(np.arange(horizon), n)[:, None]
    models = []
    for d in range(D):
        A = np.repeat(ctx_tr[:n, -lookback:, d], horizon, axis=0)
        Xf = np.hstack([A, step])
        yf = tgt_tr[:n, :, d].reshape(-1)
        models.append(HistGradientBoostingRegressor(random_state=seed).fit(Xf, yf))
    return models, dict(n_train_windows=n, capped=n < len(ctx_tr),
                        avail_windows=int(len(ctx_tr)), rows_per_feature=n * horizon)


def apply_gbm(models, ctx, lookback, horizon):
    n = len(ctx)
    step = np.tile(np.arange(horizon), n)[:, None]
    out = np.empty((n, horizon, len(models)))
    for d, m in enumerate(models):
        A = np.repeat(ctx[:, -lookback:, d], horizon, axis=0)
        out[:, :, d] = m.predict(np.hstack([A, step])).reshape(n, horizon)
    return out


# ---------------------------------------------------------------------------------------------
# Dispatcher used by both arms of the gate
# ---------------------------------------------------------------------------------------------
def predict(name, ctx_eval, lookback, horizon, train=None, dataset=None, seed=0):
    """Predictions for `ctx_eval` under baseline `name`, on whatever scale the caller passed in.

    Args:
        ctx_eval: (N, >=lookback, D) evaluation contexts.
        train:    (ctx_tr, tgt_tr) training windows on the SAME scale. Required by every baseline
                  except seasonal_naive, which fits nothing.
        dataset:  dataset name, used only to look up the seasonal period.
    Returns:
        (pred (N, horizon, D), info dict) -- info is recorded alongside the number so that a cap or
        a selected penalty is visible in the results file rather than only in this source.
    """
    if name == "seasonal_naive":
        return seasonal_naive(ctx_eval, lookback, horizon, season_for(dataset))
    if train is None:
        raise ValueError(f"baseline {name!r} needs train=(ctx_tr, tgt_tr)")
    ctx_tr, tgt_tr = train
    if name == "constant":
        return constant_mean(tgt_tr, len(ctx_eval), horizon)
    if name == "ar":
        return apply_ar(fit_ar(ctx_tr, tgt_tr, lookback), ctx_eval, lookback, horizon), dict()
    if name == "ridge_tuned":
        coef, info = fit_ridge_tuned(ctx_tr, tgt_tr, lookback, horizon)
        return apply_linear_map(coef, ctx_eval, lookback, horizon), info
    if name == "gbm":
        models, info = fit_gbm(ctx_tr, tgt_tr, lookback, horizon, seed=seed)
        return apply_gbm(models, ctx_eval, lookback, horizon), info
    raise ValueError(f"unknown baseline {name!r}; expected one of {NAMES}")


def mse(name, ctx_eval, tgt_eval, lookback, horizon, train=None, dataset=None, seed=0):
    """Convenience: the baseline's MSE against `tgt_eval`, plus the info dict."""
    pred, info = predict(name, ctx_eval, lookback, horizon,
                         train=train, dataset=dataset, seed=seed)
    return float(np.mean((pred - tgt_eval) ** 2)), info
