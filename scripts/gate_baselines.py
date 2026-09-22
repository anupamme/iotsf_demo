#!/usr/bin/env python3
"""
Alternative baselines for the value gate, so the negative result does not rest on one estimator.

WHY THIS EXISTS. The gate is R2_task = 1 - MSE_zeroshot / MSE_baseline, and the paper's own
correction appendix shows the conclusion is far more sensitive to WHICH baseline is used than to
where the threshold sits: swapping a fitted ridge-OLS map for a per-window trend extrapolation moves
17 of 21 Moirai cells across the line. A negative result whose denominator comes from a single
estimator family is therefore under-evidenced -- if the ridge map is simply a strong baseline on
these series, "the checkpoint has no demonstrated advantage" would be a statement about ridge
regression rather than about the checkpoint. This module supplies seven alternatives spanning
no-training, univariate-linear, decomposition-linear, tuned-linear and two nonlinear families, so the
claim can be checked against a family rather than a point.

THE LADDER, WEAKEST FIRST. The point of ordering it is that the gate stops being a binary test and
becomes a graded one: the question is not "does the checkpoint beat the baseline" but "how far up the
ladder does its advantage survive".

  persistence      repeat the last observed value            no parameters
  seasonal_naive   repeat the last season                    no parameters
  ar               per-channel linear on own lags            L per channel
  dlinear          per-channel linear on trend+seasonal      2L per channel
  fitted           multivariate linear, lam fixed at 1e-4    (D*L) x (D*h)   <- the paper's
  ridge_tuned      multivariate linear, lam selected         (D*L) x (D*h)
  mlp              per-channel 2-layer MLP, hidden 128       nonlinear
  gbm              per-channel boosted trees                 nonlinear

CHANNEL-INDEPENDENT VS MULTIVARIATE, because it is easy to misread the ladder as monotone in strength.
`ar`, `dlinear`, `mlp` and `gbm` are channel-independent: each feature is predicted from its own lags
only. `fitted` and `ridge_tuned` are multivariate. So a channel-independent rung is not simply weaker
than a multivariate one -- it sees less information but estimates far fewer parameters, and on these
sample sizes it sometimes wins. That is why the reported quantity is the best over admissible rungs
rather than the last rung standing.

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

NO TRAINING DATA LEAKS INTO SELECTION. `ridge_tuned` and `mlp` pick their penalty on a chronological
holdout carved out of the TRAINING windows, never on the evaluation windows. Splitting chronologically
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
# M4-Monthly is monthly -> 12 steps per year. It is the one entry here that is not a sub-daily
# sampling rate, and it is also the one where the seasonal rung is genuinely competitive: the M4
# competition's own benchmark table has seasonal-naive beating several published forecasters on the
# monthly subset, so this rung is not a formality on that cell.
SEASON_OF = {
    "etth1": 24, "etth2": 24, "weather": 24, "electricity": 24, "electricity7": 24,
    "ettm2": 96, "ili": 52, "m4_monthly": 12,
}

# Weakest first, so the order here is the order the ladder is reported in. `constant` leads because it
# is the admissibility floor rather than a rung.
NAMES = ("constant", "persistence", "seasonal_naive", "ar", "dlinear", "ridge_tuned", "mlp", "gbm")

# Baselines that fit nothing, so a caller need not build TRAIN windows for them. "trend" lives here
# too although it is not in NAMES: it is the superseded per-window extrapolation, handled directly by
# gate_all_cells, and callers branch on the same question for it.
# ONE list, read by every call site. This started as five separate hardcoded ("trend",
# "seasonal_naive") tuples across gate_all_cells.py and gate_baseline_sensitivity.py; adding
# `persistence` would have had to touch all five, and the consequence of missing one is silent --
# train windows get built and then ignored, which costs time but changes no number, so nothing fails
# loudly and the next no-training rung inherits a list that is already wrong in four places.
NO_TRAINING = ("trend", "persistence", "seasonal_naive")

# Rungs that fit one model per channel from that channel's own lags. Recorded as data because the
# distinction changes how a column is read (see the CHANNEL-INDEPENDENT note above) and because
# writing it out in prose only has already let one table describe a per-feature number as pooled.
CHANNEL_INDEPENDENT = ("persistence", "seasonal_naive", "ar", "dlinear", "mlp", "gbm")

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
# 1a. Persistence -- the bottom rung, and the one every forecaster is expected to clear
# ---------------------------------------------------------------------------------------------
def persistence(ctx, horizon):
    """Hold the last observed value flat across the whole horizon. No parameters, no training.

    This is the weakest defensible forecaster, and it is in the ladder for a reason that is about
    reading the other rungs rather than about competing with them: a cell where the checkpoint fails
    to clear even persistence is a cell where the screen is telling us something about the series (a
    near-random-walk) and not about the checkpoint. Separating it from `seasonal_naive` matters
    because the two coincide only when the seasonal period is 1, and on these datasets it never is.
    """
    last = ctx[:, -1:, :]
    return np.broadcast_to(last, (ctx.shape[0], horizon, ctx.shape[2])).copy(), dict()


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
# 2b. DLinear -- the standard linear forecasting baseline, solved exactly rather than by SGD
# ---------------------------------------------------------------------------------------------
DLINEAR_KERNEL = 25          # the moving-average window in Zeng et al.'s reference implementation


def _moving_avg(x, kernel):
    """Centred moving average along time with edge replication, as in DLinear's series_decomp.

    The reference implementation pads by repeating the first and last step ((k-1)//2 on the left,
    k//2 on the right) so the trend has the same length as the input. Reproducing that padding
    matters: zero-padding instead would drag the trend towards zero at both ends of every window,
    which on a z-scored series is a visible bias in the first and last few steps.
    """
    k = min(int(kernel), x.shape[1])
    left, right = (k - 1) // 2, k // 2
    padded = np.concatenate([np.repeat(x[:, :1, :], left, axis=1), x,
                            np.repeat(x[:, -1:, :], right, axis=1)], axis=1)
    csum = np.cumsum(padded, axis=1)
    csum = np.concatenate([np.zeros_like(csum[:, :1, :]), csum], axis=1)
    return (csum[:, k:, :] - csum[:, :-k, :]) / k


def _dlinear_features(ctx, lookback, kernel):
    """(N, 2*lookback) per channel: the trend component followed by the seasonal remainder."""
    win = ctx[:, -lookback:, :]
    trend = _moving_avg(win, kernel)
    return np.concatenate([trend, win - trend], axis=1)      # (N, 2*lookback, D)


def fit_dlinear(ctx_tr, tgt_tr, lookback, kernel=DLINEAR_KERNEL, lam=1e-4, individual=False):
    """DLinear: linear maps on the trend and the seasonal remainder, summed. Solved in closed form.

    SOLVED EXACTLY, NOT BY SGD. DLinear is two linear layers whose outputs are added, and a sum of
    linear maps of trend and seasonal is one unconstrained linear map on their concatenation. So the
    ridge solve below is the exact minimiser of the published architecture's own objective, and is at
    least as strong as the original's SGD fit. That direction matters: this rung is used to argue that
    a checkpoint has no advantage, and a weakly-fitted rival would make that argument by handicap.
    Nothing here depends on a learning rate or an epoch count.

    WHAT ACTUALLY MAKES THIS RUNG DISTINCT -- and it is not the decomposition. Writing the features as
    [M x | (I-M) x] for the moving-average operator M means the design is `x` passed through the
    stacked operator [M ; I-M], which has rank exactly `lookback` (checked in
    check_gate_baselines.py). The decomposition is therefore a REPARAMETRISATION, not a wider
    function class: with `individual=True` this rung spans precisely the same functions as `ar`, and
    the two differ only in how their ridge penalties are distributed across the redundant coordinates.
    Measured, they agree to six digits on the training windows -- which is a property of the algebra,
    not a coincidence.

    Hence the default `individual=False`, which is also the published default: ONE map shared across
    all channels rather than one per channel. That is a genuinely different estimator from `ar`
    (pooled instead of per-channel, so D times fewer parameters and D times more rows) and from
    `fitted` (no cross-channel terms at all), and it is what standard DLinear benchmark numbers refer
    to. Pooling is only meaningful because the caller has already normalised every channel; on a raw
    scale a shared map would be dominated by whichever channel is largest.
    """
    Ftr = _dlinear_features(ctx_tr, lookback, kernel)
    D = ctx_tr.shape[2]
    info = dict(kernel=min(int(kernel), lookback), lam=lam, individual=individual,
                channel_independent=True, channel_shared_weights=not individual)
    if individual:
        blocks = [(Ftr[:, :, d], tgt_tr[:, :, d]) for d in range(D)]
    else:
        # Stack channels as extra ROWS: one map, fitted on D times the data.
        blocks = [(np.concatenate([Ftr[:, :, d] for d in range(D)], axis=0),
                   np.concatenate([tgt_tr[:, :, d] for d in range(D)], axis=0))]
    coefs = []
    for A, B in blocks:
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ B)
        coefs.append((W, B.mean(axis=0) - A.mean(axis=0) @ W))
    return coefs, info


def apply_dlinear(coefs, ctx, lookback, horizon, kernel=DLINEAR_KERNEL):
    """Apply per-channel maps, or a single shared map to every channel when only one was fitted."""
    F = _dlinear_features(ctx, lookback, kernel)
    if len(coefs) == 1:
        W, b = coefs[0]
        return np.stack([F[:, :, d] @ W + b for d in range(F.shape[2])], axis=2)
    return np.stack([F[:, :, d] @ W + b for d, (W, b) in enumerate(coefs)], axis=2)


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
# 3b. Two-layer MLP -- the cheap nonlinear rung between dlinear and the trees
# ---------------------------------------------------------------------------------------------
MLP_HIDDEN = (128,)
MLP_MAX_ITER = 200
MLP_ALPHA_GRID = (1e-4, 1e-2, 1e0)
# One model per channel is a full fit per feature, and Weather has 21. This caps the training windows
# so a 21-feature cell does not dominate the whole sweep; the cap is returned in the info dict and must
# be reported, because a silent cap reads as full coverage.
MLP_MAX_WINDOWS = 20_000


def fit_mlp(ctx_tr, tgt_tr, lookback, horizon, hidden=MLP_HIDDEN, alphas=MLP_ALPHA_GRID,
            max_windows=MLP_MAX_WINDOWS, frac=0.8, seed=0):
    """One multi-output MLP per channel: that channel's `lookback` lags -> its `horizon` future steps.

    Channel-independent and direct-multistep, so it is `ar` with a nonlinearity rather than a new
    forecasting design -- which is what makes the pair informative: if the gate survives `ar` but not
    this, the checkpoint's advantage over a linear model is a nonlinear one.

    The L2 penalty is selected on a CHRONOLOGICAL holdout of the training windows, the same protocol
    `ridge_tuned` uses and for the same reason: a shuffled split puts near-duplicate neighbouring
    windows on both sides and would select an over-fitted penalty. Evaluation windows are never
    touched. sklearn's own `early_stopping` is deliberately left off, because it carves its validation
    split out at random and would reintroduce exactly that leak.

    Iteration count is a fixed budget rather than a selected one; `tol`/`n_iter_no_change` stop the
    optimiser early when the loss plateaus. Whether the budget bound was hit is returned per channel
    so an under-trained baseline is visible rather than assumed away -- an under-trained rival would
    flatter the checkpoint, which is the direction this whole module exists to guard against.
    """
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPRegressor

    n_avail = len(ctx_tr)
    n = min(n_avail, max_windows)
    k = max(1, int(frac * n))
    models, chosen, hit_budget = [], [], 0
    with warnings.catch_warnings():
        # A hit iteration budget is recorded in the info dict instead; the warning is per-fit noise
        # that would otherwise bury the sweep's own output.
        warnings.simplefilter("ignore", ConvergenceWarning)
        for d in range(ctx_tr.shape[2]):
            A, B = ctx_tr[:n, -lookback:, d], tgt_tr[:n, :, d]
            best_a, best_mse = alphas[0], np.inf
            if n - k >= 1 and len(alphas) > 1:
                for a in alphas:
                    m = MLPRegressor(hidden_layer_sizes=hidden, alpha=a, max_iter=MLP_MAX_ITER,
                                     early_stopping=False, random_state=seed).fit(A[:k], B[:k])
                    mse_a = float(np.mean((m.predict(A[k:]) - B[k:]) ** 2))
                    if mse_a < best_mse:
                        best_a, best_mse = a, mse_a
            m = MLPRegressor(hidden_layer_sizes=hidden, alpha=best_a, max_iter=MLP_MAX_ITER,
                             early_stopping=False, random_state=seed).fit(A, B)
            models.append(m)
            chosen.append(best_a)
            hit_budget += int(m.n_iter_ >= MLP_MAX_ITER)
    return models, dict(hidden=list(hidden), max_iter=MLP_MAX_ITER, alpha_grid=list(alphas),
                        alpha_per_channel=chosen, n_train_windows=n, capped=n < n_avail,
                        avail_windows=int(n_avail), n_sel=n - k,
                        channels_hitting_iter_budget=hit_budget, channel_independent=True)


def apply_mlp(models, ctx, lookback, horizon):
    return np.stack([m.predict(ctx[:, -lookback:, d]).reshape(len(ctx), horizon)
                     for d, m in enumerate(models)], axis=2)


# ---------------------------------------------------------------------------------------------
# 4. Gradient-boosted trees -- the top rung, and the other nonlinear family besides the MLP
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
                  except persistence and seasonal_naive, which fit nothing.
        dataset:  dataset name, used only to look up the seasonal period.
    Returns:
        (pred (N, horizon, D), info dict) -- info is recorded alongside the number so that a cap or
        a selected penalty is visible in the results file rather than only in this source.
    """
    if name == "persistence":
        return persistence(ctx_eval, horizon)
    if name == "seasonal_naive":
        return seasonal_naive(ctx_eval, lookback, horizon, season_for(dataset))
    if train is None:
        raise ValueError(f"baseline {name!r} needs train=(ctx_tr, tgt_tr)")
    ctx_tr, tgt_tr = train
    if name == "constant":
        return constant_mean(tgt_tr, len(ctx_eval), horizon)
    if name == "ar":
        return apply_ar(fit_ar(ctx_tr, tgt_tr, lookback), ctx_eval, lookback, horizon), dict()
    if name == "dlinear":
        coefs, info = fit_dlinear(ctx_tr, tgt_tr, lookback)
        return apply_dlinear(coefs, ctx_eval, lookback, horizon), info
    if name == "ridge_tuned":
        coef, info = fit_ridge_tuned(ctx_tr, tgt_tr, lookback, horizon)
        return apply_linear_map(coef, ctx_eval, lookback, horizon), info
    if name == "mlp":
        models, info = fit_mlp(ctx_tr, tgt_tr, lookback, horizon, seed=seed)
        return apply_mlp(models, ctx_eval, lookback, horizon), info
    if name == "gbm":
        models, info = fit_gbm(ctx_tr, tgt_tr, lookback, horizon, seed=seed)
        return apply_gbm(models, ctx_eval, lookback, horizon), info
    raise ValueError(f"unknown baseline {name!r}; expected one of {NAMES}")


def mse(name, ctx_eval, tgt_eval, lookback, horizon, train=None, dataset=None, seed=0):
    """Convenience: the baseline's MSE against `tgt_eval`, plus the info dict."""
    pred, info = predict(name, ctx_eval, lookback, horizon,
                         train=train, dataset=dataset, seed=seed)
    return float(np.mean((pred - tgt_eval) ** 2)), info
