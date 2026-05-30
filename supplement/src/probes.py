"""Linear probes on frozen encoder representations."""

import numpy as np


def linear_probe_r2(
    reps_train: np.ndarray,
    reps_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    alpha: float = 1.0,
    probe_type: str = "ridge",
    mlp_layers: int = 1,
    gbm_depth: int = 6,
):
    """Fit a probe on frozen representations and return val R-squared.

    probe_type:
        'ridge'             - Ridge regression (default)
        'mlp'               - MLPRegressor, hidden=(64,)*mlp_layers
        'linear_forecaster' - Ridge with regularisation sweep
        'gbm'               - HistGradientBoostingRegressor
        'all'               - dict with ridge, mlp, linear_forecaster
    """
    from sklearn.linear_model import Ridge

    def _flat(x):
        return x.reshape(len(x), -1)

    reps_tr_flat = _flat(reps_train)
    reps_va_flat = _flat(reps_val)
    y_tr = y_train.reshape(len(y_train), -1)
    y_va = y_val.reshape(len(y_val), -1)

    def _fit_ridge():
        reg = Ridge(alpha=alpha).fit(reps_tr_flat, y_tr)
        return float(reg.score(reps_va_flat, y_va))

    def _fit_mlp():
        from sklearn.neural_network import MLPRegressor

        reg = MLPRegressor(
            hidden_layer_sizes=tuple([64] * int(mlp_layers)),
            max_iter=500,
            alpha=1e-3,
            random_state=0,
            early_stopping=True,
            validation_fraction=0.1,
        ).fit(reps_tr_flat, y_tr)
        return float(reg.score(reps_va_flat, y_va))

    def _fit_linear_forecaster():
        best_r2 = -np.inf
        for a in (0.01, 0.1, 1.0, 10.0, 100.0):
            reg = Ridge(alpha=a).fit(reps_tr_flat, y_tr)
            r2 = float(reg.score(reps_va_flat, y_va))
            if r2 > best_r2:
                best_r2 = r2
        return best_r2

    def _fit_gbm():
        from sklearn.ensemble import HistGradientBoostingRegressor

        preds_val = np.zeros_like(y_va)
        for j in range(y_tr.shape[1]):
            gbm = HistGradientBoostingRegressor(
                max_iter=150,
                max_depth=gbm_depth,
                learning_rate=0.05,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=0,
            )
            gbm.fit(reps_tr_flat, y_tr[:, j])
            preds_val[:, j] = gbm.predict(reps_va_flat)
        ss_res = ((preds_val - y_va) ** 2).sum()
        ss_tot = ((y_va - y_va.mean(axis=0)) ** 2).sum()
        if ss_tot < 1e-12:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    if probe_type == "ridge":
        return _fit_ridge()
    if probe_type == "mlp":
        return _fit_mlp()
    if probe_type == "linear_forecaster":
        return _fit_linear_forecaster()
    if probe_type == "gbm":
        return _fit_gbm()
    if probe_type == "all":
        return {
            "ridge": _fit_ridge(),
            "mlp": _fit_mlp(),
            "linear_forecaster": _fit_linear_forecaster(),
        }
    return {"ridge": _fit_ridge(), "mlp": _fit_mlp()}
