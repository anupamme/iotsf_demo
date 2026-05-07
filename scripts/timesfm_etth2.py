"""TimesFM-2.5-200M zero-shot gate check on ETTh2 (OT target).

V11 reviewer concern 1 / Q1: the drift-utility dissociation rests on Moirai only.
This script provides the gate check: TimesFM zero-shot MSE vs. Linear baseline,
matched to the Chronos protocol (lookback 192, horizon 96, 300 test windows,
seed 42, univariate OT target, per-feature z-score normalisation consistent with
Moirai evaluation).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge

from src.data.forecasting_loader import ETTh1Loader


def build_windows(arr, lb, hz):
    X, Y = [], []
    total = lb + hz
    for i in range(len(arr) - total + 1):
        X.append(arr[i : i + lb])
        Y.append(arr[i + lb : i + total])
    return np.asarray(X), np.asarray(Y)


def timesfm_zs_predict(model, ctx: np.ndarray, hz: int) -> np.ndarray:
    """Return (B, H) point forecasts. TimesFM.forecast returns (point, quantiles)."""
    inputs = [ctx[i].astype(np.float32) for i in range(len(ctx))]
    point, _ = model.forecast(horizon=hz, inputs=inputs)
    return np.asarray(point)


def linear_baseline_mse(y_tr, y_te, lb, hz):
    Xtr, Ytr = build_windows(y_tr, lb, hz)
    Xte, Yte = build_windows(y_te, lb, hz)
    if len(Xtr) > 50000:
        idx = np.random.RandomState(42).choice(len(Xtr), 50000, replace=False)
        Xtr = Xtr[idx]; Ytr = Ytr[idx]
    mu = y_tr.mean(); sd = y_tr.std() + 1e-8
    reg = LinearRegression().fit(Xtr, Ytr)
    pred = reg.predict(Xte)
    pred_n = (pred - mu) / sd
    Y_n = (Yte - mu) / sd
    return float(np.mean((pred_n - Y_n) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/timesfm-2.5-200m-pytorch")
    ap.add_argument("--data-path", default="data/forecasting/ETTh2.csv")
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--lookback", type=int, default=192)
    ap.add_argument("--max-test-windows", type=int, default=300)
    ap.add_argument("--out", default="results/v11_timesfm_etth2.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    import timesfm

    loader = ETTh1Loader(args.data_path, lookback_window=args.lookback,
                        forecast_horizon=args.horizon, features="S")
    tr, va, te = loader.get_splits()
    y_tr = tr["OT"].values.astype(np.float32)
    y_te = te["OT"].values.astype(np.float32)
    mu = float(y_tr.mean()); sd = float(y_tr.std() + 1e-8)

    Xte, Yte = build_windows(y_te, args.lookback, args.horizon)
    if len(Xte) > args.max_test_windows:
        idx = np.linspace(0, len(Xte) - 1, args.max_test_windows).astype(int)
        Xte = Xte[idx]; Yte = Yte[idx]

    print(f"Loading {args.model_id}...")
    # huggingface_hub passes `proxies` as a leftover kwarg into the ctor on some
    # versions; sidestep by pointing at the already-cached snapshot directory.
    local_snapshot = os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--timesfm-2.5-200m-pytorch/"
        "snapshots/1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
    )
    if os.path.isdir(local_snapshot):
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(local_snapshot)
    else:
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(args.model_id)
    fc = timesfm.ForecastConfig(
        max_context=args.lookback,
        max_horizon=args.horizon,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,
        fix_quantile_crossing=True,
    )
    model.compile(fc)
    print(f"Model loaded and compiled")

    preds = timesfm_zs_predict(model, Xte, args.horizon)
    # preds may be longer than horizon if max_horizon rounded up; trim
    preds = preds[:, : args.horizon]
    pred_n = (preds - mu) / sd
    Y_n = (Yte - mu) / sd
    mse_timesfm = float(np.mean((pred_n - Y_n) ** 2))

    mse_linear = linear_baseline_mse(y_tr, y_te, args.lookback, args.horizon)
    last_n = (Xte[:, -1:].repeat(args.horizon, axis=1) - mu) / sd
    mse_last = float(np.mean((last_n - Y_n) ** 2))

    moirai_zs_etth2 = 0.126  # from paper, Moirai-Small h=96
    chronos_zs_etth2 = 0.304  # from paper, Chronos-T5-Small h=96

    adv_over_linear = 100 * (mse_linear - mse_timesfm) / mse_linear
    gate_status = "PASS" if adv_over_linear >= 20 else ("MARGINAL" if adv_over_linear >= 5 else "FAIL")

    print(f"TimesFM-2.5-200M ZS MSE (h={args.horizon}): {mse_timesfm:.4f}")
    print(f"Linear baseline MSE:                         {mse_linear:.4f}")
    print(f"Repeat-last MSE:                             {mse_last:.4f}")
    print(f"Moirai-Small ZS MSE (from paper):            {moirai_zs_etth2:.4f}")
    print(f"Chronos-T5-Small ZS MSE (from paper):        {chronos_zs_etth2:.4f}")
    print(f"TimesFM ZS advantage over Linear:            {adv_over_linear:+.1f}%")
    print(f"Pre-training value gate:                     {gate_status}")

    out = {
        "model_id": args.model_id,
        "horizon": args.horizon,
        "lookback": args.lookback,
        "seed": args.seed,
        "n_test_windows": int(len(Xte)),
        "timesfm_zs_mse": mse_timesfm,
        "linear_mse": mse_linear,
        "repeat_last_mse": mse_last,
        "moirai_zs_mse_paper": moirai_zs_etth2,
        "chronos_zs_mse_paper": chronos_zs_etth2,
        "timesfm_advantage_over_linear_pct": adv_over_linear,
        "gate_status": gate_status,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
