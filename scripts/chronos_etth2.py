"""Chronos-T5-Small zero-shot evaluation on ETTh2 (OT target).

Reviewer Concern 5: the "universal drift" claim needs a non-Moirai foundation model.
This script provides the gate check: Chronos zero-shot MSE vs. Linear baseline.
Full Chronos fine-tuning uses tokenised cross-entropy loss rather than NLL, which
differs materially from Moirai's training recipe; we therefore confine this
experiment to the value-gate / encoder-probe question and report honestly in the
response letter that fine-tuning parity would require protocol adaptation.
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


def chronos_zs_predict(pipe, ctx: np.ndarray, hz: int, num_samples: int = 20) -> np.ndarray:
    """Return (B, H) median forecasts for each context row."""
    preds = []
    for i in range(len(ctx)):
        series = torch.tensor(ctx[i], dtype=torch.float32)
        samples = pipe.predict(inputs=series, prediction_length=hz, num_samples=num_samples)
        # samples shape: (1, num_samples, hz) or (num_samples, hz)
        s = samples.squeeze(0) if samples.ndim == 3 else samples
        median = s.median(dim=0).values.cpu().numpy()
        preds.append(median)
    return np.asarray(preds)


def extract_encoder_reps(pipe, ctx: np.ndarray, max_samples: int = 500) -> np.ndarray:
    """Mean-pool Chronos T5 encoder hidden states over the context token sequence."""
    reps = []
    model = pipe.model.model
    encoder = model.encoder
    tokenizer = pipe.tokenizer
    device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(min(len(ctx), max_samples)):
            series = torch.tensor(ctx[i], dtype=torch.float32).unsqueeze(0)
            token_ids, attn_mask, _ = tokenizer.context_input_transform(series)
            token_ids = token_ids.to(device)
            attn_mask = attn_mask.to(device)
            out = encoder(input_ids=token_ids, attention_mask=attn_mask)
            h = out.last_hidden_state.squeeze(0).cpu().numpy()
            reps.append(h.mean(axis=0))
    return np.asarray(reps)


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
    ap.add_argument("--model-id", default="amazon/chronos-t5-small")
    ap.add_argument("--data-path", default="data/forecasting/ETTh2.csv")
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--lookback", type=int, default=192)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--max-test-windows", type=int, default=300)
    ap.add_argument("--out", default="results/v10_chronos_etth2.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from chronos import ChronosPipeline

    loader = ETTh1Loader(args.data_path, lookback_window=args.lookback, forecast_horizon=args.horizon, features="S")
    tr, va, te = loader.get_splits()
    y_tr = tr["OT"].values.astype(np.float32)
    y_va = va["OT"].values.astype(np.float32)
    y_te = te["OT"].values.astype(np.float32)
    mu = float(y_tr.mean()); sd = float(y_tr.std() + 1e-8)

    # Test windows
    Xte, Yte = build_windows(y_te, args.lookback, args.horizon)
    if len(Xte) > args.max_test_windows:
        idx = np.linspace(0, len(Xte) - 1, args.max_test_windows).astype(int)
        Xte = Xte[idx]; Yte = Yte[idx]

    pipe = ChronosPipeline.from_pretrained(args.model_id, torch_dtype=torch.float32)
    print(f"Loaded {args.model_id}")

    preds = chronos_zs_predict(pipe, Xte, args.horizon, num_samples=args.num_samples)
    pred_n = (preds - mu) / sd
    Y_n = (Yte - mu) / sd
    mse_chronos = float(np.mean((pred_n - Y_n) ** 2))

    mse_linear = linear_baseline_mse(y_tr, y_te, args.lookback, args.horizon)
    # Repeat-last
    last_n = (Xte[:, -1:].repeat(args.horizon, axis=1) - mu) / sd
    mse_last = float(np.mean((last_n - Y_n) ** 2))

    # Moirai ZS MSE on ETTh2 (from prior runs, already in paper)
    moirai_zs_etth2 = 0.126

    print(f"Chronos-T5-Small ZS MSE (h={args.horizon}): {mse_chronos:.4f}")
    print(f"Linear baseline MSE:                         {mse_linear:.4f}")
    print(f"Repeat-last MSE:                             {mse_last:.4f}")
    print(f"Moirai-Small ZS MSE (from paper):            {moirai_zs_etth2:.4f}")
    print(f"Chronos ZS advantage over Linear:            {100*(mse_linear - mse_chronos)/mse_linear:+.1f}%")

    # Encoder probe — CKA versus pre-trained is trivially 1.0 for ZS-only; here
    # we compute a Ridge R² sanity check on Chronos ZS reps to compare against
    # Moirai ZS R² (both should be near the noise floor around -6 to -7).
    reps = extract_encoder_reps(pipe, Xte, max_samples=200)
    # Fit ridge on first half, eval on second (matches existing Moirai protocol).
    # Use only the first len(reps) target windows to match probe data.
    n_probe = len(reps)
    n_half = n_probe // 2
    Y_probe = Y_n[:n_probe]
    reps_tr, reps_va = reps[:n_half], reps[n_half:n_probe]
    Y_tr_flat = Y_probe[:n_half].reshape(n_half, -1)
    Y_va_flat = Y_probe[n_half:n_probe].reshape(n_probe - n_half, -1)
    probe = Ridge(alpha=1.0).fit(reps_tr, Y_tr_flat)
    r2_chronos_zs = float(probe.score(reps_va, Y_va_flat))
    print(f"Chronos ZS Ridge R² on ETTh2 OT:             {r2_chronos_zs:.4f}")

    out = {
        "model_id": args.model_id,
        "horizon": args.horizon,
        "lookback": args.lookback,
        "seed": args.seed,
        "n_test_windows": int(len(Xte)),
        "chronos_zs_mse": mse_chronos,
        "linear_mse": mse_linear,
        "repeat_last_mse": mse_last,
        "moirai_zs_mse_paper": moirai_zs_etth2,
        "chronos_advantage_over_linear_pct": 100 * (mse_linear - mse_chronos) / mse_linear,
        "chronos_zs_ridge_r2": r2_chronos_zs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
