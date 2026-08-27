"""MOMENT-1-base gate check on ETTh2 OT.

MOMENT paper/docs: "Only reconstruction head is pre-trained. Forecasting head
must be fine-tuned." Therefore the honest gate analogue is:
  (a) frozen-encoder linear probe on ETTh2 OT at matched protocol (lookback=512
      as MOMENT expects; reshape to match); Ridge regression readout.
  (b) Lightweight forecasting-head fine-tune (MOMENT default recipe) at n=500
      seed 42, comparing to Linear baseline 0.213 on OT.

Output: results/v12_moment_etth2.json with ZS-linear-probe MSE, FT MSE, and
CKA pre/post FT encoder.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/tmp/moment_repo")

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge
from torch.utils.data import DataLoader, TensorDataset

from src.data.forecasting_loader import ETTh1Loader


MOMENT_SEQ_LEN = 512


def build_windows(arr, lb, hz):
    X, Y = [], []
    total = lb + hz
    for i in range(len(arr) - total + 1):
        X.append(arr[i : i + lb])
        Y.append(arr[i + lb : i + total])
    return np.asarray(X), np.asarray(Y)


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


def linear_CKA(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = float(np.sum((X.T @ Y) ** 2))
    hsic_xx = float(np.sum((X.T @ X) ** 2))
    hsic_yy = float(np.sum((Y.T @ Y) ** 2))
    denom = (hsic_xx * hsic_yy) ** 0.5
    return hsic_xy / denom if denom > 0 else 0.0


def extract_embeddings(model, X, device, batch_size=16):
    """Run MOMENT embedding task on (N, seq_len) windows.

    Returns (N, d_model) mean-pooled encoder embeddings.
    """
    reps = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).float().to(device)
            # (B, 1, seq_len)
            xb = xb.unsqueeze(1)
            input_mask = torch.ones(xb.shape[0], xb.shape[-1], device=device)
            out_em = model(x_enc=xb, input_mask=input_mask, reduction="mean")
            emb = out_em.embeddings.detach().cpu().numpy()
            reps.append(emb)
    return np.concatenate(reps, axis=0)


def forecast_with_model(model, X, device, batch_size=16):
    """Run MOMENT forecast on (N, seq_len) windows; returns (N, horizon)."""
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).float().to(device)
            xb = xb.unsqueeze(1)  # (B, 1, seq_len)
            input_mask = torch.ones(xb.shape[0], xb.shape[-1], device=device)
            out_fc = model(x_enc=xb, input_mask=input_mask)
            pred = out_fc.forecast.squeeze(1).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="AutonLab/MOMENT-1-base")
    ap.add_argument("--data-path", default="data/forecasting/ETTh2.csv")
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--max-test-windows", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-train-samples", type=int, default=500)
    ap.add_argument("--out", default="results/v12_moment_etth2.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--mode", default="full", choices=["gate", "full"],
                    help="gate: ZS probe only; full: gate + FT at n=500")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    # Data: univariate OT, use MOMENT_SEQ_LEN=512 lookback
    loader = ETTh1Loader(
        args.data_path, lookback_window=MOMENT_SEQ_LEN,
        forecast_horizon=args.horizon, features="S",
    )
    tr, va, te = loader.get_splits()
    y_tr = tr["OT"].values.astype(np.float32)
    y_va = va["OT"].values.astype(np.float32)
    y_te = te["OT"].values.astype(np.float32)
    mu = float(y_tr.mean()); sd = float(y_tr.std() + 1e-8)

    Xtr, Ytr = build_windows(y_tr, MOMENT_SEQ_LEN, args.horizon)
    Xva, Yva = build_windows(y_va, MOMENT_SEQ_LEN, args.horizon)
    Xte, Yte = build_windows(y_te, MOMENT_SEQ_LEN, args.horizon)
    if len(Xte) > args.max_test_windows:
        idx = np.linspace(0, len(Xte) - 1, args.max_test_windows).astype(int)
        Xte = Xte[idx]; Yte = Yte[idx]

    # Normalised targets for MSE reporting
    Y_te_n = (Yte - mu) / sd

    # Subsample training
    rng = np.random.RandomState(args.seed)
    if args.max_train_samples > 0 and args.max_train_samples < len(Xtr):
        sel = rng.choice(len(Xtr), args.max_train_samples, replace=False)
        Xtr_sub = Xtr[sel]; Ytr_sub = Ytr[sel]
    else:
        Xtr_sub = Xtr; Ytr_sub = Ytr

    # Baselines
    mse_linear = linear_baseline_mse(y_tr, y_te, 96, args.horizon)  # lb=96 to match Moirai protocol
    print(f"Linear baseline MSE (lb=96 vs MOMENT lb=512):  {mse_linear:.4f}")

    # Load MOMENT forecasting model
    from momentfm import MOMENTPipeline
    print("Loading MOMENT...")
    model = MOMENTPipeline.from_pretrained(
        args.model_id,
        model_kwargs={"task_name": "forecasting", "forecast_horizon": args.horizon,
                      "head_dropout": 0.1, "freeze_encoder": False, "freeze_embedder": False,
                      "freeze_head": False},
    )
    model.init()
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MOMENT loaded: {n_params:,} params, device={device}")

    # Save pre-trained state for CKA and FT reset
    pretrained_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # ----- Gate check: linear probe on frozen embeddings -----
    print("Extracting ZS embeddings...")
    model_embed = MOMENTPipeline.from_pretrained(
        args.model_id, model_kwargs={"task_name": "embedding"},
    )
    model_embed.init()
    model_embed = model_embed.to(device)

    probe_train_n = min(600, len(Xtr_sub))
    probe_test_n = len(Xte)
    sel_probe = rng.choice(len(Xtr_sub), probe_train_n, replace=False) if probe_train_n < len(Xtr_sub) else np.arange(len(Xtr_sub))
    Xprobe_tr = Xtr_sub[sel_probe]
    Yprobe_tr = Ytr_sub[sel_probe]

    pt_reps_tr = extract_embeddings(model_embed, Xprobe_tr, device)
    pt_reps_te = extract_embeddings(model_embed, Xte, device)
    print(f"PT embeddings: train={pt_reps_tr.shape}, test={pt_reps_te.shape}")

    # Ridge probe (frozen embedding → horizon target)
    y_tr_flat = Yprobe_tr.reshape(len(Yprobe_tr), -1)
    y_te_flat = Yte.reshape(len(Yte), -1)
    probe_ridge = Ridge(alpha=1.0).fit(pt_reps_tr, y_tr_flat)
    probe_pred = probe_ridge.predict(pt_reps_te)
    probe_pred_n = (probe_pred - mu) / sd
    mse_probe_zs = float(np.mean((probe_pred_n - Y_te_n) ** 2))
    r2_zs = float(probe_ridge.score(pt_reps_te, y_te_flat))
    print(f"MOMENT ZS-probe MSE: {mse_probe_zs:.4f}  (R²={r2_zs:.4f})")
    print(f"Gate vs Linear 0.213:  {100*(0.213 - mse_probe_zs)/0.213:+.1f}%")

    result = {
        "model_id": args.model_id,
        "horizon": args.horizon,
        "seed": args.seed,
        "device": device,
        "n_test_windows": int(len(Xte)),
        "linear_baseline_mse_lb96": mse_linear,
        "moment_zs_probe_mse": mse_probe_zs,
        "moment_zs_probe_r2": r2_zs,
        "gate_pass_vs_linear": mse_probe_zs < 0.213,
        "gate_margin_pct": 100 * (0.213 - mse_probe_zs) / 0.213,
    }

    if args.mode == "gate":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"saved: {args.out}")
        return

    # ----- Forecasting-head fine-tune at n=500 -----
    print(f"\nFine-tuning MOMENT forecasting head + encoder at n={args.max_train_samples}, {args.epochs} epochs...")
    # Convert to tensors
    X_tr_t = torch.from_numpy(Xtr_sub).float()
    Y_tr_t = torch.from_numpy(Ytr_sub).float()
    ds = TensorDataset(X_tr_t, Y_tr_t)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = torch.nn.MSELoss()

    history = {"train_loss": [], "val_mse": []}
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in dl:
            xb = xb.to(device).unsqueeze(1)  # (B, 1, 512)
            yb = yb.to(device).unsqueeze(1)  # (B, 1, H)
            input_mask = torch.ones(xb.shape[0], xb.shape[-1], device=device)
            out = model(x_enc=xb, input_mask=input_mask)
            pred = out.forecast  # (B, 1, H)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        train_loss = float(np.mean(losses))

        # Eval
        preds = forecast_with_model(model, Xte, device)
        preds_n = (preds - mu) / sd
        val_mse = float(np.mean((preds_n - Y_te_n) ** 2))
        history["train_loss"].append(train_loss)
        history["val_mse"].append(val_mse)
        print(f"  epoch {epoch+1}: train_loss={train_loss:.4f}  val_MSE={val_mse:.4f}")

    # Final MSE
    preds_final = forecast_with_model(model, Xte, device)
    preds_final_n = (preds_final - mu) / sd
    mse_ft = float(np.mean((preds_final_n - Y_te_n) ** 2))

    # CKA (pre-trained vs fine-tuned encoder) — extract embeddings on test windows
    ft_reps_te = extract_embeddings(model_embed, Xte, device)  # still pretrained
    # Actually we need FT embedding model; reload embedding model, then copy FT state into it
    ft_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model_embed_ft = MOMENTPipeline.from_pretrained(
        args.model_id, model_kwargs={"task_name": "embedding"},
    )
    model_embed_ft.init()
    model_embed_ft = model_embed_ft.to(device)
    # load overlapping keys (encoder backbone only)
    ft_state_filtered = {k: v for k, v in ft_state.items() if k in model_embed_ft.state_dict()}
    model_embed_ft.load_state_dict(ft_state_filtered, strict=False)

    ft_reps_te = extract_embeddings(model_embed_ft, Xte, device)
    cka = linear_CKA(pt_reps_te, ft_reps_te)
    print(f"\nFT MSE: {mse_ft:.4f}  (ZS-probe {mse_probe_zs:.4f}, Linear 0.213)")
    print(f"CKA (PT vs FT encoder): {cka:.4f}")
    forgetting = 100 * (mse_ft - mse_probe_zs) / mse_probe_zs

    result.update({
        "moment_ft_mse": mse_ft,
        "moment_ft_vs_linear_pct": 100 * (0.213 - mse_ft) / 0.213,
        "moment_ft_cka_vs_pretrained": cka,
        "forgetting_pct_vs_probe_zs": forgetting,
        "epochs": args.epochs,
        "max_train_samples": args.max_train_samples,
        "history": history,
    })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
