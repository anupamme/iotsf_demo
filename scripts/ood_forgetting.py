#!/usr/bin/env python3
"""
OOD Forgetting Experiment — Workshop Paper Revision.

Fine-tune Moirai-Small on ETTh2 (n=10k, h=96, conditions B and D), then
evaluate forecasting MSE on ETTh1 and ETTm2 (OOD datasets never seen during
fine-tuning). Tests whether "beneficial specialization" on ETTh2 comes at the
cost of degrading cross-dataset temporal capabilities.

Protocol:
  Condition B: full fine-tune (encoder + MSE head)
  Condition D: frozen encoder, head only

For each condition × dataset:
  ZS MSE  = pretrained encoder + Ridge head trained on ETTh2 train, evaluated on dataset test
  FT MSE  = fine-tuned encoder + MSE head evaluated on dataset test
  forgetting% = (FT_MSE / ZS_MSE - 1) × 100  (positive = worse, negative = better)

Usage:
    python ood_forgetting.py [--seeds 42 123 202] [--results-dir results/ood_forgetting]
"""

import argparse
import copy
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

# ── Constants ──────────────────────────────────────────────────────────────────
LOOKBACK = 96
HORIZON  = 96
PAST_LEN = LOOKBACK + HORIZON  # 192 — MoiraiForecast extended lookback
N_TRAIN  = 10_000
BATCH    = 64
MAX_EPOCHS  = 15
LR          = 1e-4
PATIENCE    = 3
D_MODEL     = 384  # Moirai-Small


# ── Data ───────────────────────────────────────────────────────────────────────

ETT_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}


def download_ett(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in ETT_URLS.items():
        p = data_dir / f"{name}.csv"
        if not p.exists():
            print(f"  Downloading {name}...")
            urllib.request.urlretrieve(url, p)


def load_ett_splits(csv_path: Path):
    """Return (train_series, val_series, test_series) as float32 arrays."""
    df = pd.read_csv(csv_path)
    s = df["OT"].values.astype(np.float32)
    n = len(s)
    return s[:int(n * 0.6)], s[int(n * 0.6):int(n * 0.8)], s[int(n * 0.8):]


def make_windows(series: np.ndarray, lookback: int, horizon: int, max_n: int = None):
    """Sliding windows → (contexts, targets), each (N, L) and (N, H)."""
    W = lookback + horizon
    ctx, tgt = [], []
    for i in range(len(series) - W + 1):
        ctx.append(series[i:i + lookback])
        tgt.append(series[i + lookback:i + W])
    ctx, tgt = np.array(ctx), np.array(tgt)
    if max_n and len(ctx) > max_n:
        idx = np.random.choice(len(ctx), max_n, replace=False)
        ctx, tgt = ctx[idx], tgt[idx]
    return ctx, tgt


def make_past_windows(series: np.ndarray, past_len: int, horizon: int, max_n: int = None):
    """
    Extended-lookback windows for MoiraiForecast inference.
    Returns (past_windows, targets) where past = (context + horizon) length.
    """
    W = past_len + horizon
    past, tgt = [], []
    for i in range(len(series) - W + 1):
        past.append(series[i:i + past_len])
        tgt.append(series[i + past_len:i + W])
    past, tgt = np.array(past), np.array(tgt)
    if max_n and len(past) > max_n:
        idx = np.random.choice(len(past), max_n, replace=False)
        past, tgt = past[idx], tgt[idx]
    return past, tgt


# ── Moirai loading ─────────────────────────────────────────────────────────────

def build_moirai_forecast(module):
    from uni2ts.model.moirai import MoiraiForecast
    return MoiraiForecast(
        module=module,
        prediction_length=HORIZON,
        context_length=LOOKBACK,
        patch_size="auto",
        num_samples=1,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )


def get_repr(forecast_model, ctx_np: np.ndarray, device, batch_size=128):
    """
    Extract mean-pooled representations from Moirai encoder's last layer.
    ctx_np: (N, LOOKBACK) — context windows (not extended).
    Returns: (N, D_MODEL).
    """
    module = forecast_model.module
    captured = {}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach().cpu()

    handle = module.encoder.layers[-1].register_forward_hook(hook)

    reprs = []
    forecast_model.eval()
    forecast_model.to(device)

    for i in range(0, len(ctx_np), batch_size):
        batch = ctx_np[i:i + batch_size]
        B = len(batch)
        # Build past_target: (B, PAST_LEN, 1) — pad future with zeros
        past = np.zeros((B, PAST_LEN), dtype=np.float32)
        past[:, :LOOKBACK] = batch
        past_t   = torch.tensor(past, dtype=torch.float32).unsqueeze(-1).to(device)
        past_obs = torch.ones(B, PAST_LEN, 1, dtype=torch.bool).to(device)
        past_pad = torch.zeros(B, PAST_LEN, dtype=torch.bool).to(device)

        with torch.no_grad():
            _ = forecast_model(
                past_target=past_t,
                past_observed_target=past_obs,
                past_is_pad=past_pad,
            )

        if "h" in captured:
            h = captured["h"]
            if h.dim() == 3:
                h = h.mean(dim=1)
            reprs.append(h.numpy())

    handle.remove()
    return np.concatenate(reprs, axis=0)  # (N, D_MODEL)


def eval_zs_mse(forecast_model, past_np: np.ndarray, tgt_np: np.ndarray,
                device, batch_size=64):
    """
    Zero-shot MSE: median of MoiraiForecast samples vs target.
    past_np: (N, PAST_LEN); tgt_np: (N, HORIZON).
    """
    forecast_model.eval()
    forecast_model.to(device)
    preds = []
    for i in range(0, len(past_np), batch_size):
        b = past_np[i:i + batch_size]
        B = len(b)
        past_t   = torch.tensor(b, dtype=torch.float32).unsqueeze(-1).to(device)
        past_obs = torch.ones(B, PAST_LEN, 1, dtype=torch.bool).to(device)
        past_pad = torch.zeros(B, PAST_LEN, dtype=torch.bool).to(device)
        with torch.no_grad():
            out = forecast_model(
                past_target=past_t,
                past_observed_target=past_obs,
                past_is_pad=past_pad,
                num_samples=20,
            )  # (B, 20, H)
        pred = out.median(dim=1).values.squeeze(-1).cpu().numpy()  # (B, H)
        preds.append(pred)
    preds = np.concatenate(preds, axis=0)
    return float(np.mean((preds - tgt_np) ** 2))


# ── Head training ──────────────────────────────────────────────────────────────

class MSEHead(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.fc(x)


def train_head(forecast_model, train_ctx, train_tgt, val_ctx, val_tgt,
               condition: str, device, seed: int):
    """
    Fine-tune Moirai-Small on ETTh2 with an MSE head.
    Condition B: encoder + head. Condition D: head only.
    Returns (best_head, best_forecast_model).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    fm = copy.deepcopy(forecast_model)
    fm.to(device)
    head = MSEHead(D_MODEL, HORIZON).to(device)

    if condition == "D":
        for p in fm.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW(head.parameters(), lr=LR)
    else:
        opt = torch.optim.AdamW(list(fm.parameters()) + list(head.parameters()), lr=LR)

    # Subsample
    n = len(train_ctx)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, min(n, N_TRAIN), replace=False)
    ctx_t = torch.tensor(train_ctx[idx], dtype=torch.float32)
    tgt_t = torch.tensor(train_tgt[idx], dtype=torch.float32)
    loader = DataLoader(TensorDataset(ctx_t, tgt_t), batch_size=BATCH, shuffle=True)

    val_ctx_t = torch.tensor(val_ctx, dtype=torch.float32)
    val_tgt_t = torch.tensor(val_tgt, dtype=torch.float32)

    best_val = float("inf")
    best_head_st, best_fm_st = None, None
    patience = 0

    for epoch in range(MAX_EPOCHS):
        if condition == "B":
            fm.train()
        else:
            fm.eval()
        head.train()
        running = 0.0

        for ctx_b, tgt_b in loader:
            ctx_b, tgt_b = ctx_b.to(device), tgt_b.to(device)
            opt.zero_grad()
            r = _batch_repr(fm, ctx_b, device)
            pred = head(r)
            loss = nn.functional.mse_loss(pred, tgt_b)
            loss.backward()
            nn.utils.clip_grad_norm_(list(fm.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            running += loss.item()

        # Validation
        fm.eval(); head.eval()
        with torch.no_grad():
            vr = _batch_repr(fm, val_ctx_t.to(device), device)
            vp = head(vr)
            vloss = nn.functional.mse_loss(vp, val_tgt_t.to(device)).item()

        print(f"    Epoch {epoch+1:2d}: train={running/len(loader):.4f}, val={vloss:.4f}")

        if vloss < best_val:
            best_val = vloss
            best_head_st = copy.deepcopy(head.state_dict())
            best_fm_st   = copy.deepcopy(fm.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"    Early stop at epoch {epoch+1}")
                break

    head.load_state_dict(best_head_st)
    fm.load_state_dict(best_fm_st)
    return head, fm


def eval_ft_mse(fm, head, ctx_np, tgt_np, device, batch_size=128):
    fm.eval(); head.eval()
    ctx_t = torch.tensor(ctx_np, dtype=torch.float32)
    tgt_t = tgt_np
    preds = []
    with torch.no_grad():
        for i in range(0, len(ctx_np), batch_size):
            r = _batch_repr(fm, ctx_t[i:i+batch_size].to(device), device)
            p = head(r).cpu().numpy()
            preds.append(p)
    preds = np.concatenate(preds, axis=0)
    return float(np.mean((preds - tgt_t) ** 2))


def _batch_repr(fm, ctx_b: torch.Tensor, device):
    """Get mean-pooled representations for a context batch (B, LOOKBACK)."""
    module = fm.module
    captured = {}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h

    handle = module.encoder.layers[-1].register_forward_hook(hook)
    B = ctx_b.shape[0]
    past = torch.zeros(B, PAST_LEN, 1, dtype=torch.float32, device=device)
    past[:, :LOOKBACK, 0] = ctx_b
    past_obs = torch.ones(B, PAST_LEN, 1, dtype=torch.bool, device=device)
    past_pad = torch.zeros(B, PAST_LEN, dtype=torch.bool, device=device)
    _ = fm(past_target=past, past_observed_target=past_obs, past_is_pad=past_pad)
    handle.remove()
    h = captured["h"]
    if h.dim() == 3:
        h = h.mean(dim=1)
    return h


def zs_mse_via_ridge(fm_zs, train_ctx, train_tgt, test_ctx, test_tgt, device, seed):
    """Zero-shot MSE: fit Ridge on ZS representations of train_ctx, eval on test_ctx."""
    train_repr = get_repr(fm_zs, train_ctx, device)
    test_repr  = get_repr(fm_zs, test_ctx,  device)
    ridge = Ridge(alpha=1.0)
    ridge.fit(train_repr, train_tgt)
    pred = ridge.predict(test_repr)
    return float(np.mean((pred - test_tgt) ** 2))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 202])
    parser.add_argument("--data-dir",    default="data/ett")
    parser.add_argument("--results-dir", default="results/ood_forgetting")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir    = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Download data
    print("Downloading ETT data...")
    download_ett(data_dir)

    # Build windows
    print("Building windows...")
    h2_tr, h2_va, h2_te = load_ett_splits(data_dir / "ETTh2.csv")
    h1_tr, h1_va, h1_te = load_ett_splits(data_dir / "ETTh1.csv")
    m2_tr, m2_va, m2_te = load_ett_splits(data_dir / "ETTm2.csv")

    train_ctx, train_tgt = make_windows(h2_tr, LOOKBACK, HORIZON, max_n=N_TRAIN)
    val_ctx,   val_tgt   = make_windows(h2_va, LOOKBACK, HORIZON)

    # Test windows (plain context/target, for head-based eval)
    te_h2_ctx, te_h2_tgt = make_windows(h2_te, LOOKBACK, HORIZON)
    te_h1_ctx, te_h1_tgt = make_windows(h1_te, LOOKBACK, HORIZON)
    te_m2_ctx, te_m2_tgt = make_windows(m2_te, LOOKBACK, HORIZON)

    print(f"  ETTh2 train={len(train_ctx)}, val={len(val_ctx)}, test={len(te_h2_ctx)}")
    print(f"  ETTh1 test={len(te_h1_ctx)}, ETTm2 test={len(te_m2_ctx)}")

    # Load pretrained Moirai-Small
    print("\nLoading Moirai-Small...")
    from uni2ts.model.moirai import MoiraiModule
    pt_module = MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small")
    pt_module.eval()
    fm_pretrained = build_moirai_forecast(pt_module)
    fm_pretrained.to(device)
    print("  Loaded.")

    all_results = {}

    for seed in args.seeds:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        seed_results = {}

        for condition in ["B", "D"]:
            print(f"\n--- Condition {condition} ---")
            ft_head, fm_ft = train_head(
                fm_pretrained, train_ctx, train_tgt,
                val_ctx, val_tgt,
                condition, device, seed,
            )

            # ZS MSE baseline: Ridge on pretrained representations, evaluated per dataset
            mse_zs_h2 = zs_mse_via_ridge(fm_pretrained, train_ctx, train_tgt, te_h2_ctx, te_h2_tgt, device, seed)
            mse_zs_h1 = zs_mse_via_ridge(fm_pretrained, train_ctx, train_tgt, te_h1_ctx, te_h1_tgt, device, seed)
            mse_zs_m2 = zs_mse_via_ridge(fm_pretrained, train_ctx, train_tgt, te_m2_ctx, te_m2_tgt, device, seed)

            # FT MSE: fine-tuned encoder + head
            mse_ft_h2 = eval_ft_mse(fm_ft, ft_head, te_h2_ctx, te_h2_tgt, device)
            mse_ft_h1 = eval_ft_mse(fm_ft, ft_head, te_h1_ctx, te_h1_tgt, device)
            mse_ft_m2 = eval_ft_mse(fm_ft, ft_head, te_m2_ctx, te_m2_tgt, device)

            def forg(zs, ft): return (ft / zs - 1) * 100

            r = {
                "ETTh2": {"zs": mse_zs_h2, "ft": mse_ft_h2, "forg_pct": forg(mse_zs_h2, mse_ft_h2)},
                "ETTh1": {"zs": mse_zs_h1, "ft": mse_ft_h1, "forg_pct": forg(mse_zs_h1, mse_ft_h1)},
                "ETTm2": {"zs": mse_zs_m2, "ft": mse_ft_m2, "forg_pct": forg(mse_zs_m2, mse_ft_m2)},
            }
            seed_results[condition] = r

            for ds, v in r.items():
                print(f"  {ds}: ZS={v['zs']:.4f}  FT={v['ft']:.4f}  forg={v['forg_pct']:+.1f}%")

        all_results[str(seed)] = seed_results
        with open(results_dir / "results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved → {results_dir}/results.json")

    # Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for cond in ["B", "D"]:
        for ds in ["ETTh2", "ETTh1", "ETTm2"]:
            vals = [all_results[str(s)][cond][ds]["forg_pct"] for s in args.seeds]
            tag = "(in-dist)" if ds == "ETTh2" else "(OOD)"
            print(f"  Cond {cond} / {ds} {tag}: {np.mean(vals):+.1f} ± {np.std(vals):.1f}%")


if __name__ == "__main__":
    main()
