#!/usr/bin/env python3
"""
TimesFM-2.5-200M full diagnostic chain on ETT datasets (h=24).

TimesFM gate-passes at h=24 (45-77% improvement over linear), so we can run
the full chain: CKA, Ridge probes, orthogonal probes, frozen-encoder control.

Architecture: decoder-only, 20 Transformer blocks (stacked_xf), d_model=1280.
Representation extraction: hook on stacked_xf[-1] output, mean-pool over patches.

Fine-tuning: direct MSE loss via a linear regression head on the hooked reps
(TimesFM's native loss would require replicating its full tokenization pipeline;
MSE head is the same approach used in chronos_mse_finetune.py and matches the
Moirai fine-tuning paradigm).

Usage:
    python scripts/timesfm_diagnostic.py --dataset ETTh1 --seed 42 --device cuda
    python scripts/timesfm_diagnostic.py --phase batch --device cuda
"""

import argparse
import copy
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, LinearRegression
from torch.utils.data import DataLoader, TensorDataset

DATASETS = {
    "ETTh1": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "target_col": "OT",
        "n_train": 8640,
        "n_val": 2880,
        "lookback": 96,
        "horizon": 24,
    },
    "ETTh2": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
        "target_col": "OT",
        "n_train": 8640,
        "n_val": 2880,
        "lookback": 96,
        "horizon": 24,
    },
}

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
D_MODEL = 1280  # TimesFM-2.5-200M hidden dim


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def download_dataset(name, cache_dir="/tmp/gate_screen_data"):
    os.makedirs(cache_dir, exist_ok=True)
    cfg = DATASETS[name]
    local_path = os.path.join(cache_dir, f"{name}.csv")
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(cfg["url"], local_path)
    return local_path


def load_series(name):
    import pandas as pd
    cfg = DATASETS[name]
    df = pd.read_csv(download_dataset(name))
    v = df[cfg["target_col"]].values.astype(np.float64)
    n_tr, n_va = cfg["n_train"], cfg["n_val"]
    return v[:n_tr], v[n_tr:n_tr+n_va], v[n_tr+n_va:]


def build_windows(series, lookback, horizon, max_windows=None, seed=42):
    n = len(series) - lookback - horizon + 1
    if n <= 0:
        return np.empty((0, lookback)), np.empty((0, horizon))
    ctx = np.array([series[i:i+lookback] for i in range(n)])
    tgt = np.array([series[i+lookback:i+lookback+horizon] for i in range(n)])
    if max_windows and len(ctx) > max_windows:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(ctx), max_windows, replace=False)
        ctx, tgt = ctx[idx], tgt[idx]
    return ctx, tgt


# ---------------------------------------------------------------------------
# CKA + probes
# ---------------------------------------------------------------------------

def linear_CKA(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(Y.T @ X, 'fro') ** 2
    denom = np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro')
    return float(num / denom) if denom > 1e-12 else 0.0


def ridge_r2(reps_tr, tgt_tr, reps_va, tgt_va, alpha=1.0):
    tgt_tr = tgt_tr.reshape(len(tgt_tr), -1)
    tgt_va = tgt_va.reshape(len(tgt_va), -1)
    return float(Ridge(alpha=alpha).fit(reps_tr, tgt_tr).score(reps_va, tgt_va))


def ortho_probes(reps_tr, reps_va, ctx_tr, ctx_va, alpha=1.0):
    def lag1(c):
        mu = c.mean(1, keepdims=True)
        return (((c[:,1:]-mu)*(c[:,:-1]-mu)).mean(1) /
                ((c-mu)**2).mean(1).clip(1e-8)).reshape(-1,1)
    res = {}
    for name, fn in [("lag1", lag1),
                     ("mean", lambda c: c.mean(1, keepdims=True)),
                     ("var",  lambda c: c.var(1,  keepdims=True))]:
        res[name] = float(Ridge(alpha=alpha).fit(reps_tr, fn(ctx_tr)).score(reps_va, fn(ctx_va)))
    return res


# ---------------------------------------------------------------------------
# Representation extraction via hook on stacked_xf[-1]
# ---------------------------------------------------------------------------

def extract_reps(tfm_model, module, contexts, device, max_samples=500, batch_size=16):
    """
    Extract representations from TimesFM by hooking stacked_xf[-1].
    Returns (N, D_MODEL) mean-pooled over patch dimension.
    """
    import timesfm as tfm_lib

    tfm_model.eval()
    n = min(len(contexts), max_samples)
    all_reps = []
    captured = {}

    def hook(mod, inp, out):
        # out is (batch, n_patches, d_model) or tuple; always take first element
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach().cpu()

    handle = module.stacked_xf[-1].register_forward_hook(hook)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = [contexts[j].astype(np.float32) for j in range(i, min(i+batch_size, n))]
            tfm_model.forecast(horizon=DATASETS[list(DATASETS.keys())[0]]["horizon"],
                               inputs=batch)
            h = captured["h"]  # (1, batch, d_model) or (batch, patches, d_model)
            # Squeeze leading 1 if present
            if h.dim() == 3 and h.shape[0] == 1:
                h = h.squeeze(0)  # (batch, d_model) or (batch, patches, d_model)
            if h.dim() == 3:
                h = h.mean(1)  # mean over patches → (batch, d_model)
            all_reps.append(h.numpy())

    handle.remove()
    return np.concatenate(all_reps, axis=0)


def extract_reps_flexible(tfm_model, module, contexts, horizon, device,
                          max_samples=500, batch_size=16):
    """Extract reps via direct module.forward() call — avoids forecast() preprocessing."""
    module.eval()
    n = min(len(contexts), max_samples)
    all_reps = []
    captured = {}

    def hook(mod, inp, out):
        # stacked_xf[-1] returns (output_embeddings, cache); we want out[0]
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach().cpu()

    handle = module.stacked_xf[-1].register_forward_hook(hook)

    patch_size = 32
    mod_device = next(module.parameters()).device
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_ctx = torch.tensor(contexts[i:i+batch_size], dtype=torch.float32).to(mod_device)
            bs, lb = batch_ctx.shape
            n_patches = lb // patch_size
            inputs_patched = batch_ctx.reshape(bs, n_patches, patch_size)
            masks = torch.ones_like(inputs_patched)
            module(inputs_patched, masks)
            h = captured["h"]  # (batch, n_patches, d_model)
            h = h.mean(1)  # mean over patches → (batch, d_model)
            all_reps.append(h.numpy())

    handle.remove()
    return np.concatenate(all_reps, axis=0)


# ---------------------------------------------------------------------------
# Zero-shot MSE
# ---------------------------------------------------------------------------

def zs_mse(tfm_model, contexts, targets, horizon):
    preds = []
    tfm_model.model.eval()
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch = [contexts[j].astype(np.float32) for j in range(i, min(i+batch_size, len(contexts)))]
            point, _ = tfm_model.forecast(horizon=horizon, inputs=batch)
            preds.append(np.array(point)[:, :horizon])
    preds = np.concatenate(preds, axis=0)
    mses = []
    for i in range(len(contexts)):
        mu, sd = contexts[i].mean(), contexts[i].std() + 1e-8
        mses.append(float(np.mean(((preds[i]-mu)/sd - (targets[i]-mu)/sd)**2)))
    return float(np.mean(mses))


def lin_mse(contexts, targets):
    mses = []
    for i in range(len(contexts)):
        ctx, tgt = contexts[i], targets[i]
        mu, sd = ctx.mean(), ctx.std() + 1e-8
        x = np.arange(len(ctx)).reshape(-1,1)
        pred = LinearRegression().fit(x, ctx).predict(
            np.arange(len(ctx), len(ctx)+len(tgt)).reshape(-1,1))
        mses.append(float(np.mean(((pred-mu)/sd - (tgt-mu)/sd)**2)))
    return float(np.mean(mses))


# ---------------------------------------------------------------------------
# MSE head (for fine-tuning)
# ---------------------------------------------------------------------------

class MSEHead(nn.Module):
    def __init__(self, d_in, horizon):
        super().__init__()
        self.fc = nn.Linear(d_in, horizon)

    def forward(self, h):
        # h: (batch, patches, d_model) or (batch, d_model)
        if h.dim() == 3:
            h = h.mean(1)
        return self.fc(h)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(tfm_module, head, loader, optimizer, device, freeze_backbone=False):
    tfm_module.train()
    head.train()
    if freeze_backbone:
        tfm_module.eval()

    captured = {}
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h

    handle = tfm_module.stacked_xf[-1].register_forward_hook(hook)

    total_loss, count = 0.0, 0
    for ctx_b, tgt_b in loader:
        ctx_b = ctx_b.to(device, dtype=torch.float32)
        tgt_b = tgt_b.to(device, dtype=torch.float32)

        patch_size = 32  # TimesFM-2.5 uses 32-step patches
        n_patches = ctx_b.shape[1] // patch_size
        inputs_patched = ctx_b.reshape(ctx_b.shape[0], n_patches, patch_size)
        # masks same shape as inputs: (batch, n_patches, patch_size), all ones = fully observed
        masks = torch.ones_like(inputs_patched)

        # Call through the full module (triggers hook on stacked_xf[-1])
        _ = tfm_module(inputs_patched, masks)
        h = captured["h"]  # (batch, n_patches, d_model)
        if h.dim() == 3 and h.shape[0] == 1:
            h = h.squeeze(0)

        pred = head(h)

        ctx_mean = ctx_b.mean(1, keepdim=True)
        ctx_std = ctx_b.std(1, keepdim=True) + 1e-8
        tgt_norm = (tgt_b - ctx_mean) / ctx_std

        loss = nn.functional.mse_loss(pred, tgt_norm)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(tfm_module.parameters()) + list(head.parameters()), 1.0)
        optimizer.step()

        total_loss += loss.item()
        count += 1

    handle.remove()
    return total_loss / max(count, 1)


def eval_head_mse(tfm_module, head, loader, device):
    tfm_module.eval()
    head.eval()
    captured = {}
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h
    handle = tfm_module.stacked_xf[-1].register_forward_hook(hook)

    total, count = 0.0, 0
    with torch.no_grad():
        for ctx_b, tgt_b in loader:
            ctx_b = ctx_b.to(device, dtype=torch.float32)
            tgt_b = tgt_b.to(device, dtype=torch.float32)
            patch_size = 32
            n_patches = ctx_b.shape[1] // patch_size
            inputs_patched = ctx_b.reshape(ctx_b.shape[0], n_patches, patch_size)
            masks = torch.ones_like(inputs_patched)
            _ = tfm_module(inputs_patched, masks)
            h = captured["h"]
            if h.dim() == 3 and h.shape[0] == 1:
                h = h.squeeze(0)
            pred = head(h)
            ctx_mean = ctx_b.mean(1, keepdim=True)
            ctx_std = ctx_b.std(1, keepdim=True) + 1e-8
            tgt_norm = (tgt_b - ctx_mean) / ctx_std
            total += nn.functional.mse_loss(pred, tgt_norm, reduction='sum').item()
            count += len(ctx_b)
    handle.remove()
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(args, dataset_name):
    import timesfm as tfm_lib

    cfg = DATASETS[dataset_name]
    lookback, horizon = cfg["lookback"], cfg["horizon"]

    print(f"\n{'='*60}")
    print(f"TIMESFM DIAGNOSTIC: {dataset_name}, seed={args.seed}, "
          f"condition={args.condition}")
    print(f"{'='*60}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    results_dir = Path(args.results_dir) / f"timesfm_{dataset_name.lower()}" / f"seed{args.seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    tr_s, va_s, te_s = load_series(dataset_name)
    ctx_tr, tgt_tr = build_windows(tr_s, lookback, horizon,
                                   max_windows=args.max_train_samples, seed=args.seed)
    ctx_va, tgt_va = build_windows(va_s, lookback, horizon, max_windows=200, seed=args.seed)

    n_pval = max(int(len(ctx_tr)*0.2), 10)
    n_ptr = len(ctx_tr) - n_pval
    probe_tr_ctx, probe_va_ctx = ctx_tr[:n_ptr], ctx_tr[n_ptr:]
    probe_tr_tgt, probe_va_tgt = tgt_tr[:n_ptr], tgt_tr[n_ptr:]
    eval_ctx, eval_tgt = (ctx_va, tgt_va) if len(ctx_va) >= 20 else (ctx_tr[:200], tgt_tr[:200])
    print(f"  Train: {n_ptr}, Probe-val: {n_pval}, Eval: {len(eval_ctx)}")

    # Load TimesFM
    print(f"  Loading {MODEL_ID}...")
    tfm = tfm_lib.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
    fc = tfm_lib.ForecastConfig(
        max_context=lookback, max_horizon=horizon, normalize_inputs=True,
        use_continuous_quantile_head=True, force_flip_invariance=True,
        infer_is_positive=False, fix_quantile_crossing=True,
    )
    tfm.compile(fc)
    module = tfm.model

    # Store pretrained state
    pretrained_state = copy.deepcopy(module.state_dict())

    # Zero-shot baseline
    print("  Computing zero-shot MSE...")
    zs = zs_mse(tfm, eval_ctx, eval_tgt, horizon)
    lm = lin_mse(eval_ctx, eval_tgt)
    gate_pct = (lm - zs) / lm * 100 if lm > 1e-10 else 0.0
    print(f"  ZS MSE: {zs:.4f}, Linear: {lm:.4f}, Gate: {gate_pct:.1f}%")

    # Pre-trained reps
    print("  Extracting pre-trained representations...")
    reps_pt_tr = extract_reps_flexible(tfm, module, probe_tr_ctx, horizon, args.device, max_samples=500)
    reps_pt_va = extract_reps_flexible(tfm, module, probe_va_ctx, horizon, args.device, max_samples=200)
    r2_pt = ridge_r2(reps_pt_tr, probe_tr_tgt[:len(reps_pt_tr)],
                     reps_pt_va, probe_va_tgt[:len(reps_pt_va)])
    ortho_pt = ortho_probes(reps_pt_tr, reps_pt_va,
                            probe_tr_ctx[:len(reps_pt_tr)], probe_va_ctx[:len(reps_pt_va)])
    print(f"  PT Ridge R²: {r2_pt:.4f}, Orthogonal: {ortho_pt}")

    if args.condition == 'A':
        out = {"condition":"A","dataset":dataset_name,"seed":args.seed,
               "zs_mse":zs,"linear_mse":lm,"gate_pct":gate_pct,
               "r2_pt":r2_pt,"ortho_pt":ortho_pt}
        with open(results_dir/f"condition_A_s{args.seed}.json","w") as f:
            json.dump(out,f,indent=2)
        return out

    # Fine-tuning setup
    freeze_backbone = (args.condition == 'D')
    if freeze_backbone:
        print("  Freezing backbone (condition D)...")
        for p in module.parameters():
            p.requires_grad = False

    head = MSEHead(D_MODEL, horizon)
    if args.device != "cpu":
        module_device = next(module.parameters()).device
        head = head.to(module_device)

    # Use the device where module lives
    actual_device = next(module.parameters()).device

    ctx_t = torch.tensor(probe_tr_ctx, dtype=torch.float32)
    tgt_t = torch.tensor(probe_tr_tgt, dtype=torch.float32)
    ctx_v = torch.tensor(eval_ctx[:200], dtype=torch.float32)
    tgt_v = torch.tensor(eval_tgt[:200], dtype=torch.float32)
    train_ds = TensorDataset(ctx_t, tgt_t)
    val_ds = TensorDataset(ctx_v, tgt_v)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    params = list(head.parameters())
    if not freeze_backbone:
        params += list(module.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    best_val, best_epoch = float('inf'), 0
    best_mod_state = copy.deepcopy(module.state_dict())
    best_head_state = copy.deepcopy(head.state_dict())
    patience_counter = 0

    print(f"\n  Training: {args.epochs} epochs, lr={args.lr}")
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tl = train_epoch(module, head, train_loader, optimizer, actual_device, freeze_backbone)
        vl = eval_head_mse(module, head, val_loader, actual_device)

        reps_ft_va = extract_reps_flexible(tfm, module, probe_va_ctx, horizon,
                                           actual_device, max_samples=200)
        cka = linear_CKA(reps_pt_va, reps_ft_va)
        print(f"    Ep {epoch:2d}: loss={tl:.4f} val={vl:.4f} CKA={cka:.3f} ({time.time()-t0:.1f}s)")

        if vl < best_val:
            best_val, best_epoch = vl, epoch
            best_mod_state = copy.deepcopy(module.state_dict())
            best_head_state = copy.deepcopy(head.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    Early stopping at epoch {epoch}")
                break

    print(f"  Restoring best epoch {best_epoch}")
    module.load_state_dict(best_mod_state)
    head.load_state_dict(best_head_state)

    # Post-training diagnostics
    reps_ft_tr = extract_reps_flexible(tfm, module, probe_tr_ctx, horizon,
                                       actual_device, max_samples=500)
    reps_ft_va = extract_reps_flexible(tfm, module, probe_va_ctx, horizon,
                                       actual_device, max_samples=200)
    final_cka = linear_CKA(reps_pt_va, reps_ft_va)
    r2_ft = ridge_r2(reps_ft_tr, probe_tr_tgt[:len(reps_ft_tr)],
                     reps_ft_va, probe_va_tgt[:len(reps_ft_va)])
    delta_r2 = r2_ft - r2_pt
    ortho_ft = ortho_probes(reps_ft_tr, reps_ft_va,
                            probe_tr_ctx[:len(reps_ft_tr)], probe_va_ctx[:len(reps_ft_va)])
    ortho_delta = {k: ortho_ft[k]-ortho_pt[k] for k in ortho_pt}

    # Weight drift
    cur = module.state_dict()
    wdrift = sum((cur[k]-pretrained_state[k]).float().pow(2).sum().item()
                 for k in pretrained_state if k in cur) ** 0.5

    has_trained_gain = delta_r2 > 0
    has_ortho_loss = all(ortho_delta[k] <= 0 for k in ortho_delta)
    probe_asym = has_trained_gain and has_ortho_loss
    has_drift = final_cka < 0.95

    print(f"\n  {'='*50}")
    print(f"  RESULT: TimesFM × {dataset_name}, seed {args.seed}, cond {args.condition}")
    print(f"  {'='*50}")
    print(f"  Gate:         {gate_pct:.1f}%")
    print(f"  CKA:          {final_cka:.3f} {'DRIFT' if has_drift else 'stable'}")
    print(f"  Weight drift: {wdrift:.1f}")
    print(f"  ΔR² trained:  {delta_r2:+.4f} {'OK' if has_trained_gain else 'neg'}")
    print(f"  ΔR² orthog:   {ortho_delta}")
    print(f"  Probe asym:   {'YES <<<' if probe_asym else 'no'}")
    print(f"  DISSOCIATION: {'YES <<<' if (has_drift and probe_asym) else 'no'}")
    print(f"  {'='*50}")

    result = {
        "dataset": dataset_name, "condition": args.condition, "seed": args.seed,
        "model": MODEL_ID, "loss_type": "mse_head_on_stacked_xf",
        "best_epoch": best_epoch, "lr": args.lr,
        "max_train_samples": args.max_train_samples,
        "gate_improvement_pct": gate_pct, "zs_mse": zs, "linear_mse": lm,
        "final_cka": final_cka, "weight_drift": wdrift, "has_drift": has_drift,
        "linear_probe": {"r2_pt": r2_pt, "r2_ft": r2_ft, "delta_r2": delta_r2},
        "orthogonal_probes": {"pretrained": ortho_pt, "finetuned": ortho_ft, "delta": ortho_delta},
        "probe_asymmetry": probe_asym,
        "dissociation": has_drift and probe_asym,
    }
    out_path = results_dir / f"condition_{args.condition}_s{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    return result


def run_batch(args):
    all_results = []
    for ds in ["ETTh1", "ETTh2"]:
        for seed in range(42, 42 + args.n_seeds):
            for cond in ["B", "D"]:
                args.seed = seed
                args.condition = cond
                try:
                    r = run(args, ds)
                    all_results.append(r)
                except Exception as e:
                    print(f"ERROR {ds}/s{seed}/{cond}: {e}")
                    import traceback; traceback.print_exc()

    print("\n" + "="*60)
    print("TIMESFM DIAGNOSTIC — SUMMARY")
    print("="*60)
    for r in all_results:
        if r and "final_cka" in r:
            d = "DISSOC" if r.get("dissociation") else "no"
            pa = "ASYM" if r.get("probe_asymmetry") else "no"
            print(f"  {r['dataset']:6s} s{r['seed']} {r['condition']}: "
                  f"CKA={r['final_cka']:.3f} ΔR²={r['linear_probe']['delta_r2']:+.3f} "
                  f"asym={pa} → {d}")

    out_path = Path(args.results_dir) / "timesfm_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='single', choices=['single', 'batch'])
    parser.add_argument('--dataset', default='ETTh1')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--condition', default='B', choices=['A', 'B', 'D'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max-train-samples', type=int, default=8000)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--results-dir', default='results/timesfm_diagnostic')
    args = parser.parse_args()

    if args.phase == 'batch':
        run_batch(args)
    else:
        run(args, args.dataset)


if __name__ == "__main__":
    main()
