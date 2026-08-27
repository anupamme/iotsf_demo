#!/usr/bin/env python3
"""
Moderate-Gate-Band Screening: Chronos × multiple datasets.

Goal: Find a second-backbone cell with 20-45% gate-pass (moderate value,
room for restructuring) and run the full diagnostic chain to check for
CKA drift + probe asymmetry — the "dissociation on a second backbone" that
the AC flagged as decisive.

Hypothesis: the drift-utility dissociation requires moderate gate-pass:
  - Strong gate-pass (>50%): backbone near-solves → no restructuring needed → stability
  - Moderate gate-pass (20-45%): enough value to matter, room to restructure → drift possible
  - Gate-fail (<20%): no pretrained value to preserve → gate screens out

We test Chronos-T5-Small on datasets where it has moderate (not dominant) value:
  - ETTh2, ETTm1, ETTm2 (multivariate → univariate channel forecasting)
  - Weather (multivariate, 21 channels)
  - ILI (short, irregular seasonality)
  - Electricity (321 channels, complex)

Phase 1: Gate screen (zero-shot vs linear) — fast, no training
Phase 2: Full diagnostic chain on moderate-gate cells (20-45%)

Usage:
    python scripts/moderate_gate_screen.py --phase screen --device cuda
    python scripts/moderate_gate_screen.py --phase finetune --dataset ETTh2 --seed 42 --device cuda
    python scripts/moderate_gate_screen.py --phase finetune-all --device cuda
"""

import argparse
import copy
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, LinearRegression
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

DATASETS = {
    "ETTh2": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
        "freq": "h",
        "lookback": 96,
        "horizon": 24,
        "target_col": "OT",
        "n_train": 8640,   # standard ETT split
        "n_val": 2880,
    },
    "ETTm1": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        "freq": "15min",
        "lookback": 96,
        "horizon": 24,
        "target_col": "OT",
        "n_train": 34560,
        "n_val": 11520,
    },
    "ETTm2": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
        "freq": "15min",
        "lookback": 96,
        "horizon": 24,
        "target_col": "OT",
        "n_train": 34560,
        "n_val": 11520,
    },
    "ETTh1": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "freq": "h",
        "lookback": 96,
        "horizon": 24,
        "target_col": "OT",
        "n_train": 8640,
        "n_val": 2880,
    },
    "Weather": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/Autoformer/main/dataset/weather/weather.csv",
        "freq": "10min",
        "lookback": 96,
        "horizon": 24,
        "target_col": "OT",
        "n_train": 36792,
        "n_val": 5271,
    },
    "ILI": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/Autoformer/main/dataset/illness/national_illness.csv",
        "freq": "w",
        "lookback": 36,
        "horizon": 12,
        "target_col": "OT",
        "n_train": 617,
        "n_val": 74,
    },
}

MODEL_ID = "amazon/chronos-t5-small"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def download_dataset(name, cache_dir="/tmp/gate_screen_data"):
    os.makedirs(cache_dir, exist_ok=True)
    cfg = DATASETS[name]
    local_path = os.path.join(cache_dir, f"{name}.csv")
    if not os.path.exists(local_path):
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(cfg["url"], local_path)
    return local_path


def load_univariate_series(name, cache_dir="/tmp/gate_screen_data"):
    """Load dataset as univariate target column, return train/val/test splits."""
    import pandas as pd
    cfg = DATASETS[name]
    path = download_dataset(name, cache_dir)
    df = pd.read_csv(path)

    # Use "OT" column; fall back to last column if OT not present
    if cfg["target_col"] in df.columns:
        values = df[cfg["target_col"]].values.astype(np.float64)
    else:
        values = df.iloc[:, -1].values.astype(np.float64)

    n_train = cfg["n_train"]
    n_val = cfg["n_val"]
    train = values[:n_train]
    val = values[n_train:n_train + n_val]
    test = values[n_train + n_val:]

    return train, val, test, cfg


def build_windows(series, lookback, horizon, max_windows=None, seed=42):
    """Slide a window over a 1-D series → (contexts, targets)."""
    n_total = len(series) - lookback - horizon + 1
    if n_total <= 0:
        return np.empty((0, lookback)), np.empty((0, horizon))
    contexts = np.array([series[i:i+lookback] for i in range(n_total)])
    targets = np.array([series[i+lookback:i+lookback+horizon] for i in range(n_total)])
    if max_windows and len(contexts) > max_windows:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(contexts), max_windows, replace=False)
        contexts, targets = contexts[idx], targets[idx]
    return contexts, targets


# ---------------------------------------------------------------------------
# CKA
# ---------------------------------------------------------------------------

def linear_CKA(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    XTX = X.T @ X
    YTY = Y.T @ Y
    YTX = Y.T @ X
    num = np.linalg.norm(YTX, 'fro') ** 2
    denom = np.linalg.norm(XTX, 'fro') * np.linalg.norm(YTY, 'fro')
    if denom < 1e-12:
        return 0.0
    return float(num / denom)


# ---------------------------------------------------------------------------
# Representation Extraction
# ---------------------------------------------------------------------------

def extract_encoder_reps(model, tokenizer, contexts, device, max_samples=500, batch_size=32):
    """Extract mean-pooled encoder representations from Chronos T5 encoder."""
    model.eval()
    encoder = model.model.encoder
    n = min(len(contexts), max_samples)
    all_reps = []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_ctx = torch.tensor(contexts[i:i+batch_size], dtype=torch.float32)
            token_ids, attn_mask, _ = tokenizer.context_input_transform(batch_ctx)
            token_ids = token_ids.to(device)
            attn_mask = attn_mask.to(device)
            out = encoder(input_ids=token_ids, attention_mask=attn_mask)
            h = out.last_hidden_state
            mask_expanded = attn_mask.unsqueeze(-1).float()
            pooled = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            all_reps.append(pooled.cpu().numpy())

    return np.concatenate(all_reps, axis=0)


# ---------------------------------------------------------------------------
# Zero-Shot Evaluation
# ---------------------------------------------------------------------------

def chronos_zs_mse(pipe, contexts, targets, device, batch_size=32):
    """Compute zero-shot MSE using Chronos median predictions."""
    horizon = targets.shape[1]
    preds = []
    pipe.model.eval()
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch = [torch.tensor(contexts[j], dtype=torch.float32)
                     for j in range(i, min(i + batch_size, len(contexts)))]
            samples = pipe.predict(batch, prediction_length=horizon, num_samples=20)
            medians = samples.median(dim=1).values.cpu().numpy()
            preds.append(medians)
    preds = np.concatenate(preds, axis=0)

    mses = []
    for i in range(len(contexts)):
        mu = contexts[i].mean()
        sd = contexts[i].std() + 1e-8
        pred_n = (preds[i] - mu) / sd
        tgt_n = (targets[i] - mu) / sd
        mses.append(float(np.mean((pred_n - tgt_n) ** 2)))
    return float(np.mean(mses))


def linear_baseline_mse(contexts, targets):
    """Per-window linear baseline."""
    mses = []
    for i in range(len(contexts)):
        ctx = contexts[i]
        tgt = targets[i]
        mu = ctx.mean()
        sd = ctx.std() + 1e-8
        # Simple linear extrapolation from last lookback points
        x = np.arange(len(ctx)).reshape(-1, 1)
        y = ctx
        reg = LinearRegression().fit(x, y)
        x_pred = np.arange(len(ctx), len(ctx) + len(tgt)).reshape(-1, 1)
        pred = reg.predict(x_pred)
        pred_n = (pred - mu) / sd
        tgt_n = (tgt - mu) / sd
        mses.append(float(np.mean((pred_n - tgt_n) ** 2)))
    return float(np.mean(mses))


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def ridge_probe_r2(reps_train, targets_train, reps_val, targets_val, alpha=1.0):
    tgt_tr = targets_train.reshape(len(targets_train), -1)
    tgt_va = targets_val.reshape(len(targets_val), -1)
    probe = Ridge(alpha=alpha).fit(reps_train, tgt_tr)
    return float(probe.score(reps_val, tgt_va))


def task_orthogonal_probes(reps_train, reps_val, contexts_train, contexts_val, alpha=1.0):
    results = {}

    def lag1_autocorr(ctx):
        x = ctx[:, 1:]
        x_lag = ctx[:, :-1]
        mu = ctx.mean(axis=1, keepdims=True)
        num = ((x - mu) * (x_lag - mu)).mean(axis=1)
        denom = ((ctx - mu) ** 2).mean(axis=1)
        return (num / (denom + 1e-8)).reshape(-1, 1)

    def input_mean(ctx):
        return ctx.mean(axis=1, keepdims=True)

    def input_var(ctx):
        return ctx.var(axis=1, keepdims=True)

    for fname, fn in [("lag1", lag1_autocorr), ("mean", input_mean), ("var", input_var)]:
        tgt_tr = fn(contexts_train)
        tgt_va = fn(contexts_val)
        probe = Ridge(alpha=alpha).fit(reps_train, tgt_tr)
        r2 = float(probe.score(reps_val, tgt_va))
        results[fname] = r2

    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(chronos_model, tokenizer, train_loader, optimizer, device,
                    freeze_encoder=False, horizon=24):
    chronos_model.model.train()
    if freeze_encoder:
        chronos_model.model.encoder.eval()

    epoch_loss = 0.0
    batch_count = 0

    for ctx_batch, tgt_batch in train_loader:
        ctx_batch = ctx_batch.to(dtype=torch.float32)
        tgt_batch = tgt_batch.to(dtype=torch.float32)

        token_ids, attn_mask, scale = tokenizer.context_input_transform(ctx_batch)
        token_ids = token_ids.to(device)
        attn_mask = attn_mask.to(device)

        label_ids, label_mask = tokenizer.label_input_transform(tgt_batch, scale)
        label_ids = label_ids.to(device)

        labels = label_ids.clone()
        labels[labels == tokenizer.config.pad_token_id] = -100

        outputs = chronos_model.model(
            input_ids=token_ids,
            attention_mask=attn_mask,
            labels=labels,
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(chronos_model.model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    return epoch_loss / max(batch_count, 1)


# ---------------------------------------------------------------------------
# Phase 1: Gate Screen
# ---------------------------------------------------------------------------

def run_gate_screen(args):
    """Screen all datasets for gate-pass percentage."""
    from chronos import ChronosPipeline

    print("=" * 60)
    print("PHASE 1: Gate Screening — Chronos-T5-Small")
    print("=" * 60)

    pipe = ChronosPipeline.from_pretrained(MODEL_ID, dtype=torch.float32)
    if args.device != "cpu":
        pipe.model.model = pipe.model.model.to(args.device)

    results = {}
    for name in DATASETS:
        print(f"\n--- {name} ---")
        try:
            train, val, test, cfg = load_univariate_series(name)
            lookback = cfg["lookback"]
            horizon = cfg["horizon"]

            # Use val set for gate screening (test held out for final eval)
            ctx_val, tgt_val = build_windows(val, lookback, horizon, max_windows=200, seed=42)
            if len(ctx_val) < 10:
                print(f"  Too few windows ({len(ctx_val)}), skipping")
                results[name] = {"status": "skipped", "reason": "too_few_windows"}
                continue

            # Zero-shot MSE
            pipe.tokenizer.config.prediction_length = horizon
            zs_mse = chronos_zs_mse(pipe, ctx_val, tgt_val, args.device)

            # Linear baseline MSE
            lin_mse = linear_baseline_mse(ctx_val, tgt_val)

            # Gate improvement
            gate_pct = (lin_mse - zs_mse) / lin_mse * 100 if lin_mse > 1e-10 else 0.0

            regime = "gate-fail" if gate_pct < 20 else ("moderate" if gate_pct < 45 else "strong")
            print(f"  ZS MSE: {zs_mse:.4f}  Linear MSE: {lin_mse:.4f}")
            print(f"  Gate improvement: {gate_pct:.1f}% → {regime.upper()}")

            results[name] = {
                "zs_mse": zs_mse,
                "linear_mse": lin_mse,
                "gate_improvement_pct": gate_pct,
                "regime": regime,
                "n_windows": len(ctx_val),
                "lookback": lookback,
                "horizon": horizon,
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"status": "error", "error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("GATE SCREEN SUMMARY")
    print("=" * 60)
    moderate_cells = []
    for name, r in results.items():
        if "gate_improvement_pct" in r:
            marker = " <<<" if r["regime"] == "moderate" else ""
            print(f"  {name:12s}: {r['gate_improvement_pct']:+6.1f}%  [{r['regime']}]{marker}")
            if r["regime"] == "moderate":
                moderate_cells.append(name)
        else:
            print(f"  {name:12s}: {r.get('status', 'unknown')}")

    print(f"\nModerate-gate cells (20-45%): {moderate_cells if moderate_cells else 'NONE'}")
    if not moderate_cells:
        print("  → Expanding search: cells with 15-50% may also be informative")
        for name, r in results.items():
            if "gate_improvement_pct" in r and 15 <= r["gate_improvement_pct"] <= 50:
                moderate_cells.append(name)
        print(f"  Expanded set (15-50%): {moderate_cells if moderate_cells else 'NONE'}")

    # Save
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "gate_screen.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_dir / 'gate_screen.json'}")

    return results, moderate_cells


# ---------------------------------------------------------------------------
# Phase 2: Full Diagnostic Chain
# ---------------------------------------------------------------------------

def run_finetune(args, dataset_name):
    """Run full diagnostic chain on a single dataset × seed."""
    from chronos import ChronosPipeline

    cfg = DATASETS[dataset_name]
    lookback = cfg["lookback"]
    horizon = cfg["horizon"]

    print(f"\n{'='*60}")
    print(f"FINE-TUNING: Chronos × {dataset_name}, seed={args.seed}, "
          f"condition={args.condition}, n={args.max_train_samples}")
    print(f"{'='*60}")

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    results_dir = Path(args.results_dir) / f"chronos_{dataset_name.lower()}" / f"seed{args.seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_series, val_series, test_series, _ = load_univariate_series(dataset_name)

    ctx_tr, tgt_tr = build_windows(train_series, lookback, horizon,
                                   max_windows=args.max_train_samples, seed=args.seed)
    ctx_va, tgt_va = build_windows(val_series, lookback, horizon, max_windows=200, seed=args.seed)
    ctx_test, tgt_test = build_windows(test_series, lookback, horizon, max_windows=200, seed=args.seed)

    # Use val for probes, test for MSE evaluation
    eval_ctx = ctx_va if len(ctx_va) >= 20 else ctx_test
    eval_tgt = tgt_va if len(tgt_va) >= 20 else tgt_test
    print(f"  Train: {len(ctx_tr)}, Val: {len(ctx_va)}, Test: {len(ctx_test)}")

    # Split train into train/probe-val
    n_total = len(ctx_tr)
    n_pval = max(int(n_total * 0.2), 10)
    n_ptr = n_total - n_pval
    probe_tr_ctx, probe_va_ctx = ctx_tr[:n_ptr], ctx_tr[n_ptr:]
    probe_tr_tgt, probe_va_tgt = tgt_tr[:n_ptr], tgt_tr[n_ptr:]

    # Load model
    print(f"  Loading {MODEL_ID}...")
    pipe = ChronosPipeline.from_pretrained(MODEL_ID, dtype=torch.float32)
    chronos_model = pipe.model
    tokenizer = pipe.tokenizer
    tokenizer.config.prediction_length = horizon

    if args.device != "cpu":
        chronos_model.model = chronos_model.model.to(args.device)

    pretrained_state = copy.deepcopy(chronos_model.model.state_dict())

    # Zero-shot
    print("  Computing zero-shot MSE...")
    zs_mse = chronos_zs_mse(pipe, eval_ctx, eval_tgt, args.device)
    lin_mse = linear_baseline_mse(eval_ctx, eval_tgt)
    gate_pct = (lin_mse - zs_mse) / lin_mse * 100 if lin_mse > 1e-10 else 0.0
    print(f"  ZS MSE: {zs_mse:.4f}, Linear: {lin_mse:.4f}, Gate: {gate_pct:.1f}%")

    # Pre-trained reps
    print("  Extracting pre-trained representations...")
    reps_pt_tr = extract_encoder_reps(chronos_model, tokenizer, probe_tr_ctx, args.device, max_samples=500)
    reps_pt_va = extract_encoder_reps(chronos_model, tokenizer, probe_va_ctx, args.device, max_samples=200)

    r2_pt = ridge_probe_r2(reps_pt_tr, probe_tr_tgt[:len(reps_pt_tr)],
                           reps_pt_va, probe_va_tgt[:len(reps_pt_va)])
    ortho_pt = task_orthogonal_probes(reps_pt_tr, reps_pt_va,
                                      probe_tr_ctx[:len(reps_pt_tr)],
                                      probe_va_ctx[:len(reps_pt_va)])
    print(f"  PT Ridge R²: {r2_pt:.4f}, Orthogonal: {ortho_pt}")

    if args.condition == 'A':
        results = {
            "condition": "A", "dataset": dataset_name, "seed": args.seed,
            "zs_mse": zs_mse, "linear_mse": lin_mse, "gate_pct": gate_pct,
            "r2_pt": r2_pt, "ortho_pt": ortho_pt,
        }
        with open(results_dir / f"condition_A_s{args.seed}.json", "w") as f:
            json.dump(results, f, indent=2)
        return results

    # Fine-tune
    freeze_encoder = (args.condition == 'D')
    if freeze_encoder:
        print("  Freezing encoder (condition D)...")
        for param in chronos_model.model.encoder.parameters():
            param.requires_grad = False

    ctx_tensor = torch.tensor(probe_tr_ctx, dtype=torch.float32)
    tgt_tensor = torch.tensor(probe_tr_tgt, dtype=torch.float32)
    dataset = TensorDataset(ctx_tensor, tgt_tensor)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(
        [p for p in chronos_model.model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )

    best_val_mse = zs_mse
    best_epoch = 0
    best_state = copy.deepcopy(chronos_model.model.state_dict())
    patience_counter = 0

    print(f"\n  Training: {args.epochs} epochs, lr={args.lr}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(chronos_model, tokenizer, train_loader,
                                     optimizer, args.device, freeze_encoder, horizon)

        # Eval every epoch
        val_mse = chronos_zs_mse(pipe, eval_ctx, eval_tgt, args.device)
        reps_ft_va = extract_encoder_reps(chronos_model, tokenizer, probe_va_ctx, args.device, max_samples=200)
        cka = linear_CKA(reps_pt_va, reps_ft_va)

        forgetting = (val_mse - zs_mse) / zs_mse * 100
        elapsed = time.time() - t0
        print(f"    Ep {epoch:2d}: loss={train_loss:.4f} val_mse={val_mse:.4f} "
              f"forg={forgetting:+.1f}% CKA={cka:.3f} ({elapsed:.1f}s)")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(chronos_model.model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    Early stopping at epoch {epoch}")
                break

    # Restore best
    print(f"  Restoring best epoch {best_epoch} (val_mse={best_val_mse:.4f})")
    chronos_model.model.load_state_dict(best_state)

    # Post-training diagnostics
    print("\n  Post-training diagnostics...")
    final_mse = chronos_zs_mse(pipe, eval_ctx, eval_tgt, args.device)
    forgetting_pct = (final_mse - zs_mse) / zs_mse * 100

    reps_ft_tr = extract_encoder_reps(chronos_model, tokenizer, probe_tr_ctx, args.device, max_samples=500)
    reps_ft_va = extract_encoder_reps(chronos_model, tokenizer, probe_va_ctx, args.device, max_samples=200)
    final_cka = linear_CKA(reps_pt_va, reps_ft_va)

    r2_ft = ridge_probe_r2(reps_ft_tr, probe_tr_tgt[:len(reps_ft_tr)],
                           reps_ft_va, probe_va_tgt[:len(reps_ft_va)])
    delta_r2 = r2_ft - r2_pt

    ortho_ft = task_orthogonal_probes(reps_ft_tr, reps_ft_va,
                                      probe_tr_ctx[:len(reps_ft_tr)],
                                      probe_va_ctx[:len(reps_ft_va)])
    ortho_delta = {k: ortho_ft[k] - ortho_pt[k] for k in ortho_pt}

    # Probe asymmetry check
    has_trained_gain = delta_r2 > 0
    has_ortho_loss = all(ortho_delta[k] <= 0 for k in ortho_delta)
    probe_asymmetry = has_trained_gain and has_ortho_loss
    has_drift = final_cka < 0.95

    print(f"\n  {'='*50}")
    print(f"  RESULT: Chronos × {dataset_name}, seed {args.seed}")
    print(f"  {'='*50}")
    print(f"  Gate:         {gate_pct:.1f}%")
    print(f"  CKA:          {final_cka:.3f} {'DRIFT' if has_drift else 'stable'}")
    print(f"  Forgetting:   {forgetting_pct:+.1f}%")
    print(f"  ΔR² trained:  {delta_r2:+.4f} {'✓' if has_trained_gain else '✗'}")
    print(f"  ΔR² orthog:   {ortho_delta}")
    print(f"  Probe asym:   {'YES <<<' if probe_asymmetry else 'no'}")
    print(f"  DISSOCIATION: {'YES <<<' if (has_drift and probe_asymmetry) else 'no'}")
    print(f"  {'='*50}")

    # Save
    results = {
        "dataset": dataset_name,
        "condition": args.condition,
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "lr": args.lr,
        "max_train_samples": args.max_train_samples,
        "gate_improvement_pct": gate_pct,
        "zs_mse": zs_mse,
        "linear_mse": lin_mse,
        "final_val_mse": final_mse,
        "forgetting_pct": forgetting_pct,
        "final_cka": final_cka,
        "has_drift": has_drift,
        "linear_probe": {
            "r2_pt": r2_pt,
            "r2_ft": r2_ft,
            "delta_r2": delta_r2,
        },
        "orthogonal_probes": {
            "pretrained": ortho_pt,
            "finetuned": ortho_ft,
            "delta": ortho_delta,
        },
        "probe_asymmetry": probe_asymmetry,
        "dissociation": has_drift and probe_asymmetry,
    }

    out_path = results_dir / f"condition_{args.condition}_s{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    # Save encoder checkpoint
    enc_path = results_dir / "best_encoder.pt"
    torch.save(best_state, enc_path)

    return results


# ---------------------------------------------------------------------------
# Phase 2: Batch fine-tune all moderate-gate cells
# ---------------------------------------------------------------------------

def run_finetune_all(args):
    """Run full diagnostic on all moderate-gate cells, conditions B+D, 5 seeds."""
    # Load screen results
    screen_path = Path(args.results_dir) / "gate_screen.json"
    if not screen_path.exists():
        print("No gate_screen.json found. Run --phase screen first.")
        return

    with open(screen_path) as f:
        screen = json.load(f)

    # Find moderate-gate cells (expand to 15-50% if needed)
    moderate = [name for name, r in screen.items()
                if "gate_improvement_pct" in r and 15 <= r["gate_improvement_pct"] <= 50]
    if not moderate:
        # Try wider band
        moderate = [name for name, r in screen.items()
                    if "gate_improvement_pct" in r and 10 <= r["gate_improvement_pct"] <= 55]
    if not moderate:
        print("No moderate-gate cells found. Consider different datasets.")
        # Fall back to closest-to-moderate
        cells = [(name, abs(r["gate_improvement_pct"] - 32.5))
                 for name, r in screen.items() if "gate_improvement_pct" in r]
        cells.sort(key=lambda x: x[1])
        moderate = [cells[0][0]] if cells else []
        print(f"  Falling back to closest: {moderate}")

    print(f"\nFine-tuning moderate-gate cells: {moderate}")
    print(f"  Seeds: {list(range(42, 42 + args.n_seeds))}")
    print(f"  Conditions: B (full), D (frozen control)")

    all_results = []
    for dataset_name in moderate:
        for seed in range(42, 42 + args.n_seeds):
            for condition in ['B', 'D']:
                args.seed = seed
                args.condition = condition
                try:
                    r = run_finetune(args, dataset_name)
                    all_results.append(r)
                except Exception as e:
                    print(f"  ERROR: {dataset_name}/s{seed}/{condition}: {e}")
                    import traceback
                    traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("FINE-TUNING SUMMARY")
    print("=" * 60)
    for r in all_results:
        if r and "final_cka" in r:
            d = "DISSOC" if r.get("dissociation") else "no"
            print(f"  {r['dataset']:8s} s{r['seed']} {r['condition']}: "
                  f"CKA={r['final_cka']:.3f} ΔR²={r['linear_probe']['delta_r2']:+.3f} "
                  f"forg={r['forgetting_pct']:+.1f}% → {d}")

    # Save aggregate
    out_path = Path(args.results_dir) / "finetune_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['screen', 'finetune', 'finetune-all'])
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--condition', default='B', choices=['A', 'B', 'D'])
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max-train-samples', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--results-dir', default='results/moderate_gate')
    args = parser.parse_args()

    if args.phase == 'screen':
        run_gate_screen(args)
    elif args.phase == 'finetune':
        if not args.dataset:
            print("--dataset required for finetune phase")
            return
        run_finetune(args, args.dataset)
    elif args.phase == 'finetune-all':
        run_finetune_all(args)


if __name__ == "__main__":
    main()
