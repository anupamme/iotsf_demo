#!/usr/bin/env python3
"""
Chronos-T5-Small fine-tuning with DIRECT MSE LOSS (bypassing tokenizer).

Hypothesis: Chronos's native tokenized CE loss shields the encoder from
specialization pressure — the tokenizer's discrete bins structure the output
space so the decoder can produce good forecasts without encoder specialization.
A direct MSE regression head on encoder mean-pooled outputs forces the encoder
to carry task-specific information, creating Moirai-like specialization pressure.

If this produces probe asymmetry (trained ΔR² > 0 while orthogonal ΔR² ≤ 0),
it's a second-backbone dissociation — same Chronos encoder, different training
objective reveals the mechanism.

Usage:
    python scripts/chronos_mse_finetune.py --dataset ETTh1 --seed 42 --device cuda
    python scripts/chronos_mse_finetune.py --dataset ETTh1 --seed 42 --condition D --device cuda
    python scripts/chronos_mse_finetune.py --phase batch --device cuda
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
# Dataset configs (same as moderate_gate_screen.py)
# ---------------------------------------------------------------------------

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
    "ETTm2": {
        "local": "data/forecasting/ETTm2.csv",
        "target_col": "OT",
        "n_train": 34560,
        "n_val": 11520,
        "lookback": 96,
        "horizon": 24,
    },
    "Weather": {
        "local": "data/forecasting/Weather.csv",
        "target_col": "OT",
        "n_train": 49064,   # 70% of 70,092
        "n_val": 7009,      # 10%
        "lookback": 96,
        "horizon": 24,
    },
    "Electricity": {
        "local": "data/forecasting/Electricity.csv",
        "target_col": "OT",
        "n_train": 18345,   # 70% of 26,208
        "n_val": 2620,      # 10%
        "lookback": 96,
        "horizon": 24,
    },
    "ETTm1": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        "target_col": "OT",
        "n_train": 34560,
        "n_val": 11520,
        "lookback": 96,
        "horizon": 24,
    },
}

MODEL_ID = "amazon/chronos-t5-small"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def download_dataset(name, cache_dir="/tmp/gate_screen_data"):
    os.makedirs(cache_dir, exist_ok=True)
    cfg = DATASETS[name]
    # A dataset may name a file already in the repo instead of a URL. Preferred where we have the
    # CSV locally: it removes a network dependency and pins the exact data the runs used.
    if cfg.get("local"):
        if not os.path.exists(cfg["local"]):
            raise FileNotFoundError(f"{name}: {cfg['local']} not found")
        return cfg["local"]
    local_path = os.path.join(cache_dir, f"{name}.csv")
    if not os.path.exists(local_path):
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(cfg["url"], local_path)
    return local_path


def load_series(name, cache_dir="/tmp/gate_screen_data"):
    import pandas as pd
    cfg = DATASETS[name]
    path = download_dataset(name, cache_dir)
    df = pd.read_csv(path)
    values = df[cfg["target_col"]].values.astype(np.float64)
    n_train = cfg["n_train"]
    n_val = cfg["n_val"]
    return values[:n_train], values[n_train:n_train+n_val], values[n_train+n_val:]


# Context windows flatter than this carry no scale, and the protocol normalises targets by the
# context std. Dividing by ~0 turns a normal target into a ~1e8 one and detonates training: on
# Chronos/ETTm2 both the ridge and the AdamW head diverged to ~1e15. The threshold is not tuned --
# degenerate windows sit at 0 or 2.2e-16 (machine epsilon) while the smallest genuine context std
# in any dataset we use is ~0.1, eight orders of magnitude away. It is a no-op on ETTh1, ETTh2 and
# Weather, which contain no constant windows.
MIN_CONTEXT_STD = 1e-8


def build_windows(series, lookback, horizon, max_windows=None, seed=42):
    n_total = len(series) - lookback - horizon + 1
    if n_total <= 0:
        return np.empty((0, lookback)), np.empty((0, horizon))
    contexts = np.array([series[i:i+lookback] for i in range(n_total)])
    targets = np.array([series[i+lookback:i+lookback+horizon] for i in range(n_total)])
    keep = contexts.std(axis=1) > MIN_CONTEXT_STD
    if not keep.all():
        print(f"    dropped {int((~keep).sum())}/{len(keep)} constant context windows "
              f"(std <= {MIN_CONTEXT_STD:g}; they cannot be scale-normalised)")
        contexts, targets = contexts[keep], targets[keep]
        if len(contexts) == 0:
            return np.empty((0, lookback)), np.empty((0, horizon))
    if max_windows and len(contexts) > max_windows:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(contexts), max_windows, replace=False)
        contexts, targets = contexts[idx], targets[idx]
    return contexts, targets


# ---------------------------------------------------------------------------
# CKA + Probes (reused)
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
# Representation Extraction (same as before — direct encoder access)
# ---------------------------------------------------------------------------

def extract_encoder_reps(model_t5, tokenizer, contexts, device, max_samples=500, batch_size=32):
    """Extract mean-pooled encoder representations from the T5 encoder."""
    model_t5.eval()
    encoder = model_t5.encoder
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
# MSE Regression Head
# ---------------------------------------------------------------------------

class MSEForecastHead(nn.Module):
    """Linear regression head on mean-pooled encoder output → horizon predictions."""

    def __init__(self, d_model, horizon):
        super().__init__()
        self.head = nn.Linear(d_model, horizon)

    def forward(self, encoder_output, attention_mask):
        # Mean pool over sequence dim
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (encoder_output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch_mse(t5_model, tokenizer, head, train_loader, optimizer,
                        device, freeze_encoder=False):
    """Train one epoch with direct MSE loss on encoder → linear head."""
    t5_model.train()
    head.train()
    if freeze_encoder:
        t5_model.encoder.eval()

    epoch_loss = 0.0
    batch_count = 0

    for ctx_batch, tgt_batch in train_loader:
        ctx_batch = ctx_batch.to(dtype=torch.float32)
        tgt_batch = tgt_batch.to(device, dtype=torch.float32)

        # Tokenize context (needed for encoder input)
        token_ids, attn_mask, scale = tokenizer.context_input_transform(ctx_batch)
        token_ids = token_ids.to(device)
        attn_mask = attn_mask.to(device)

        # Forward through encoder
        enc_out = t5_model.encoder(input_ids=token_ids, attention_mask=attn_mask)
        h = enc_out.last_hidden_state

        # MSE head prediction
        pred = head(h, attn_mask)

        # Normalize targets same way for fair MSE
        # Use per-sample z-score (context mean/std)
        ctx_mean = ctx_batch.mean(dim=1, keepdim=True).to(device)
        ctx_std = ctx_batch.std(dim=1, keepdim=True).to(device) + 1e-8
        tgt_norm = (tgt_batch - ctx_mean) / ctx_std
        pred_norm = pred  # head predicts in normalized space

        loss = nn.functional.mse_loss(pred_norm, tgt_norm)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(t5_model.parameters()) + list(head.parameters()), 1.0
        )
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    return epoch_loss / max(batch_count, 1)


def eval_mse(t5_model, tokenizer, head, contexts, targets, device, batch_size=32):
    """Evaluate MSE on a set of windows."""
    t5_model.eval()
    head.eval()
    total_mse = 0.0
    count = 0

    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            ctx = torch.tensor(contexts[i:i+batch_size], dtype=torch.float32)
            tgt = torch.tensor(targets[i:i+batch_size], dtype=torch.float32).to(device)

            token_ids, attn_mask, _ = tokenizer.context_input_transform(ctx)
            token_ids = token_ids.to(device)
            attn_mask = attn_mask.to(device)

            enc_out = t5_model.encoder(input_ids=token_ids, attention_mask=attn_mask)
            pred = head(enc_out.last_hidden_state, attn_mask)

            ctx_mean = ctx.mean(dim=1, keepdim=True).to(device)
            ctx_std = ctx.std(dim=1, keepdim=True).to(device) + 1e-8
            tgt_norm = (tgt - ctx_mean) / ctx_std

            mse = nn.functional.mse_loss(pred, tgt_norm, reduction='sum').item()
            total_mse += mse
            count += len(ctx)

    return total_mse / max(count, 1)


# ---------------------------------------------------------------------------
# Zero-shot MSE (using Chronos pipeline for fair comparison)
# ---------------------------------------------------------------------------

def chronos_zs_mse(pipe, contexts, targets, device, batch_size=32):
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


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args, dataset_name):
    from chronos import ChronosPipeline

    cfg = DATASETS[dataset_name]
    lookback = cfg["lookback"]
    horizon = cfg["horizon"]

    print(f"\n{'='*60}")
    print(f"MSE-LOSS CHRONOS: {dataset_name}, seed={args.seed}, "
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

    results_dir = Path(args.results_dir) / f"mse_{dataset_name.lower()}" / f"seed{args.seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_s, val_s, test_s = load_series(dataset_name)
    ctx_tr, tgt_tr = build_windows(train_s, lookback, horizon,
                                   max_windows=args.max_train_samples, seed=args.seed)
    ctx_va, tgt_va = build_windows(val_s, lookback, horizon, max_windows=200, seed=args.seed)
    # Held-out test windows. Fixed subsample seed (not args.seed) so every run scores the SAME
    # 200 windows -- removes test-set resampling from the seed-to-seed variance, identically
    # for B and D. Nothing selected on these: early stopping stays on eval_ctx (validation).
    ctx_test, tgt_test = build_windows(test_s, lookback, horizon, max_windows=200,
                                       seed=args.test_seed)

    eval_ctx = ctx_va if len(ctx_va) >= 20 else ctx_test
    eval_tgt = tgt_va if len(tgt_va) >= 20 else tgt_test

    # Guard the protocol: if validation were ever too small, eval_ctx would fall back to the test
    # windows and selection would silently leak into the reported held-out score. Fail loudly.
    if len(ctx_va) < 20:
        raise RuntimeError(
            "validation split too small; eval would fall back to the test windows, which would "
            "make the held-out score a selection set. Refusing to run.")

    # Train/probe split
    n_total = len(ctx_tr)
    n_pval = max(int(n_total * 0.2), 10)
    n_ptr = n_total - n_pval
    probe_tr_ctx, probe_va_ctx = ctx_tr[:n_ptr], ctx_tr[n_ptr:]
    probe_tr_tgt, probe_va_tgt = tgt_tr[:n_ptr], tgt_tr[n_ptr:]
    print(f"  Train: {n_ptr}, Probe-val: {n_pval}, Eval: {len(eval_ctx)}")

    # Load model
    print(f"  Loading {MODEL_ID}...")
    pipe = ChronosPipeline.from_pretrained(MODEL_ID, dtype=torch.float32)
    chronos_model = pipe.model
    tokenizer = pipe.tokenizer
    tokenizer.config.prediction_length = horizon
    t5_model = chronos_model.model  # the actual T5ForConditionalGeneration

    if args.device != "cpu":
        t5_model = t5_model.to(args.device)

    # Get encoder d_model
    d_model = t5_model.config.d_model
    print(f"  Encoder d_model: {d_model}")

    # Store pretrained state
    pretrained_state = copy.deepcopy(t5_model.encoder.state_dict())

    # Zero-shot baseline (using full Chronos pipeline)
    print("  Computing zero-shot MSE...")
    zs_mse = chronos_zs_mse(pipe, eval_ctx, eval_tgt, args.device)
    lin_mse = float(np.mean([
        np.mean(((LinearRegression().fit(
            np.arange(lookback).reshape(-1,1), eval_ctx[i]
        ).predict(np.arange(lookback, lookback+horizon).reshape(-1,1)) - eval_ctx[i].mean()) / (eval_ctx[i].std()+1e-8)
        - (eval_tgt[i] - eval_ctx[i].mean()) / (eval_ctx[i].std()+1e-8)) ** 2)
        for i in range(len(eval_ctx))
    ]))
    gate_pct = (lin_mse - zs_mse) / lin_mse * 100 if lin_mse > 1e-10 else 0.0
    print(f"  ZS MSE: {zs_mse:.4f}, Linear: {lin_mse:.4f}, Gate: {gate_pct:.1f}%")


    # Pre-trained encoder reps
    print("  Extracting pre-trained representations...")
    reps_pt_tr = extract_encoder_reps(t5_model, tokenizer, probe_tr_ctx, args.device, max_samples=500)
    reps_pt_va = extract_encoder_reps(t5_model, tokenizer, probe_va_ctx, args.device, max_samples=200)
    r2_pt = ridge_probe_r2(reps_pt_tr, probe_tr_tgt[:len(reps_pt_tr)],
                           reps_pt_va, probe_va_tgt[:len(reps_pt_va)])
    ortho_pt = task_orthogonal_probes(reps_pt_tr, reps_pt_va,
                                      probe_tr_ctx[:len(reps_pt_tr)],
                                      probe_va_ctx[:len(reps_pt_va)])
    print(f"  PT Ridge R²: {r2_pt:.4f}, Orthogonal: {ortho_pt}")

    if args.condition == 'A':
        results = {
            "condition": "A", "dataset": dataset_name, "seed": args.seed,
            "loss_type": "mse_head", "zs_mse": zs_mse, "linear_mse": lin_mse,
            "gate_pct": gate_pct, "r2_pt": r2_pt, "ortho_pt": ortho_pt,
        }
        with open(results_dir / f"condition_A_s{args.seed}.json", "w") as f:
            json.dump(results, f, indent=2)
        return results

    # Create MSE head
    head = MSEForecastHead(d_model, horizon).to(args.device)

    # Freeze encoder if condition D
    freeze_encoder = (args.condition == 'D')
    if freeze_encoder:
        print("  Freezing encoder (condition D)...")
        for param in t5_model.encoder.parameters():
            param.requires_grad = False

    # DataLoader
    ctx_tensor = torch.tensor(probe_tr_ctx, dtype=torch.float32)
    tgt_tensor = torch.tensor(probe_tr_tgt, dtype=torch.float32)
    dataset = TensorDataset(ctx_tensor, tgt_tensor)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Optimizer: encoder + head
    params = list(head.parameters())
    if not freeze_encoder:
        params += list(t5_model.encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    best_encoder_state = copy.deepcopy(t5_model.encoder.state_dict())
    best_head_state = copy.deepcopy(head.state_dict())
    patience_counter = 0

    print(f"\n  Training: {args.epochs} epochs, lr={args.lr}, MSE loss")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch_mse(t5_model, tokenizer, head, train_loader,
                                         optimizer, args.device, freeze_encoder)

        val_loss = eval_mse(t5_model, tokenizer, head, eval_ctx, eval_tgt, args.device)

        # CKA check
        reps_ft_va = extract_encoder_reps(t5_model, tokenizer, probe_va_ctx, args.device, max_samples=200)
        cka = linear_CKA(reps_pt_va, reps_ft_va)

        elapsed = time.time() - t0
        print(f"    Ep {epoch:2d}: loss={train_loss:.4f} val={val_loss:.4f} "
              f"CKA={cka:.3f} ({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_encoder_state = copy.deepcopy(t5_model.encoder.state_dict())
            best_head_state = copy.deepcopy(head.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    Early stopping at epoch {epoch}")
                break

    # Restore best
    print(f"  Restoring best epoch {best_epoch} (val_loss={best_val_loss:.4f})")
    t5_model.encoder.load_state_dict(best_encoder_state)
    head.load_state_dict(best_head_state)

    # Held-out test score of the checkpoint that validation selected. Selection (early stopping)
    # used eval_ctx only; this is the first and only time the test windows touch the model.
    #
    # Both test-side measurements happen HERE, after training, and never before it. chronos_zs_mse
    # calls pipe.predict(num_samples=20), which draws from the global torch RNG -- computing it up
    # front would advance that RNG and silently change head init and batch shuffling, so the run
    # would no longer reproduce the published validation numbers for the same seed (measured: a
    # 1.3% shift in best_val_loss on ETTh1/42). Ordering the extra measurement after training brings
    # these runs back to the published validation figures to within MPS run-to-run float noise
    # (0.04% on that same seed; torch.use_deterministic_algorithms is CUDA-only here). They are the
    # same models under the same protocol, additionally scored on windows nothing was selected on --
    # not bit-identical reruns.
    test_loss = eval_mse(t5_model, tokenizer, head, ctx_test, tgt_test, args.device)

    # The zero-shot reference must come from the PRETRAINED encoder, but `pipe` shares the module
    # object with t5_model, whose encoder now holds the fine-tuned weights. Swap the pretrained
    # weights back in for the duration of the measurement, then restore -- the post-training
    # diagnostics below (CKA, probes) need the fine-tuned encoder.
    t5_model.encoder.load_state_dict(pretrained_state)
    zs_mse_test = chronos_zs_mse(pipe, ctx_test, tgt_test, args.device)
    t5_model.encoder.load_state_dict(best_encoder_state)
    print(f"  ZS MSE (held-out test): {zs_mse_test:.4f}")
    print(f"  Held-out test loss: {test_loss:.4f} "
          f"(per-element {test_loss / horizon:.4f}, "
          f"forgetting {(test_loss / horizon - zs_mse_test) / zs_mse_test * 100:+.2f}%)")

    # Post-training diagnostics
    print("\n  Post-training diagnostics...")
    reps_ft_tr = extract_encoder_reps(t5_model, tokenizer, probe_tr_ctx, args.device, max_samples=500)
    reps_ft_va = extract_encoder_reps(t5_model, tokenizer, probe_va_ctx, args.device, max_samples=200)
    final_cka = linear_CKA(reps_pt_va, reps_ft_va)

    r2_ft = ridge_probe_r2(reps_ft_tr, probe_tr_tgt[:len(reps_ft_tr)],
                           reps_ft_va, probe_va_tgt[:len(reps_ft_va)])
    delta_r2 = r2_ft - r2_pt

    ortho_ft = task_orthogonal_probes(reps_ft_tr, reps_ft_va,
                                      probe_tr_ctx[:len(reps_ft_tr)],
                                      probe_va_ctx[:len(reps_ft_va)])
    ortho_delta = {k: ortho_ft[k] - ortho_pt[k] for k in ortho_pt}

    # Probe asymmetry
    has_trained_gain = delta_r2 > 0
    has_ortho_loss = all(ortho_delta[k] <= 0 for k in ortho_delta)
    probe_asymmetry = has_trained_gain and has_ortho_loss
    has_drift = final_cka < 0.95

    # Encoder weight drift
    final_encoder_state = t5_model.encoder.state_dict()
    weight_drift = sum(
        (final_encoder_state[k] - pretrained_state[k]).float().pow(2).sum().item()
        for k in pretrained_state if k in final_encoder_state
    ) ** 0.5

    print(f"\n  {'='*50}")
    print(f"  RESULT: MSE-Chronos × {dataset_name}, seed {args.seed}, cond {args.condition}")
    print(f"  {'='*50}")
    print(f"  Loss type:    DIRECT MSE (encoder → linear head)")
    print(f"  Gate:         {gate_pct:.1f}%")
    print(f"  CKA:          {final_cka:.3f} {'DRIFT' if has_drift else 'stable'}")
    print(f"  Weight drift: {weight_drift:.1f}")
    print(f"  ΔR² trained:  {delta_r2:+.4f} {'✓' if has_trained_gain else '✗'}")
    print(f"  ΔR² orthog:   {ortho_delta}")
    print(f"  Probe asym:   {'YES <<<' if probe_asymmetry else 'no'}")
    print(f"  DISSOCIATION: {'YES <<<' if (has_drift and probe_asymmetry) else 'no'}")
    print(f"  {'='*50}")

    results = {
        "dataset": dataset_name,
        "condition": args.condition,
        "seed": args.seed,
        "loss_type": "mse_head",
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "lr": args.lr,
        "max_train_samples": args.max_train_samples,
        "d_model": d_model,
        "gate_improvement_pct": gate_pct,
        "zs_mse": zs_mse,
        "linear_mse": lin_mse,
        "best_val_loss": best_val_loss,
        "final_cka": final_cka,
        "weight_drift": weight_drift,
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
        # ---- held-out test side (selection was on validation only) ----
        "test_seed": args.test_seed,
        "n_test_windows": int(len(ctx_test)),
        "zs_mse_test": zs_mse_test,
        "test_val_loss": test_loss,
        "test_mse_per_element": test_loss / horizon,
        "test_forgetting_pct": (test_loss / horizon - zs_mse_test) / zs_mse_test * 100,
    }

    out_path = results_dir / f"condition_{args.condition}_s{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    # Save encoder
    torch.save(best_encoder_state, results_dir / "best_encoder.pt")
    # The head was previously discarded, which forced a full re-run to score any new split.
    torch.save(best_head_state, results_dir / "best_head.pt")

    return results


def run_batch(args):
    """Run full experiment: 5 seeds × 2 datasets × conditions B+D."""
    datasets = ["ETTh1", "ETTh2"]
    seeds = list(range(42, 42 + args.n_seeds))
    conditions = ["B", "D"]

    all_results = []
    for dataset_name in datasets:
        for seed in seeds:
            for condition in conditions:
                args.seed = seed
                args.condition = condition
                try:
                    r = run_experiment(args, dataset_name)
                    all_results.append(r)
                except Exception as e:
                    print(f"  ERROR: {dataset_name}/s{seed}/{condition}: {e}")
                    import traceback
                    traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("MSE-LOSS CHRONOS — SUMMARY")
    print("=" * 60)
    for r in all_results:
        if r and "final_cka" in r:
            d = "DISSOC" if r.get("dissociation") else "no"
            pa = "ASYM" if r.get("probe_asymmetry") else "no"
            print(f"  {r['dataset']:6s} s{r['seed']} {r['condition']}: "
                  f"CKA={r['final_cka']:.3f} ΔR²={r['linear_probe']['delta_r2']:+.3f} "
                  f"asym={pa} → {d}")

    out_path = Path(args.results_dir) / "mse_chronos_summary.json"
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
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-train-samples', type=int, default=8000)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--test-seed', type=int, default=0,
                        help='subsample seed for the held-out test windows; fixed across runs')
    parser.add_argument('--results-dir', default='results/chronos_mse')
    args = parser.parse_args()

    if args.phase == 'batch':
        run_batch(args)
    else:
        run_experiment(args, args.dataset)


if __name__ == "__main__":
    main()
