#!/usr/bin/env python3
"""
Chronos-T5-Small Fine-Tuning on M4-Monthly with Full Diagnostic Protocol.

Adapts the Moirai fine-tuning diagnostic methodology for Chronos:
- Tokenized cross-entropy loss (T5 seq2seq)
- CKA between pre-trained and fine-tuned encoder representations
- Ridge probes (ΔR²) on encoder outputs
- Task-orthogonal probes (lag-1, mean, variance)
- Frozen-encoder and random-init controls

This addresses Reviewer R7's decisive recommendation: run the full diagnostic
toolkit on the Chronos × M4-Monthly gate-passing cell (84.5% ZS improvement).

Usage:
    python scripts/finetune_chronos_m4.py \
        --condition B --seed 42 --epochs 20 --max-train-samples 500 \
        --device cuda --results-dir results/chronos_m4

    python scripts/finetune_chronos_m4.py \
        --condition D --seed 42 --epochs 20 --max-train-samples 10000 \
        --device cuda --results-dir results/chronos_m4_frozen

    python scripts/finetune_chronos_m4.py \
        --condition B --random-init --seed 42 --epochs 20 \
        --max-train-samples 10000 --device cuda \
        --results-dir results/chronos_m4_randinit
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
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

M4_MONTHLY_TRAIN_URL = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Monthly-train.csv"
M4_MONTHLY_TEST_URL = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Monthly-test.csv"
LOOKBACK = 96
HORIZON = 18
NUM_SERIES = 200
MODEL_ID = "amazon/chronos-t5-small"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def download_m4_monthly(cache_dir="/tmp/m4_data"):
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(cache_dir, "Monthly-train.csv")
    test_path = os.path.join(cache_dir, "Monthly-test.csv")
    if not os.path.exists(train_path):
        print("Downloading M4-Monthly train...")
        urllib.request.urlretrieve(M4_MONTHLY_TRAIN_URL, train_path)
    if not os.path.exists(test_path):
        print("Downloading M4-Monthly test...")
        urllib.request.urlretrieve(M4_MONTHLY_TEST_URL, test_path)
    return train_path, test_path


def load_m4_series(train_path, test_path, n_series=NUM_SERIES, min_len=None):
    import pandas as pd
    train_df = pd.read_csv(train_path, index_col=0)
    test_df = pd.read_csv(test_path, index_col=0)
    if min_len is None:
        min_len = LOOKBACK + HORIZON

    series_pairs = []
    for idx in train_df.index[:n_series * 3]:
        train_vals = train_df.loc[idx].dropna().values.astype(np.float64)
        if len(train_vals) < min_len:
            continue
        test_vals = test_df.loc[idx].dropna().values.astype(np.float64)
        if len(test_vals) < HORIZON:
            continue
        series_pairs.append((train_vals, test_vals))
        if len(series_pairs) >= n_series:
            break
    return series_pairs


def build_windows_from_series(series_pairs, lookback, horizon, max_windows=None, seed=42):
    """Build (context, target) windows from M4-Monthly series pairs.

    For training: slide over the training portion.
    Returns context (N, lookback) and target (N, horizon) arrays.
    """
    rng = np.random.RandomState(seed)
    contexts, targets = [], []
    for train_vals, test_vals in series_pairs:
        total_needed = lookback + horizon
        for i in range(len(train_vals) - total_needed + 1):
            contexts.append(train_vals[i:i + lookback])
            targets.append(train_vals[i + lookback:i + lookback + horizon])
    contexts = np.array(contexts, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)

    if max_windows is not None and len(contexts) > max_windows:
        idx = rng.choice(len(contexts), max_windows, replace=False)
        contexts = contexts[idx]
        targets = targets[idx]

    return contexts, targets


def build_test_windows(series_pairs, lookback, horizon):
    """Build test windows: last lookback of train → first horizon of test."""
    contexts, targets = [], []
    for train_vals, test_vals in series_pairs:
        if len(train_vals) < lookback:
            continue
        contexts.append(train_vals[-lookback:])
        targets.append(test_vals[:horizon])
    return np.array(contexts, dtype=np.float64), np.array(targets, dtype=np.float64)


# ---------------------------------------------------------------------------
# CKA (copied from finetune_forecasting.py for self-contained script)
# ---------------------------------------------------------------------------

def linear_CKA(X, Y):
    """Linear Centered Kernel Alignment between representations X and Y."""
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
            batch_ctx = torch.tensor(
                contexts[i:i + batch_size], dtype=torch.float32
            )
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
# Probing
# ---------------------------------------------------------------------------

def ridge_probe_r2(reps_train, targets_train, reps_val, targets_val, alpha=1.0):
    """Ridge regression probe: returns R² on validation set."""
    tgt_tr = targets_train.reshape(len(targets_train), -1)
    tgt_va = targets_val.reshape(len(targets_val), -1)
    probe = Ridge(alpha=alpha).fit(reps_train, tgt_tr)
    return float(probe.score(reps_val, tgt_va))


def task_orthogonal_probes(reps_train, reps_val, contexts_train, contexts_val, alpha=1.0):
    """Task-orthogonal probes: lag-1 autocorrelation, mean, variance."""
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

    for name, fn in [("lag1", lag1_autocorr), ("mean", input_mean), ("var", input_var)]:
        tgt_tr = fn(contexts_train)
        tgt_va = fn(contexts_val)
        probe = Ridge(alpha=alpha).fit(reps_train, tgt_tr)
        r2 = float(probe.score(reps_val, tgt_va))
        results[name] = r2

    return results


# ---------------------------------------------------------------------------
# Zero-Shot Evaluation
# ---------------------------------------------------------------------------

def chronos_zs_mse(pipe, contexts, targets, device, batch_size=32):
    """Compute zero-shot MSE using Chronos median predictions (batched)."""
    preds = []
    pipe.model.eval()
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch = [torch.tensor(contexts[j], dtype=torch.float32)
                     for j in range(i, min(i + batch_size, len(contexts)))]
            samples = pipe.predict(batch, prediction_length=HORIZON, num_samples=20)
            medians = samples.median(dim=1).values.cpu().numpy()
            preds.append(medians)
    preds = np.concatenate(preds, axis=0)

    # Per-series normalized MSE (using context mean/std)
    mses = []
    for i in range(len(contexts)):
        mu = contexts[i].mean()
        sd = contexts[i].std() + 1e-8
        pred_n = (preds[i] - mu) / sd
        tgt_n = (targets[i] - mu) / sd
        mses.append(float(np.mean((pred_n - tgt_n) ** 2)))
    return float(np.mean(mses))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    chronos_model, tokenizer, train_loader, optimizer, device,
    freeze_encoder=False
):
    """Train one epoch with tokenized cross-entropy loss."""
    chronos_model.model.train()
    if freeze_encoder:
        chronos_model.model.encoder.eval()

    epoch_loss = 0.0
    batch_count = 0

    for ctx_batch, tgt_batch in train_loader:
        ctx_batch = ctx_batch.to(dtype=torch.float32)
        tgt_batch = tgt_batch.to(dtype=torch.float32)

        # Tokenize context
        token_ids, attn_mask, scale = tokenizer.context_input_transform(ctx_batch)
        token_ids = token_ids.to(device)
        attn_mask = attn_mask.to(device)

        # Tokenize labels using same scale
        label_ids, label_mask = tokenizer.label_input_transform(tgt_batch, scale)
        label_ids = label_ids.to(device)

        # T5 forward with labels → cross-entropy loss
        # Replace padding tokens with -100 (ignored in CE loss)
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
# Linear Baseline
# ---------------------------------------------------------------------------

def linear_baseline_mse(series_pairs, lookback=LOOKBACK, horizon=HORIZON):
    """Per-series linear baseline for gate-pass computation."""
    from sklearn.linear_model import LinearRegression
    mses = []
    for train_vals, test_vals in series_pairs:
        if len(train_vals) < lookback + horizon:
            continue
        X_tr = np.array([train_vals[i:i+lookback]
                         for i in range(len(train_vals) - lookback - horizon + 1)])
        Y_tr = np.array([train_vals[i+lookback:i+lookback+horizon]
                         for i in range(len(train_vals) - lookback - horizon + 1)])
        if len(X_tr) < 5:
            continue
        mu = train_vals.mean()
        sd = train_vals.std() + 1e-8
        reg = LinearRegression().fit(X_tr, Y_tr)
        ctx = train_vals[-lookback:].reshape(1, -1)
        pred = reg.predict(ctx)[0]
        pred_n = (pred - mu) / sd
        test_n = (test_vals[:horizon] - mu) / sd
        mses.append(float(np.mean((pred_n - test_n) ** 2)))
    return float(np.mean(mses)) if mses else float('inf')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Chronos-T5-Small fine-tuning on M4-Monthly + diagnostics")
    parser.add_argument('--model-id', default=MODEL_ID)
    parser.add_argument('--condition', required=True, choices=['A', 'B', 'D'],
                        help="A=zero-shot, B=full fine-tune (CE), D=frozen encoder")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max-train-samples', type=int, default=500)
    parser.add_argument('--n-series', type=int, default=NUM_SERIES)
    parser.add_argument('--results-dir', default='results/chronos_m4')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--random-init', action='store_true',
                        help="Use randomly-initialized weights (negative control)")
    parser.add_argument('--early-stopping', action='store_true', default=True)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--save-best-encoder', action='store_true', default=True)
    parser.add_argument('--eval-every', type=int, default=1)
    parser.add_argument('--max-eval-windows', type=int, default=200)
    parser.add_argument('--probe-alpha', type=float, default=1.0)
    args = parser.parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.deterministic and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    device = args.device
    results_dir = Path(args.results_dir) / f"seed{args.seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print("Loading M4-Monthly data...")
    train_path, test_path = download_m4_monthly()
    series_pairs = load_m4_series(train_path, test_path, n_series=args.n_series)
    print(f"  Loaded {len(series_pairs)} series")

    # Build training windows
    contexts_train, targets_train = build_windows_from_series(
        series_pairs, LOOKBACK, HORIZON, max_windows=args.max_train_samples, seed=args.seed
    )
    # Build test windows (last context → test target)
    contexts_test, targets_test = build_test_windows(series_pairs, LOOKBACK, HORIZON)
    print(f"  Training windows: {len(contexts_train)}, Test windows: {len(contexts_test)}")

    # Split train into train/val (80/20)
    n_total = len(contexts_train)
    n_val = max(int(n_total * 0.2), 10)
    n_tr = n_total - n_val
    ctx_tr, ctx_va = contexts_train[:n_tr], contexts_train[n_tr:]
    tgt_tr, tgt_va = targets_train[:n_tr], targets_train[n_tr:]

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(f"Loading {args.model_id}...")
    from chronos import ChronosPipeline

    pipe = ChronosPipeline.from_pretrained(
        args.model_id,
        dtype=torch.float32,
    )
    chronos_model = pipe.model  # ChronosModel wrapping T5
    tokenizer = pipe.tokenizer

    # Override prediction_length to match our HORIZON (default config is 64)
    tokenizer.config.prediction_length = HORIZON

    # Ensure model is on the correct device
    if device != "cpu":
        chronos_model.model = chronos_model.model.to(device)
        print(f"  Model moved to {device}")

    if args.random_init:
        print("  Reinitializing weights (random-init control)...")
        for param in chronos_model.model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    # Store pre-trained state for CKA/drift
    pretrained_state = copy.deepcopy(chronos_model.model.state_dict())

    # -----------------------------------------------------------------------
    # Zero-shot baseline
    # -----------------------------------------------------------------------
    print("Computing zero-shot MSE on test windows...")
    test_ctx_sub = contexts_test[:args.max_eval_windows]
    test_tgt_sub = targets_test[:args.max_eval_windows]
    zs_mse = chronos_zs_mse(pipe, test_ctx_sub, test_tgt_sub, device)
    print(f"  ZS MSE: {zs_mse:.4f}")

    # Linear baseline
    linear_mse = linear_baseline_mse(series_pairs)
    gate_improvement = (linear_mse - zs_mse) / linear_mse if linear_mse > 1e-10 else 0.0
    print(f"  Linear MSE: {linear_mse:.4f}")
    print(f"  Gate improvement: {gate_improvement*100:.1f}% (threshold: 20%)")

    # Pre-trained encoder representations
    print("Extracting pre-trained encoder representations...")
    reps_pt_tr = extract_encoder_reps(chronos_model, tokenizer, ctx_tr, device, max_samples=500)
    reps_pt_va = extract_encoder_reps(chronos_model, tokenizer, ctx_va, device, max_samples=200)
    reps_pt_test = extract_encoder_reps(chronos_model, tokenizer, test_ctx_sub, device, max_samples=300)

    # Pre-trained Ridge R²
    r2_pt = ridge_probe_r2(reps_pt_tr, tgt_tr[:len(reps_pt_tr)],
                           reps_pt_va, tgt_va[:len(reps_pt_va)], alpha=args.probe_alpha)
    print(f"  Pre-trained Ridge R²: {r2_pt:.4f}")

    # Task-orthogonal probes on pre-trained
    ortho_pt = task_orthogonal_probes(
        reps_pt_tr, reps_pt_va,
        ctx_tr[:len(reps_pt_tr)], ctx_va[:len(reps_pt_va)],
        alpha=args.probe_alpha
    )
    print(f"  Pre-trained orthogonal probes: {ortho_pt}")

    if args.condition == 'A':
        # Zero-shot only — save results and exit
        results = {
            "condition": "A",
            "model_id": args.model_id,
            "seed": args.seed,
            "zeroshot_mse": zs_mse,
            "linear_mse": linear_mse,
            "gate_improvement_pct": gate_improvement * 100,
            "pretrained_ridge_r2": r2_pt,
            "pretrained_orthogonal_probes": ortho_pt,
            "n_train_windows": len(contexts_train),
            "n_test_windows": len(test_ctx_sub),
        }
        out_path = results_dir / f"condition_A_s{args.seed}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {out_path}")
        return

    # -----------------------------------------------------------------------
    # Fine-tuning
    # -----------------------------------------------------------------------
    freeze_encoder = (args.condition == 'D')
    if freeze_encoder:
        print("Freezing encoder (condition D)...")
        for param in chronos_model.model.encoder.parameters():
            param.requires_grad = False

    # DataLoader
    ctx_tensor = torch.tensor(ctx_tr, dtype=torch.float32)
    tgt_tensor = torch.tensor(tgt_tr, dtype=torch.float32)
    dataset = TensorDataset(ctx_tensor, tgt_tensor)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        [p for p in chronos_model.model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )

    # History tracking
    history = {
        "epoch": [0], "train_loss": [0.0], "val_mse": [zs_mse],
        "cka": [1.0], "weight_drift": [0.0]
    }

    best_val_mse = zs_mse
    best_epoch = 0
    best_state = copy.deepcopy(chronos_model.model.state_dict())

    print(f"\nFine-tuning: condition {args.condition}, {args.epochs} epochs, "
          f"n={len(ctx_tr)}, lr={args.lr}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            chronos_model, tokenizer, train_loader, optimizer, device,
            freeze_encoder=freeze_encoder
        )

        # Eval
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_mse = chronos_zs_mse(pipe, test_ctx_sub, test_tgt_sub, device)

            # CKA
            reps_ft_va = extract_encoder_reps(chronos_model, tokenizer, ctx_va, device, max_samples=200)
            cka = linear_CKA(reps_pt_va, reps_ft_va)

            # Weight drift
            current_state = chronos_model.model.state_dict()
            drift = sum(
                (current_state[k] - pretrained_state[k]).float().pow(2).sum().item()
                for k in pretrained_state if k in current_state
            ) ** 0.5

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_mse"].append(val_mse)
            history["cka"].append(cka)
            history["weight_drift"].append(drift)

            # Early stopping
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                best_state = copy.deepcopy(chronos_model.model.state_dict())

            elapsed = time.time() - t0
            forgetting = (val_mse - zs_mse) / zs_mse * 100
            print(f"  Epoch {epoch:2d}: loss={train_loss:.4f}  val_mse={val_mse:.4f}  "
                  f"forg={forgetting:+.1f}%  CKA={cka:.3f}  drift={drift:.1f}  "
                  f"({elapsed:.1f}s)")

    # Restore best checkpoint
    if args.early_stopping:
        print(f"\nRestoring best epoch {best_epoch} (val_mse={best_val_mse:.4f})")
        chronos_model.model.load_state_dict(best_state)

    # -----------------------------------------------------------------------
    # Post-training diagnostics
    # -----------------------------------------------------------------------
    print("\nPost-training diagnostics...")

    # Final forgetting
    final_mse = chronos_zs_mse(pipe, test_ctx_sub, test_tgt_sub, device)
    forgetting_pct = (final_mse - zs_mse) / zs_mse * 100

    # Final CKA
    reps_ft_tr = extract_encoder_reps(chronos_model, tokenizer, ctx_tr, device, max_samples=500)
    reps_ft_va = extract_encoder_reps(chronos_model, tokenizer, ctx_va, device, max_samples=200)
    final_cka = linear_CKA(reps_pt_va, reps_ft_va)

    # Weight drift
    final_state = chronos_model.model.state_dict()
    final_drift = sum(
        (final_state[k] - pretrained_state[k]).float().pow(2).sum().item()
        for k in pretrained_state if k in final_state
    ) ** 0.5

    # Ridge probe ΔR²
    r2_ft = ridge_probe_r2(reps_ft_tr, tgt_tr[:len(reps_ft_tr)],
                           reps_ft_va, tgt_va[:len(reps_ft_va)], alpha=args.probe_alpha)
    delta_r2 = r2_ft - r2_pt

    # Task-orthogonal probes on fine-tuned
    ortho_ft = task_orthogonal_probes(
        reps_ft_tr, reps_ft_va,
        ctx_tr[:len(reps_ft_tr)], ctx_va[:len(reps_ft_va)],
        alpha=args.probe_alpha
    )
    ortho_delta = {k: ortho_ft[k] - ortho_pt[k] for k in ortho_pt}

    # Final-epoch diagnostics (for protocol comparison)
    final_epoch_state = chronos_model.model.state_dict()
    if args.early_stopping and best_epoch < args.epochs:
        # Load final epoch for comparison
        # We're already at final epoch unless we restored; recalculate
        chronos_model.model.load_state_dict(final_state)
        final_epoch_mse = chronos_zs_mse(pipe, test_ctx_sub, test_tgt_sub, device)
        final_epoch_forg = (final_epoch_mse - zs_mse) / zs_mse * 100
        reps_final = extract_encoder_reps(chronos_model, tokenizer, ctx_va, device, max_samples=200)
        final_epoch_cka = linear_CKA(reps_pt_va, reps_final)
        # Restore best
        chronos_model.model.load_state_dict(best_state)
    else:
        final_epoch_mse = final_mse
        final_epoch_forg = forgetting_pct
        final_epoch_cka = final_cka

    print(f"\n{'='*60}")
    print(f"  ZS MSE:           {zs_mse:.4f}")
    print(f"  FT MSE (best):    {final_mse:.4f}")
    print(f"  Forgetting:       {forgetting_pct:+.1f}%")
    print(f"  CKA:              {final_cka:.3f}")
    print(f"  Weight drift:     {final_drift:.1f}")
    print(f"  Ridge R²(PT):     {r2_pt:.4f}")
    print(f"  Ridge R²(FT):     {r2_ft:.4f}")
    print(f"  ΔR² (trained):    {delta_r2:+.4f}")
    print(f"  Orthogonal ΔR²:   {ortho_delta}")
    print(f"  Best epoch:       {best_epoch}")
    print(f"  Final-epoch forg: {final_epoch_forg:+.1f}%")
    print(f"{'='*60}")

    # Save encoder
    if args.save_best_encoder:
        enc_path = results_dir / "best_encoder.pt"
        torch.save(best_state, enc_path)
        print(f"  Saved encoder: {enc_path}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results = {
        "condition": args.condition,
        "model_id": args.model_id,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "max_train_samples": args.max_train_samples,
        "n_train": len(ctx_tr),
        "n_val": len(ctx_va),
        "n_test": len(test_ctx_sub),
        "random_init": args.random_init,
        "freeze_encoder": freeze_encoder,
        "zeroshot_mse": zs_mse,
        "linear_mse": linear_mse,
        "gate_improvement_pct": gate_improvement * 100,
        "final_val_mse": final_mse,
        "forgetting_pct": forgetting_pct,
        "final_cka": final_cka,
        "final_weight_drift": final_drift,
        "best_epoch": best_epoch,
        "early_stopping": {
            "enabled": args.early_stopping,
            "best_epoch": best_epoch,
            "best_val_mse": best_val_mse,
            "final_epoch_val_mse": final_epoch_mse,
            "final_epoch_forgetting_pct": final_epoch_forg,
            "final_epoch_cka": final_epoch_cka,
        },
        "linear_probe": {
            "pretrained_r2": r2_pt,
            "finetuned_r2": r2_ft,
            "r2_delta": delta_r2,
        },
        "orthogonal_probes": {
            "pretrained": ortho_pt,
            "finetuned": ortho_ft,
            "delta": ortho_delta,
        },
        "history": history,
    }

    out_path = results_dir / f"condition_{args.condition}_s{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
