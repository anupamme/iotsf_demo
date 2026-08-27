#!/usr/bin/env python3
"""
Condition D (frozen encoder) for Chronos-T5, trained to CONVERGENCE.

WHY THIS SCRIPT EXISTS
----------------------
scripts/chronos_mse_finetune.py gives both conditions the same 30-epoch budget. That budget is
generous for condition B, which early-stops at epoch 9 with its best epoch at 2, but it truncates
condition D badly: in results/v37_chronos_etth2 all three D seeds restored "best epoch 30" with
validation loss still falling nearly linearly (43.06 -> 41.32 over 30 epochs, no plateau). The
resulting B-D of -77.8 pp is therefore an artifact of the epoch budget, not a measurement of how
much encoder adaptation contributes. B-D only means something when BOTH conditions are trained to
their own convergence.

WHY CACHING IS EXACT
--------------------
In condition D the encoder is frozen (requires_grad=False) AND placed in eval() mode
(chronos_mse_finetune.py:213), so dropout is off and the mean-pooled encoder output is a
deterministic function of the context window. MSEForecastHead is a single nn.Linear on that pooled
vector, so caching the pooled features and training the head on them is mathematically identical to
re-running the encoder every epoch -- and about 500x faster, which is what makes training to
convergence affordable.

The validation metric is replicated exactly from eval_mse(): MSE summed over the horizon and
averaged over windows (i.e. per-element MSE x horizon), so numbers from this script are directly
comparable to the best_val_loss written by chronos_mse_finetune.py.

Usage:
  python scripts/chronos_frozen_converged.py --dataset ETTh2 --seed 42 --results-dir DIR
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from chronos_mse_finetune import (  # noqa: E402
    DATASETS, MODEL_ID, MSEForecastHead, build_windows, chronos_zs_mse,
    extract_encoder_reps, linear_CKA, load_series, ridge_probe_r2,
    task_orthogonal_probes,
)
from sklearn.linear_model import LinearRegression  # noqa: E402


def pooled_features(t5_model, tokenizer, contexts, device, batch_size=32):
    """Masked mean-pool of the frozen encoder's last hidden state -- the head's actual input."""
    t5_model.encoder.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            ctx = torch.tensor(contexts[i:i + batch_size], dtype=torch.float32)
            token_ids, attn_mask, _ = tokenizer.context_input_transform(ctx)
            token_ids, attn_mask = token_ids.to(device), attn_mask.to(device)
            h = t5_model.encoder(input_ids=token_ids, attention_mask=attn_mask).last_hidden_state
            m = attn_mask.unsqueeze(-1).float()
            out.append(((h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)).cpu())
    return torch.cat(out, dim=0)


def norm_targets(contexts, targets):
    """Per-window z-score of targets by context mean/std -- matches train_one_epoch_mse."""
    ctx = torch.tensor(contexts, dtype=torch.float32)
    tgt = torch.tensor(targets, dtype=torch.float32)
    mu = ctx.mean(dim=1, keepdim=True)
    sd = ctx.std(dim=1, keepdim=True) + 1e-8
    return (tgt - mu) / sd


RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0,
                100.0, 1e3, 1e4, 1e5)


def ridge_optimum(f_tr, y_tr, f_ev, y_ev, f_te=None, y_te=None, alphas=RIDGE_ALPHAS):
    """
    Best linear head obtainable on the frozen features, in closed form.

    This is the point of the whole exercise. MSEForecastHead is a single nn.Linear on the pooled
    features, so ridge searches exactly the same function class the AdamW head does, but reaches
    its optimum in closed form -- which removes the training budget as a confound entirely.

    alpha is selected on the same eval windows condition B early-stops on, so D gets eval-set
    selection too (over 16 alphas rather than over ~1500 checkpoints). That keeps B - D a
    comparison of two similarly-selected models rather than a selected one against an unselected
    one. The grid is log-spaced and the optimum must be interior -- a boundary optimum means the
    grid was too narrow and the number should not be trusted.

    Metric matches eval_head(): squared error summed over the horizon, averaged over windows.
    """
    from sklearn.linear_model import Ridge
    Xtr, Ytr = f_tr.numpy(), y_tr.numpy()
    Xev, Yev = f_ev.numpy(), y_ev.numpy()
    grid, fits = {}, {}
    for a in alphas:
        fits[a] = Ridge(alpha=a).fit(Xtr, Ytr)
        pred = fits[a].predict(Xev)
        grid[a] = float(((pred - Yev) ** 2).sum(axis=1).mean())
    best_a = min(grid, key=grid.get)
    out = {"alpha": best_a, "best_val_loss": grid[best_a],
           "interior": bool(alphas[0] < best_a < alphas[-1]),
           "grid": {str(a): v for a, v in grid.items()}}
    # Score the alpha that VALIDATION chose on the untouched test windows. The test set plays no
    # part in choosing alpha -- that is the whole point of scoring it separately.
    if f_te is not None:
        pred_te = fits[best_a].predict(f_te.numpy())
        out["test_loss"] = float(((pred_te - y_te.numpy()) ** 2).sum(axis=1).mean())
    return out


def eval_head(head, feats, tgt_norm, device, batch_size=256):
    """Replicates eval_mse(): sum of squared error over horizon, averaged over windows."""
    head.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(feats), batch_size):
            f = feats[i:i + batch_size].to(device)
            t = tgt_norm[i:i + batch_size].to(device)
            total += nn.functional.mse_loss(head.head(f), t, reduction="sum").item()
            count += len(f)
    return total / max(count, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ETTh2", choices=list(DATASETS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="mps")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--max-train-samples", type=int, default=8000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=4000)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--test-seed", type=int, default=0,
                   help="subsample seed for the held-out test windows; fixed across runs")
    args = p.parse_args()

    from chronos import ChronosPipeline

    cfg = DATASETS[args.dataset]
    lookback, horizon = cfg["lookback"], cfg["horizon"]

    print(f"\n{'='*64}")
    print(f"CHRONOS FROZEN-ENCODER (D), CONVERGED: {args.dataset}, seed={args.seed}")
    print(f"{'='*64}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.results_dir) / f"mse_{args.dataset.lower()}" / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data: identical construction to run_experiment() ----
    train_s, val_s, test_s = load_series(args.dataset)
    ctx_tr, tgt_tr = build_windows(train_s, lookback, horizon,
                                   max_windows=args.max_train_samples, seed=args.seed)
    ctx_va, tgt_va = build_windows(val_s, lookback, horizon, max_windows=200, seed=args.seed)
    # Fixed subsample seed (not args.seed) so every run scores the SAME 200 test windows,
    # identically for B and D. Nothing is selected on them.
    ctx_te, tgt_te = build_windows(test_s, lookback, horizon, max_windows=200,
                                   seed=args.test_seed)
    eval_ctx = ctx_va if len(ctx_va) >= 20 else ctx_te
    eval_tgt = tgt_va if len(tgt_va) >= 20 else tgt_te

    # Guard the protocol: if validation were ever too small, eval_ctx would fall back to the test
    # windows and selection would silently leak into the reported held-out score. Fail loudly.
    if len(ctx_va) < 20:
        raise RuntimeError(
            "validation split too small; eval would fall back to the test windows, which would "
            "make the held-out score a selection set. Refusing to run.")

    n_pval = max(int(len(ctx_tr) * 0.2), 10)
    n_ptr = len(ctx_tr) - n_pval
    probe_tr_ctx, probe_va_ctx = ctx_tr[:n_ptr], ctx_tr[n_ptr:]
    probe_tr_tgt, probe_va_tgt = tgt_tr[:n_ptr], tgt_tr[n_ptr:]
    print(f"  Train: {n_ptr}, Probe-val: {n_pval}, Eval: {len(eval_ctx)}")

    print(f"  Loading {MODEL_ID}...")
    # Loaded exactly as chronos_mse_finetune.py:360 does (no device_map: needs accelerate).
    pipe = ChronosPipeline.from_pretrained(MODEL_ID, dtype=torch.float32)
    tokenizer = pipe.tokenizer
    tokenizer.config.prediction_length = horizon
    t5_model = pipe.model.model
    if args.device != "cpu":
        t5_model = t5_model.to(args.device)
    d_model = t5_model.config.d_model

    # ---- gate (condition-independent: pretrained checkpoint vs linear baseline) ----
    print("  Computing zero-shot MSE...")
    zs_mse = chronos_zs_mse(pipe, eval_ctx, eval_tgt, args.device)
    lin_mse = float(np.mean([
        np.mean(((LinearRegression().fit(
            np.arange(lookback).reshape(-1, 1), eval_ctx[i]
        ).predict(np.arange(lookback, lookback + horizon).reshape(-1, 1))
            - eval_ctx[i].mean()) / (eval_ctx[i].std() + 1e-8)
            - (eval_tgt[i] - eval_ctx[i].mean()) / (eval_ctx[i].std() + 1e-8)) ** 2)
        for i in range(len(eval_ctx))]))
    gate_pct = (lin_mse - zs_mse) / lin_mse * 100 if lin_mse > 1e-10 else 0.0
    print(f"  ZS MSE: {zs_mse:.4f}, Linear: {lin_mse:.4f}, Gate: {gate_pct:.1f}%")

    # ---- freeze, then cache the head's exact input ----
    for prm in t5_model.encoder.parameters():
        prm.requires_grad = False
    t5_model.encoder.eval()

    # extract_encoder_reps rounds up to a whole batch, so slice targets by len(reps) as
    # run_experiment() does rather than by max_samples.
    reps_pt_tr = extract_encoder_reps(t5_model, tokenizer, probe_tr_ctx, args.device,
                                      max_samples=500)
    reps_pt_va = extract_encoder_reps(t5_model, tokenizer, probe_va_ctx, args.device,
                                      max_samples=200)
    n_tr, n_va = len(reps_pt_tr), len(reps_pt_va)
    r2_pt = ridge_probe_r2(reps_pt_tr, probe_tr_tgt[:n_tr],
                           reps_pt_va, probe_va_tgt[:n_va])
    ortho_pt = task_orthogonal_probes(reps_pt_tr, reps_pt_va,
                                      probe_tr_ctx[:n_tr], probe_va_ctx[:n_va])

    print("  Caching frozen pooled features (encoder in eval mode -> deterministic)...")
    t0 = time.time()
    f_tr = pooled_features(t5_model, tokenizer, probe_tr_ctx, args.device)
    f_ev = pooled_features(t5_model, tokenizer, eval_ctx, args.device)
    f_te = pooled_features(t5_model, tokenizer, ctx_te, args.device)
    y_tr = norm_targets(probe_tr_ctx, probe_tr_tgt)
    y_ev = norm_targets(eval_ctx, eval_tgt)
    y_te = norm_targets(ctx_te, tgt_te)
    print(f"    cached {tuple(f_tr.shape)} train / {tuple(f_ev.shape)} eval in {time.time()-t0:.1f}s")

    # ---- closed-form best linear head on the frozen features (budget-free upper bound) ----
    print("  Closed-form ridge optimum on frozen features...")
    t0 = time.time()
    ols = ridge_optimum(f_tr, y_tr, f_ev, y_ev, f_te, y_te)
    assert "test_loss" in ols, "ridge_optimum must be given the test features"
    print(f"    alpha={ols['alpha']:g}  val={ols['best_val_loss']:.4f}  "
          f"(val/elem {ols['best_val_loss']/horizon:.4f}, "
          f"forgetting {(ols['best_val_loss']/horizon - zs_mse)/zs_mse*100:+.2f}%)  "
          f"interior={ols['interior']}  {time.time()-t0:.1f}s")
    if not ols["interior"]:
        print("    WARNING: ridge optimum on the grid boundary; widen RIDGE_ALPHAS")

    # ---- train the linear head to convergence (protocol-matched AdamW) ----
    head = MSEForecastHead(d_model, horizon).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    loader = DataLoader(TensorDataset(f_tr, y_tr), batch_size=args.batch_size, shuffle=True)

    best_val, best_epoch, best_state, bad = float("inf"), 0, None, 0
    print(f"\n  Training head: max {args.max_epochs} epochs, patience {args.patience}, "
          f"lr={args.lr}")
    t0 = time.time()
    for epoch in range(1, args.max_epochs + 1):
        head.train()
        tot, nb = 0.0, 0
        for fb, yb in loader:
            fb, yb = fb.to(args.device), yb.to(args.device)
            loss = nn.functional.mse_loss(head.head(fb), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        val = eval_head(head, f_ev, y_ev, args.device)
        if val < best_val - 1e-9:
            best_val, best_epoch, bad = val, epoch, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"    Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break
        if epoch % 100 == 0 or epoch <= 5:
            print(f"    Ep {epoch:>4}: loss={tot/max(nb,1):.4f} val={val:.4f} "
                  f"(best {best_val:.4f} @ {best_epoch})")
    print(f"  Restoring best epoch {best_epoch} (val_loss={best_val:.4f}); "
          f"{time.time()-t0:.0f}s total")
    head.load_state_dict(best_state)

    # Test-side measurements happen only after training. chronos_zs_mse samples (num_samples=20)
    # from the global torch RNG, so computing it up front would perturb head init and shuffling and
    # move the validation numbers away from the published runs. Caching f_te earlier is safe -- the
    # frozen encoder runs under no_grad in eval mode and draws no randomness.
    zs_mse_test = chronos_zs_mse(pipe, ctx_te, tgt_te, args.device)
    print(f"  ZS MSE (held-out test): {zs_mse_test:.4f}")
    test_adamw = eval_head(head, f_te, y_te, args.device)

    # Only count as converged if early stopping actually fired: the val minimum must be followed
    # by a full patience window of no improvement. best_epoch < epoch is too weak -- 1499/1500
    # passes it while the budget is plainly still binding.
    converged = best_epoch <= epoch - args.patience
    if not converged:
        print(f"  WARNING: best epoch {best_epoch} of {epoch} run; early stopping never fired, "
              f"so the AdamW leg is still budget-bound. The ridge optimum is unaffected.")

    # frozen encoder => CKA is 1.0 and the probes are the pretrained ones, by construction
    final_cka = linear_CKA(reps_pt_va,
                           extract_encoder_reps(t5_model, tokenizer, probe_va_ctx,
                                                args.device, max_samples=200))

    # Interiority is not sufficiency. On ETTm2 the ridge optimum pinned to the grid max with a loss
    # of 1e11; on Electricity it sat at an INTERIOR alpha and still returned 1.7e7. Both are broken
    # fits and only the first trips the interior flag. The reliable check is against the AdamW head:
    # both search the same function class, so a closed-form optimum that loses to gradient descent
    # by a wide margin is not an optimum. Recorded so the analysis can apply the pre-committed rule
    # (headline = whichever estimator validation preferred) instead of trusting ridge blindly.
    ridge_sane = bool(ols["best_val_loss"] <= best_val * 1.05)
    if not ridge_sane:
        print(f"  WARNING: ridge val {ols['best_val_loss']:.4g} > AdamW val {best_val:.4g}; "
              f"ridge degenerate on this cell -- validation prefers the AdamW head")

    results = {
        "dataset": args.dataset, "condition": "D", "seed": args.seed,
        "ridge_sane": ridge_sane, "val_prefers": "ridge" if ridge_sane else "adamw",
        "loss_type": "mse_head", "protocol": "frozen_cached_features_converged",
        "max_epochs": args.max_epochs, "patience": args.patience,
        "epochs_run": epoch, "best_epoch": best_epoch, "converged": bool(converged),
        "lr": args.lr, "max_train_samples": args.max_train_samples, "d_model": d_model,
        "gate_improvement_pct": gate_pct, "zs_mse": zs_mse, "linear_mse": lin_mse,
        # AdamW-to-convergence (protocol-matched)
        "best_val_loss": best_val,
        "val_mse_per_element": best_val / horizon,
        "forgetting_pct": (best_val / horizon - zs_mse) / zs_mse * 100,
        # closed-form optimum: the headline D, since no training budget can beat it
        "ridge_optimum": ols,
        "best_val_loss_ols": ols["best_val_loss"],
        "val_mse_per_element_ols": ols["best_val_loss"] / horizon,
        "forgetting_pct_ols": (ols["best_val_loss"] / horizon - zs_mse) / zs_mse * 100,
        # ---- held-out test side (alpha and early stopping were chosen on validation only) ----
        "test_seed": args.test_seed,
        "n_test_windows": int(len(ctx_te)),
        "zs_mse_test": zs_mse_test,
        "test_loss_ols": ols["test_loss"],
        "test_mse_per_element_ols": ols["test_loss"] / horizon,
        "test_forgetting_pct_ols": (ols["test_loss"] / horizon - zs_mse_test) / zs_mse_test * 100,
        "test_loss_adamw": test_adamw,
        "test_mse_per_element_adamw": test_adamw / horizon,
        "test_forgetting_pct_adamw": (test_adamw / horizon - zs_mse_test) / zs_mse_test * 100,
        "final_cka": final_cka,
        "linear_probe": {"r2_pt": r2_pt, "r2_ft": r2_pt, "delta_r2": 0.0},
        "orthogonal_probes": {"pretrained": ortho_pt, "finetuned": ortho_pt,
                              "delta": {k: 0.0 for k in ortho_pt}},
    }
    path = out_dir / f"condition_D_s{args.seed}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  ZS MSE {zs_mse:.4f}   CKA {final_cka:.4f}")
    print(f"  D (AdamW, {epoch} ep): val/elem {best_val/horizon:.4f}  "
          f"forgetting {results['forgetting_pct']:+.2f}%")
    print(f"  D (ridge optimum)    : val/elem {ols['best_val_loss']/horizon:.4f}  "
          f"forgetting {results['forgetting_pct_ols']:+.2f}%   <-- headline")
    print(f"  D held-out test: ridge {ols['test_loss']/horizon:.4f} "
          f"({results['test_forgetting_pct_ols']:+.2f}%)   "
          f"AdamW {test_adamw/horizon:.4f} ({results['test_forgetting_pct_adamw']:+.2f}%)")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
