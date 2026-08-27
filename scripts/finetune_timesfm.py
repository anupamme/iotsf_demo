#!/usr/bin/env python3
"""
The third-backbone intervention: fine-tune TimesFM 2.5 (200M, torch) under conditions A / B / D / H.

WHY THIS EXISTS
---------------
Fourteen of the paper's nineteen cells are Moirai and five are Chronos, so "the value of encoder
adaptation is not predicted by representation drift" rests on two backbones, one of which (Chronos)
needs an attached MSE head for a matched B/D comparison. This arm adds a third backbone under its
OWN output head and its OWN inference path, so the generality argument does not depend on Moirai
and is not open to the "you attached a head, so this is not normal TSFM fine-tuning" objection.

CONDITIONS (chosen to be the like-for-like analogues of the Moirai arm's, not new ones)
  A  nothing trains          -- the released checkpoint, scored through model.forecast()
  B  everything trains       -- tokenizer + 20-layer stack + output projections
  D  the stack is frozen     -- tokenizer and output projections still train  == Moirai D
  H  stack + tokenizer frozen -- output projections only; CKA = 1.0 by construction == Moirai H

The stack (`stacked_xf`, 196.71M of 231.29M parameters) is the encoder for this arm's purposes.
Condition D leaves the input tokenizer trainable exactly as Moirai's condition D leaves `in_proj`
trainable, which is why condition H exists in both arms: it is the control for "is 'freezing wins'
really the input projection re-fitting?".

OBJECTIVE -- no head is attached
  The loss is TimesFM's own point-head output structure: MSE on the mean channel plus pinball loss
  on the nine decile channels the released `output_projection_point` emits (timesfm_common.
  native_loss). Predictions and targets are per-window z-scored by the CONTEXT's mean/std before
  the loss, matching this arm's evaluation metric and the Chronos arm's convention; TimesFM is
  scale-equivariant by construction, so this is a per-window reweighting, not a change of head.
  The continuous quantile head (`output_projection_quantiles`, 27.85M) only rewrites non-median
  channels at inference and therefore receives no gradient: condition B updates 203.44M of the
  231.29M parameters. The appendix states this.

  Training goes through timesfm_common.native_forecast, a differentiable re-implementation of the
  released inference stack -- necessary because `module.decode()` is wrapped in `torch.no_grad()`.
  verify_native_path() asserts it reproduces `model.forecast()` on real windows at the start of
  EVERY run, so a re-implementation drift can never reach the paper silently.

PROTOCOL
  Same univariate-OT series, lookback 96, h = 24, and the SAME 200-window seed-0 test subsample as
  the Chronos arm and as gate_all_cells.timesfm_gates(), so the gate, the zero-shot reference and
  B-D are all read on one window set. Early stopping selects on validation only; the test windows
  are scored once, after training, and nothing is selected on them.

  Hyperparameters are the ones the other two arms already use -- lr 1e-4, AdamW, grad-norm clip
  1.0, 20 epochs, patience 7, 1000 training windows -- and are IDENTICAL for B and D. Nothing here
  is tuned per condition; B-D would be meaningless if it were.

Run:  HF_HUB_OFFLINE=1 .venv-probe/bin/python scripts/finetune_timesfm.py \
          --dataset ETTh1 --condition B --seed 42 --device mps \
          --results-dir results/v46_timesfm/ETTh1_h24/condition_B
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chronos_mse_finetune import build_windows, linear_CKA, load_series
from timesfm_common import (HEAD_ATTRS, STACK_ATTR, TOKENIZER_ATTR, assert_no_ar,
                            batched_point_mse, load_timesfm, native_forecast, native_loss,
                            pooled_reps, verify_native_path, window_norm_mse)

MAX_HORIZON = 128       # = output patch size o, so num_decode_steps == 0 (see timesfm_common)


def freeze_for(module, condition):
    """Freeze by condition and return (n_trainable, n_total, frozen_module_names)."""
    for p in module.parameters():
        p.requires_grad = True
    frozen = []
    if condition in ("D", "H"):
        frozen.append(STACK_ATTR)
    if condition == "H":
        frozen.append(TOKENIZER_ATTR)
    for name in frozen:
        for p in getattr(module, name).parameters():
            p.requires_grad = False
    n_tr = sum(p.numel() for p in module.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in module.parameters())
    return n_tr, n_all, frozen


def _norm_pair(forecast, target, context):
    """Per-window z-score by the context's own mean/std -- the evaluation metric's scale."""
    mu = context.mean(dim=1, keepdim=True)
    sd = context.std(dim=1, keepdim=True) + 1e-8
    return (forecast - mu.unsqueeze(-1)) / sd.unsqueeze(-1), (target - mu) / sd


def train_one_epoch(model, ctx, tgt, horizon, optimizer, device, batch_size, rng):
    module = model.model
    module.train()
    order = rng.permutation(len(ctx))
    total, nb = 0.0, 0
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        c = torch.from_numpy(ctx[idx].astype(np.float32)).to(device)
        t = torch.from_numpy(tgt[idx].astype(np.float32)).to(device)
        fc = native_forecast(model, c, horizon)
        fc_n, t_n = _norm_pair(fc, t, c)
        loss = native_loss(fc_n, t_n)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in module.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        total += float(loss.item())
        nb += 1
    return total / max(nb, 1)


def run(args):
    dev = args.device
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    tag = f"{args.dataset}_h{args.horizon}"
    print("=" * 72)
    print(f"TimesFM 2.5 200M  |  {tag}  |  condition {args.condition}  |  seed {args.seed}")
    print("=" * 72)

    train_s, val_s, test_s = load_series(args.dataset)
    ctx_tr, tgt_tr = build_windows(train_s, args.lookback, args.horizon,
                                   max_windows=args.max_train_samples, seed=args.seed)
    ctx_va, tgt_va = build_windows(val_s, args.lookback, args.horizon,
                                   max_windows=200, seed=args.seed)
    # Fixed subsample seed (not args.seed): every condition and every seed scores the SAME 200
    # held-out windows, and they are the same 200 the gate screen used.
    ctx_te, tgt_te = build_windows(test_s, args.lookback, args.horizon,
                                   max_windows=200, seed=args.test_seed)
    if len(ctx_va) < 20:
        raise RuntimeError("validation split too small; selection would leak into the held-out "
                           "score. Refusing to run.")
    print(f"  windows: train {len(ctx_tr)}  val {len(ctx_va)}  test {len(ctx_te)}")

    model, fc = load_timesfm(max_context=args.lookback, max_horizon=MAX_HORIZON, device=dev)
    assert_no_ar(model)
    module = model.model
    verify_native_path(model, ctx_te, args.horizon, dev)

    pretrained = copy.deepcopy({k: v.cpu().clone() for k, v in module.state_dict().items()})

    print("  zero-shot reference (pretrained weights)...")
    zs_val = batched_point_mse(model, ctx_va, tgt_va, args.horizon, dev, args.batch_size)
    zs_test = batched_point_mse(model, ctx_te, tgt_te, args.horizon, dev, args.batch_size)
    print(f"    ZS val {zs_val:.4f}   ZS test {zs_test:.4f}")
    reps_pt = pooled_reps(model, ctx_va, args.horizon, dev, args.batch_size)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"condition_{args.condition}_h{args.horizon}_s{args.seed}.json"

    common = dict(backbone="timesfm", model_id="google/timesfm-2.5-200m-pytorch",
                  dataset=args.dataset, horizon=args.horizon, lookback=args.lookback,
                  condition=args.condition, seed=args.seed, test_seed=args.test_seed,
                  loss_type="native_point_head", n_test_windows=int(len(ctx_te)),
                  n_val_windows=int(len(ctx_va)),
                  zeroshot_mse=zs_val, zeroshot_test_mse=zs_test)

    if args.condition == "A":
        out_path.write_text(json.dumps(common, indent=2) + "\n")
        print(f"  wrote {out_path}")
        return

    n_tr, n_all, frozen = freeze_for(module, args.condition)
    print(f"  condition {args.condition}: {n_tr:,}/{n_all:,} params trainable"
          f"{'  frozen: ' + ', '.join(frozen) if frozen else ''}")
    frozen_ref = {name: copy.deepcopy({k: v.cpu().clone()
                                       for k, v in getattr(module, name).state_dict().items()})
                  for name in frozen}

    optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=0.01)

    best_val, best_epoch, best_state, patience = float("inf"), 0, None, 0
    print(f"\n  training: {args.epochs} epochs, lr={args.lr}, batch {args.batch_size}")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tl = train_one_epoch(model, ctx_tr, tgt_tr, args.horizon, optimizer, dev,
                             args.batch_size, rng)
        vl = batched_point_mse(model, ctx_va, tgt_va, args.horizon, dev, args.batch_size)
        cka = linear_CKA(reps_pt, pooled_reps(model, ctx_va, args.horizon, dev, args.batch_size))
        print(f"    Ep {ep:2d}: loss={tl:.4f} val={vl:.4f} CKA={cka:.4f} ({time.time()-t0:.1f}s)",
              flush=True)
        if vl < best_val:
            best_val, best_epoch, patience = vl, ep, 0
            best_state = {k: v.cpu().clone() for k, v in module.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print(f"    early stop at epoch {ep}")
                break

    print(f"  restoring best epoch {best_epoch} (val {best_val:.4f})")
    module.load_state_dict(best_state)

    # The frozen modules must be bit-identical to the checkpoint. If they are not, "frozen encoder"
    # is a false label and every B-D in this arm is wrong -- so this is an assertion, not a print.
    for name, ref in frozen_ref.items():
        now = getattr(module, name).state_dict()
        bad = [k for k in ref if not torch.equal(ref[k], now[k].cpu())]
        if bad:
            raise AssertionError(f"condition {args.condition}: {name} changed during training "
                                 f"({len(bad)} tensors, e.g. {bad[:3]})")
        print(f"  verified {name} bit-identical to the released checkpoint")

    test_mse = batched_point_mse(model, ctx_te, tgt_te, args.horizon, dev, args.batch_size)
    reps_ft = pooled_reps(model, ctx_va, args.horizon, dev, args.batch_size)
    final_cka = linear_CKA(reps_pt, reps_ft)
    now = module.state_dict()
    stack_keys = [k for k in pretrained if k.startswith(STACK_ATTR + ".")]
    drift = sum((now[k].cpu() - pretrained[k]).float().pow(2).sum().item()
                for k in stack_keys) ** 0.5

    res = dict(common, final_val_mse=best_val, test_mse=test_mse, best_epoch=best_epoch,
               epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
               max_train_samples=int(len(ctx_tr)), n_trainable=n_tr, n_params=n_all,
               frozen_modules=frozen, final_cka=final_cka, final_weight_drift=drift,
               has_drift=bool(final_cka < 0.95),
               forgetting_pct_test=(test_mse - zs_test) / zs_test * 100,
               forgetting_pct_val=(best_val - zs_val) / zs_val * 100)
    out_path.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\n  {'-'*60}")
    print(f"  RESULT  TimesFM/{tag} cond {args.condition} seed {args.seed}")
    print(f"  CKA {final_cka:.4f} {'DRIFT' if final_cka < 0.95 else 'stable'}   "
          f"stack weight drift {drift:.2f}")
    print(f"  val  {best_val:.4f} vs ZS {zs_val:.4f}   forgetting {res['forgetting_pct_val']:+.2f}%")
    print(f"  test {test_mse:.4f} vs ZS {zs_test:.4f}   forgetting {res['forgetting_pct_test']:+.2f}%")
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--condition", required=True, choices=["A", "B", "D", "H"])
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--lookback", type=int, default=96)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-train-samples", type=int, default=1000)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-seed", type=int, default=0,
                   help="subsample seed for the held-out windows; MUST stay 0 to match the gate")
    p.add_argument("--device", default="cpu")
    p.add_argument("--results-dir", default="results/v46_timesfm/ETTh1_h24/condition_B")
    run(p.parse_args())


if __name__ == "__main__":
    main()
