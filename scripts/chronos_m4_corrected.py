#!/usr/bin/env python3
"""Chronos-T5-Small on M4-Monthly, conditions B and D, under the corrected three-split protocol.

WHAT THIS IS FOR. app:chronos_detail reported ten seeds, two sample sizes and a random-init control
for this cell, and no file under results/ could produce any of it. This script produces the records,
or the numbers leave the paper. It replaces scripts/finetune_chronos_m4.py, which is kept only as the
historical artefact that generated the unreproducible appendix.

THREE THINGS IT DOES DIFFERENTLY, all fixed in results/chronos_m4/preregistration.json before the
gate was computed:

  1. Three splits with disjoint targets (scripts/m4_monthly_data.py), instead of a positional cut
     through a randomly subsampled window pool whose neighbours overlap in 95 of 96 context steps.
  2. Early stopping selects on the SELECTION windows. finetune_chronos_m4.py:522 recomputed its
     val_mse on `test_ctx_sub` -- the same windows its forgetting number was reported on -- so the
     checkpoint was chosen to minimise the quantity being reported.
  3. The best epoch must be at least 1. That script set `best_val_mse = zs_mse` and
     `best_epoch = 0` with the pre-trained weights already saved as `best_state` (lines 505-507), so
     the pre-trained checkpoint was itself an early-stopping candidate; a run that selects it reports
     CKA 1.000 and zero drift by definition, which is most of why that appendix read as "this
     backbone does not drift".

Held-out windows (official M4 test values) touch the model exactly once, after training. Note also
that the old script's `weight_drift` was an l2 over the WHOLE model state dict; ours is over the
encoder, as `weight_drift_scope` in the output records.

CONDITION D FREEZES WHAT THE OTHER CHRONOS CELLS FREEZE: t5.encoder.parameters(), which is what
chronos_mse_finetune.py:476 does. On Chronos that includes the shared token embedding, so unlike
Moirai's condition D -- which leaves the input projection trainable -- the encoder's representation is
pinned by construction here and CKA is expected to be exactly 1. That is a protocol difference between
the backbones, not a measurement, and the paper states it rather than presenting D's CKA as evidence.

Run (per seed, per condition):
    .venv-probe/bin/python scripts/chronos_m4_corrected.py --condition B --seed 42 --device mps
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from gate_all_cells import _window_linear_mse                    # noqa: E402
from m4_monthly_data import HORIZON, LOOKBACK, build_splits, load_series  # noqa: E402

MODEL_ID = "amazon/chronos-t5-small"
DATASET = "m4_monthly"
CKA_WINDOWS = 200           # the 200 selection contexts; the non-Moirai arms' convention


def linear_CKA(X, Y):
    """Linear CKA, identical to chronos_mse_finetune.linear_CKA and finetune_forecasting's."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Y.T @ X, 'fro') ** 2
    denom = np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro')
    return float(num / denom) if denom >= 1e-12 else 0.0


def encoder_reps(model, tokenizer, contexts, device, batch_size=32):
    """Final encoder output, mean-pooled over tokens under the attention mask.

    The body's primary CKA object, chosen there because it is the comparison a practitioner holding
    two checkpoints can make without picking a layer or a token position. Mean-pooled, not
    token-flattened: the flattened variant reads uniformly high and cannot order these encoders
    (app:layerunfreeze).
    """
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            ctx = torch.tensor(contexts[i:i + batch_size], dtype=torch.float32)
            ids, mask, _ = tokenizer.context_input_transform(ctx)
            h = model.encoder(input_ids=ids.to(device),
                              attention_mask=mask.to(device)).last_hidden_state
            m = mask.to(device).unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
    return np.concatenate(out)


def zs_mse(pipe, contexts, targets, batch_size=32):
    """Median-of-20-samples MSE on the per-window z-scored scale (chronos_mse_finetune's aggregation)."""
    horizon = targets.shape[1]
    preds = []
    pipe.model.eval()
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch = [torch.tensor(contexts[j], dtype=torch.float32)
                     for j in range(i, min(i + batch_size, len(contexts)))]
            preds.append(pipe.predict(batch, prediction_length=horizon,
                                      num_samples=20).median(dim=1).values.cpu().numpy())
    preds = np.concatenate(preds)
    return float(np.mean([
        np.mean(((preds[i] - contexts[i].mean()) / (contexts[i].std() + 1e-8)
                 - (targets[i] - contexts[i].mean()) / (contexts[i].std() + 1e-8)) ** 2)
        for i in range(len(contexts))]))


def train_epoch(model, tokenizer, loader, optimizer, device, freeze_encoder=False, grad_clip=1.0):
    """One epoch of Chronos's native tokenised cross-entropy."""
    model.train()
    if freeze_encoder:
        # finetune_chronos_m4.py:258 does this too, and it is not cosmetic: a frozen encoder left in
        # train mode still applies dropout, so the decoder's gradients would be computed against a
        # stochastic version of the representation condition D is supposed to hold fixed.
        model.encoder.eval()
    total, n = 0.0, 0
    for ctx, tgt in loader:
        ids, mask, scale = tokenizer.context_input_transform(ctx.to(dtype=torch.float32))
        labels, _ = tokenizer.label_input_transform(tgt.to(dtype=torch.float32), scale)
        labels = labels.to(device)
        labels = labels.masked_fill(labels == tokenizer.config.pad_token_id, -100)
        loss = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels).loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["B", "D"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--results-dir", default="results/chronos_m4")
    a = ap.parse_args()

    reg = ROOT / "results/chronos_m4/preregistration.json"
    if not reg.exists():
        sys.exit("refusing to run: results/chronos_m4/preregistration.json is missing")
    r = json.load(open(reg))
    if a.seed not in r["runs"]["seeds"] or a.condition not in r["runs"]["conditions"]:
        sys.exit(f"seed/condition not registered: {a.condition}/{a.seed} vs "
                 f"{r['runs']['conditions']}/{r['runs']['seeds']}")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = a.device

    series = load_series()
    train_all, sel, ho, info_all = build_splits(series, max_train=None)
    (ctx_tr, tgt_tr), _, _, info = build_splits(series, max_train=a.n_train, seed=a.seed)
    ctx_sel, tgt_sel = sel
    ctx_ho, tgt_ho = ho
    print(f"train {len(ctx_tr)} of {info_all['train_windows_used']} available, "
          f"selection {len(ctx_sel)}, held-out {len(ctx_ho)}")

    from chronos import ChronosPipeline
    pipe = ChronosPipeline.from_pretrained(MODEL_ID, dtype=torch.float32)
    t5 = pipe.model.model                      # the HF T5 inside Chronos's wrapper
    tokenizer = pipe.tokenizer
    tokenizer.config.prediction_length = HORIZON
    if device != "cpu":
        pipe.model.model = t5 = t5.to(device)

    pretrained_encoder = {k: v.detach().clone() for k, v in t5.encoder.state_dict().items()}
    reps_pt = encoder_reps(t5, tokenizer, ctx_sel[:CKA_WINDOWS], device)

    # --- the gate, recomputed per run so the numerator and denominator share this seed's sampling
    zs_sel = zs_mse(pipe, ctx_sel, tgt_sel)
    lin_sel = _window_linear_mse(ctx_sel, tgt_sel, LOOKBACK, HORIZON, train=train_all,
                                 baseline="fitted", dataset=DATASET)
    gate = 1 - zs_sel / lin_sel
    zs_ho = zs_mse(pipe, ctx_ho, tgt_ho)
    print(f"  selection ZS {zs_sel:.4f}  fitted {lin_sel:.4f}  gate {gate:+.4f}   "
          f"held-out ZS {zs_ho:.4f}")

    if a.condition == "D":
        # Exactly chronos_mse_finetune.py:476's freeze, so the six Chronos cells share one definition
        # of condition D. On Chronos this includes the shared token embedding.
        for p in t5.encoder.parameters():
            p.requires_grad = False
    n_train_params = sum(p.numel() for p in t5.parameters() if p.requires_grad)
    print(f"  condition {a.condition}: {n_train_params:,} trainable parameters")

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(ctx_tr, dtype=torch.float32),
                                       torch.tensor(tgt_tr, dtype=torch.float32)),
        batch_size=a.batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW([p for p in t5.parameters() if p.requires_grad],
                                  lr=a.lr, weight_decay=0.01)

    # best_val starts at inf, not at the zero-shot score: a fine-tune that selects the checkpoint it
    # started from is not a fine-tune, and epoch 0 is therefore not a candidate (registration,
    # runs.early_stopping). The zero-shot score is recorded as the comparison, not as a candidate.
    history = dict(epoch=[], train_loss=[], sel_mse=[], cka=[], drift=[])
    best = dict(sel_mse=float("inf"), epoch=0)
    best_state = None
    stale = 0
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        loss = train_epoch(t5, tokenizer, loader, optimizer, device,
                           freeze_encoder=(a.condition == "D"))
        sel = zs_mse(pipe, ctx_sel, tgt_sel)
        reps = encoder_reps(t5, tokenizer, ctx_sel[:CKA_WINDOWS], device)
        cka = linear_CKA(reps_pt, reps)
        enc = t5.encoder.state_dict()
        drift = float(sum((enc[k] - pretrained_encoder[k]).float().pow(2).sum().item()
                          for k in pretrained_encoder) ** 0.5)
        for k, v in dict(epoch=epoch, train_loss=loss, sel_mse=sel, cka=cka, drift=drift).items():
            history[k].append(v)
        if sel < best["sel_mse"]:
            best = dict(sel_mse=sel, epoch=epoch)
            best_state = copy.deepcopy(t5.state_dict())
            stale = 0
        else:
            stale += 1
        print(f"  epoch {epoch:2d}: loss {loss:.4f}  sel_mse {sel:.4f}  "
              f"CKA {cka:.4f}  drift {drift:8.2f}  ({time.time() - t0:.0f}s)")
        if stale >= a.patience:
            print(f"  early stop: {a.patience} epochs without improvement")
            break

    stopped_epoch = history["epoch"][-1]
    t5.load_state_dict(best_state)

    # --- held-out measurements, after training, once
    ft_ho = zs_mse(pipe, ctx_ho, tgt_ho)
    forgetting = (ft_ho - zs_ho) / zs_ho * 100
    reps_ft = encoder_reps(t5, tokenizer, ctx_sel[:CKA_WINDOWS], device)
    final_cka = linear_CKA(reps_pt, reps_ft)
    enc = t5.encoder.state_dict()
    final_drift = float(sum((enc[k] - pretrained_encoder[k]).float().pow(2).sum().item()
                            for k in pretrained_encoder) ** 0.5)
    sel_at_best = best["sel_mse"]

    print(f"\n  {'=' * 56}")
    print(f"  Chronos/M4-Monthly  cond {a.condition}  seed {a.seed}")
    print(f"  gate (selection, fitted)  {gate:+.4f}")
    print(f"  held-out ZS {zs_ho:.4f} -> FT {ft_ho:.4f}   forgetting {forgetting:+.2f}%")
    print(f"  CKA {final_cka:.4f}   encoder drift {final_drift:.2f}   best epoch {best['epoch']}")
    print(f"  {'=' * 56}")

    out = dict(
        cell="chronos_m4monthly", dataset="M4-Monthly", arm="chronos_m4", model_id=MODEL_ID,
        condition=a.condition, seed=a.seed, objective="tokenised_cross_entropy",
        lookback=LOOKBACK, horizon=HORIZON, lr=a.lr, batch_size=a.batch_size,
        epochs_requested=a.epochs, patience=a.patience, stopped_epoch=stopped_epoch,
        best_epoch=best["epoch"], best_selection_mse=sel_at_best,
        n_train_windows=int(len(ctx_tr)), n_selection_windows=int(len(ctx_sel)),
        n_heldout_windows=int(len(ctx_ho)), n_trainable_params=int(n_train_params),
        freeze_encoder=(a.condition == "D"),
        frozen_scope=("t5.encoder.parameters(), including the shared token embedding"
                      if a.condition == "D" else None),
        zs_mse=zs_sel, linear_mse=lin_sel, gate_r2_task=gate,
        gate_baseline="fitted ridge lam=1e-4 on all train windows, selection split",
        zs_mse_heldout=zs_ho, ft_mse_heldout=ft_ho, forgetting_pct=forgetting,
        final_cka=final_cka, cka_object="final encoder output, mean-pooled over tokens",
        cka_windows=int(min(CKA_WINDOWS, len(ctx_sel))),
        weight_drift=final_drift, weight_drift_scope="encoder state dict, l2",
        data={k: v for k, v in info.items() if k != "series_ids"},
        history=history,
    )
    d = ROOT / a.results_dir / f"cond_{a.condition}" / f"seed{a.seed}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"condition_{a.condition}_s{a.seed}.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(f"  wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
