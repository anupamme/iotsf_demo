#!/usr/bin/env python3
"""
Score a checkpoint fine-tuned on TASK B against TASK A's held-out windows.

This is the positive control's measuring instrument. The main matrix never needs it, because there task
A and task B are the same series and the runner's own `forgetting_pct` already reports the retention
loss. In the positive control they are DIFFERENT series -- that is the whole construction -- so
retention on task A has to be measured after the fact, on a checkpoint trained elsewhere.

WHAT IS IMPORTED RATHER THAN REIMPLEMENTED, and why it matters here more than usual. Every quantity
below comes from scripts/finetune_forecasting.py itself:

    make_eval_sequences    the window geometry, so task A's inputs are the same 96+h past steps the
                           gate and the main matrix scored, down to the index arithmetic
    evaluate_forecasting   the forward pass, the sampling, and the train-split normalisation
    extract_representations / linear_CKA    the CKA the paper's negative claim is about
    compute_weight_drift   the L2 drift reported alongside it

A reimplementation would make the positive control's numbers incomparable to the matrix's, which would
defeat the point of running it: the arm exists to say whether CKA -- the same CKA, measured the same
way -- orders damage when damage is large. So this file contains no metric of its own.

WHERE CKA IS MEASURED, AND WHY IT IS THE DESIGN CHOICE TO PROBE. On TASK A's inputs, as registered.
In the main matrix the distinction is empty because the two tasks coincide. Here it is the entire
question: we are asking whether the representation OF THE CAPABILITY BEING LOST has changed. Measured
on task B's inputs the answer would be trivially yes -- the model just spent twenty epochs adapting to
them -- and would say nothing about task A.

TIER C: needs a checkpoint (*.pt, gitignored). The JSON it writes is what downstream emitters read.

Usage:
  .venv-probe/bin/python scripts/eval_retention.py \
      --state results/positive_control/lr1e-2_B_s42/final_state.pt \
      --task-a-data data/forecasting/ETTh2.csv --model-size small --horizon 192 \
      --device mps --out results/positive_control/lr1e-2_B_s42/retention_A_final.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Module-level in finetune_forecasting.py since 21 Sep 2026 precisely so this file can reach them.
from finetune_forecasting import (                                        # noqa: E402
    make_eval_sequences, evaluate_forecasting, extract_representations,
    linear_CKA, compute_weight_drift, _patch_packed_scaler_for_mps,
)
from src.data.forecasting_loader import get_forecasting_loader            # noqa: E402


def load_task_a(data_path, lookback, horizon, features, eval_limit):
    """Task A's train statistics and its HELD-OUT windows, built exactly as the runner builds them."""
    loader = get_forecasting_loader(data_path, lookback_window=lookback,
                                    forecast_horizon=horizon, features=features)
    train_df, _val_df, test_df = loader.get_splits()
    cols = ["OT"] if features == "S" else loader.FEATURE_COLUMNS
    train_vals, test_vals = train_df[cols].values, test_df[cols].values
    # Normalisation from the TRAIN split only, as everywhere else in the paper: held-out statistics
    # must not enter the scale that the numerator and denominator are both measured on.
    train_mean = train_vals.mean(axis=0)
    train_std = train_vals.std(axis=0) + 1e-8
    X, y = make_eval_sequences(test_vals, lookback + horizon, horizon)
    return X[:eval_limit], y[:eval_limit], train_mean, train_std, len(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="state dict saved by --save-final-state or --save-best-encoder")
    ap.add_argument("--task-a-data", default="data/forecasting/ETTh2.csv")
    ap.add_argument("--model-size", default="small", choices=["small", "base", "large"])
    ap.add_argument("--horizon", type=int, default=192)
    ap.add_argument("--lookback", type=int, default=96)
    ap.add_argument("--features", default="M", choices=["M", "S", "MS"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-eval-sequences", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    state_path = Path(a.state)
    if not state_path.exists():
        sys.exit(f"no checkpoint at {state_path}")

    if a.device == "mps":
        _patch_packed_scaler_for_mps()

    X, y_raw, train_mean, train_std, n_features = load_task_a(
        a.task_a_data, a.lookback, a.horizon, a.features, a.max_eval_sequences)
    X_t = torch.from_numpy(X).float()
    print(f"task A: {a.task_a_data}  {len(X)} held-out windows, input {X.shape[1]} steps, "
          f"target {y_raw.shape[1]} steps, {n_features} features")

    from src.models.moirai_detector import MoiraiAnomalyDetector
    detector = MoiraiAnomalyDetector(model_size=a.model_size, context_length=a.lookback,
                                     prediction_length=a.horizon, target_dim=n_features,
                                     num_samples=20, device=a.device)
    detector.initialize()
    model = detector.model

    # ZERO-SHOT first, on the same windows, from the same freshly-initialised pre-trained weights the
    # fine-tuning run started from. This is the denominator of retention_A, and it is recomputed here
    # rather than read from the run record because the run record's zero-shot is on TASK B.
    pretrained_params = {n: p.data.clone() for n, p in model.named_parameters()}
    zs_reps = extract_representations(model, X_t, None, device=a.device)
    zs = evaluate_forecasting(model, X_t, y_raw, train_mean, train_std, a.horizon, device=a.device)
    print(f"zero-shot on task A:   MSE={zs['mse']:.6f}  MAE={zs['mae']:.6f}")

    # THE FINE-TUNED STATE. strict=True on purpose: a key mismatch means the checkpoint was not
    # produced by this geometry, and silently ignoring it would score a partly-pre-trained model.
    sd = torch.load(state_path, map_location="cpu")
    model.load_state_dict({k: v.to(a.device) for k, v in sd.items()}, strict=True)

    ft_reps = extract_representations(model, X_t, None, device=a.device)
    ft = evaluate_forecasting(model, X_t, y_raw, train_mean, train_std, a.horizon, device=a.device)
    n = min(len(zs_reps), len(ft_reps))
    cka = linear_CKA(zs_reps[:n], ft_reps[:n]) if n > 0 else None
    drift = compute_weight_drift(model, pretrained_params)

    retention = (ft["mse"] - zs["mse"]) / zs["mse"] * 100.0
    print(f"fine-tuned on task A:  MSE={ft['mse']:.6f}  MAE={ft['mae']:.6f}")
    print(f"retention_A = {retention:+.2f}%   (positive = the capability got WORSE)")
    print(f"CKA on task A's inputs = {cka:.4f}   weight drift = {drift:.2f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        state=str(state_path), task_a_data=a.task_a_data, model_size=a.model_size,
        horizon=a.horizon, lookback=a.lookback, features=a.features,
        n_eval_windows=int(len(X)), n_features=int(n_features),
        mse_zeroshot=zs["mse"], mae_zeroshot=zs["mae"],
        mse_finetuned=ft["mse"], mae_finetuned=ft["mae"],
        retention_A_pct=retention,
        cka_on_task_a_inputs=cka, weight_drift=drift,
        cka_measured_on=("task A's held-out inputs -- NOT task B's; see the pre-registration"),
    ), indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
