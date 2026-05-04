#!/usr/bin/env python3
"""Re-probe saved Moirai encoders without retraining.

Addresses V17-round-3 reviewer priority 3: the reported ΔR² is a comparison of
two negative absolute-R² floors on the 96-step-ahead forecasting head (both
R²(PT) and R²(FT) are negative). This script evaluates each saved best_encoder.pt
against multiple probe types (ridge / mlp / gbm) and multiple head types
(forecast96 — the paper's primary; delta1 — next-step delta prediction where
R²(PT) > 0 is plausible) without paying the ~1.5h retrain per seed.

Usage:
    python scripts/reprobe_saved_encoders.py \
        --encoder-dir results/v17r2_etth2_n10k_es_mlp/seed303 \
        --probe-types ridge,mlp,gbm \
        --head-types forecast96,delta1 \
        --data-path data/forecasting/ETTh2.csv \
        --out-path results/v18_reprobe/seed303.json

Passing --zero-shot uses the pre-trained Moirai-Small encoder (no best_encoder.pt
load); this is the "PT" reference for every probe×head combination.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.finetune_forecasting import (
    _patch_packed_scaler_for_mps,
    extract_representations,
    linear_probe_r2,
)
from src.data.forecasting_loader import get_forecasting_loader


def build_heads(y_val_eval_raw: np.ndarray, X_val_eval: np.ndarray,
                head_types: list[str], features_mode: str) -> dict:
    """Construct regression targets per head type.

    forecast96: (N, H, F) full 96-step-ahead forecast horizon (paper's head).
    delta1    : (N, F) one-step delta x_{t+1} - x_t where x_t is the last
                observed timestep. Predicts the immediate next-step change;
                R²(PT) on this should be positive because Moirai's pre-trained
                encoder carries strong next-step information.
    """
    heads = {}
    if 'forecast96' in head_types:
        heads['forecast96'] = y_val_eval_raw
    if 'forecast48' in head_types:
        if y_val_eval_raw.shape[1] < 48:
            raise ValueError(f"forecast48 requires horizon >= 48, got {y_val_eval_raw.shape[1]}")
        heads['forecast48'] = y_val_eval_raw[:, :48, :]
    if 'delta1' in head_types:
        # x_{t+1} is the first step of the target; x_t is the last step of context.
        # X_val_eval shape: (N, ext_lookback, F); y shape: (N, H, F).
        last_ctx = X_val_eval[:, -1, :]  # (N, F)
        first_tgt = y_val_eval_raw[:, 0, :]  # (N, F)
        heads['delta1'] = first_tgt - last_ctx
    return heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder-dir', default=None,
                    help="Directory containing best_encoder.pt. Mutually exclusive with --zero-shot.")
    ap.add_argument('--zero-shot', action='store_true',
                    help="Use unmodified pre-trained Moirai-Small (reference for ΔR²).")
    ap.add_argument('--probe-types', default='ridge,mlp,gbm',
                    help="Comma-list from {ridge, mlp, gbm, linear_forecaster}.")
    ap.add_argument('--head-types', default='forecast96,delta1',
                    help="Comma-list from {forecast96, delta1}.")
    ap.add_argument('--mlp-layers', default='1,2,5',
                    help="Comma-list of MLP hidden-layer depths.")
    ap.add_argument('--gbm-depth', type=int, default=6,
                    help="max_depth for HistGradientBoostingRegressor GBM probe (default 6).")
    ap.add_argument('--ridge-alpha', type=float, default=1.0,
                    help="Ridge regularisation alpha for noise-floor sensitivity analysis (default 1.0).")
    ap.add_argument('--pooling', default='mean', choices=['mean', 'last'],
                    help="Pooling over sequence dimension: 'mean' (default) or 'last' token.")
    ap.add_argument('--data-path', default='data/forecasting/ETTh2.csv')
    ap.add_argument('--horizon', type=int, default=96)
    ap.add_argument('--lookback', type=int, default=96)
    ap.add_argument('--features', default='M', choices=['M', 'S', 'MS'])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--model-size', default='small')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-path', required=True)
    args = ap.parse_args()

    if args.zero_shot == bool(args.encoder_dir):
        raise SystemExit("Provide exactly one of --encoder-dir or --zero-shot.")

    if args.device == 'mps':
        _patch_packed_scaler_for_mps()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    probe_types = [p.strip() for p in args.probe_types.split(',') if p.strip()]
    head_types = [h.strip() for h in args.head_types.split(',') if h.strip()]
    mlp_layers = [int(k) for k in args.mlp_layers.split(',') if k.strip()]
    gbm_depth = args.gbm_depth
    ridge_alpha = args.ridge_alpha
    pooling = args.pooling

    # Load data (mirror finetune_forecasting.py exactly so probe train/val match).
    loader = get_forecasting_loader(
        args.data_path, lookback_window=args.lookback,
        forecast_horizon=args.horizon, features=args.features)
    train_df, val_df, test_df = loader.get_splits()

    feature_cols = ['OT'] if args.features == 'S' else loader.FEATURE_COLUMNS
    n_features = len(feature_cols)
    val_vals = val_df[feature_cols].values
    test_vals = test_df[feature_cols].values

    extended_lookback = args.lookback + args.horizon

    def make_eval_sequences(data, ext_lb, hz):
        X, y = [], []
        total = ext_lb + hz
        for i in range(len(data) - total + 1):
            X.append(data[i:i+ext_lb])
            y.append(data[i+ext_lb:i+total])
        return np.array(X), np.array(y)

    X_val_eval, y_val_eval_raw = make_eval_sequences(val_vals, extended_lookback, args.horizon)
    X_test_eval, y_test_eval_raw = make_eval_sequences(test_vals, extended_lookback, args.horizon)

    n_probe_train = min(300, len(X_val_eval))
    n_probe_val = min(200, len(X_test_eval))
    X_probe_train_t = torch.from_numpy(X_val_eval[:n_probe_train]).float()
    X_probe_val_t = torch.from_numpy(X_test_eval[:n_probe_val]).float()

    # Build heads (after we know n_probe_* sizes).
    y_probe_train_by_head = build_heads(
        y_val_eval_raw[:n_probe_train], X_val_eval[:n_probe_train],
        head_types, args.features)
    y_probe_val_by_head = build_heads(
        y_test_eval_raw[:n_probe_val], X_test_eval[:n_probe_val],
        head_types, args.features)

    # Load Moirai-Small.
    from src.models.moirai_detector import MoiraiAnomalyDetector
    detector = MoiraiAnomalyDetector(
        model_size=args.model_size,
        context_length=args.lookback,
        prediction_length=args.horizon,
        target_dim=n_features,
        num_samples=20,
        device=args.device)
    detector.initialize()
    model = detector.model

    # If encoder-dir supplied, load best_encoder.pt onto the model.
    if args.encoder_dir:
        enc_path = Path(args.encoder_dir) / 'best_encoder.pt'
        if not enc_path.exists():
            raise SystemExit(f"Missing: {enc_path}")
        state = torch.load(enc_path, map_location=args.device)
        device_state = {k: v.to(args.device) for k, v in state.items()}
        model.load_state_dict(device_state)
        logger.info(f"Loaded saved encoder from {enc_path}")
    else:
        logger.info("Using pre-trained Moirai-Small encoder (zero-shot reference)")

    # Extract representations once; reuse across all (probe × head) combos.
    # pooling='last' uses the final sequence token instead of mean-pooling.
    keep_seq = (pooling == 'last')
    reps_train_raw = extract_representations(
        model, X_probe_train_t, None, device=args.device,
        max_samples=n_probe_train, keep_sequence=keep_seq)
    reps_val_raw = extract_representations(
        model, X_probe_val_t, None, device=args.device,
        max_samples=n_probe_val, keep_sequence=keep_seq)
    if pooling == 'last':
        reps_train = reps_train_raw[:, -1, :]
        reps_val = reps_val_raw[:, -1, :]
    else:
        reps_train = reps_train_raw
        reps_val = reps_val_raw
    logger.info(f"Reps ({pooling}-pool): train={reps_train.shape}, val={reps_val.shape}")

    out = {
        'encoder_dir': args.encoder_dir,
        'zero_shot': bool(args.zero_shot),
        'data_path': args.data_path,
        'horizon': args.horizon,
        'features': args.features,
        'n_probe_train': int(n_probe_train),
        'n_probe_val': int(n_probe_val),
        'probe_types': probe_types,
        'head_types': head_types,
        'mlp_layers': mlp_layers,
        'gbm_depth': gbm_depth,
        'ridge_alpha': ridge_alpha,
        'pooling': pooling,
        'results': {},
    }

    for head in head_types:
        y_tr = y_probe_train_by_head[head]
        y_va = y_probe_val_by_head[head]
        head_out = {}
        for probe in probe_types:
            if probe == 'mlp':
                for k_depth in mlp_layers:
                    r2 = linear_probe_r2(
                        reps_train, reps_val, y_tr, y_va,
                        probe_type='mlp', mlp_layers=k_depth)
                    head_out[f'mlp_k{k_depth}'] = float(r2)
                    logger.info(f"[{head}] mlp k={k_depth}: R²={r2:+.4f}")
            elif probe == 'gbm':
                r2 = linear_probe_r2(
                    reps_train, reps_val, y_tr, y_va,
                    probe_type='gbm', gbm_depth=gbm_depth)
                head_out[f'gbm_d{gbm_depth}'] = float(r2)
                logger.info(f"[{head}] gbm depth={gbm_depth}: R²={r2:+.4f}")
            else:
                r2 = linear_probe_r2(
                    reps_train, reps_val, y_tr, y_va,
                    probe_type=probe, alpha=ridge_alpha)
                head_out[probe] = float(r2)
                logger.info(f"[{head}] {probe} (α={ridge_alpha}): R²={r2:+.4f}")
        out['results'][head] = head_out

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, 'w') as f:
        json.dump(out, f, indent=2)
    logger.info(f"saved: {args.out_path}")


if __name__ == '__main__':
    main()
