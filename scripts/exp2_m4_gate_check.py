"""Gate check: Chronos/MOMENT zero-shot on M4-Monthly.

Tests the corpus-coverage hypothesis: M4 is documented in both Chronos
(public M competition data) and MOMENT corpora. If either gate-passes
(ZS outperforms Linear by ≥20%), corpus composition is doing real work.

Usage:
    python scripts/exp2_m4_gate_check.py \
        --backbone chronos \
        --model-name amazon/chronos-t5-small \
        --out-path results/exp2_corpus_coverage/chronos_m4_monthly.json

    python scripts/exp2_m4_gate_check.py \
        --backbone moment \
        --model-name AutonLab/MOMENT-1-large \
        --out-path results/exp2_corpus_coverage/moment_m4_monthly.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

# M4 Monthly subset: use datasetsforecast or download directly
M4_MONTHLY_URL = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Monthly-train.csv"
M4_MONTHLY_TEST_URL = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Monthly-test.csv"

LOOKBACK = 96
HORIZON = 18  # M4-Monthly standard forecast horizon
GATE_THRESHOLD = 0.20  # 20% improvement required
NUM_SERIES = 200  # Subsample for tractability


def download_m4_monthly(cache_dir: str = "/tmp/m4_data"):
    """Download M4-Monthly train/test if not cached."""
    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(cache_dir, "Monthly-train.csv")
    test_path = os.path.join(cache_dir, "Monthly-test.csv")

    if not os.path.exists(train_path):
        print(f"Downloading M4-Monthly train...")
        urllib.request.urlretrieve(M4_MONTHLY_URL, train_path)
    if not os.path.exists(test_path):
        print(f"Downloading M4-Monthly test...")
        urllib.request.urlretrieve(M4_MONTHLY_TEST_URL, test_path)

    return train_path, test_path


def load_m4_monthly(train_path, test_path, n_series=NUM_SERIES, min_len=None):
    """Load M4-Monthly: each row is a time series (variable length, NaN-padded).

    Returns list of (train_array, test_array) tuples for series with enough history.
    """
    import pandas as pd
    train_df = pd.read_csv(train_path, index_col=0)
    test_df = pd.read_csv(test_path, index_col=0)

    if min_len is None:
        min_len = LOOKBACK + HORIZON

    series_pairs = []
    for idx in train_df.index[:n_series * 3]:  # scan more to find enough valid
        train_vals = train_df.loc[idx].dropna().values.astype(np.float64)
        if len(train_vals) < min_len:
            continue
        test_vals = test_df.loc[idx].dropna().values.astype(np.float64)
        if len(test_vals) < HORIZON:
            continue
        series_pairs.append((train_vals, test_vals))
        if len(series_pairs) >= n_series:
            break

    print(f"Loaded {len(series_pairs)} M4-Monthly series with min_len >= {min_len}")
    return series_pairs


def linear_baseline_mse(series_pairs):
    """Per-series linear baseline: fit on train, predict on test, normalised MSE."""
    mses = []
    for train_vals, test_vals in series_pairs:
        # Use last LOOKBACK points as context, predict HORIZON
        if len(train_vals) < LOOKBACK:
            continue
        ctx = train_vals[-LOOKBACK:]
        X_tr = np.array([train_vals[i:i+LOOKBACK] for i in range(len(train_vals) - LOOKBACK - HORIZON + 1)])
        Y_tr = np.array([train_vals[i+LOOKBACK:i+LOOKBACK+HORIZON] for i in range(len(train_vals) - LOOKBACK - HORIZON + 1)])

        if len(X_tr) < 5:
            continue

        mu = train_vals.mean()
        sd = train_vals.std() + 1e-8

        reg = LinearRegression().fit(X_tr, Y_tr)
        pred = reg.predict(ctx.reshape(1, -1))[0]

        pred_n = (pred - mu) / sd
        test_n = (test_vals[:HORIZON] - mu) / sd
        mses.append(float(np.mean((pred_n - test_n) ** 2)))

    return float(np.mean(mses)) if mses else float('inf')


def chronos_zs_mse(model_name, series_pairs, device="cuda"):
    """Chronos zero-shot: predict each series, compute normalised MSE."""
    from chronos import ChronosPipeline

    pipe = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32,
    )

    mses = []
    for train_vals, test_vals in series_pairs:
        if len(train_vals) < LOOKBACK:
            continue
        ctx = torch.tensor(train_vals[-LOOKBACK:], dtype=torch.float32)
        samples = pipe.predict(ctx, prediction_length=HORIZON, num_samples=20)
        median_pred = samples.squeeze(0).median(dim=0).values.cpu().numpy()

        mu = train_vals.mean()
        sd = train_vals.std() + 1e-8
        pred_n = (median_pred - mu) / sd
        test_n = (test_vals[:HORIZON] - mu) / sd
        mses.append(float(np.mean((pred_n - test_n) ** 2)))

    return float(np.mean(mses)) if mses else float('inf')


def moment_zs_mse(model_name, series_pairs, device="cuda"):
    """MOMENT zero-shot via frozen-encoder Ridge probe (MOMENT has no native forecast head)."""
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={"task_name": "embedding"},
    )
    model = model.to(device)
    model.eval()

    # MOMENT needs fixed-length input (512 or 1024); pad/truncate
    seq_len = 512

    # Collect embeddings + targets for a Ridge probe approach
    embeddings_train = []
    targets_train = []
    embeddings_test = []
    targets_test = []

    with torch.no_grad():
        for train_vals, test_vals in series_pairs:
            if len(train_vals) < seq_len + HORIZON:
                # Pad with zeros at the start
                padded = np.zeros(seq_len + HORIZON)
                padded[-(len(train_vals)):] = train_vals
                train_vals_use = padded
            else:
                train_vals_use = train_vals[-(seq_len + HORIZON):]

            # Train embedding: use [:-HORIZON] as context
            ctx_train = train_vals_use[:seq_len]
            tgt_train = train_vals_use[seq_len:seq_len + HORIZON]

            x = torch.tensor(ctx_train, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            input_mask = torch.ones(1, seq_len, device=device)
            out = model(x_enc=x, input_mask=input_mask, reduction="mean")
            emb = out.embeddings.detach().cpu().numpy().flatten()

            embeddings_train.append(emb)
            targets_train.append(tgt_train)

            # Test: use last seq_len of train as context, predict test
            ctx_test = train_vals[-seq_len:]
            x2 = torch.tensor(ctx_test, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            out2 = model(x_enc=x2, input_mask=input_mask, reduction="mean")
            emb2 = out2.embeddings.detach().cpu().numpy().flatten()

            embeddings_test.append(emb2)
            targets_test.append(test_vals[:HORIZON])

    # Fit Ridge on train embeddings → train targets
    from sklearn.linear_model import Ridge as RidgeReg
    X_tr = np.array(embeddings_train)
    Y_tr = np.array(targets_train)
    X_te = np.array(embeddings_test)
    Y_te = np.array(targets_test)

    reg = RidgeReg(alpha=1.0).fit(X_tr, Y_tr)
    preds = reg.predict(X_te)

    # Per-series normalised MSE
    mses = []
    for i, (train_vals, test_vals) in enumerate(series_pairs):
        if i >= len(preds):
            break
        mu = train_vals.mean()
        sd = train_vals.std() + 1e-8
        pred_n = (preds[i] - mu) / sd
        test_n = (Y_te[i] - mu) / sd
        mses.append(float(np.mean((pred_n - test_n) ** 2)))

    return float(np.mean(mses)) if mses else float('inf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["chronos", "moment"])
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--n-series", type=int, default=NUM_SERIES)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-path", required=True)
    args = ap.parse_args()

    # Download and load M4-Monthly
    train_path, test_path = download_m4_monthly()
    series_pairs = load_m4_monthly(train_path, test_path, n_series=args.n_series)

    if len(series_pairs) < 10:
        print(f"ERROR: Only {len(series_pairs)} valid series found. Need at least 10.")
        sys.exit(1)

    # Linear baseline
    print("Computing linear baseline MSE...")
    linear_mse = linear_baseline_mse(series_pairs)
    print(f"Linear baseline MSE (normalised): {linear_mse:.4f}")

    # Zero-shot model
    print(f"Computing {args.backbone} zero-shot MSE...")
    if args.backbone == "chronos":
        zs_mse = chronos_zs_mse(args.model_name, series_pairs, device=args.device)
    else:
        zs_mse = moment_zs_mse(args.model_name, series_pairs, device=args.device)
    print(f"{args.backbone} ZS MSE (normalised): {zs_mse:.4f}")

    # Gate check
    if linear_mse > 1e-10:
        improvement = (linear_mse - zs_mse) / linear_mse
    else:
        improvement = 0.0

    gate_pass = improvement >= GATE_THRESHOLD
    gate_status = "PASS" if gate_pass else "FAIL"

    print(f"\nGate check: ZS improvement over Linear = {improvement*100:.1f}%")
    print(f"Threshold: {GATE_THRESHOLD*100:.0f}%")
    print(f"Status: {gate_status}")

    results = {
        "backbone": args.backbone,
        "model_name": args.model_name,
        "dataset": "M4-Monthly",
        "n_series": len(series_pairs),
        "horizon": HORIZON,
        "lookback": LOOKBACK,
        "linear_mse_norm": linear_mse,
        "zs_mse_norm": zs_mse,
        "improvement_pct": improvement * 100,
        "gate_threshold_pct": GATE_THRESHOLD * 100,
        "gate_status": gate_status,
    }

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.out_path}")


if __name__ == "__main__":
    main()
