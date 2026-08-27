#!/usr/bin/env python3
"""
Zero-Shot Moirai Evaluation on ETTh1 Forecasting Benchmark

Evaluates pre-trained Moirai (no fine-tuning) against Linear/DLinear baselines
to establish that pre-trained features provide measurable value on this task.

This is the critical Phase 1 gate for Path 2: if Moirai >> Linear by >20%,
pre-trained features are demonstrably useful and we can test whether
contrastive fine-tuning causes catastrophic forgetting.

Usage:
    python scripts/evaluate_moirai_zeroshot_forecasting.py \
        --data-path data/forecasting/ETTh1.csv \
        --horizons 96,192,336,720 \
        --results-dir results/forecasting_baselines
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.forecasting_loader import ETTh1Loader, ForecastingDataset, compute_forecast_metrics


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------

class LinearForecaster:
    """Simple linear regression baseline: predict y = W @ x + b."""

    def __init__(self, lookback: int, horizon: int, n_features: int):
        self.lookback = lookback
        self.horizon = horizon
        self.n_features = n_features
        self.W = None  # (horizon * n_feat, lookback * n_feat)
        self.b = None  # (horizon * n_feat,)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit via closed-form OLS: W = (X^T X)^{-1} X^T y.

        Args:
            X: (n, lookback, n_features)
            y: (n, horizon, n_features)
        """
        n = X.shape[0]
        X_flat = X.reshape(n, -1)  # (n, lookback * n_feat)
        y_flat = y.reshape(n, -1)  # (n, horizon * n_feat)

        # Ridge regression with small regularisation for stability
        lam = 1e-4
        XtX = X_flat.T @ X_flat + lam * np.eye(X_flat.shape[1])
        Xty = X_flat.T @ y_flat
        self.W = np.linalg.solve(XtX, Xty)  # (lookback*feat, horizon*feat)
        self.b = y_flat.mean(axis=0) - X_flat.mean(axis=0) @ self.W

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        X_flat = X.reshape(n, -1)
        y_flat = X_flat @ self.W + self.b
        return y_flat.reshape(n, self.horizon, self.n_features)


class DLinearForecaster:
    """
    DLinear: Decomposition-Linear.

    Decomposes series into trend (moving average) and seasonal (residual),
    applies separate linear layers to each, then sums.
    """

    def __init__(self, lookback: int, horizon: int, n_features: int, kernel_size: int = 25):
        self.lookback = lookback
        self.horizon = horizon
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.W_trend = None
        self.b_trend = None
        self.W_seasonal = None
        self.b_seasonal = None

    def _decompose(self, X: np.ndarray) -> tuple:
        """Moving-average decomposition: trend + seasonal."""
        # X: (n, lookback, n_features)
        k = self.kernel_size
        pad = k // 2

        # Pad and compute moving average per feature
        trend = np.zeros_like(X)
        for i in range(X.shape[2]):
            for j in range(X.shape[0]):
                padded = np.pad(X[j, :, i], (pad, pad), mode='edge')
                trend[j, :, i] = np.convolve(padded, np.ones(k) / k, mode='valid')[:self.lookback]

        seasonal = X - trend
        return trend, seasonal

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        trend_X, seasonal_X = self._decompose(X)

        # Decompose targets too (for fitting)
        # But we predict full y from each component
        trend_flat = trend_X.reshape(n, -1)
        seasonal_flat = seasonal_X.reshape(n, -1)
        y_flat = y.reshape(n, -1)

        lam = 1e-4
        d = trend_flat.shape[1]

        # Fit trend component
        TtT = trend_flat.T @ trend_flat + lam * np.eye(d)
        Tty = trend_flat.T @ y_flat
        self.W_trend = np.linalg.solve(TtT, Tty)
        self.b_trend = y_flat.mean(axis=0) - trend_flat.mean(axis=0) @ self.W_trend

        # Fit seasonal component
        StS = seasonal_flat.T @ seasonal_flat + lam * np.eye(d)
        Sty = seasonal_flat.T @ y_flat
        self.W_seasonal = np.linalg.solve(StS, Sty)
        self.b_seasonal = y_flat.mean(axis=0) - seasonal_flat.mean(axis=0) @ self.W_seasonal

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        trend_X, seasonal_X = self._decompose(X)

        trend_flat = trend_X.reshape(n, -1)
        seasonal_flat = seasonal_X.reshape(n, -1)

        y_trend = trend_flat @ self.W_trend + self.b_trend
        y_seasonal = seasonal_flat @ self.W_seasonal + self.b_seasonal

        y_flat = y_trend + y_seasonal
        return y_flat.reshape(n, self.horizon, self.n_features)


class RepeatLastForecaster:
    """Naive baseline: repeat the last value of the lookback window."""

    def __init__(self, horizon: int):
        self.horizon = horizon

    def fit(self, X, y):
        pass  # No-op

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X: (n, lookback, n_features) — repeat the last timestep
        last_val = X[:, -1:, :]  # (n, 1, n_features)
        return np.tile(last_val, (1, self.horizon, 1))


# ---------------------------------------------------------------------------
# Moirai zero-shot evaluation
# ---------------------------------------------------------------------------

def evaluate_moirai_zeroshot(
    loader: ETTh1Loader,
    horizon: int,
    device: str = 'cpu',
    batch_size: int = 32,
    num_samples: int = 20,
    max_test_sequences: int = 0
) -> dict:
    """
    Evaluate Moirai zero-shot (no fine-tuning) on ETTh1 test set.

    Args:
        num_samples: Number of forecast samples for probabilistic prediction (default 20)
        max_test_sequences: If >0, subsample test set for faster evaluation

    Returns dict with MSE, MAE, RMSE.
    """
    try:
        from src.models.moirai_detector import MoiraiAnomalyDetector
    except ImportError:
        logger.error("uni2ts not available; skipping Moirai evaluation")
        return {'mse': float('nan'), 'mae': float('nan'), 'rmse': float('nan')}

    lookback = 96
    n_features = len(loader.FEATURE_COLUMNS)

    # Reinitialise loader with this horizon
    loader_h = ETTh1Loader(
        data_path=str(loader.data_path),
        lookback_window=lookback,
        forecast_horizon=horizon,
        features='M'
    )

    # Get splits for normalization statistics and test data
    train_df, val_df, test_df = loader_h.get_splits()

    # Compute TRAINING normalization statistics (for fair comparison with published numbers)
    train_vals = train_df[loader.FEATURE_COLUMNS].values
    train_mean = train_vals.mean(axis=0)
    train_std = train_vals.std(axis=0) + 1e-8

    # MoiraiForecast.forward with patch_size='auto' expects:
    #   past_target shape: (batch, past_length, features) where past_length = context_length + prediction_length
    #   - _val_loss uses first past_length timesteps for patch size selection
    #   - _get_distr uses LAST context_length timesteps as actual input for prediction
    extended_lookback = lookback + horizon  # past_length = context_length + prediction_length
    values = test_df[loader.FEATURE_COLUMNS].values  # raw scale
    total_window = extended_lookback + horizon

    X_extended, y_raw = [], []
    for i in range(len(values) - total_window + 1):
        X_extended.append(values[i:i+extended_lookback])
        y_raw.append(values[i+extended_lookback:i+total_window])

    context = torch.from_numpy(np.array(X_extended)).float()
    target_raw = np.array(y_raw)
    logger.info(f"Created {len(context)} sequences with extended lookback={extended_lookback}")

    # Subsample if requested (for faster iteration)
    if max_test_sequences > 0 and max_test_sequences < len(context):
        indices = np.linspace(0, len(context)-1, max_test_sequences, dtype=int)
        context = context[indices]
        target_raw = target_raw[indices]
        logger.info(f"Subsampled to {len(context)} test sequences")

    logger.info(f"Moirai eval: {len(context)} test sequences, horizon={horizon}, num_samples={num_samples}")

    # Load pre-trained Moirai
    detector = MoiraiAnomalyDetector(
        model_size='small',
        context_length=lookback,
        prediction_length=horizon,
        target_dim=n_features,
        num_samples=num_samples,
        device=device
    )
    detector.initialize()

    # Run zero-shot inference
    all_preds = []
    t0 = time.time()

    with torch.no_grad():
        for i in range(0, len(context), batch_size):
            batch_ctx = context[i:i+batch_size]  # (b, extended_lookback, features)
            b = batch_ctx.shape[0]

            # past_target is the extended context (context_length + prediction_length)
            # Moirai's _get_distr uses the last context_length timesteps for prediction
            past_target = batch_ctx.to(device)
            past_observed = torch.ones_like(past_target, dtype=torch.bool)
            past_is_pad = torch.zeros(b, extended_lookback, dtype=torch.bool, device=device)

            # Forward pass — returns (batch, num_samples, horizon, n_features)
            forecast_samples = detector.model.forward(
                past_target=past_target,
                past_observed_target=past_observed,
                past_is_pad=past_is_pad,
                num_samples=num_samples
            )

            # Median forecast (robust to heavy-tailed mixture components)
            # Moirai paper uses median for point forecasts
            median_forecast = forecast_samples.median(dim=1).values.cpu().numpy()  # (batch, horizon, n_feat)
            all_preds.append(median_forecast)

            batch_num = i // batch_size
            if batch_num % 10 == 0:
                elapsed = time.time() - t0
                total_batches = (len(context) + batch_size - 1) // batch_size
                logger.info(f"  Moirai: batch {batch_num+1}/{total_batches} "
                           f"({i+b}/{len(context)} seqs, {elapsed:.0f}s elapsed)")

    elapsed = time.time() - t0
    predictions_raw = np.concatenate(all_preds, axis=0)

    # Normalize both predictions and targets using TRAINING statistics
    # This matches standard TSF benchmarking protocol for fair comparison
    predictions_norm = (predictions_raw - train_mean) / train_std
    targets_norm = (target_raw - train_mean) / train_std

    mse_norm = float(np.mean((predictions_norm - targets_norm) ** 2))
    mae_norm = float(np.mean(np.abs(predictions_norm - targets_norm)))

    # Also compute raw-scale metrics for reference
    mse_raw = float(np.mean((predictions_raw - target_raw) ** 2))

    logger.info(f"Moirai zero-shot (horizon={horizon}): "
               f"MSE(norm)={mse_norm:.6f}, MAE(norm)={mae_norm:.6f}, "
               f"MSE(raw)={mse_raw:.6f}, time={elapsed:.1f}s")

    metrics = {'mse': mse_norm, 'mae': mae_norm, 'rmse': float(np.sqrt(mse_norm)),
               'mse_raw': mse_raw}
    return metrics


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------

def evaluate_baselines(
    loader: ETTh1Loader,
    horizon: int
) -> dict:
    """
    Evaluate Linear, DLinear, and Repeat-Last baselines.

    Uses standard protocol: normalize ALL data using TRAINING mean/std,
    compute MSE/MAE on normalized predictions vs normalized targets.

    Returns dict of {model_name: {mse, mae, rmse}}.
    """
    lookback = 96
    n_features = len(loader.FEATURE_COLUMNS)

    # Reinitialise loader with this horizon
    loader_h = ETTh1Loader(
        data_path=str(loader.data_path),
        lookback_window=lookback,
        forecast_horizon=horizon,
        features='M'
    )

    # Standard protocol: normalize using TRAINING statistics
    train_df, val_df, test_df = loader_h.get_splits()
    train_vals = train_df[loader.FEATURE_COLUMNS].values
    test_vals = test_df[loader.FEATURE_COLUMNS].values

    train_mean = train_vals.mean(axis=0)
    train_std = train_vals.std(axis=0) + 1e-8

    # Normalize both train and test using training statistics
    train_norm = (train_vals - train_mean) / train_std
    test_norm = (test_vals - train_mean) / train_std

    # Create sliding-window sequences
    def _make_sequences(data, lb, hz):
        X, y = [], []
        for i in range(len(data) - lb - hz + 1):
            X.append(data[i:i+lb])
            y.append(data[i+lb:i+lb+hz])
        return np.array(X), np.array(y)

    X_train, y_train = _make_sequences(train_norm, lookback, horizon)
    X_test, y_test = _make_sequences(test_norm, lookback, horizon)

    logger.info(f"Baselines: train={len(X_train)}, test={len(X_test)}, horizon={horizon}")

    results = {}

    # --- Repeat Last ---
    model = RepeatLastForecaster(horizon)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = compute_forecast_metrics(preds, y_test)  # already normalized, no scaler needed
    results['repeat_last'] = metrics
    logger.info(f"  Repeat-Last (h={horizon}): MSE={metrics['mse']:.6f}")

    # --- Linear ---
    model = LinearForecaster(lookback, horizon, n_features)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = compute_forecast_metrics(preds, y_test)
    results['linear'] = metrics
    logger.info(f"  Linear (h={horizon}): MSE={metrics['mse']:.6f}")

    # --- DLinear ---
    model = DLinearForecaster(lookback, horizon, n_features)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = compute_forecast_metrics(preds, y_test)
    results['dlinear'] = metrics
    logger.info(f"  DLinear (h={horizon}): MSE={metrics['mse']:.6f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Zero-shot Moirai vs baselines on ETTh1")
    parser.add_argument('--data-path', default='data/forecasting/ETTh1.csv')
    parser.add_argument('--horizons', default='96,192,336,720',
                        help="Comma-separated forecast horizons")
    parser.add_argument('--results-dir', default='results/forecasting_baselines')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--skip-moirai', action='store_true',
                        help="Skip Moirai evaluation (baselines only)")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-samples', type=int, default=20,
                        help="Number of forecast samples for Moirai (default 20)")
    parser.add_argument('--max-test-sequences', type=int, default=0,
                        help="Subsample test set for faster Moirai eval (0=all)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    horizons = [int(h) for h in args.horizons.split(',')]

    # Initial loader (horizon will be overridden per experiment)
    loader = ETTh1Loader(
        data_path=args.data_path,
        lookback_window=96,
        forecast_horizon=96,
        features='M'
    )

    all_results = {}

    for horizon in horizons:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating forecast horizon = {horizon}")
        logger.info(f"{'='*60}")

        horizon_results = {}

        # Baselines
        baseline_results = evaluate_baselines(loader, horizon)
        horizon_results.update(baseline_results)

        # Moirai zero-shot
        if not args.skip_moirai:
            moirai_results = evaluate_moirai_zeroshot(
                loader, horizon,
                device=args.device,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                max_test_sequences=args.max_test_sequences
            )
            horizon_results['moirai_zeroshot'] = moirai_results

        all_results[f'horizon_{horizon}'] = horizon_results

        # Compute gap if Moirai ran
        if 'moirai_zeroshot' in horizon_results and not np.isnan(horizon_results['moirai_zeroshot']['mse']):
            linear_mse = horizon_results['linear']['mse']
            moirai_mse = horizon_results['moirai_zeroshot']['mse']
            gap_pct = (linear_mse - moirai_mse) / linear_mse * 100

            logger.info(f"\n  Gap (horizon={horizon}): "
                       f"Linear MSE={linear_mse:.6f}, Moirai MSE={moirai_mse:.6f}, "
                       f"Improvement={gap_pct:.1f}%")

            if gap_pct < 10:
                logger.warning(f"  ⚠ Gap <10%: pre-trained features may not be sufficiently valuable")
            elif gap_pct < 20:
                logger.warning(f"  ⚠ Gap 10-20%: marginal pre-trained feature value")
            else:
                logger.info(f"  ✓ Gap >20%: pre-trained features demonstrably valuable")

    # Save results
    output_path = results_dir / 'metrics.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Print summary table
    logger.info("\n" + "="*80)
    logger.info("SUMMARY: MSE by Model and Horizon")
    logger.info("="*80)

    models = set()
    for h_results in all_results.values():
        models.update(h_results.keys())
    models = sorted(models)

    header = f"{'Model':<20}" + "".join(f"{'h=' + str(h):>12}" for h in horizons)
    logger.info(header)
    logger.info("-" * len(header))

    for model_name in models:
        row = f"{model_name:<20}"
        for horizon in horizons:
            key = f'horizon_{horizon}'
            if key in all_results and model_name in all_results[key]:
                mse = all_results[key][model_name]['mse']
                row += f"{mse:>12.6f}"
            else:
                row += f"{'---':>12}"
        logger.info(row)


if __name__ == '__main__':
    main()
