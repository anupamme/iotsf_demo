"""Data loading and sequence creation for forecasting experiments."""

import numpy as np
import pandas as pd
from pathlib import Path


ETT_FEATURE_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def load_ett_data(
    data_path: str,
    features: str = "M",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
):
    """Load ETT dataset and split into train/val/test.

    Args:
        data_path: Path to ETTh2.csv (or ETTh1, ETTm2).
        features: 'M' (multivariate), 'S' (univariate OT only).
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.

    Returns:
        (train_vals, val_vals, test_vals, feature_cols) as numpy arrays.
    """
    df = pd.read_csv(data_path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    feature_cols = ["OT"] if features == "S" else ETT_FEATURE_COLUMNS
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = df[feature_cols].values.astype(np.float32)
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]

    return train, val, test, feature_cols


def load_ili_data(
    data_path: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
):
    """Load ILI (National Illness) dataset.

    Returns:
        (train, val, test, feature_cols) as numpy arrays.
    """
    df = pd.read_csv(data_path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    feature_cols = list(df.columns)
    data = df.values.astype(np.float32)
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]

    return train, val, test, feature_cols


def make_train_sequences(data: np.ndarray, ctx_len: int, horizon: int):
    """Create (context, target) pairs for NLL training.

    Returns:
        X: (N, ctx_len, D) context windows.
        y: (N, horizon, D) target windows.
    """
    X, y = [], []
    total = ctx_len + horizon
    for i in range(len(data) - total + 1):
        X.append(data[i : i + ctx_len])
        y.append(data[i + ctx_len : i + total])
    return np.array(X), np.array(y)


def make_eval_sequences(data: np.ndarray, extended_lookback: int, horizon: int):
    """Create extended-lookback sequences for inference evaluation.

    Returns:
        X: (N, extended_lookback, D) context windows.
        y: (N, horizon, D) target windows.
    """
    X, y = [], []
    total = extended_lookback + horizon
    for i in range(len(data) - total + 1):
        X.append(data[i : i + extended_lookback])
        y.append(data[i + extended_lookback : i + total])
    return np.array(X), np.array(y)


def linear_baseline_mse(contexts: list, targets: list) -> float:
    """Per-feature Ridge regression baseline MSE for value-gate comparison."""
    from sklearn.linear_model import Ridge

    n_windows = len(contexts)
    lookback = contexts[0].shape[0]
    horizon = targets[0].shape[0]
    n_features = contexts[0].shape[1]

    all_preds = []
    all_targets = []

    for feat in range(n_features):
        X_train = np.array([c[:, feat] for c in contexts[: n_windows // 2]])
        y_train = np.array([t[:, feat] for t in targets[: n_windows // 2]])
        X_test = np.array([c[:, feat] for c in contexts[n_windows // 2 :]])
        y_test = np.array([t[:, feat] for t in targets[n_windows // 2 :]])

        reg = Ridge(alpha=1.0).fit(X_train, y_train)
        preds = reg.predict(X_test)
        all_preds.append(preds)
        all_targets.append(y_test)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return float(np.mean((all_preds - all_targets) ** 2))
