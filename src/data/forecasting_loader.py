"""
ETTh1 Forecasting Dataset Loader

Provides functionality to load ETTh1 (Electricity Transformer Temperature - Hourly)
dataset for time-series forecasting experiments.

Dataset details:
- 7 features: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT (oil temperature + 6 load features)
- Hourly granularity
- ~17,420 timesteps (2 years)
- Standard split: 12 months train / 4 months val / 4 months test
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from loguru import logger
import torch
from torch.utils.data import Dataset


class ETTh1Loader:
    """
    Loader for ETTh1 (Electricity Transformer Temperature - Hourly) dataset.

    Standard forecasting benchmark with 7 features and hourly granularity.
    """

    # Feature columns in ETTh1
    FEATURE_COLUMNS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

    # Standard forecast horizons (in timesteps)
    FORECAST_HORIZONS = [96, 192, 336, 720]  # 4h, 8h, 14h, 30h ahead

    def __init__(
        self,
        data_path: str,
        lookback_window: int = 96,
        forecast_horizon: int = 96,
        features: str = 'M'  # 'M': multivariate, 'S': univariate (OT only), 'MS': multivariate predict univariate
    ):
        """
        Initialize ETTh1 loader.

        Args:
            data_path: Path to ETTh1.csv file
            lookback_window: Number of past timesteps to use as input (context)
            forecast_horizon: Number of future timesteps to predict
            features: 'M' (multivariate), 'S' (univariate OT), or 'MS' (multivariate input, univariate output)
        """
        self.data_path = Path(data_path)
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.features = features

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"ETTh1 data file not found at {self.data_path}. "
                "Download from: https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTh1.csv"
            )

        logger.info(f"Loading ETTh1 from {self.data_path}")
        self.data = pd.read_csv(self.data_path)

        # Validate columns
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.set_index('date')

        missing_cols = [col for col in self.FEATURE_COLUMNS if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        logger.info(f"ETTh1 loaded: {len(self.data)} timesteps, {len(self.FEATURE_COLUMNS)} features")

    def get_splits(
        self,
        train_ratio: float = 12/(12+4+4),  # 12 months train
        val_ratio: float = 4/(12+4+4),     # 4 months val
        test_ratio: float = 4/(12+4+4)     # 4 months test
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test using standard chronological split.

        Returns:
            (train_df, val_df, test_df)
        """
        n = len(self.data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = self.data.iloc[:train_end]
        val_df = self.data.iloc[train_end:val_end]
        test_df = self.data.iloc[val_end:]

        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df

    def create_sequences(
        self,
        data: pd.DataFrame,
        scale: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
        """
        Create (input, target) sequences for forecasting.

        Args:
            data: DataFrame to create sequences from
            scale: Whether to apply normalization (mean=0, std=1)

        Returns:
            (X, y, scaler_params) where:
            - X: (n_sequences, lookback_window, n_features) input sequences
            - y: (n_sequences, forecast_horizon, n_features) target sequences
            - scaler_params: {'mean': ..., 'std': ...} for inverse transform
        """
        # Select features based on mode
        if self.features == 'S':
            # Univariate: only OT (oil temperature)
            values = data[['OT']].values
        elif self.features == 'M':
            # Multivariate: all features
            values = data[self.FEATURE_COLUMNS].values
        elif self.features == 'MS':
            # Multivariate input, univariate output
            input_values = data[self.FEATURE_COLUMNS].values
            target_values = data[['OT']].values
        else:
            raise ValueError(f"Invalid features mode: {self.features}")

        # Normalization
        scaler_params = None
        if scale:
            if self.features == 'MS':
                # Scale input and target separately
                input_mean, input_std = input_values.mean(axis=0), input_values.std(axis=0)
                target_mean, target_std = target_values.mean(axis=0), target_values.std(axis=0)
                input_values = (input_values - input_mean) / (input_std + 1e-8)
                target_values = (target_values - target_mean) / (target_std + 1e-8)
                scaler_params = {
                    'input_mean': input_mean, 'input_std': input_std,
                    'target_mean': target_mean, 'target_std': target_std
                }
            else:
                mean, std = values.mean(axis=0), values.std(axis=0)
                values = (values - mean) / (std + 1e-8)
                scaler_params = {'mean': mean, 'std': std}

        # Create sliding windows
        X, y = [], []
        total_window = self.lookback_window + self.forecast_horizon

        if self.features == 'MS':
            for i in range(len(input_values) - total_window + 1):
                X.append(input_values[i:i+self.lookback_window])
                y.append(target_values[i+self.lookback_window:i+total_window])
        else:
            for i in range(len(values) - total_window + 1):
                X.append(values[i:i+self.lookback_window])
                y.append(values[i+self.lookback_window:i+total_window])

        X = np.array(X)  # (n_sequences, lookback, n_features)
        y = np.array(y)  # (n_sequences, horizon, n_features)

        logger.info(f"Created {len(X)} sequences: X {X.shape}, y {y.shape}")
        return X, y, scaler_params

    def get_moirai_format(
        self,
        split: str = 'test',
        n_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get data in Moirai-compatible format for inference.

        Args:
            split: 'train', 'val', or 'test'
            n_samples: Optional limit on number of sequences

        Returns:
            (context, target, metadata) where:
            - context: (n, lookback, n_features) tensor
            - target: (n, horizon, n_features) tensor
            - metadata: scaler parameters for inverse transform
        """
        train_df, val_df, test_df = self.get_splits()

        if split == 'train':
            data_df = train_df
        elif split == 'val':
            data_df = val_df
        elif split == 'test':
            data_df = test_df
        else:
            raise ValueError(f"Invalid split: {split}")

        X, y, scaler_params = self.create_sequences(data_df, scale=True)

        if n_samples is not None and n_samples < len(X):
            # Random sampling for fine-tuning experiments
            indices = np.random.choice(len(X), size=n_samples, replace=False)
            X, y = X[indices], y[indices]
            logger.info(f"Sampled {n_samples} sequences from {split} split")

        # Convert to PyTorch tensors
        context = torch.from_numpy(X).float()
        target = torch.from_numpy(y).float()

        return context, target, scaler_params


class WeatherLoader(ETTh1Loader):
    """
    Loader for the Autoformer/Informer Weather benchmark (Jena climate).

    14 features (13 meteorological + OT=temperature target), hourly resolution
    from 10-min raw, 7:1:2 train/val/test split per LTSF convention.
    """

    FEATURE_COLUMNS = [
        'p (mbar)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)',
        'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)',
        'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)', 'OT'
    ]

    def get_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return super().get_splits(train_ratio, val_ratio, test_ratio)


class ElectricityLoader(ETTh1Loader):
    """
    Loader for the Autoformer/Informer Electricity benchmark.

    370 hourly series (MT_001..MT_370) over 26,208 timesteps; the last column
    (MT_370, renamed OT) is the canonical target per LTSF convention.
    """

    FEATURE_COLUMNS = [f"MT_{i:03d}" for i in range(1, 370)] + ['OT']

    def get_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return super().get_splits(train_ratio, val_ratio, test_ratio)


class TrafficLoader(ETTh1Loader):
    """
    Loader for the LTSF-Linear Traffic benchmark (PEMS-BAY, 862 sensors).

    862 hourly series over 17,544 timesteps; last column is the target (OT).
    Uses univariate (S) mode by default for the ZS gate check.
    Column layout: date + 862 numeric sensor columns, last named OT.
    """

    # Traffic has 862 sensor columns; only OT (last) is needed for the gate check.
    # Column names are auto-detected from the CSV header at load time.
    FEATURE_COLUMNS = None  # overridden in __init__

    def __init__(self, data_path: str, **kwargs):
        self.data_path = Path(data_path)
        self.lookback_window = kwargs.get('lookback_window', 96)
        self.forecast_horizon = kwargs.get('forecast_horizon', 96)
        self.features = kwargs.get('features', 'S')

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Traffic data file not found at {self.data_path}. "
                "Download from the LTSF-Linear benchmark: "
                "https://github.com/thuml/LTSF-Linear"
            )

        self.data = pd.read_csv(data_path)
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.set_index('date')

        # Accept any column layout; use last column as OT target
        if 'OT' not in self.data.columns:
            self.data = self.data.rename(columns={self.data.columns[-1]: 'OT'})
        self.FEATURE_COLUMNS = list(self.data.columns)
        logger.info(f"Traffic loaded: {len(self.data)} timesteps, {len(self.FEATURE_COLUMNS)} features")

    def get_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return super().get_splits(train_ratio, val_ratio, test_ratio)


def get_forecasting_loader(data_path: str, **kwargs):
    """Factory: pick loader by filename stem."""
    stem = Path(data_path).stem.lower()
    if stem.startswith('weather'):
        return WeatherLoader(data_path, **kwargs)
    if stem.startswith('electricity'):
        return ElectricityLoader(data_path, **kwargs)
    if stem.startswith('traffic'):
        return TrafficLoader(data_path, **kwargs)
    return ETTh1Loader(data_path, **kwargs)


class ForecastingDataset(Dataset):
    """
    PyTorch Dataset wrapper for forecasting sequences.
    """

    def __init__(
        self,
        context: torch.Tensor,
        target: torch.Tensor
    ):
        """
        Args:
            context: (n, lookback, n_features) input sequences
            target: (n, horizon, n_features) target sequences
        """
        self.context = context
        self.target = target

        assert len(context) == len(target), "Context and target must have same length"

    def __len__(self) -> int:
        return len(self.context)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.context[idx], self.target[idx]


def compute_forecast_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    scaler_params: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Compute standard forecasting metrics (MSE, MAE, RMSE).

    Args:
        predictions: (n, horizon, n_features) predicted values
        targets: (n, horizon, n_features) ground truth values
        scaler_params: Optional scaler params for inverse transform

    Returns:
        Dict with 'mse', 'mae', 'rmse' keys
    """
    # Inverse transform if scaler provided
    if scaler_params is not None:
        if 'target_mean' in scaler_params:
            # MS mode: use target scaler
            mean, std = scaler_params['target_mean'], scaler_params['target_std']
        else:
            # M or S mode: use single scaler
            mean, std = scaler_params['mean'], scaler_params['std']

        predictions = predictions * std + mean
        targets = targets * std + mean

    # Compute metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)

    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse)
    }


if __name__ == '__main__':
    # Example usage
    loader = ETTh1Loader(
        data_path='data/ETTh1.csv',
        lookback_window=96,
        forecast_horizon=96,
        features='M'
    )

    # Get train/val/test splits
    context, target, scaler = loader.get_moirai_format(split='test', n_samples=100)
    print(f"Context shape: {context.shape}")
    print(f"Target shape: {target.shape}")

    # Create PyTorch dataset
    dataset = ForecastingDataset(context, target)
    print(f"Dataset length: {len(dataset)}")
