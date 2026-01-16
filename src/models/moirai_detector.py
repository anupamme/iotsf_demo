"""Moirai-based anomaly detection using forecast errors."""

import numpy as np
import torch
from typing import Dict, Optional
from loguru import logger


class MoiraiDetector:
    """
    Moirai foundation model for time-series anomaly detection.

    Uses forecast-based approach: predict future values and compute
    reconstruction error as anomaly score. High prediction error indicates
    anomalous patterns.

    Attributes:
        model_size: Moirai model size ('small', 'base', 'large')
        context_length: Number of time steps for context
        prediction_length: Number of time steps to forecast
        threshold: Anomaly score threshold for binary classification
        model_: Loaded Moirai model
        device_: Torch device (CPU/GPU)
    """

    def __init__(
        self,
        model_size: str = "small",
        context_length: int = 100,
        prediction_length: int = 28,
        threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Initialize Moirai detector.

        Args:
            model_size: Model size - 'small', 'base', or 'large'
            context_length: Context window size for prediction
            prediction_length: Number of steps to forecast
            threshold: Anomaly score threshold for binary classification
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        valid_sizes = ['small', 'base', 'large']
        if model_size not in valid_sizes:
            raise ValueError(f"model_size must be one of {valid_sizes}")

        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")

        self.model_size = model_size
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.threshold = threshold

        # Set device
        if device is None:
            self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device_ = torch.device(device)

        self.model_: Optional[object] = None
        self._is_initialized = False

        logger.info(
            f"Initialized MoiraiDetector (model_size={model_size}, "
            f"context_length={context_length}, prediction_length={prediction_length}, "
            f"device={self.device_})"
        )

    def _initialize_model(self):
        """Load Moirai model from uni2ts."""
        if self._is_initialized:
            return

        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

            logger.info(f"Loading Moirai-{self.model_size} model...")

            # Model name mapping
            model_map = {
                'small': 'Salesforce/moirai-1.0-R-small',
                'base': 'Salesforce/moirai-1.0-R-base',
                'large': 'Salesforce/moirai-1.0-R-large'
            }
            model_name = model_map[self.model_size]

            # Load model
            self.model_ = MoiraiForecast.load_from_checkpoint(
                checkpoint_path=model_name,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                map_location=self.device_
            )

            self._is_initialized = True
            logger.info(f"✅ Moirai model loaded successfully on {self.device_}")

        except ImportError as e:
            logger.error(
                "uni2ts not available. Install with: pip install uni2ts\n"
                "Note: Requires Python 3.12"
            )
            raise ImportError(
                "uni2ts (Moirai) requires Python 3.12. "
                "See requirements-py312.txt"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load Moirai model: {e}")
            raise

    def detect(self, samples: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect anomalies using forecast-based approach.

        For each sample, uses first context_length steps to forecast
        prediction_length steps, then computes MSE against actual values.

        Args:
            samples: Traffic samples. Shape: (n_samples, seq_length, n_features)

        Returns:
            Dictionary containing:
                - predictions: Binary predictions (0=benign, 1=attack). Shape: (n_samples,)
                - scores: Anomaly scores in [0, 1]. Shape: (n_samples,)
                - forecasts: Predicted values. Shape: (n_samples, prediction_length, n_features)
        """
        # Initialize model if needed
        if not self._is_initialized:
            self._initialize_model()

        if samples.ndim != 3:
            raise ValueError(
                f"samples must be 3D (n_samples, seq_length, n_features), "
                f"got shape {samples.shape}"
            )

        n_samples, seq_length, n_features = samples.shape

        if seq_length < self.context_length + self.prediction_length:
            raise ValueError(
                f"seq_length ({seq_length}) must be >= "
                f"context_length + prediction_length "
                f"({self.context_length + self.prediction_length})"
            )

        logger.info(f"Running Moirai detection on {n_samples} samples...")

        scores = np.zeros(n_samples)
        forecasts = np.zeros((n_samples, self.prediction_length, n_features))

        for i in range(n_samples):
            sample = samples[i]  # Shape: (seq_length, n_features)

            # Split into context and target
            context = sample[:self.context_length]  # For prediction
            target = sample[self.context_length:self.context_length + self.prediction_length]

            # Forecast each feature independently
            feature_forecasts = []
            feature_errors = []

            for feat_idx in range(n_features):
                # Get context for this feature
                context_feat = context[:, feat_idx]

                # Prepare input for Moirai (expects specific format)
                forecast = self._forecast_feature(context_feat)

                # Compute prediction error
                actual = target[:, feat_idx]
                mse = np.mean((forecast - actual) ** 2)

                feature_forecasts.append(forecast)
                feature_errors.append(mse)

            # Stack forecasts
            forecasts[i] = np.stack(feature_forecasts, axis=1)

            # Anomaly score = normalized mean squared error across features
            # Use robust normalization based on data scale
            data_scale = np.std(sample) + 1e-8
            normalized_error = np.mean(feature_errors) / (data_scale ** 2)
            scores[i] = min(normalized_error, 1.0)  # Clip to [0, 1]

        # Binary predictions
        predictions = (scores > self.threshold).astype(int)

        logger.info(f"Detected {predictions.sum()}/{n_samples} anomalies")
        logger.debug(f"Anomaly scores: {scores}")

        return {
            'predictions': predictions,
            'scores': scores,
            'forecasts': forecasts
        }

    def _forecast_feature(self, context: np.ndarray) -> np.ndarray:
        """
        Forecast single feature using Moirai.

        Args:
            context: Historical values. Shape: (context_length,)

        Returns:
            Forecast values. Shape: (prediction_length,)
        """
        try:
            # Convert to torch tensor
            context_tensor = torch.from_numpy(context).float().unsqueeze(0).unsqueeze(-1)
            # Shape: (1, context_length, 1) - batch, time, features

            # Move to device
            context_tensor = context_tensor.to(self.device_)

            # Generate forecast
            with torch.no_grad():
                # Moirai expects dictionary input
                forecast = self.model_(context_tensor)

                # Extract forecast (model returns dict or tuple)
                if isinstance(forecast, dict):
                    forecast_values = forecast['forecast']
                elif isinstance(forecast, tuple):
                    forecast_values = forecast[0]
                else:
                    forecast_values = forecast

            # Convert back to numpy
            forecast_np = forecast_values.cpu().numpy().squeeze()

            # Ensure correct shape
            if forecast_np.ndim == 0:
                forecast_np = np.array([forecast_np])
            elif len(forecast_np) != self.prediction_length:
                # Pad or truncate if needed
                if len(forecast_np) < self.prediction_length:
                    forecast_np = np.pad(
                        forecast_np,
                        (0, self.prediction_length - len(forecast_np)),
                        mode='edge'
                    )
                else:
                    forecast_np = forecast_np[:self.prediction_length]

            return forecast_np

        except Exception as e:
            logger.warning(f"Forecast failed, using fallback: {e}")
            # Fallback: use last value (persistence model)
            return np.full(self.prediction_length, context[-1])

    def get_params(self) -> Dict:
        """Get detector parameters."""
        return {
            'model_size': self.model_size,
            'context_length': self.context_length,
            'prediction_length': self.prediction_length,
            'threshold': self.threshold,
            'device': str(self.device_),
            'is_initialized': self._is_initialized
        }
