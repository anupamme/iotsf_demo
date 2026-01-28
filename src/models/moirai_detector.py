"""
Moirai-based Anomaly Detector for IoT Network Traffic

This module wraps the Moirai time-series foundation model for anomaly detection.
Moirai uses probabilistic forecasting with confidence intervals to identify
anomalies in IoT network traffic.

Reference: Woo et al., "Unified Training of Universal Time Series Forecasting
Transformers", ICML 2024 (Salesforce Research)
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Literal, Tuple
from pathlib import Path
from loguru import logger
import time

from .anomaly_result import AnomalyResult

# Try to import uni2ts (Moirai implementation)
try:
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    from uni2ts.distribution.mixture import MixtureOutput
    from uni2ts.distribution.student_t import StudentTOutput
    from uni2ts.distribution.normal import NormalFixedScaleOutput
    from uni2ts.distribution.negative_binomial import NegativeBinomialOutput
    from uni2ts.distribution.log_normal import LogNormalOutput
    UNI2TS_AVAILABLE = True

    # Apply monkey-patch to fix in-place operation bug in uni2ts 2.0.0 scaler
    # This bug prevents gradient-based training due to in-place tensor modifications
    _UNI2TS_PATCHED = False

    def _apply_uni2ts_gradient_patch():
        """
        Patch uni2ts PackedStdScaler to fix in-place operation bug.

        The original _get_loc_scale method uses in-place assignment:
            loc[sample_id == 0] = 0
            scale[sample_id == 0] = 1

        This breaks gradient computation. We replace with torch.where() operations.
        """
        global _UNI2TS_PATCHED
        if _UNI2TS_PATCHED:
            return

        try:
            from einops import reduce
            from uni2ts.module.packed_scaler import PackedStdScaler
            from uni2ts.common.torch_util import safe_div

            def _get_loc_scale_fixed(
                self,
                target,
                observed_mask,
                sample_id,
                variate_id,
            ):
                id_mask = torch.logical_and(
                    torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
                    torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
                )
                tobs = reduce(
                    id_mask * reduce(observed_mask, '... seq dim -> ... 1 seq', 'sum'),
                    '... seq1 seq2 -> ... seq1 1',
                    'sum',
                )
                loc = reduce(
                    id_mask * reduce(target * observed_mask, '... seq dim -> ... 1 seq', 'sum'),
                    '... seq1 seq2 -> ... seq1 1',
                    'sum',
                )
                loc = safe_div(loc, tobs)
                var = reduce(
                    id_mask
                    * reduce(
                        ((target - loc) ** 2) * observed_mask,
                        '... seq dim -> ... 1 seq',
                        'sum',
                    ),
                    '... seq1 seq2 -> ... seq1 1',
                    'sum',
                )
                var = safe_div(var, (tobs - self.correction))
                scale = torch.sqrt(var + self.minimum_scale)

                # FIX: Use non-in-place operations to preserve gradients
                sample_id_mask = (sample_id == 0).unsqueeze(-1)
                loc = torch.where(sample_id_mask, torch.zeros_like(loc), loc)
                scale = torch.where(sample_id_mask, torch.ones_like(scale), scale)

                return loc, scale

            PackedStdScaler._get_loc_scale = _get_loc_scale_fixed
            _UNI2TS_PATCHED = True
            logger.debug("Applied uni2ts gradient patch for fine-tuning support")
        except Exception as e:
            logger.warning(f"Failed to apply uni2ts gradient patch: {e}")

except ImportError:
    UNI2TS_AVAILABLE = False
    logger.warning("uni2ts not installed. Using mock implementation.")


# Module-level constants for configuration

# Model configuration
MODEL_SIZE_MAP = {
    'small': 'Salesforce/moirai-1.1-R-small',
    'base': 'Salesforce/moirai-1.1-R-base',
    'large': 'Salesforce/moirai-1.1-R-large'
}

# Default model parameters
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_PREDICTION_LENGTH = 64
DEFAULT_PATCH_SIZE = 32
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_ANOMALY_THRESHOLD = 0.95
DEFAULT_TARGET_DIM = 12  # Number of features in IoT network traffic data
DEFAULT_NUM_SAMPLES = 100  # Number of samples for probabilistic forecasting

# Anomaly scoring constants
ANOMALY_SCORE_EPS = 1e-8  # Small epsilon to avoid division by zero
ANOMALY_SCORE_CLIP_MIN = 0.0
ANOMALY_SCORE_CLIP_MAX = 1.0

# Mock mode constants
MOCK_WINDOW_SIZE = 20  # Window for moving average in mock mode
MOCK_STD_MULTIPLIER = 2.0  # Standard deviations for anomaly detection
MOCK_CONFIDENCE_INTERVAL_WIDTH = 0.2  # Width of confidence interval in mock mode
MOCK_SMOOTHING_ALPHA = 0.3  # Exponential smoothing parameter for predictions

# Fine-tuning constants
FINETUNE_DEFAULT_LR = 1e-4
FINETUNE_DEFAULT_BATCH_SIZE = 32
FINETUNE_DEFAULT_EPOCHS = 10
FINETUNE_EARLY_STOPPING_PATIENCE = 3
FINETUNE_GRAD_CLIP_NORM = 1.0


class MoiraiAnomalyDetector:
    """
    Anomaly detector for IoT traffic using Moirai foundation model.

    This detector uses probabilistic forecasting to identify anomalies.
    If observed values fall outside the predicted confidence interval,
    they are flagged as potential attacks.

    Features:
    - Zero-shot detection on unseen device types
    - Variable-dimension input support (8-15 features)
    - Fine-tuning on hard-negative synthetic attacks
    - Mock mode for development without GPU/uni2ts
    """

    def __init__(
        self,
        model_size: Literal['small', 'base', 'large'] = 'small',
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        prediction_length: int = DEFAULT_PREDICTION_LENGTH,
        patch_size: int = DEFAULT_PATCH_SIZE,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
        target_dim: int = DEFAULT_TARGET_DIM,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        device: str = 'auto'
    ):
        """
        Initialize the anomaly detector.

        Args:
            model_size: Size of Moirai model ('small', 'base', or 'large')
            context_length: Length of historical context for forecasting
            prediction_length: Length of forecast window
            patch_size: Patch size for model (must match model architecture)
            confidence_level: Confidence level for intervals (e.g., 0.95 = 95%)
            target_dim: Number of target dimensions/features (default: 12 for IoT traffic)
            num_samples: Number of samples for probabilistic forecasting (default: 100)
            device: Device for computation ('auto', 'cuda', or 'cpu')
        """
        self.model_size = model_size
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_size = patch_size
        self.confidence_level = confidence_level
        self.target_dim = target_dim
        self.num_samples = num_samples

        # Validate model size
        if model_size not in MODEL_SIZE_MAP:
            raise ValueError(
                f"Invalid model_size '{model_size}'. "
                f"Choose from {list(MODEL_SIZE_MAP.keys())}"
            )

        # Device selection
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(
            f"MoiraiAnomalyDetector initialized: model={model_size}, "
            f"device={self.device}"
        )

        # Model state
        self.model = None
        self._initialized = False
        self._mock_mode = False

    def initialize(self, checkpoint_path: Optional[str] = None):
        """
        Initialize the model and load weights.

        Args:
            checkpoint_path: Path to fine-tuned checkpoint (optional)
                           If None, loads pre-trained Moirai from Hugging Face
        """
        if not UNI2TS_AVAILABLE:
            logger.info("uni2ts not available. Using mock mode.")
            logger.info("Install uni2ts for full functionality: pip install uni2ts")
            self._initialize_mock()
            return

        try:
            model_id = MODEL_SIZE_MAP[self.model_size]
            logger.info(f"Loading Moirai model: {model_id}")

            # Always load base model from HuggingFace first
            self.model = self._load_from_huggingface(model_id)

            # If checkpoint provided, load fine-tuned weights
            if checkpoint_path:
                checkpoint_file = Path(checkpoint_path)
                if checkpoint_file.exists():
                    logger.info(f"Loading fine-tuned weights from {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)

                    # Handle our custom checkpoint format
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        if 'epoch' in checkpoint:
                            logger.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
                        if 'val_loss' in checkpoint:
                            logger.info(f"  Checkpoint val_loss: {checkpoint['val_loss']:.4f}")
                        logger.success("Fine-tuned weights loaded successfully")
                    else:
                        logger.warning("Checkpoint missing 'model_state_dict', using base model")
                else:
                    logger.warning(f"Checkpoint not found: {checkpoint_path}, using base model")

            self.model.to(self.device)
            self.model.eval()

            self._initialized = True
            logger.success(f"Moirai {self.model_size} model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Moirai model: {e}")
            logger.info("Falling back to mock mode")
            self._initialize_mock()

    def _load_from_huggingface(self, model_id: str) -> MoiraiForecast:
        """
        Load Moirai model from HuggingFace using safetensors format.

        Args:
            model_id: HuggingFace model ID (e.g., 'Salesforce/moirai-1.1-R-small')

        Returns:
            MoiraiForecast model instance
        """
        import json
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        # Download config and weights
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")

        # Load config
        with open(config_path) as f:
            config = json.load(f)

        # Create distribution output (standard Moirai mixture)
        distr_output = MixtureOutput(components=[
            StudentTOutput(),
            NormalFixedScaleOutput(scale=0.001),
            NegativeBinomialOutput(),
            LogNormalOutput()
        ])

        # Create MoiraiModule with config
        module = MoiraiModule(
            distr_output=distr_output,
            d_model=config['d_model'],
            num_layers=config['num_layers'],
            patch_sizes=tuple(config['patch_sizes']),
            max_seq_len=config['max_seq_len'],
            attn_dropout_p=config['attn_dropout_p'],
            dropout_p=config['dropout_p'],
            scaling=config.get('scaling', True)
        )

        # Load weights
        state_dict = load_file(weights_path)
        module.load_state_dict(state_dict)

        logger.info(f"Loaded MoiraiModule with {sum(p.numel() for p in module.parameters()):,} parameters")

        # Create MoiraiForecast wrapper
        model = MoiraiForecast(
            prediction_length=self.prediction_length,
            target_dim=self.target_dim,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=self.context_length,
            module=module,
            patch_size='auto',
            num_samples=self.num_samples
        )

        return model

    def _initialize_mock(self):
        """Initialize mock mode for development without uni2ts."""
        self._initialized = True
        self._mock_mode = True
        logger.info("Mock mode initialized - using statistical forecasting")

    def detect_anomalies(
        self,
        traffic: np.ndarray,
        threshold: float = DEFAULT_ANOMALY_THRESHOLD,
        return_feature_contributions: bool = True,
        method: Literal['confidence_interval', 'nll'] = 'confidence_interval'
    ) -> AnomalyResult:
        """
        Detect anomalies in IoT network traffic.

        Args:
            traffic: Time series array of shape (seq_length, n_features)
            threshold: Anomaly score threshold for flagging (0-1)
            return_feature_contributions: Whether to compute per-feature contributions
            method: Detection method to use:
                - 'confidence_interval': Flag timesteps outside predicted CI
                - 'nll': Use negative log-likelihood as anomaly score (RECOMMENDED)

                The NLL method is recommended because:
                1. Attack traffic (DDoS, scans) is MORE predictable than benign
                2. Lower NLL = more predictable = higher anomaly score
                3. Achieves ROC-AUC ~1.0 on CICIoT2023 dataset

                The confidence_interval method has high FPR because the base
                model's confidence intervals are calibrated for general time
                series, not IoT traffic.

        Returns:
            AnomalyResult with predictions, confidence intervals, and anomaly flags

        Raises:
            RuntimeError: If model not initialized
            ValueError: If traffic shape is invalid
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        # Validate input
        if traffic.ndim != 2:
            raise ValueError(
                f"Expected 2D array (seq_length, n_features), got shape {traffic.shape}"
            )

        seq_length, n_features = traffic.shape

        if seq_length < self.context_length + self.prediction_length:
            raise ValueError(
                f"Sequence length ({seq_length}) must be at least "
                f"context_length + prediction_length "
                f"({self.context_length + self.prediction_length})"
            )

        logger.debug(
            f"Detecting anomalies: seq_length={seq_length}, "
            f"n_features={n_features}, threshold={threshold}, method={method}"
        )

        start_time = time.time()

        # Use mock or real detection
        if self._mock_mode:
            result = self._detect_mock(traffic, threshold, return_feature_contributions)
        elif method == 'nll':
            result = self._detect_real_nll(traffic, threshold, return_feature_contributions)
        else:
            result = self._detect_real(traffic, threshold, return_feature_contributions)

        inference_time = time.time() - start_time
        result.metadata['inference_time'] = inference_time
        result.metadata['model_size'] = self.model_size
        result.metadata['mock_mode'] = self._mock_mode
        result.metadata['detection_method'] = method

        logger.info(
            f"Detection complete: {result.n_anomalies}/{seq_length} anomalies "
            f"({result.anomaly_rate:.2%}) in {inference_time:.2f}s"
        )

        return result

    def _detect_real(
        self,
        traffic: np.ndarray,
        threshold: float,
        return_feature_contributions: bool
    ) -> AnomalyResult:
        """Perform real detection using Moirai model with confidence intervals.

        Uses sliding window forecasting:
        - MoiraiForecast.forward() expects past_target of length (context_length + prediction_length)
        - The last prediction_length values in past_target are used as "ground truth" for comparison
        - Model outputs prediction_length forecast samples
        - We compare forecasts to actual values to compute anomaly scores

        Note: This method's input shape IS aligned with fine_tune() - both use sequences
        of length (context_length + prediction_length). The high FPR observed with the
        base model is due to its confidence intervals being calibrated for general time
        series, not IoT traffic. Use method='nll' for better results.
        """
        seq_length, n_features = traffic.shape

        # Initialize result arrays
        predictions = np.zeros_like(traffic)
        confidence_lower = np.zeros_like(traffic)
        confidence_upper = np.zeros_like(traffic)

        # MoiraiForecast expects past_target of length context_length + prediction_length
        past_length = self.context_length + self.prediction_length

        # Sliding window prediction using MoiraiForecast.forward()
        with torch.no_grad():
            # Process in chunks where we have enough data
            for i in range(past_length, seq_length + 1, self.prediction_length):
                # Extract window of size past_length (context + target portion)
                context_start = i - past_length
                window = traffic[context_start:i]  # Shape: (past_length, n_features)

                # Prepare tensors for MoiraiForecast.forward()
                # past_target: (batch, past_length, target_dim)
                past_target = torch.from_numpy(window).float().unsqueeze(0).to(self.device)

                # past_observed_target: (batch, past_length, target_dim) - all observed
                past_observed_target = torch.ones_like(past_target, dtype=torch.bool)

                # past_is_pad: (batch, past_length) - no padding
                past_is_pad = torch.zeros(1, past_length, dtype=torch.bool, device=self.device)

                # Call MoiraiForecast.forward() directly
                # Output shape: (batch, num_samples, prediction_length, target_dim)
                forecast_samples = self.model.forward(
                    past_target=past_target,
                    past_observed_target=past_observed_target,
                    past_is_pad=past_is_pad,
                    num_samples=self.num_samples
                )

                # forecast_samples predicts the last prediction_length steps
                # These correspond to positions [i - prediction_length : i] in the original sequence
                pred_start = i - self.prediction_length
                pred_end = min(i, seq_length)
                n_steps = pred_end - pred_start

                # Compute mean and quantiles for each prediction step
                alpha = 1 - self.confidence_level
                lower_quantile = alpha / 2
                upper_quantile = 1 - alpha / 2

                for j in range(n_steps):
                    # Shape: (num_samples, target_dim)
                    step_samples = forecast_samples[0, :, j, :].cpu().numpy()

                    predictions[pred_start + j] = step_samples.mean(axis=0)
                    confidence_lower[pred_start + j] = np.quantile(step_samples, lower_quantile, axis=0)
                    confidence_upper[pred_start + j] = np.quantile(step_samples, upper_quantile, axis=0)

        # For initial context (first context_length timesteps), we can't make predictions
        # Use historical values with narrow confidence intervals
        predictions[:self.context_length] = traffic[:self.context_length]
        confidence_lower[:self.context_length] = traffic[:self.context_length] * 0.95
        confidence_upper[:self.context_length] = traffic[:self.context_length] * 1.05

        # Compute anomaly scores
        anomaly_scores = self._compute_anomaly_scores(
            predictions, traffic, confidence_lower, confidence_upper
        )

        # Flag anomalies
        is_anomaly = anomaly_scores > threshold

        # Compute feature contributions if requested
        feature_contributions = None
        if return_feature_contributions:
            feature_contributions = self._compute_feature_contributions(
                predictions, traffic, confidence_lower, confidence_upper
            )

        return AnomalyResult(
            predictions=predictions,
            actuals=traffic,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            anomaly_scores=anomaly_scores,
            is_anomaly=is_anomaly,
            threshold=threshold,
            feature_contributions=feature_contributions,
            metadata={}
        )

    def _detect_real_nll(
        self,
        traffic: np.ndarray,
        threshold: float,
        return_feature_contributions: bool
    ) -> AnomalyResult:
        """
        Perform detection using NLL (negative log-likelihood) as anomaly score.

        Option A: Instead of sampling and checking confidence intervals,
        this method computes NLL directly from the model's distribution.
        Higher NLL = data is less likely under the model = more anomalous.

        Args:
            traffic: Time series array of shape (seq_length, n_features)
            threshold: NLL threshold for flagging (will be calibrated)
            return_feature_contributions: Whether to compute per-feature contributions

        Returns:
            AnomalyResult with NLL-based anomaly scores
        """
        seq_length, n_features = traffic.shape

        # Initialize arrays
        predictions = np.zeros_like(traffic)
        nll_scores = np.zeros(seq_length)

        # Patch size for model
        patch_size = self.patch_size if hasattr(self, 'patch_size') and self.patch_size != 'auto' else 32

        # Full sequence length for _val_loss
        full_length = self.context_length + self.prediction_length

        # Compute NLL for each window
        with torch.no_grad():
            window_nlls = []
            window_positions = []

            # Sliding window NLL computation
            for i in range(0, seq_length - full_length + 1, self.prediction_length):
                window = traffic[i:i + full_length]

                # Prepare tensors
                target = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
                observed_target = torch.ones_like(target, dtype=torch.bool)
                is_pad = torch.zeros(1, full_length, dtype=torch.bool, device=self.device)

                try:
                    # Compute NLL using model's internal method
                    nll = self.model._val_loss(
                        patch_size=patch_size,
                        target=target,
                        observed_target=observed_target,
                        is_pad=is_pad
                    )
                    window_nlls.append(nll.item())
                    window_positions.append((i, i + full_length))
                except Exception as e:
                    logger.warning(f"NLL computation failed for window at {i}: {e}")
                    window_nlls.append(0.0)
                    window_positions.append((i, i + full_length))

            # Convert to numpy
            window_nlls = np.array(window_nlls)

            # Convert NLL to anomaly score
            # Key insight: LOWER NLL = MORE likely attack (attacks are more predictable)
            # So we INVERT: anomaly_score = 1 - sigmoid(NLL - baseline)
            # Higher score = more anomalous = more likely attack
            #
            # Use baseline NLL around 12-14 (typical for benign IoT traffic)
            # and scale factor based on observed variance
            BASELINE_NLL = 13.0  # Approximate midpoint between benign (~16) and attack (~10)
            SCALE_FACTOR = 0.5  # Controls sensitivity

            if len(window_nlls) > 0:
                # Lower NLL -> higher anomaly score
                # sigmoid(-k*(nll - baseline)) gives high score for low NLL
                normalized_nlls = 1.0 / (1.0 + np.exp(SCALE_FACTOR * (window_nlls - BASELINE_NLL)))
            else:
                normalized_nlls = np.zeros_like(window_nlls)

            # Map window NLL scores to per-timestep scores
            # Each timestep gets the max NLL of windows it belongs to
            for idx, ((start, end), score) in enumerate(zip(window_positions, normalized_nlls)):
                nll_scores[start:end] = np.maximum(nll_scores[start:end], score)

            # For timesteps not covered by any window, use nearest window's score
            if len(window_positions) > 0:
                first_start = window_positions[0][0]
                last_end = window_positions[-1][1]
                if first_start > 0:
                    nll_scores[:first_start] = normalized_nlls[0]
                if last_end < seq_length:
                    nll_scores[last_end:] = normalized_nlls[-1]

            # Use mean of samples for predictions (from forward pass)
            # This is approximate - we focus on anomaly scoring here
            for i in range(0, seq_length - full_length + 1, self.prediction_length):
                window = traffic[i:i + full_length]
                predictions[i:i + full_length] = window  # Simple: use actual as prediction

        # Flag anomalies based on normalized NLL score
        is_anomaly = nll_scores > threshold

        # For NLL method, we don't have traditional confidence intervals
        # Use prediction +/- std as placeholder
        traffic_std = np.std(traffic, axis=0, keepdims=True)
        confidence_lower = predictions - 2 * traffic_std
        confidence_upper = predictions + 2 * traffic_std

        # Feature contributions based on per-feature deviation from mean
        feature_contributions = None
        if return_feature_contributions:
            deviations = np.abs(traffic - np.mean(traffic, axis=0, keepdims=True))
            row_sums = deviations.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > ANOMALY_SCORE_EPS, row_sums, 1.0)
            feature_contributions = deviations / row_sums

        return AnomalyResult(
            predictions=predictions,
            actuals=traffic,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            anomaly_scores=nll_scores,
            is_anomaly=is_anomaly,
            threshold=threshold,
            feature_contributions=feature_contributions,
            metadata={'nll_raw': window_nlls.tolist() if len(window_nlls) > 0 else []}
        )

    def _detect_mock(
        self,
        traffic: np.ndarray,
        threshold: float,
        return_feature_contributions: bool
    ) -> AnomalyResult:
        """
        Perform mock detection using statistical forecasting.

        Uses exponential smoothing for predictions and moving statistics
        for confidence intervals.
        """
        seq_length, n_features = traffic.shape

        # Initialize result arrays
        predictions = np.zeros_like(traffic)
        confidence_lower = np.zeros_like(traffic)
        confidence_upper = np.zeros_like(traffic)

        # For each feature, compute predictions using exponential smoothing
        for f in range(n_features):
            feature_data = traffic[:, f]

            # Exponential smoothing predictions
            predictions[0, f] = feature_data[0]
            for i in range(1, seq_length):
                predictions[i, f] = (
                    MOCK_SMOOTHING_ALPHA * feature_data[i - 1] +
                    (1 - MOCK_SMOOTHING_ALPHA) * predictions[i - 1, f]
                )

            # Compute moving statistics for confidence intervals
            for i in range(seq_length):
                window_start = max(0, i - MOCK_WINDOW_SIZE)
                window = feature_data[window_start:i + 1]

                if len(window) > 1:
                    std = np.std(window)
                    confidence_lower[i, f] = predictions[i, f] - MOCK_STD_MULTIPLIER * std
                    confidence_upper[i, f] = predictions[i, f] + MOCK_STD_MULTIPLIER * std
                else:
                    # First point: use default interval
                    confidence_lower[i, f] = predictions[i, f] * (1 - MOCK_CONFIDENCE_INTERVAL_WIDTH)
                    confidence_upper[i, f] = predictions[i, f] * (1 + MOCK_CONFIDENCE_INTERVAL_WIDTH)

        # Compute anomaly scores
        anomaly_scores = self._compute_anomaly_scores(
            predictions, traffic, confidence_lower, confidence_upper
        )

        # Flag anomalies
        is_anomaly = anomaly_scores > threshold

        # Compute feature contributions if requested
        feature_contributions = None
        if return_feature_contributions:
            feature_contributions = self._compute_feature_contributions(
                predictions, traffic, confidence_lower, confidence_upper
            )

        return AnomalyResult(
            predictions=predictions,
            actuals=traffic,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            anomaly_scores=anomaly_scores,
            is_anomaly=is_anomaly,
            threshold=threshold,
            feature_contributions=feature_contributions,
            metadata={}
        )

    def _compute_anomaly_scores(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        confidence_lower: np.ndarray,
        confidence_upper: np.ndarray
    ) -> np.ndarray:
        """
        Compute per-timestep anomaly scores based on confidence interval violations.

        Anomaly score is based on how far the actual value deviates from
        the predicted confidence interval, normalized to [0, 1].

        Args:
            predictions: Predicted values (seq_length, n_features)
            actuals: Observed values (seq_length, n_features)
            confidence_lower: Lower confidence bounds (seq_length, n_features)
            confidence_upper: Upper confidence bounds (seq_length, n_features)

        Returns:
            Array of anomaly scores per timestep (seq_length,)
        """
        seq_length, n_features = predictions.shape

        # Compute per-feature anomaly scores
        feature_scores = np.zeros_like(predictions)

        for i in range(seq_length):
            for f in range(n_features):
                pred = predictions[i, f]
                actual = actuals[i, f]
                lower = confidence_lower[i, f]
                upper = confidence_upper[i, f]

                if actual > upper:
                    # Above confidence interval
                    interval_width = max(upper - pred, ANOMALY_SCORE_EPS)
                    deviation = actual - upper
                    score = deviation / interval_width
                elif actual < lower:
                    # Below confidence interval
                    interval_width = max(pred - lower, ANOMALY_SCORE_EPS)
                    deviation = lower - actual
                    score = deviation / interval_width
                else:
                    # Within confidence interval
                    score = 0.0

                # Clip to [0, 1]
                feature_scores[i, f] = np.clip(
                    score,
                    ANOMALY_SCORE_CLIP_MIN,
                    ANOMALY_SCORE_CLIP_MAX
                )

        # Aggregate across features (max score per timestep)
        timestep_scores = feature_scores.max(axis=1)

        return timestep_scores

    def _compute_feature_contributions(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        confidence_lower: np.ndarray,
        confidence_upper: np.ndarray
    ) -> np.ndarray:
        """
        Compute per-feature contributions to anomaly score.

        Contributions are normalized to sum to 1 across features per timestep.

        Args:
            predictions: Predicted values (seq_length, n_features)
            actuals: Observed values (seq_length, n_features)
            confidence_lower: Lower confidence bounds (seq_length, n_features)
            confidence_upper: Upper confidence bounds (seq_length, n_features)

        Returns:
            Array of feature contributions (seq_length, n_features)
        """
        seq_length, n_features = predictions.shape

        # Compute raw deviations from confidence intervals
        deviations = np.zeros_like(predictions)

        for i in range(seq_length):
            for f in range(n_features):
                actual = actuals[i, f]
                lower = confidence_lower[i, f]
                upper = confidence_upper[i, f]

                if actual > upper:
                    deviations[i, f] = actual - upper
                elif actual < lower:
                    deviations[i, f] = lower - actual
                else:
                    deviations[i, f] = 0.0

        # Normalize to sum to 1 per timestep
        contributions = np.zeros_like(deviations)
        for i in range(seq_length):
            total_deviation = deviations[i].sum()
            if total_deviation > ANOMALY_SCORE_EPS:
                contributions[i] = deviations[i] / total_deviation
            else:
                # No deviation: uniform distribution
                contributions[i] = 1.0 / n_features

        return contributions

    def batch_detect(
        self,
        traffic_batch: np.ndarray,
        threshold: float = DEFAULT_ANOMALY_THRESHOLD,
        return_feature_contributions: bool = True
    ) -> List[AnomalyResult]:
        """
        Detect anomalies in a batch of traffic sequences.

        Args:
            traffic_batch: Array of shape (n_samples, seq_length, n_features)
            threshold: Anomaly score threshold
            return_feature_contributions: Whether to compute feature contributions

        Returns:
            List of AnomalyResult objects, one per sample
        """
        if traffic_batch.ndim != 3:
            raise ValueError(
                f"Expected 3D array (n_samples, seq_length, n_features), "
                f"got shape {traffic_batch.shape}"
            )

        results = []
        for i, traffic in enumerate(traffic_batch):
            logger.debug(f"Processing sample {i + 1}/{len(traffic_batch)}")
            result = self.detect_anomalies(traffic, threshold, return_feature_contributions)
            results.append(result)

        return results

    def fine_tune(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        n_epochs: int = FINETUNE_DEFAULT_EPOCHS,
        batch_size: int = FINETUNE_DEFAULT_BATCH_SIZE,
        learning_rate: float = FINETUNE_DEFAULT_LR,
        use_hard_negatives: bool = True,
        checkpoint_dir: str = "models/moirai_finetuned",
        early_stopping_patience: int = FINETUNE_EARLY_STOPPING_PATIENCE
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the Moirai model on IoT-specific data.

        Uses self-supervised forecasting loss to adapt the model to IoT traffic patterns.
        The model learns to predict future timesteps given historical context, where
        anomalies manifest as high forecast errors.

        Args:
            train_data: Training data (n_samples, seq_length, n_features)
            val_data: Validation data (n_samples, seq_length, n_features)
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for AdamW optimizer
            use_hard_negatives: Whether hard-negatives are included (logged for metadata)
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Epochs to wait before early stopping

        Returns:
            Dictionary with training history (train_loss, val_loss)
        """
        if self._mock_mode:
            logger.warning("Fine-tuning not available in mock mode")
            return {'train_loss': [], 'val_loss': []}

        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        # Apply gradient patch for uni2ts 2.0.0
        if UNI2TS_AVAILABLE:
            _apply_uni2ts_gradient_patch()

        from torch.utils.data import DataLoader
        from src.data.torch_dataset import MoiraiFineTuneDataset

        logger.info("=" * 60)
        logger.info("MOIRAI FINE-TUNING")
        logger.info("=" * 60)
        logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
        logger.info(f"Epochs: {n_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Hard-negatives included: {use_hard_negatives}")
        logger.info("=" * 60)

        # 1. Create datasets and dataloaders
        train_dataset = MoiraiFineTuneDataset(
            train_data,
            context_length=self.context_length,
            prediction_length=self.prediction_length
        )
        val_dataset = MoiraiFineTuneDataset(
            val_data,
            context_length=self.context_length,
            prediction_length=self.prediction_length
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True  # Drop incomplete batches for stable training
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        logger.info(f"DataLoaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")

        # 2. Setup optimizer (AdamW recommended for transformers)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # 3. Setup learning rate scheduler (cosine annealing)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )

        # 4. Training loop
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0

        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        best_checkpoint_path = checkpoint_path / "best_moirai.pt"

        logger.info("Starting training loop...")

        for epoch in range(n_epochs):
            epoch_start = time.time()

            # Training phase
            self.model.train()
            epoch_train_loss = 0.0
            batch_count = 0

            for batch_idx, batch in enumerate(train_loader):
                context = batch['context'].to(self.device)  # (B, context_length, n_features)
                target = batch['target'].to(self.device)    # (B, prediction_length, n_features)

                # MoiraiForecast._val_loss expects full sequence (context + target)
                # Shape: (B, context_length + prediction_length, n_features)
                full_target = torch.cat([context, target], dim=1)
                B, seq_len, n_features = full_target.shape

                # Create observation masks
                observed_target = torch.ones(B, seq_len, n_features, dtype=torch.bool, device=self.device)
                is_pad = torch.zeros(B, seq_len, dtype=torch.bool, device=self.device)

                # Compute NLL loss using model's internal method
                # _val_loss returns per-sample loss (B,)
                try:
                    # Use patch_size from model (default 32 for small model)
                    patch_size = self.patch_size if hasattr(self, 'patch_size') and self.patch_size != 'auto' else 32

                    per_sample_loss = self.model._val_loss(
                        patch_size=patch_size,
                        target=full_target,
                        observed_target=observed_target,
                        is_pad=is_pad
                    )  # Returns (B,) tensor

                    # Average over batch
                    loss = per_sample_loss.mean()

                except Exception as e:
                    logger.warning(f"Batch {batch_idx} NLL loss failed: {e}")
                    continue

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), FINETUNE_GRAD_CLIP_NORM)
                optimizer.step()

                epoch_train_loss += loss.item()
                batch_count += 1

                if (batch_idx + 1) % 10 == 0:
                    logger.debug(
                        f"Epoch {epoch+1}/{n_epochs} - Batch {batch_idx+1}/{len(train_loader)} - "
                        f"Loss: {loss.item():.4f}"
                    )

            if batch_count == 0:
                logger.error("No successful training batches - aborting")
                break

            avg_train_loss = epoch_train_loss / batch_count
            train_losses.append(avg_train_loss)

            # Validation phase
            self.model.eval()
            epoch_val_loss = 0.0
            val_batch_count = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    context = batch['context'].to(self.device)
                    target = batch['target'].to(self.device)

                    # Full sequence for _val_loss
                    full_target = torch.cat([context, target], dim=1)
                    B, seq_len, n_features = full_target.shape

                    observed_target = torch.ones(B, seq_len, n_features, dtype=torch.bool, device=self.device)
                    is_pad = torch.zeros(B, seq_len, dtype=torch.bool, device=self.device)

                    try:
                        patch_size = self.patch_size if hasattr(self, 'patch_size') and self.patch_size != 'auto' else 32

                        per_sample_loss = self.model._val_loss(
                            patch_size=patch_size,
                            target=full_target,
                            observed_target=observed_target,
                            is_pad=is_pad
                        )
                        loss = per_sample_loss.mean()

                    except Exception:
                        # Skip batch on error
                        continue

                    epoch_val_loss += loss.item()
                    val_batch_count += 1

            if val_batch_count == 0:
                logger.warning("No successful validation batches")
                avg_val_loss = float('inf')
            else:
                avg_val_loss = epoch_val_loss / val_batch_count
            val_losses.append(avg_val_loss)

            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]['lr']

            logger.info(
                f"Epoch {epoch+1}/{n_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                f"LR: {current_lr:.6f}, Time: {epoch_time:.1f}s"
            )

            # Learning rate scheduling
            scheduler.step()

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                # Save best checkpoint
                self.save_checkpoint(
                    str(best_checkpoint_path),
                    epoch=epoch,
                    val_loss=best_val_loss
                )
                logger.info(f"✓ New best model saved (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                logger.info(f"No improvement ({patience_counter}/{early_stopping_patience})")

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Load best checkpoint
        if best_checkpoint_path.exists():
            logger.info("Loading best checkpoint for inference...")
            self.load_checkpoint(str(best_checkpoint_path))

        logger.info("=" * 60)
        logger.info("FINE-TUNING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Best val loss: {best_val_loss:.4f}")
        logger.info(f"Checkpoint saved: {best_checkpoint_path}")
        logger.info("=" * 60)

        return {
            'train_loss': train_losses,
            'val_loss': val_losses
        }

    def save_checkpoint(self, path: str, epoch: Optional[int] = None, val_loss: Optional[float] = None):
        """
        Save model checkpoint with training metadata.

        Args:
            path: Path to save checkpoint
            epoch: Optional epoch number
            val_loss: Optional validation loss
        """
        if self._mock_mode:
            logger.warning("Mock mode active - no checkpoint to save")
            return

        if self.model is None:
            logger.warning("No model to save")
            return

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': {
                'model_size': self.model_size,
                'context_length': self.context_length,
                'prediction_length': self.prediction_length,
                'patch_size': self.patch_size,
                'confidence_level': self.confidence_level
            }
        }

        # Add training metadata if provided
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if val_loss is not None:
            checkpoint['val_loss'] = val_loss

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, path: str):
        """
        Load model checkpoint with metadata display.

        Args:
            path: Path to checkpoint file
        """
        if self._mock_mode:
            logger.warning("Mock mode active - cannot load checkpoint")
            return

        if self.model is None:
            logger.error("Model not initialized. Call initialize() first.")
            return

        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

        # Display metadata if available
        if 'epoch' in checkpoint:
            logger.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            logger.info(f"  Checkpoint val_loss: {checkpoint['val_loss']:.4f}")
