"""
Diffusion-TS Wrapper for IoT Attack Generation

This module wraps the Diffusion-TS model for generating synthetic
IoT network traffic. The key innovation is the decomposition-aware
architecture that separately models trend and seasonality.

Reference: Zhang et al., "Diffusion-TS: Interpretable Diffusion for
General Time Series Generation", ICLR 2024
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple
from pathlib import Path
from loguru import logger

# Import from Diffusion-TS repo (needs to be installed)
# git clone https://github.com/Y-debug-sys/Diffusion-TS
# pip install -e Diffusion-TS/

try:
    from diffusion_ts.model import DiffusionTS
    from diffusion_ts.diffusion import GaussianDiffusion
    DIFFUSION_TS_AVAILABLE = True
except ImportError:
    DIFFUSION_TS_AVAILABLE = False
    logger.warning("Diffusion-TS not installed. Using mock implementation.")


# Feature indices for IoT network traffic (12 features total)
# These correspond to typical network flow features:
# 0-4: Basic flow statistics (duration, packet counts, etc.)
# 5-7: Packet rate features (packets/sec, burst indicators, etc.)
# 8-9: Byte transfer features (outbound bytes, inbound bytes)
# 10-11: Timing features (inter-arrival times, jitter)
FEATURE_PACKET_RATE_START = 5
FEATURE_PACKET_RATE_END = 8
FEATURE_OUTBOUND_BYTES = 8
FEATURE_INBOUND_BYTES = 9
FEATURE_TIMING_IAT = 10
FEATURE_TIMING_JITTER = 11

# Attack pattern configuration
BEACON_INTERVAL = 16  # Time steps between C2 beacon signals
LOTL_BURST_POSITIONS = [32, 64, 96]  # Positions for living-off-the-land micro-bursts

# Attack pattern intensity parameters (as fractions/multipliers)
SLOW_EXFIL_TREND_MAGNITUDE = 0.03  # 3% gradual increase over sequence
SLOW_EXFIL_INBOUND_SCALING = 0.5  # Inbound bytes increase at 50% of outbound rate
LOTL_BURST_MAGNITUDE = 1.15  # 15% increase in packet rate during bursts
PROTOCOL_ANOMALY_NOISE_FACTOR = 0.05  # 5% random variation in timing features
BEACON_DIP_MAGNITUDE = 0.98  # 2% decrease at beacon intervals (1.0 - 0.02)

# Mock generation parameters
MOCK_TIME_RANGE = 4 * np.pi  # Time range for sinusoidal seasonality patterns
MOCK_TREND_MAGNITUDE = 0.1  # Maximum trend increase over sequence
MOCK_SEASONALITY_AMPLITUDE = 0.3  # Amplitude of sinusoidal seasonality
MOCK_NOISE_STD = 0.1  # Standard deviation of Gaussian noise
MOCK_PHASE_RANGE = 2 * np.pi  # Phase variation range for different features

# Hard-negative generation parameters
HARD_NEG_STD_DEVIATION_SCALING = 0.1  # Scaling factor for statistical deviation
HARD_NEG_GUIDANCE_SCALE_MULTIPLIER = 5  # Multiplier for stealth level to guidance scale

# Decomposition configuration
MIN_SEQUENCE_LENGTH_FOR_DECOMPOSITION = 5  # Minimum length for meaningful decomposition
SAVGOL_DEFAULT_WINDOW = 21  # Default window length for Savitzky-Golay filter
SEASONALITY_TOP_FREQUENCIES = 5  # Number of top FFT frequencies to keep


class IoTDiffusionGenerator:
    """
    Generator for synthetic IoT traffic using Diffusion-TS.

    This class wraps the Diffusion-TS model and provides:
    - Easy-to-use generation interface
    - Statistical property targeting (for hard-negative generation)
    - Trend/seasonality decomposition visualization
    - Checkpoint management
    """

    def __init__(
        self,
        seq_length: int = 128,
        feature_dim: int = 12,
        n_diffusion_steps: int = 1000,
        device: str = 'auto'
    ):
        """
        Initialize the generator.

        Args:
            seq_length: Length of generated sequences
            feature_dim: Number of features (12 for our IoT traffic)
            n_diffusion_steps: Number of diffusion steps (more = better quality)
            device: 'auto', 'cuda', or 'cpu'
        """
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.n_diffusion_steps = n_diffusion_steps

        # Device selection
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"IoTDiffusionGenerator initialized with device: {self.device}")

        # Initialize model
        self.model = None
        self.diffusion = None
        self._initialized = False
        self._mock_mode = False

    def initialize(self, checkpoint_path: Optional[str] = None):
        """
        Initialize the model architecture and optionally load weights.

        Args:
            checkpoint_path: Path to pre-trained weights (optional)
        """
        if not DIFFUSION_TS_AVAILABLE:
            logger.info("Using mock Diffusion-TS mode. Install real package for production.")
            logger.info("Installation: bash scripts/install_diffusion_ts.sh")
            self._initialize_mock()
            return

        # Model configuration
        model_config = {
            'seq_length': self.seq_length,
            'feature_dim': self.feature_dim,
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 4,
            'dropout': 0.1,
            # Decomposition parameters
            'trend_poly_degree': 3,
            'seasonality_n_harmonics': 5
        }

        # Initialize model
        self.model = DiffusionTS(**model_config).to(self.device)

        # Initialize diffusion process
        self.diffusion = GaussianDiffusion(
            model=self.model,
            seq_length=self.seq_length,
            n_steps=self.n_diffusion_steps,
            loss_type='l2'
        )

        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)

        self._initialized = True
        logger.info("Diffusion-TS model initialized")

    def _initialize_mock(self):
        """Initialize a mock model for development without Diffusion-TS."""
        self._initialized = True
        self._mock_mode = True
        logger.info("Mock mode initialized - using statistical generation")

    def generate(
        self,
        n_samples: int,
        target_statistics: Optional[Dict] = None,
        guidance_scale: float = 1.0,
        n_inference_steps: int = 50,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate synthetic traffic sequences.

        Args:
            n_samples: Number of sequences to generate
            target_statistics: Dict of target statistics for guidance
                              e.g., {'mean': 0.5, 'std': 0.1, 'trend_slope': 0.01}
            guidance_scale: Strength of guidance (higher = closer to target)
            n_inference_steps: Number of denoising steps (fewer = faster)
            seed: Random seed for reproducibility

        Returns:
            Array of shape (n_samples, seq_length, feature_dim)
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        if self._mock_mode:
            return self._generate_mock(n_samples, target_statistics)

        with torch.no_grad():
            # Start from random noise
            x_T = torch.randn(
                n_samples, self.seq_length, self.feature_dim,
                device=self.device
            )

            # Reverse diffusion with optional guidance
            x_0 = self.diffusion.sample(
                x_T,
                n_steps=n_inference_steps,
                guidance_fn=self._create_guidance_fn(target_statistics, guidance_scale)
                if target_statistics else None
            )

            return x_0.cpu().numpy()

    def _generate_mock(
        self,
        n_samples: int,
        target_statistics: Optional[Dict]
    ) -> np.ndarray:
        """
        Mock generation for development.

        Uses module-level constants for generation parameters:
        - MOCK_TIME_RANGE, MOCK_TREND_MAGNITUDE, MOCK_SEASONALITY_AMPLITUDE
        - MOCK_NOISE_STD, MOCK_PHASE_RANGE
        """
        # Generate realistic-looking mock data
        t = np.linspace(0, MOCK_TIME_RANGE, self.seq_length)

        samples = []
        for _ in range(n_samples):
            # Trend component (slight upward trend)
            trend = np.linspace(0, MOCK_TREND_MAGNITUDE, self.seq_length)[:, None]
            trend = np.tile(trend, (1, self.feature_dim))

            # Seasonality component (sinusoidal pattern)
            seasonality = np.sin(t)[:, None] * MOCK_SEASONALITY_AMPLITUDE
            seasonality = np.tile(seasonality, (1, self.feature_dim))

            # Add some phase variations per feature
            for i in range(self.feature_dim):
                phase = np.random.uniform(0, MOCK_PHASE_RANGE)
                seasonality[:, i] = np.sin(t + phase) * MOCK_SEASONALITY_AMPLITUDE

            # Noise component
            noise = np.random.randn(self.seq_length, self.feature_dim) * MOCK_NOISE_STD

            # Combine components
            sample = trend + seasonality + noise

            # Apply target statistics if provided
            if target_statistics:
                if 'mean' in target_statistics:
                    current_mean = sample.mean()
                    sample = sample - current_mean + target_statistics['mean']
                if 'std' in target_statistics:
                    current_std = sample.std()
                    if current_std > 1e-8:
                        sample = (sample - sample.mean()) / current_std * target_statistics['std'] + sample.mean()

            samples.append(sample)

        return np.array(samples)

    def _create_guidance_fn(
        self,
        target_statistics: Dict,
        guidance_scale: float
    ):
        """
        Create a guidance function for conditional generation.

        This enables generating hard-negatives by guiding the generation
        to match specific statistical properties of benign traffic.
        """
        def guidance_fn(x_t, t):
            # Compute gradient towards target statistics
            grad = torch.zeros_like(x_t)

            if 'mean' in target_statistics:
                target_mean = target_statistics['mean']
                current_mean = x_t.mean(dim=(1, 2), keepdim=True)
                grad += (target_mean - current_mean) * guidance_scale

            if 'std' in target_statistics:
                target_std = target_statistics['std']
                current_std = x_t.std(dim=(1, 2), keepdim=True)
                # Gradient to adjust variance
                grad += (x_t - x_t.mean(dim=(1,2), keepdim=True)) * \
                        (target_std / (current_std + 1e-8) - 1) * guidance_scale * HARD_NEG_STD_DEVIATION_SCALING

            return grad

        return guidance_fn

    def generate_hard_negative(
        self,
        benign_sample: np.ndarray,
        attack_pattern: str = 'slow_exfiltration',
        stealth_level: float = 0.95
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate a hard-negative attack that mimics benign traffic.

        Args:
            benign_sample: Reference benign traffic to mimic
            attack_pattern: Type of attack pattern to inject
            stealth_level: 0-1, higher means more similar to benign

        Returns:
            Tuple of (generated_attack, attack_metadata)

        Uses module-level constants:
        - HARD_NEG_STD_DEVIATION_SCALING: Controls statistical deviation
        - HARD_NEG_GUIDANCE_SCALE_MULTIPLIER: Converts stealth level to guidance scale
        """
        # Extract statistics from benign sample
        target_stats = {
            'mean': float(benign_sample.mean()),
            'std': float(benign_sample.std()) * (1 + (1 - stealth_level) * HARD_NEG_STD_DEVIATION_SCALING),
            # Match variance within threshold
        }

        # Generate with guidance
        generated = self.generate(
            n_samples=1,
            target_statistics=target_stats,
            guidance_scale=stealth_level * HARD_NEG_GUIDANCE_SCALE_MULTIPLIER
        )[0]

        # Inject attack pattern (subtle modifications)
        generated = self._inject_attack_pattern(generated, attack_pattern)

        metadata = {
            'attack_type': attack_pattern,
            'stealth_level': stealth_level,
            'target_stats': target_stats,
            'mean_diff': abs(generated.mean() - benign_sample.mean()),
            'std_diff': abs(generated.std() - benign_sample.std())
        }

        return generated, metadata

    def _inject_attack_pattern(
        self,
        traffic: np.ndarray,
        pattern: str
    ) -> np.ndarray:
        """
        Inject subtle attack patterns into generated traffic.

        These patterns are designed to be statistically hidden but
        detectable by sophisticated anomaly detection.

        Uses feature indices and attack parameters defined at module level:
        - Feature indices: FEATURE_OUTBOUND_BYTES, FEATURE_INBOUND_BYTES, etc.
        - Attack intensities: SLOW_EXFIL_TREND_MAGNITUDE, LOTL_BURST_MAGNITUDE, etc.
        """
        traffic = traffic.copy()

        if pattern == 'slow_exfiltration':
            # Gradual increase in outbound/inbound bytes
            trend = np.linspace(0, SLOW_EXFIL_TREND_MAGNITUDE, traffic.shape[0])
            if traffic.shape[1] > FEATURE_OUTBOUND_BYTES:
                traffic[:, FEATURE_OUTBOUND_BYTES] += trend * traffic[:, FEATURE_OUTBOUND_BYTES].std()
            if traffic.shape[1] > FEATURE_INBOUND_BYTES:
                traffic[:, FEATURE_INBOUND_BYTES] += trend * SLOW_EXFIL_INBOUND_SCALING * traffic[:, FEATURE_INBOUND_BYTES].std()

        elif pattern == 'lotl_mimicry':
            # Living-off-the-land: Periodic micro-bursts
            # Inject at positions that look like legitimate polling
            for pos in LOTL_BURST_POSITIONS:
                if pos < len(traffic):
                    if traffic.shape[1] > FEATURE_PACKET_RATE_END:
                        traffic[pos, FEATURE_PACKET_RATE_START:FEATURE_PACKET_RATE_END] *= LOTL_BURST_MAGNITUDE

        elif pattern == 'protocol_anomaly':
            # Subtle protocol timing anomaly
            # Slightly irregular inter-arrival times and jitter
            if traffic.shape[1] > FEATURE_TIMING_IAT:
                traffic[:, FEATURE_TIMING_IAT] *= (1 + np.random.randn(traffic.shape[0]) * PROTOCOL_ANOMALY_NOISE_FACTOR)
            if traffic.shape[1] > FEATURE_TIMING_JITTER:
                traffic[:, FEATURE_TIMING_JITTER] *= (1 + np.random.randn(traffic.shape[0]) * PROTOCOL_ANOMALY_NOISE_FACTOR)

        elif pattern == 'beacon':
            # C2 beacon pattern - very regular intervals
            for i in range(0, len(traffic), BEACON_INTERVAL):
                traffic[i, :] *= BEACON_DIP_MAGNITUDE
        else:
            logger.warning(f"Unknown attack pattern: {pattern}. No pattern injected.")

        return traffic

    def get_decomposition(self, sample: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get trend and seasonality decomposition for visualization.

        Args:
            sample: Array of shape (seq_length, feature_dim)

        Returns:
            Dict with 'trend', 'seasonality', 'residual' components

        Raises:
            RuntimeError: If model not initialized
            ValueError: If sequence is too short for decomposition
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized")

        # Check minimum sequence length
        seq_length = sample.shape[0]
        if seq_length < MIN_SEQUENCE_LENGTH_FOR_DECOMPOSITION:
            raise ValueError(
                f"Sequence too short for decomposition. "
                f"Got length {seq_length}, need at least {MIN_SEQUENCE_LENGTH_FOR_DECOMPOSITION}. "
                f"Decomposition requires sufficient data points for trend and seasonality extraction."
            )

        # Simple decomposition for visualization
        # In real Diffusion-TS, this comes from the model's internal representation
        from scipy.signal import savgol_filter
        from scipy.fft import rfft, irfft

        # Extract trend using Savitzky-Golay filter
        window_length = min(SAVGOL_DEFAULT_WINDOW, seq_length if seq_length % 2 == 1 else seq_length - 1)
        if window_length < 3:
            window_length = 3
        polyorder = min(3, window_length - 1)

        trend = savgol_filter(sample, window_length=window_length, polyorder=polyorder, axis=0)

        # Remove trend
        detrended = sample - trend

        # Extract dominant frequencies for seasonality
        fft = rfft(detrended, axis=0)
        # Keep only top frequencies
        n_keep = SEASONALITY_TOP_FREQUENCIES
        fft_filtered = np.zeros_like(fft)
        magnitudes = np.abs(fft)
        for i in range(sample.shape[1]):
            if magnitudes.shape[0] > n_keep:
                top_indices = np.argsort(magnitudes[:, i])[-n_keep:]
                fft_filtered[top_indices, i] = fft[top_indices, i]
            else:
                fft_filtered[:, i] = fft[:, i]
        seasonality = irfft(fft_filtered, n=sample.shape[0], axis=0)

        # Residual is what's left
        residual = sample - trend - seasonality

        return {
            'trend': trend,
            'seasonality': seasonality,
            'residual': residual,
            'original': sample
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        if self._mock_mode:
            logger.warning("Mock mode active - no checkpoint to save")
            return

        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': {
                    'seq_length': self.seq_length,
                    'feature_dim': self.feature_dim,
                    'n_diffusion_steps': self.n_diffusion_steps
                }
            }, path)
            logger.info(f"Checkpoint saved to {path}")
        else:
            logger.warning("No model to save")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        if self._mock_mode:
            logger.warning("Mock mode active - cannot load checkpoint")
            return

        if self.model is None:
            logger.error("Model not initialized. Call initialize() first.")
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Checkpoint loaded from {path}")
