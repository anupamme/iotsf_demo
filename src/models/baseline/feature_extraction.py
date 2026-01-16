"""
Feature Extraction Utilities for Traditional IDS

Converts time series sequences into statistical feature vectors
that traditional IDS methods can process effectively.
"""

import numpy as np
from typing import Tuple, Dict
from scipy.stats import linregress

# Feature names for CICIoT2023 dataset
FEATURE_NAMES = [
    'flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot',
    'fwd_data_pkts_tot', 'bwd_data_pkts_tot',
    'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'flow_pkts_per_sec',
    'fwd_byts_b_avg', 'bwd_byts_b_avg',
    'fwd_iat_mean', 'bwd_iat_mean'
]

# Statistic names (order matters - must match extract_sequence_features)
STAT_NAMES = ['mean', 'std', 'min', 'max', 'slope', 'peak_to_peak']


def extract_sequence_features(sequence: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from a single time series sequence.

    For each of the feature_dim features, computes:
    - mean: Average value over time
    - std: Standard deviation (variability)
    - min: Minimum value
    - max: Maximum value
    - slope: Linear trend (via linear regression)
    - peak_to_peak: max - min (amplitude)

    Args:
        sequence: Time series of shape (seq_length, feature_dim)

    Returns:
        Feature vector of shape (feature_dim * 6,)

    Example:
        >>> seq = np.random.randn(128, 12)  # 128 timesteps, 12 features
        >>> features = extract_sequence_features(seq)
        >>> features.shape
        (72,)  # 12 features × 6 statistics
    """
    seq_length, feature_dim = sequence.shape
    features = []

    for feature_idx in range(feature_dim):
        feature_values = sequence[:, feature_idx]

        # Basic statistics
        mean_val = np.mean(feature_values)
        std_val = np.std(feature_values)
        min_val = np.min(feature_values)
        max_val = np.max(feature_values)

        # Trend analysis (linear regression slope)
        time_steps = np.arange(seq_length)
        slope, _, _, _, _ = linregress(time_steps, feature_values)

        # Amplitude
        peak_to_peak = max_val - min_val

        # Append all statistics for this feature
        features.extend([mean_val, std_val, min_val, max_val, slope, peak_to_peak])

    return np.array(features)


def extract_structured_features(sequence: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Extract statistical features as a structured dictionary.

    This provides a more maintainable interface than the flat array.
    Features can be accessed by name, e.g., features['fwd_pkts_per_sec']['mean']

    Args:
        sequence: Time series of shape (seq_length, feature_dim)

    Returns:
        Nested dictionary: {feature_name: {stat_name: value}}

    Example:
        >>> seq = np.random.randn(128, 12)
        >>> features = extract_structured_features(seq)
        >>> fwd_pkt_rate_mean = features['fwd_pkts_per_sec']['mean']
        >>> fwd_byte_avg_max = features['fwd_byts_b_avg']['max']
    """
    seq_length, feature_dim = sequence.shape
    structured_features = {}

    for feature_idx in range(feature_dim):
        feature_name = FEATURE_NAMES[feature_idx] if feature_idx < len(FEATURE_NAMES) else f'feature_{feature_idx}'
        feature_values = sequence[:, feature_idx]

        # Basic statistics
        mean_val = np.mean(feature_values)
        std_val = np.std(feature_values)
        min_val = np.min(feature_values)
        max_val = np.max(feature_values)

        # Trend analysis (linear regression slope)
        time_steps = np.arange(seq_length)
        slope, _, _, _, _ = linregress(time_steps, feature_values)

        # Amplitude
        peak_to_peak = max_val - min_val

        # Store as nested dictionary
        structured_features[feature_name] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'slope': slope,
            'peak_to_peak': peak_to_peak
        }

    return structured_features


def get_feature_value(features_flat: np.ndarray, feature_name: str, stat_name: str) -> float:
    """
    Access a specific feature value from the flat feature array.

    This helper function provides name-based access to features extracted
    by extract_sequence_features(), making the code more maintainable.

    Args:
        features_flat: Flat feature array from extract_sequence_features()
        feature_name: Name of the feature (e.g., 'fwd_pkts_per_sec')
        stat_name: Name of the statistic (e.g., 'mean', 'std')

    Returns:
        The requested feature value

    Raises:
        ValueError: If feature_name or stat_name is invalid

    Example:
        >>> seq = np.random.randn(128, 12)
        >>> features = extract_sequence_features(seq)
        >>> pkt_rate_mean = get_feature_value(features, 'fwd_pkts_per_sec', 'mean')
    """
    if feature_name not in FEATURE_NAMES:
        raise ValueError(f"Invalid feature name: {feature_name}. Must be one of {FEATURE_NAMES}")

    if stat_name not in STAT_NAMES:
        raise ValueError(f"Invalid stat name: {stat_name}. Must be one of {STAT_NAMES}")

    feature_idx = FEATURE_NAMES.index(feature_name)
    stat_idx = STAT_NAMES.index(stat_name)

    # Calculate flat array index
    flat_idx = feature_idx * len(STAT_NAMES) + stat_idx

    return features_flat[flat_idx]


def extract_batch_features(sequences: np.ndarray) -> np.ndarray:
    """
    Vectorized feature extraction for multiple sequences.

    Args:
        sequences: Batch of time series of shape (n_samples, seq_length, feature_dim)

    Returns:
        Feature matrix of shape (n_samples, feature_dim * 6)

    Example:
        >>> seqs = np.random.randn(100, 128, 12)  # 100 sequences
        >>> features = extract_batch_features(seqs)
        >>> features.shape
        (100, 72)  # 100 samples × 72 features
    """
    n_samples = sequences.shape[0]
    feature_list = []

    for i in range(n_samples):
        feature_list.append(extract_sequence_features(sequences[i]))

    return np.array(feature_list)


def compute_asymmetry_ratio(sequence: np.ndarray, fwd_idx: int, bwd_idx: int) -> float:
    """
    Compute asymmetry ratio between forward and backward traffic features.

    High asymmetry (e.g., >10:1) can indicate DDoS or exfiltration.

    Args:
        sequence: Time series of shape (seq_length, feature_dim)
        fwd_idx: Index of forward traffic feature
        bwd_idx: Index of backward traffic feature

    Returns:
        Asymmetry ratio (fwd_mean / bwd_mean), or 0 if bwd_mean is zero
    """
    fwd_mean = np.mean(sequence[:, fwd_idx])
    bwd_mean = np.mean(sequence[:, bwd_idx])

    if bwd_mean < 1e-6:  # Avoid division by zero
        return 0.0

    return fwd_mean / bwd_mean


def detect_periodicity(sequence: np.ndarray, feature_idx: int) -> Tuple[float, float]:
    """
    Detect periodicity in a time series using FFT.

    Regular periodic patterns (e.g., C2 beaconing) will have strong
    frequency components.

    Args:
        sequence: Time series of shape (seq_length, feature_dim)
        feature_idx: Index of feature to analyze

    Returns:
        Tuple of (dominant_frequency, power_ratio)
        - dominant_frequency: Frequency with highest power
        - power_ratio: Ratio of dominant power to total power
    """
    feature_values = sequence[:, feature_idx]

    # Compute FFT
    fft_values = np.fft.fft(feature_values)
    power_spectrum = np.abs(fft_values) ** 2

    # Ignore DC component (index 0)
    power_spectrum[0] = 0

    # Find dominant frequency
    dominant_idx = np.argmax(power_spectrum)
    dominant_power = power_spectrum[dominant_idx]
    total_power = np.sum(power_spectrum)

    # Compute frequency and power ratio
    seq_length = len(feature_values)
    dominant_frequency = dominant_idx / seq_length
    power_ratio = dominant_power / total_power if total_power > 0 else 0.0

    return dominant_frequency, power_ratio


def compute_traffic_volume(sequence: np.ndarray, packet_idx: int, byte_idx: int) -> Tuple[float, float]:
    """
    Compute total traffic volume and rate.

    Args:
        sequence: Time series of shape (seq_length, feature_dim)
        packet_idx: Index of packet count feature
        byte_idx: Index of byte count feature

    Returns:
        Tuple of (total_packets, total_bytes)
    """
    total_packets = np.sum(sequence[:, packet_idx])
    total_bytes = np.sum(sequence[:, byte_idx])

    return total_packets, total_bytes


def check_monotonic_trend(sequence: np.ndarray, feature_idx: int, threshold: float = 0.8) -> bool:
    """
    Check if a feature has a strong monotonic increasing or decreasing trend.

    Useful for detecting slow exfiltration (gradual increase in outbound data).

    Args:
        sequence: Time series of shape (seq_length, feature_dim)
        feature_idx: Index of feature to analyze
        threshold: Minimum R² value to consider trend significant (default: 0.8)

    Returns:
        True if monotonic trend detected, False otherwise
    """
    feature_values = sequence[:, feature_idx]
    time_steps = np.arange(len(feature_values))

    # Linear regression
    slope, intercept, r_value, _, _ = linregress(time_steps, feature_values)

    # Check if R² is above threshold and slope is significant
    r_squared = r_value ** 2

    return r_squared >= threshold and abs(slope) > 1e-6
