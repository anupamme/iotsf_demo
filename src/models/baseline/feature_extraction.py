"""
Feature Extraction Utilities for Traditional IDS

Converts time series sequences into statistical feature vectors
that traditional IDS methods can process effectively.
"""

import numpy as np
from typing import Tuple
from scipy.stats import linregress


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
