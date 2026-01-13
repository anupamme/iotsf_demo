#!/usr/bin/env python3
"""
Test script for visualization components.

This script generates all visualization types with synthetic data
and saves them as HTML files for visual inspection.

Usage:
    python scripts/test_visualizations.py
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.components.plots import (
    plot_time_series,
    plot_comparison_grid,
    plot_anomaly_scores,
    plot_detection_comparison,
    plot_prediction_vs_actual,
    plot_decomposition,
    plot_metrics_comparison,
    plot_diffusion_animation
)


def create_output_dir():
    """Create output directory for test plots."""
    output_dir = ROOT_DIR / "outputs" / "test_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_synthetic_data(seq_length=128, n_features=12):
    """Generate synthetic time-series data."""
    t = np.linspace(0, 4*np.pi, seq_length)
    data = np.zeros((seq_length, n_features))

    for i in range(n_features):
        # Trend
        trend = np.linspace(0, 0.1 * (i + 1), seq_length)
        # Seasonality
        seasonality = np.sin(t + i * 0.5) * 0.3
        # Noise
        noise = np.random.randn(seq_length) * 0.1
        # Combine
        data[:, i] = trend + seasonality + noise

    return data


def test_time_series_plot(output_dir):
    """Test time-series plotting."""
    print("Testing plot_time_series()...")

    # Test 1: Basic benign traffic
    data_benign = generate_synthetic_data()
    fig1 = plot_time_series(data_benign, label="Benign Traffic", is_attack=False)
    fig1.write_html(output_dir / "01_time_series_benign.html")

    # Test 2: Attack traffic
    data_attack = generate_synthetic_data() * 1.5  # Simulate attack with higher values
    fig2 = plot_time_series(data_attack, label="DDoS Attack", is_attack=True)
    fig2.write_html(output_dir / "02_time_series_attack.html")

    # Test 3: Show all features with dropdown
    fig3 = plot_time_series(data_benign, label="All Features", show_all=True, is_attack=False)
    fig3.write_html(output_dir / "03_time_series_all_features.html")

    print("  ✓ Created 3 time-series plots")


def test_comparison_grid(output_dir):
    """Test comparison grid plotting."""
    print("Testing plot_comparison_grid()...")

    # Generate 6 samples (3 benign, 3 attack)
    samples = []
    true_labels = []

    for i in range(6):
        is_attack = (i % 2 == 1)  # Alternate benign/attack
        data = generate_synthetic_data()
        if is_attack:
            data *= 1.3  # Simulate attack
        samples.append(data)
        true_labels.append(is_attack)

    # Test 1: Hidden labels (challenge mode)
    fig1 = plot_comparison_grid(samples, true_labels=true_labels, reveal=False)
    fig1.write_html(output_dir / "04_comparison_grid_hidden.html")

    # Test 2: Revealed labels
    fig2 = plot_comparison_grid(samples, true_labels=true_labels, reveal=True)
    fig2.write_html(output_dir / "05_comparison_grid_revealed.html")

    print("  ✓ Created 2 comparison grid plots")


def test_anomaly_scores(output_dir):
    """Test anomaly score plotting."""
    print("Testing plot_anomaly_scores()...")

    # Generate synthetic scores
    np.random.seed(42)
    n_samples = 10
    scores = np.concatenate([
        np.random.uniform(0.2, 0.5, 5),  # Benign samples
        np.random.uniform(0.6, 0.9, 5)   # Attack samples
    ])
    ground_truth = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    threshold = 0.55

    # Test 1: With ground truth
    fig1 = plot_anomaly_scores(scores, threshold, ground_truth=ground_truth)
    fig1.write_html(output_dir / "06_anomaly_scores_with_gt.html")

    # Test 2: Without ground truth
    fig2 = plot_anomaly_scores(scores, threshold)
    fig2.write_html(output_dir / "07_anomaly_scores_no_gt.html")

    print("  ✓ Created 2 anomaly score plots")


def test_detection_comparison(output_dir):
    """Test detection comparison plotting."""
    print("Testing plot_detection_comparison()...")

    # Simulate baseline vs Moirai results
    baseline_results = {
        'TP': 45,
        'FP': 15,
        'TN': 35,
        'FN': 5
    }

    moirai_results = {
        'TP': 48,
        'FP': 3,
        'TN': 47,
        'FN': 2
    }

    fig = plot_detection_comparison(baseline_results, moirai_results)
    fig.write_html(output_dir / "08_detection_comparison.html")

    print("  ✓ Created detection comparison plot")


def test_prediction_vs_actual(output_dir):
    """Test prediction vs actual plotting."""
    print("Testing plot_prediction_vs_actual()...")

    # Generate synthetic prediction data
    n_points = 100
    t = np.linspace(0, 4*np.pi, n_points)
    actual = np.sin(t) + np.random.randn(n_points) * 0.05
    predicted = np.sin(t) + np.random.randn(n_points) * 0.1  # More noise

    # Test 1: Without confidence intervals
    fig1 = plot_prediction_vs_actual(actual, predicted, feature_name="Packet Rate")
    fig1.write_html(output_dir / "09_prediction_no_ci.html")

    # Test 2: With confidence intervals
    ci_lower = predicted - 0.2
    ci_upper = predicted + 0.2
    fig2 = plot_prediction_vs_actual(actual, predicted, ci_lower, ci_upper, feature_name="Packet Rate")
    fig2.write_html(output_dir / "10_prediction_with_ci.html")

    print("  ✓ Created 2 prediction vs actual plots")


def test_decomposition(output_dir):
    """Test decomposition plotting."""
    print("Testing plot_decomposition()...")

    # Generate synthetic decomposition
    seq_length = 128
    t = np.linspace(0, 4*np.pi, seq_length)

    # Create components
    trend = np.linspace(0, 0.3, seq_length)
    seasonality = 0.3 * np.sin(t)
    residual = np.random.randn(seq_length) * 0.05
    original = trend + seasonality + residual

    decomposition = {
        'original': original,
        'trend': trend,
        'seasonality': seasonality,
        'residual': residual
    }

    fig = plot_decomposition(decomposition, feature_idx=0, feature_name="Flow Duration")
    fig.write_html(output_dir / "11_decomposition.html")

    print("  ✓ Created decomposition plot")


def test_metrics_comparison(output_dir):
    """Test metrics comparison plotting."""
    print("Testing plot_metrics_comparison()...")

    # Simulate metrics from multiple methods
    metrics_dict = {
        "Baseline IDS": {
            "precision": 0.65,
            "recall": 0.70,
            "f1": 0.67,
            "accuracy": 0.72
        },
        "Isolation Forest": {
            "precision": 0.78,
            "recall": 0.75,
            "f1": 0.76,
            "accuracy": 0.80
        },
        "Moirai": {
            "precision": 0.92,
            "recall": 0.95,
            "f1": 0.93,
            "accuracy": 0.94
        }
    }

    fig = plot_metrics_comparison(metrics_dict)
    fig.write_html(output_dir / "12_metrics_comparison.html")

    print("  ✓ Created metrics comparison plot")


def test_diffusion_animation(output_dir):
    """Test diffusion animation plotting."""
    print("Testing plot_diffusion_animation()...")

    # Generate synthetic diffusion steps (noise -> signal)
    seq_length = 128
    n_steps = 30
    n_features = 12

    # Final clean signal
    t = np.linspace(0, 4*np.pi, seq_length)
    clean_signal = np.sin(t) * 0.5

    # Create steps from noise to clean signal
    diffusion_steps = []
    for i in range(n_steps):
        noise_level = 1.0 - (i / n_steps)  # Decrease noise over time
        step_data = np.zeros((seq_length, n_features))
        for j in range(n_features):
            step_data[:, j] = clean_signal + np.random.randn(seq_length) * noise_level * 0.5
        diffusion_steps.append(step_data)

    fig = plot_diffusion_animation(diffusion_steps, feature_idx=0, feature_name="Flow Duration")
    fig.write_html(output_dir / "13_diffusion_animation.html")

    print("  ✓ Created diffusion animation plot")


def main():
    """Run all visualization tests."""
    print("=" * 60)
    print("Visualization Components Test Suite")
    print("=" * 60)
    print()

    # Create output directory
    output_dir = create_output_dir()
    print(f"Output directory: {output_dir}")
    print()

    # Run tests
    test_time_series_plot(output_dir)
    test_comparison_grid(output_dir)
    test_anomaly_scores(output_dir)
    test_detection_comparison(output_dir)
    test_prediction_vs_actual(output_dir)
    test_decomposition(output_dir)
    test_metrics_comparison(output_dir)
    test_diffusion_animation(output_dir)

    print()
    print("=" * 60)
    print(f"✅ All tests completed successfully!")
    print(f"📊 Generated 13 test plots in: {output_dir}")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Open HTML files in browser to visually inspect plots")
    print("2. Verify dark theme is applied consistently")
    print("3. Check interactivity (hover, zoom, pan)")
    print("4. Validate color scheme (benign=green, attack=red, detected=blue)")


if __name__ == "__main__":
    main()
