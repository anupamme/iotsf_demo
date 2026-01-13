"""Test Visualization Components Page"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add src to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.components.plots import (
    plot_time_series,
    plot_comparison_grid,
    plot_anomaly_scores,
    plot_detection_comparison,
    plot_prediction_vs_actual,
    plot_decomposition,
    plot_metrics_comparison,
)

st.set_page_config(
    page_title="Test Visualizations",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Visualization Components Test Page")

st.markdown("""
This page demonstrates all implemented visualization components with synthetic data.
Use the tabs below to explore each visualization type.
""")

# Generate synthetic data
def generate_synthetic_data(seq_length=128, n_features=12):
    """Generate synthetic time-series data."""
    t = np.linspace(0, 4*np.pi, seq_length)
    data = np.zeros((seq_length, n_features))

    for i in range(n_features):
        trend = np.linspace(0, 0.1 * (i + 1), seq_length)
        seasonality = np.sin(t + i * 0.5) * 0.3
        noise = np.random.randn(seq_length) * 0.1
        data[:, i] = trend + seasonality + noise

    return data

# Create tabs for different visualizations
tabs = st.tabs([
    "Time Series",
    "Comparison Grid",
    "Anomaly Scores",
    "Detection Comparison",
    "Prediction vs Actual",
    "Decomposition",
    "Metrics Comparison"
])

# Tab 1: Time Series
with tabs[0]:
    st.header("Time Series Visualization")
    st.markdown("Display time-series traffic data with multi-feature support.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Benign Traffic")
        data_benign = generate_synthetic_data()
        fig1 = plot_time_series(data_benign, label="Benign Traffic", is_attack=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Attack Traffic")
        data_attack = generate_synthetic_data() * 1.5
        fig2 = plot_time_series(data_attack, label="DDoS Attack", is_attack=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("All Features with Dropdown")
    show_all = st.checkbox("Show all features dropdown", value=True)
    if show_all:
        fig3 = plot_time_series(data_benign, label="All Features", show_all=True, is_attack=False)
        st.plotly_chart(fig3, use_container_width=True)

# Tab 2: Comparison Grid
with tabs[1]:
    st.header("Comparison Grid (Spot the Attack)")
    st.markdown("Grid layout for comparing multiple samples - perfect for the challenge page!")

    # Generate samples
    samples = []
    true_labels = []
    for i in range(6):
        is_attack = (i % 2 == 1)
        data = generate_synthetic_data()
        if is_attack:
            data *= 1.3
        samples.append(data)
        true_labels.append(is_attack)

    reveal = st.checkbox("Reveal answers (show color coding)", value=False)

    fig = plot_comparison_grid(samples, true_labels=true_labels, reveal=reveal)
    st.plotly_chart(fig, use_container_width=True)

    if reveal:
        st.success("Green = Benign, Red = Attack")
    else:
        st.info("Try to spot which samples are attacks!")

# Tab 3: Anomaly Scores
with tabs[2]:
    st.header("Anomaly Detection Scores")
    st.markdown("Bar chart showing anomaly scores with threshold and optional TP/FP/TN/FN markers.")

    # Generate scores
    np.random.seed(42)
    scores = np.concatenate([
        np.random.uniform(0.2, 0.5, 5),
        np.random.uniform(0.6, 0.9, 5)
    ])
    ground_truth = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    threshold = 0.55

    show_gt = st.checkbox("Show ground truth (TP/FP/TN/FN markers)", value=True)

    if show_gt:
        fig = plot_anomaly_scores(scores, threshold, ground_truth=ground_truth)
    else:
        fig = plot_anomaly_scores(scores, threshold)

    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Detection Comparison
with tabs[3]:
    st.header("Detection Performance Comparison")
    st.markdown("Compare baseline IDS vs Moirai detection results.")

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
    st.plotly_chart(fig, use_container_width=True)

    st.info("Notice how Moirai has higher TP and TN with fewer false positives!")

# Tab 5: Prediction vs Actual
with tabs[4]:
    st.header("Moirai Prediction vs Actual")
    st.markdown("Compare predicted values against actual values with optional confidence intervals.")

    n_points = 100
    t = np.linspace(0, 4*np.pi, n_points)
    actual = np.sin(t) + np.random.randn(n_points) * 0.05
    predicted = np.sin(t) + np.random.randn(n_points) * 0.1

    show_ci = st.checkbox("Show confidence intervals", value=True)

    if show_ci:
        ci_lower = predicted - 0.2
        ci_upper = predicted + 0.2
        fig = plot_prediction_vs_actual(actual, predicted, ci_lower, ci_upper, feature_name="Packet Rate")
    else:
        fig = plot_prediction_vs_actual(actual, predicted, feature_name="Packet Rate")

    st.plotly_chart(fig, use_container_width=True)

# Tab 6: Decomposition
with tabs[5]:
    st.header("Time-Series Decomposition")
    st.markdown("Break down traffic into trend, seasonality, and residual components.")

    seq_length = 128
    t = np.linspace(0, 4*np.pi, seq_length)

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
    st.plotly_chart(fig, use_container_width=True)

# Tab 7: Metrics Comparison
with tabs[6]:
    st.header("Detection Metrics Comparison")
    st.markdown("Compare precision, recall, F1, and accuracy across multiple detection methods.")

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
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.success("✅ All visualization components are working correctly!")
st.info("These components are now ready to be integrated into the main demo pages.")
