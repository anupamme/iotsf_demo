"""Reusable plot components for the Streamlit app."""

import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.visualization import ids_plots


def display_confusion_matrix(y_true, y_pred, method_name="IDS"):
    """
    Display confusion matrix plot in Streamlit.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        method_name: Name of the IDS method
    """
    fig = ids_plots.plot_confusion_matrix(y_true, y_pred, method_name)
    st.plotly_chart(fig, use_container_width=True)


def display_roc_curves(methods_results, y_test):
    """
    Display ROC curves for multiple methods.

    Args:
        methods_results: Dictionary mapping method name to results dict
        y_test: True labels
    """
    fig = ids_plots.plot_roc_curves(methods_results, y_test)
    st.plotly_chart(fig, use_container_width=True)


def display_detection_results(X_sequences, y_true, y_pred, feature_idx=7, n_show=5):
    """
    Display time series with detection overlays.

    Args:
        X_sequences: Time series sequences
        y_true: True labels
        y_pred: Predicted labels
        feature_idx: Feature index to visualize
        n_show: Number of sequences to show
    """
    fig = ids_plots.plot_detection_results(X_sequences, y_true, y_pred, feature_idx, n_show)
    st.plotly_chart(fig, use_container_width=True)


def display_method_comparison(methods_results):
    """
    Display bar chart comparing methods.

    Args:
        methods_results: Dictionary mapping method name to metrics dict
    """
    fig = ids_plots.plot_method_comparison(methods_results)
    st.plotly_chart(fig, use_container_width=True)


def display_score_distribution(y_scores, y_true, method_name="IDS"):
    """
    Display anomaly score distribution.

    Args:
        y_scores: Anomaly scores
        y_true: True labels
        method_name: Name of the IDS method
    """
    fig = ids_plots.plot_score_distribution(y_scores, y_true, method_name)
    st.plotly_chart(fig, use_container_width=True)


def display_metrics_radar(methods_results):
    """
    Display radar chart of metrics.

    Args:
        methods_results: Dictionary mapping method name to metrics dict
    """
    fig = ids_plots.plot_metrics_radar(methods_results)
    st.plotly_chart(fig, use_container_width=True)
