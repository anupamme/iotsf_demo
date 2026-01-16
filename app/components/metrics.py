"""Reusable metric display components for the Streamlit app."""

import streamlit as st
from typing import Dict, Any


def display_metrics_row(metrics: Dict[str, Any]):
    """
    Display key metrics in a row of columns.

    Args:
        metrics: Dictionary containing metrics (accuracy, precision, recall, f1, etc.)
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")

    with col2:
        st.metric("Precision", f"{metrics.get('precision', 0):.3f}")

    with col3:
        st.metric("Recall", f"{metrics.get('recall', 0):.3f}")

    with col4:
        st.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")


def display_confusion_matrix_stats(metrics: Dict[str, Any]):
    """
    Display confusion matrix statistics.

    Args:
        metrics: Dictionary containing confusion matrix values
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "True Positives",
            metrics.get('true_positives', 0),
            help="Attacks correctly detected"
        )

    with col2:
        st.metric(
            "False Positives",
            metrics.get('false_positives', 0),
            help="Benign traffic incorrectly flagged as attacks"
        )

    with col3:
        st.metric(
            "True Negatives",
            metrics.get('true_negatives', 0),
            help="Benign traffic correctly identified"
        )

    with col4:
        st.metric(
            "False Negatives",
            metrics.get('false_negatives', 0),
            help="Attacks missed (not detected)"
        )


def display_fpr_metric(metrics: Dict[str, Any]):
    """
    Display False Positive Rate with color coding.

    Args:
        metrics: Dictionary containing false_positive_rate
    """
    fpr = metrics.get('false_positive_rate', 0)

    # Color code: green if <5%, yellow if 5-10%, red if >10%
    if fpr < 0.05:
        delta_color = "normal"
        help_text = "Excellent: Low false positive rate"
    elif fpr < 0.10:
        delta_color = "off"
        help_text = "Good: Moderate false positive rate"
    else:
        delta_color = "inverse"
        help_text = "High: Many false positives"

    st.metric(
        "False Positive Rate",
        f"{fpr:.3f}",
        delta=None,
        help=help_text
    )
