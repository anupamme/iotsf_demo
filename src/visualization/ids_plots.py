"""
IDS Visualization Plots

Plotly-based visualization functions for IDS detection results.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
from sklearn.metrics import roc_curve, auc


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method_name: str = "IDS"
) -> go.Figure:
    """
    Create a confusion matrix heatmap.

    Args:
        y_true: True labels (0=benign, 1=attack)
        y_pred: Predicted labels (0=benign, 1=attack)
        method_name: Name of the IDS method for title

    Returns:
        Plotly Figure object
    """
    from sklearn.metrics import confusion_matrix

    # Explicitly specify labels to ensure 2x2 matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Benign', 'Attack'],
        y=['Benign', 'Attack'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))

    fig.update_layout(
        title=f'Confusion Matrix - {method_name}',
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
        width=500,
        height=500
    )

    return fig


def plot_roc_curves(
    methods_results: Dict[str, Dict],
    y_test: np.ndarray
) -> go.Figure:
    """
    Plot ROC curves for multiple IDS methods.

    Args:
        methods_results: Dictionary mapping method name to results dict
                        (must contain 'scores' key with probability scores)
        y_test: True labels

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    for method_name, results in methods_results.items():
        if 'scores' not in results:
            continue

        y_scores = results['scores']

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        # Add trace
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            name=f'{method_name} (AUC={roc_auc:.3f})',
            mode='lines',
            line=dict(width=2)
        ))

    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray', width=1)
    ))

    fig.update_layout(
        title='ROC Curves - Method Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate (Recall)',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=700,
        height=600,
        legend=dict(x=0.6, y=0.1)
    )

    return fig


def plot_detection_results(
    X_sequences: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_idx: int = 7,
    n_show: int = 5
) -> go.Figure:
    """
    Visualize time series sequences with detection overlays.

    Args:
        X_sequences: Sequences of shape (n_samples, seq_length, feature_dim)
        y_true: True labels (0=benign, 1=attack)
        y_pred: Predicted labels (0=benign, 1=attack)
        feature_idx: Which feature to plot (default: 7 = flow_pkts_per_sec)
        n_show: Number of sequences to show (default: 5)

    Returns:
        Plotly Figure object
    """
    n_show = min(n_show, len(X_sequences))

    fig = go.Figure()

    for i in range(n_show):
        seq = X_sequences[i, :, feature_idx]

        # Determine color and label
        true_label = 'Attack' if y_true[i] == 1 else 'Benign'
        pred_label = 'Attack' if y_pred[i] == 1 else 'Benign'
        correct = '✓' if y_pred[i] == y_true[i] else '✗'

        # Color: green for correct, red for incorrect
        color = '#00CC96' if y_pred[i] == y_true[i] else '#EF553B'

        fig.add_trace(go.Scatter(
            y=seq,
            mode='lines',
            name=f'Seq {i}: True={true_label}, Pred={pred_label} {correct}',
            line=dict(color=color, width=2),
            opacity=0.7
        ))

    fig.update_layout(
        title='Detection Results - Time Series Visualization',
        xaxis_title='Time Step',
        yaxis_title='Feature Value (flow_pkts_per_sec)',
        width=900,
        height=500,
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='top')
    )

    return fig


def plot_method_comparison(methods_results: Dict[str, Dict]) -> go.Figure:
    """
    Bar chart comparing precision, recall, F1 across methods.

    Args:
        methods_results: Dictionary mapping method name to metrics dict

    Returns:
        Plotly Figure object
    """
    methods = list(methods_results.keys())
    metrics_to_plot = ['precision', 'recall', 'f1']

    fig = go.Figure()

    for metric in metrics_to_plot:
        values = [methods_results[m][metric] for m in methods]
        fig.add_trace(go.Bar(
            name=metric.title(),
            x=methods,
            y=values,
            text=[f'{v:.3f}' for v in values],
            textposition='auto'
        ))

    fig.update_layout(
        title='IDS Performance Comparison',
        xaxis_title='Method',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1.0]),
        barmode='group',
        width=800,
        height=500,
        legend=dict(x=0.7, y=0.98)
    )

    return fig


def plot_score_distribution(
    y_scores: np.ndarray,
    y_true: np.ndarray,
    method_name: str = "IDS"
) -> go.Figure:
    """
    Plot distribution of anomaly scores for benign vs attack samples.

    Args:
        y_scores: Anomaly scores from predict_proba()
        y_true: True labels (0=benign, 1=attack)
        method_name: Name of the IDS method

    Returns:
        Plotly Figure object
    """
    # Split scores by true label
    benign_scores = y_scores[y_true == 0]
    attack_scores = y_scores[y_true == 1]

    fig = go.Figure()

    # Benign distribution
    fig.add_trace(go.Histogram(
        x=benign_scores,
        name='Benign',
        opacity=0.7,
        marker_color='#00CC96',
        nbinsx=30
    ))

    # Attack distribution
    fig.add_trace(go.Histogram(
        x=attack_scores,
        name='Attack',
        opacity=0.7,
        marker_color='#EF553B',
        nbinsx=30
    ))

    fig.update_layout(
        title=f'Anomaly Score Distribution - {method_name}',
        xaxis_title='Anomaly Score',
        yaxis_title='Count',
        barmode='overlay',
        width=700,
        height=500,
        legend=dict(x=0.7, y=0.98)
    )

    return fig


def plot_metrics_radar(methods_results: Dict[str, Dict]) -> go.Figure:
    """
    Radar chart showing multiple metrics for each method.

    Args:
        methods_results: Dictionary mapping method name to metrics dict

    Returns:
        Plotly Figure object
    """
    metrics = ['precision', 'recall', 'f1', 'accuracy']

    fig = go.Figure()

    for method_name, results in methods_results.items():
        values = [results[m] for m in metrics]
        # Close the radar chart
        values.append(values[0])

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            name=method_name,
            fill='toself',
            opacity=0.6
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title='IDS Methods - Multi-Metric Comparison',
        width=600,
        height=600,
        showlegend=True
    )

    return fig
