"""Reusable plot components for the Streamlit app."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional, List, Dict, Any
import sys
from pathlib import Path

# Add src to path for imports
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models import AnomalyResult


def plot_prediction_vs_actual(
    result: AnomalyResult,
    feature_idx: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    theme: str = "plotly_dark",
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None
) -> go.Figure:
    """
    Plot predictions vs actual values with confidence intervals.

    Args:
        result: AnomalyResult object containing detection results
        feature_idx: Specific feature to plot (if None, plots all as subplots)
        feature_names: Names for features (optional)
        theme: Plotly theme ('plotly_dark' or 'plotly_white')
        title: Custom title for the plot
        colors: Color dictionary with keys: 'benign', 'attack', 'detected'

    Returns:
        Plotly figure object
    """
    # Default colors
    if colors is None:
        colors = {
            'benign': '#00CC96',
            'attack': '#EF553B',
            'detected': '#636EFA'
        }

    # Feature names
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(result.n_features)]

    # Time axis
    time_steps = np.arange(result.seq_length)

    if feature_idx is not None:
        # Single feature plot
        fig = go.Figure()

        # Confidence interval (filled area)
        fig.add_trace(go.Scatter(
            x=np.concatenate([time_steps, time_steps[::-1]]),
            y=np.concatenate([
                result.confidence_upper[:, feature_idx],
                result.confidence_lower[:, feature_idx][::-1]
            ]),
            fill='toself',
            fillcolor='rgba(99, 110, 250, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=True,
            hoverinfo='skip'
        ))

        # Predictions
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=result.predictions[:, feature_idx],
            mode='lines',
            name='Prediction',
            line=dict(color=colors['benign'], width=2)
        ))

        # Actual values
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=result.actuals[:, feature_idx],
            mode='lines',
            name='Actual',
            line=dict(color=colors['attack'], width=2, dash='dot')
        ))

        # Highlight anomalous regions
        anomaly_indices = result.get_anomalous_timesteps()
        if len(anomaly_indices) > 0:
            for idx in anomaly_indices:
                fig.add_vrect(
                    x0=max(0, idx - 0.5),
                    x1=min(result.seq_length - 1, idx + 0.5),
                    fillcolor="red",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )

        fig.update_layout(
            template=theme,
            title=title or f"Prediction vs Actual: {feature_names[feature_idx]}",
            xaxis_title="Time Step",
            yaxis_title="Value",
            hovermode='x unified',
            height=400
        )

    else:
        # Multi-feature subplot
        n_features = result.n_features
        rows = (n_features + 2) // 3  # 3 columns
        cols = min(3, n_features)

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=feature_names,
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )

        for i in range(n_features):
            row = i // 3 + 1
            col = i % 3 + 1

            # Predictions
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=result.predictions[:, i],
                    mode='lines',
                    name='Prediction' if i == 0 else None,
                    line=dict(color=colors['benign'], width=1),
                    showlegend=(i == 0),
                    legendgroup='pred'
                ),
                row=row,
                col=col
            )

            # Actuals
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=result.actuals[:, i],
                    mode='lines',
                    name='Actual' if i == 0 else None,
                    line=dict(color=colors['attack'], width=1, dash='dot'),
                    showlegend=(i == 0),
                    legendgroup='actual'
                ),
                row=row,
                col=col
            )

        fig.update_layout(
            template=theme,
            title=title or "Predictions vs Actuals (All Features)",
            height=300 * rows,
            showlegend=True
        )

    return fig


def plot_anomaly_scores(
    result: AnomalyResult,
    theme: str = "plotly_dark",
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None
) -> go.Figure:
    """
    Plot anomaly scores over time with threshold line.

    Args:
        result: AnomalyResult object
        theme: Plotly theme
        title: Custom title
        colors: Color dictionary

    Returns:
        Plotly figure object
    """
    if colors is None:
        colors = {
            'benign': '#00CC96',
            'attack': '#EF553B',
            'detected': '#636EFA'
        }

    time_steps = np.arange(result.seq_length)

    fig = go.Figure()

    # Anomaly scores
    score_colors = np.where(result.is_anomaly, colors['attack'], colors['benign'])

    fig.add_trace(go.Scatter(
        x=time_steps,
        y=result.anomaly_scores,
        mode='lines+markers',
        name='Anomaly Score',
        line=dict(color=colors['detected'], width=2),
        marker=dict(
            size=4,
            color=score_colors,
            line=dict(width=0)
        ),
        fill='tozeroy',
        fillcolor=f'rgba(99, 110, 250, 0.1)'
    ))

    # Threshold line
    fig.add_hline(
        y=result.threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({result.threshold:.2f})",
        annotation_position="right"
    )

    # Highlight anomalous regions
    anomaly_indices = result.get_anomalous_timesteps()
    if len(anomaly_indices) > 0:
        for idx in anomaly_indices:
            fig.add_vrect(
                x0=max(0, idx - 0.5),
                x1=min(result.seq_length - 1, idx + 0.5),
                fillcolor="red",
                opacity=0.15,
                layer="below",
                line_width=0,
            )

    fig.update_layout(
        template=theme,
        title=title or f"Anomaly Scores Over Time ({result.n_anomalies} detected)",
        xaxis_title="Time Step",
        yaxis_title="Anomaly Score",
        yaxis=dict(range=[0, 1.1]),
        hovermode='x unified',
        height=400
    )

    return fig


def plot_detection_metrics(
    results: List[AnomalyResult],
    labels: Optional[List[str]] = None,
    ground_truth: Optional[List[np.ndarray]] = None,
    theme: str = "plotly_dark",
    colors: Optional[Dict[str, str]] = None
) -> go.Figure:
    """
    Plot detection metrics summary.

    Args:
        results: List of AnomalyResult objects
        labels: Labels for each result (e.g., sample names)
        ground_truth: Optional ground truth anomaly flags for accuracy metrics
        theme: Plotly theme
        colors: Color dictionary

    Returns:
        Plotly figure object
    """
    if colors is None:
        colors = {
            'benign': '#00CC96',
            'attack': '#EF553B',
            'detected': '#636EFA'
        }

    if labels is None:
        labels = [f"Sample {i+1}" for i in range(len(results))]

    # Compute metrics
    anomaly_rates = [r.anomaly_rate for r in results]
    mean_scores = [r.anomaly_scores.mean() for r in results]
    max_scores = [r.anomaly_scores.max() for r in results]

    if ground_truth is not None:
        # Compute accuracy metrics
        accuracy = []
        precision = []
        recall = []
        f1 = []

        for result, gt in zip(results, ground_truth):
            pred = result.is_anomaly
            tp = np.sum(pred & gt)
            fp = np.sum(pred & ~gt)
            fn = np.sum(~pred & gt)
            tn = np.sum(~pred & ~gt)

            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            accuracy.append(acc)
            precision.append(prec)
            recall.append(rec)
            f1.append(f1_score)

        # Create subplot with metrics
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=["Accuracy", "Precision", "Recall", "F1 Score"],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        fig.add_trace(go.Bar(x=labels, y=accuracy, name="Accuracy", marker_color=colors['benign']), row=1, col=1)
        fig.add_trace(go.Bar(x=labels, y=precision, name="Precision", marker_color=colors['detected']), row=1, col=2)
        fig.add_trace(go.Bar(x=labels, y=recall, name="Recall", marker_color=colors['attack']), row=2, col=1)
        fig.add_trace(go.Bar(x=labels, y=f1, name="F1", marker_color='purple'), row=2, col=2)

        fig.update_yaxes(range=[0, 1.1])
        fig.update_layout(
            template=theme,
            title="Detection Performance Metrics",
            showlegend=False,
            height=600
        )

    else:
        # Simple metrics without ground truth
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=labels,
            y=anomaly_rates,
            name='Anomaly Rate',
            marker_color=colors['attack'],
            text=[f"{r:.1%}" for r in anomaly_rates],
            textposition='auto'
        ))

        fig.add_trace(go.Scatter(
            x=labels,
            y=mean_scores,
            mode='lines+markers',
            name='Mean Score',
            yaxis='y2',
            marker=dict(size=10, color=colors['detected']),
            line=dict(width=2)
        ))

        fig.update_layout(
            template=theme,
            title="Anomaly Detection Summary",
            xaxis_title="Sample",
            yaxis_title="Anomaly Rate",
            yaxis2=dict(
                title="Mean Anomaly Score",
                overlaying='y',
                side='right',
                range=[0, 1]
            ),
            hovermode='x unified',
            height=400
        )

    return fig


def plot_feature_contributions(
    result: AnomalyResult,
    feature_names: Optional[List[str]] = None,
    timesteps: Optional[List[int]] = None,
    theme: str = "plotly_dark",
    colors: Optional[Dict[str, str]] = None
) -> go.Figure:
    """
    Plot feature contributions to anomaly detection.

    Args:
        result: AnomalyResult object with feature_contributions
        feature_names: Names for features
        timesteps: Specific timesteps to show (if None, shows anomalous timesteps)
        theme: Plotly theme
        colors: Color dictionary

    Returns:
        Plotly figure object
    """
    if result.feature_contributions is None:
        raise ValueError("Feature contributions not available in result")

    if feature_names is None:
        feature_names = [f"F{i}" for i in range(result.n_features)]

    if timesteps is None:
        # Show top anomalous timesteps
        anomalous_timesteps = result.get_anomalous_timesteps()
        if len(anomalous_timesteps) == 0:
            # If no anomalies, show top 5 highest scoring timesteps
            timesteps = np.argsort(result.anomaly_scores)[-5:][::-1]
        else:
            # Show up to 10 anomalous timesteps
            timesteps = anomalous_timesteps[:10]

    # Extract contributions for selected timesteps
    contributions = result.feature_contributions[timesteps]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=contributions.T,
        x=[f"t={t}" for t in timesteps],
        y=feature_names,
        colorscale='Reds',
        colorbar=dict(title="Contribution"),
        hovertemplate='Timestep: %{x}<br>Feature: %{y}<br>Contribution: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        template=theme,
        title="Feature Contributions to Anomaly Detection",
        xaxis_title="Timestep",
        yaxis_title="Feature",
        height=400
    )

    return fig
