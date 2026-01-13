"""Reusable plot components for the Streamlit app.

This module contains two categories of visualization functions:
1. IDS Baseline visualization wrappers (display_* functions) - from main branch
2. Moirai-specific visualizations (moirai_plot_* functions) - from main branch
3. General-purpose visualization components (plot_* functions) - new in this PR
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Optional, Union, Any
import sys
from pathlib import Path

# Add src to path for imports
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from src.models import AnomalyResult
    from src.visualization import ids_plots
    IDS_PLOTS_AVAILABLE = True
except ImportError:
    IDS_PLOTS_AVAILABLE = False


# ============================================================================
# IDS Baseline Visualization Functions (from main branch)
# ============================================================================

def display_confusion_matrix(y_true, y_pred, method_name="IDS"):
    """
    Display confusion matrix plot in Streamlit.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        method_name: Name of the IDS method
    """
    if not IDS_PLOTS_AVAILABLE:
        st.error("IDS plots module not available")
        return
    fig = ids_plots.plot_confusion_matrix(y_true, y_pred, method_name)
    st.plotly_chart(fig, use_container_width=True)


def display_roc_curves(methods_results, y_test):
    """
    Display ROC curves for multiple methods.

    Args:
        methods_results: Dictionary mapping method name to results dict
        y_test: True labels
    """
    if not IDS_PLOTS_AVAILABLE:
        st.error("IDS plots module not available")
        return
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
    if not IDS_PLOTS_AVAILABLE:
        st.error("IDS plots module not available")
        return
    fig = ids_plots.plot_detection_results(X_sequences, y_true, y_pred, feature_idx, n_show)
    st.plotly_chart(fig, use_container_width=True)


def display_method_comparison(methods_results):
    """
    Display bar chart comparing methods.

    Args:
        methods_results: Dictionary mapping method name to metrics dict
    """
    if not IDS_PLOTS_AVAILABLE:
        st.error("IDS plots module not available")
        return
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
    if not IDS_PLOTS_AVAILABLE:
        st.error("IDS plots module not available")
        return
    fig = ids_plots.plot_score_distribution(y_scores, y_true, method_name)
    st.plotly_chart(fig, use_container_width=True)


def display_metrics_radar(methods_results):
    """
    Display radar chart of metrics.

    Args:
        methods_results: Dictionary mapping method name to metrics dict
    """
    if not IDS_PLOTS_AVAILABLE:
        st.error("IDS plots module not available")
        return
    fig = ids_plots.plot_metrics_radar(methods_results)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Color Palette Constants for General Visualizations
# ============================================================================

# Color palette constants
COLOR_BENIGN = "#00CC96"  # Green for benign traffic
COLOR_ATTACK = "#EF553B"  # Red for attack traffic
COLOR_DETECTED = "#636EFA"  # Blue for detection results
COLOR_BASELINE = "#AB63FA"  # Purple for baseline methods
COLOR_THRESHOLD = "#FFA15A"  # Orange for thresholds

# Default feature names for IoT traffic
DEFAULT_FEATURE_NAMES = [
    'flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot',
    'fwd_data_pkts_tot', 'bwd_data_pkts_tot',
    'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'flow_pkts_per_sec',
    'fwd_byts_b_avg', 'bwd_byts_b_avg',
    'fwd_iat_mean', 'bwd_iat_mean'
]


def _get_theme_colors() -> Dict[str, str]:
    """
    Get theme colors from session state config.

    Returns:
        Dict with color configuration
    """
    if "config" in st.session_state:
        config = st.session_state.config
        colors = config.get("visualization.colors", {})
        return {
            'benign': colors.get('benign', COLOR_BENIGN),
            'attack': colors.get('attack', COLOR_ATTACK),
            'detected': colors.get('detected', COLOR_DETECTED),
        }
    return {
        'benign': COLOR_BENIGN,
        'attack': COLOR_ATTACK,
        'detected': COLOR_DETECTED,
    }


def _apply_dark_theme(fig: go.Figure, title: str = None) -> go.Figure:
    """
    Apply consistent dark theme to a Plotly figure.

    Args:
        fig: Plotly figure object
        title: Optional title for the figure

    Returns:
        Modified figure with dark theme applied
    """
    fig.update_layout(
        template="plotly_dark",
        title=title,
        title_font_size=16,
        font=dict(family="Arial, sans-serif", size=12),
        hovermode='closest',
        plot_bgcolor='#111111',
        paper_bgcolor='#111111',
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def plot_time_series(
    data: np.ndarray,
    label: str = "Traffic",
    features: Optional[List[str]] = None,
    show_all: bool = False,
    is_attack: bool = False
) -> go.Figure:
    """
    Plot time-series traffic data with multi-feature support.

    Args:
        data: Time-series array of shape (seq_length, n_features)
        label: Label for the traffic (e.g., "Benign", "DDoS Attack")
        features: List of feature names (default: DEFAULT_FEATURE_NAMES)
        show_all: If True, show all features with dropdown; if False, show first 3
        is_attack: If True, use attack color; otherwise use benign color

    Returns:
        Plotly Figure object

    Example:
        >>> data = np.random.randn(128, 12)
        >>> fig = plot_time_series(data, label="Benign Traffic", is_attack=False)
        >>> st.plotly_chart(fig)
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array (seq_length, n_features), got shape {data.shape}")

    seq_length, n_features = data.shape

    if features is None:
        features = DEFAULT_FEATURE_NAMES[:n_features]
    elif len(features) != n_features:
        features = features[:n_features] + [f"Feature {i}" for i in range(len(features), n_features)]

    colors = _get_theme_colors()
    line_color = colors['attack'] if is_attack else colors['benign']

    if show_all:
        # Create subplot for each feature with dropdown
        fig = go.Figure()

        # Add all features as separate traces
        for i, feature_name in enumerate(features):
            visible = (i == 0)  # Only first feature visible initially
            fig.add_trace(go.Scatter(
                x=np.arange(seq_length),
                y=data[:, i],
                mode='lines',
                name=feature_name,
                line=dict(color=line_color, width=2),
                visible=visible,
                hovertemplate=f'<b>{feature_name}</b><br>Time: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>'
            ))

        # Create dropdown menu
        buttons = []
        for i, feature_name in enumerate(features):
            visible_array = [False] * len(features)
            visible_array[i] = True
            buttons.append(
                dict(
                    label=feature_name,
                    method="update",
                    args=[{"visible": visible_array},
                          {"title": f"{label} - {feature_name}"}]
                )
            )

        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ],
            xaxis_title="Time Step",
            yaxis_title="Normalized Value",
        )

        _apply_dark_theme(fig, title=f"{label} - {features[0]}")

    else:
        # Show first 3 features in subplots
        n_plots = min(3, n_features)
        fig = make_subplots(
            rows=n_plots,
            cols=1,
            subplot_titles=[features[i] for i in range(n_plots)],
            vertical_spacing=0.12
        )

        for i in range(n_plots):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(seq_length),
                    y=data[:, i],
                    mode='lines',
                    name=features[i],
                    line=dict(color=line_color, width=2),
                    hovertemplate=f'<b>{features[i]}</b><br>Time: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>',
                    showlegend=False
                ),
                row=i+1,
                col=1
            )

        fig.update_xaxes(title_text="Time Step", row=n_plots, col=1)
        fig.update_yaxes(title_text="Value")

        _apply_dark_theme(fig, title=label)

    return fig


def plot_comparison_grid(
    samples: List[np.ndarray],
    labels: Optional[List[str]] = None,
    true_labels: Optional[List[bool]] = None,
    reveal: bool = False,
    feature_indices: Optional[List[int]] = None
) -> go.Figure:
    """
    Create a grid comparison of multiple traffic samples (for "Spot the Attack" challenge).

    Args:
        samples: List of time-series arrays, each of shape (seq_length, n_features)
        labels: Optional list of labels for each sample (e.g., ["Sample 1", "Sample 2"])
        true_labels: Optional list of booleans (True=attack, False=benign)
        reveal: If True, color-code borders based on true_labels
        feature_indices: Which features to plot (default: [5, 6, 8] - packet rates and bytes)

    Returns:
        Plotly Figure object with grid layout

    Example:
        >>> samples = [np.random.randn(128, 12) for _ in range(6)]
        >>> true_labels = [False, True, False, True, False, False]
        >>> fig = plot_comparison_grid(samples, true_labels=true_labels, reveal=True)
    """
    n_samples = len(samples)

    if n_samples > 9:
        raise ValueError(f"Maximum 9 samples supported, got {n_samples}")

    if labels is None:
        labels = [f"Sample {i+1}" for i in range(n_samples)]

    if feature_indices is None:
        feature_indices = [5, 6, 8]  # packet rates and bytes

    # Determine grid layout
    if n_samples <= 3:
        rows, cols = 1, n_samples
    elif n_samples <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3

    colors = _get_theme_colors()

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=labels[:n_samples],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
        shared_xaxes=True,
        shared_yaxes=False
    )

    for idx, sample in enumerate(samples):
        row = idx // cols + 1
        col = idx % cols + 1

        # Determine color
        if reveal and true_labels is not None and idx < len(true_labels):
            color = colors['attack'] if true_labels[idx] else colors['benign']
        else:
            color = '#FFFFFF'  # Neutral white

        # Plot selected features as overlaid lines
        for feat_idx in feature_indices:
            if feat_idx < sample.shape[1]:
                feature_name = DEFAULT_FEATURE_NAMES[feat_idx] if feat_idx < len(DEFAULT_FEATURE_NAMES) else f"F{feat_idx}"
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(sample.shape[0]),
                        y=sample[:, feat_idx],
                        mode='lines',
                        name=feature_name,
                        line=dict(color=color, width=1.5),
                        opacity=0.8,
                        showlegend=(idx == 0),  # Only show legend for first sample
                        hovertemplate=f'<b>{feature_name}</b><br>%{{y:.4f}}<extra></extra>'
                    ),
                    row=row,
                    col=col
                )

    # Update layout
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=True, nticks=4)

    title_text = "Traffic Pattern Comparison"
    if reveal:
        title_text += f" - <span style='color:{colors['benign']}'>Benign</span> vs <span style='color:{colors['attack']}'>Attack</span>"

    _apply_dark_theme(fig, title=title_text)
    fig.update_layout(height=200 * rows + 100)

    return fig


def plot_anomaly_scores(
    scores: np.ndarray,
    threshold: float,
    labels: Optional[List[str]] = None,
    ground_truth: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Visualize anomaly scores with threshold line and optional classification markers.

    Args:
        scores: Array of anomaly scores
        threshold: Detection threshold
        labels: Optional list of sample labels
        ground_truth: Optional array of true labels (0=benign, 1=attack) for TP/FP/TN/FN markers

    Returns:
        Plotly Figure object

    Example:
        >>> scores = np.array([0.3, 0.8, 0.4, 0.9, 0.2, 0.7])
        >>> ground_truth = np.array([0, 1, 0, 1, 0, 1])
        >>> fig = plot_anomaly_scores(scores, threshold=0.5, ground_truth=ground_truth)
    """
    n_samples = len(scores)

    if labels is None:
        labels = [f"Sample {i+1}" for i in range(n_samples)]

    colors = _get_theme_colors()

    # Determine bar colors based on score vs threshold
    bar_colors = [colors['attack'] if score >= threshold else colors['benign'] for score in scores]

    # Create figure
    fig = go.Figure()

    # Add bars
    hover_text = []
    for i, (score, label) in enumerate(zip(scores, labels)):
        predicted = "Attack" if score >= threshold else "Benign"
        text = f"<b>{label}</b><br>Score: {score:.4f}<br>Predicted: {predicted}"

        if ground_truth is not None:
            actual = "Attack" if ground_truth[i] == 1 else "Benign"
            if ground_truth[i] == 1 and score >= threshold:
                classification = "✓ True Positive"
            elif ground_truth[i] == 0 and score < threshold:
                classification = "✓ True Negative"
            elif ground_truth[i] == 1 and score < threshold:
                classification = "✗ False Negative"
            else:
                classification = "✗ False Positive"
            text += f"<br>Actual: {actual}<br>{classification}"

        hover_text.append(text)

    fig.add_trace(go.Bar(
        x=labels,
        y=scores,
        marker=dict(color=bar_colors),
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=hover_text,
        showlegend=False
    ))

    # Add threshold line
    fig.add_hline(
        y=threshold,
        line=dict(color=COLOR_THRESHOLD, width=2, dash='dash'),
        annotation_text=f"Threshold = {threshold:.3f}",
        annotation_position="right"
    )

    # Update layout
    fig.update_layout(
        xaxis_title="Samples",
        yaxis_title="Anomaly Score",
        xaxis=dict(tickangle=-45) if n_samples > 10 else dict()
    )

    _apply_dark_theme(fig, title="Anomaly Detection Scores")

    return fig


def plot_detection_comparison(
    baseline_results: Dict[str, int],
    moirai_results: Dict[str, int]
) -> go.Figure:
    """
    Compare detection results between baseline and Moirai systems.

    Args:
        baseline_results: Dict with keys 'TP', 'FP', 'TN', 'FN' for baseline method
        moirai_results: Dict with keys 'TP', 'FP', 'TN', 'FN' for Moirai method

    Returns:
        Plotly Figure object

    Example:
        >>> baseline = {'TP': 45, 'FP': 12, 'TN': 38, 'FN': 5}
        >>> moirai = {'TP': 48, 'FP': 3, 'TN': 47, 'FN': 2}
        >>> fig = plot_detection_comparison(baseline, moirai)
    """
    categories = ['True Positives', 'False Positives', 'True Negatives', 'False Negatives']
    keys = ['TP', 'FP', 'TN', 'FN']

    baseline_values = [baseline_results.get(k, 0) for k in keys]
    moirai_values = [moirai_results.get(k, 0) for k in keys]

    colors = _get_theme_colors()

    fig = go.Figure()

    # Add baseline bars
    fig.add_trace(go.Bar(
        name='Baseline IDS',
        x=categories,
        y=baseline_values,
        marker_color=COLOR_BASELINE,
        text=baseline_values,
        textposition='auto',
        hovertemplate='<b>Baseline IDS</b><br>%{x}: %{y}<extra></extra>'
    ))

    # Add Moirai bars
    fig.add_trace(go.Bar(
        name='Moirai',
        x=categories,
        y=moirai_values,
        marker_color=colors['detected'],
        text=moirai_values,
        textposition='auto',
        hovertemplate='<b>Moirai</b><br>%{x}: %{y}<extra></extra>'
    ))

    # Calculate metrics
    def calc_metrics(results):
        tp = results.get('TP', 0)
        fp = results.get('FP', 0)
        tn = results.get('TN', 0)
        fn = results.get('FN', 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    baseline_p, baseline_r, baseline_f1 = calc_metrics(baseline_results)
    moirai_p, moirai_r, moirai_f1 = calc_metrics(moirai_results)

    # Add metrics as annotations
    annotation_text = (
        f"<b>Baseline:</b> P={baseline_p:.3f}, R={baseline_r:.3f}, F1={baseline_f1:.3f}<br>"
        f"<b>Moirai:</b> P={moirai_p:.3f}, R={moirai_r:.3f}, F1={moirai_f1:.3f}"
    )

    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=0.5, y=1.15,
        showarrow=False,
        font=dict(size=12),
        align="center"
    )

    # Update layout
    fig.update_layout(
        barmode='group',
        xaxis_title="Detection Category",
        yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    _apply_dark_theme(fig, title="Detection Performance Comparison")

    return fig


def plot_prediction_vs_actual(
    actual: np.ndarray,
    predicted: np.ndarray,
    confidence_lower: Optional[np.ndarray] = None,
    confidence_upper: Optional[np.ndarray] = None,
    feature_idx: int = 0,
    feature_name: Optional[str] = None
) -> go.Figure:
    """
    Plot Moirai predictions vs actual values with optional confidence intervals.

    Args:
        actual: Actual values array
        predicted: Predicted values array
        confidence_lower: Optional lower bound of confidence interval
        confidence_upper: Optional upper bound of confidence interval
        feature_idx: Which feature to plot (if data is multi-dimensional)
        feature_name: Name of the feature being plotted

    Returns:
        Plotly Figure object

    Example:
        >>> actual = np.sin(np.linspace(0, 4*np.pi, 100))
        >>> predicted = actual + np.random.randn(100) * 0.1
        >>> ci_lower = predicted - 0.2
        >>> ci_upper = predicted + 0.2
        >>> fig = plot_prediction_vs_actual(actual, predicted, ci_lower, ci_upper)
    """
    colors = _get_theme_colors()

    if feature_name is None:
        feature_name = DEFAULT_FEATURE_NAMES[feature_idx] if feature_idx < len(DEFAULT_FEATURE_NAMES) else f"Feature {feature_idx}"

    time_steps = np.arange(len(actual))

    fig = go.Figure()

    # Add confidence interval as shaded area
    if confidence_lower is not None and confidence_upper is not None:
        fig.add_trace(go.Scatter(
            x=np.concatenate([time_steps, time_steps[::-1]]),
            y=np.concatenate([confidence_upper, confidence_lower[::-1]]),
            fill='toself',
            fillcolor='rgba(99, 110, 250, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI',
            showlegend=True,
            hoverinfo='skip'
        ))

    # Add predicted line
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=predicted,
        mode='lines',
        name='Predicted',
        line=dict(color=colors['detected'], width=2, dash='dash'),
        hovertemplate='<b>Predicted</b><br>Time: %{x}<br>Value: %{y:.4f}<extra></extra>'
    ))

    # Add actual line
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(color='#FFFFFF', width=2),
        hovertemplate='<b>Actual</b><br>Time: %{x}<br>Value: %{y:.4f}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        xaxis_title="Time Step",
        yaxis_title=f"{feature_name} Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )

    _apply_dark_theme(fig, title=f"Moirai Prediction: {feature_name}")

    return fig


def plot_decomposition(
    decomposition: Dict[str, np.ndarray],
    feature_idx: int = 0,
    feature_name: Optional[str] = None
) -> go.Figure:
    """
    Visualize time-series decomposition (trend, seasonality, residual).

    Args:
        decomposition: Dict with keys 'original', 'trend', 'seasonality', 'residual'
        feature_idx: Which feature to plot (for multi-feature data)
        feature_name: Name of the feature being plotted

    Returns:
        Plotly Figure object with 4 stacked subplots

    Example:
        >>> from src.models.diffusion_ts import IoTDiffusionGenerator
        >>> generator = IoTDiffusionGenerator()
        >>> generator.initialize()
        >>> sample = np.random.randn(128, 12)
        >>> decomp = generator.get_decomposition(sample)
        >>> fig = plot_decomposition(decomp, feature_idx=0)
    """
    if feature_name is None:
        feature_name = DEFAULT_FEATURE_NAMES[feature_idx] if feature_idx < len(DEFAULT_FEATURE_NAMES) else f"Feature {feature_idx}"

    # Extract components
    components = {
        'Original': (decomposition.get('original', decomposition.get('trend')), '#FFFFFF'),
        'Trend': (decomposition.get('trend'), '#FFD700'),  # Yellow
        'Seasonality': (decomposition.get('seasonality'), '#00CED1'),  # Cyan
        'Residual': (decomposition.get('residual'), '#FF00FF')  # Magenta
    }

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=list(components.keys()),
        vertical_spacing=0.08,
        shared_xaxes=True
    )

    for i, (name, (data, color)) in enumerate(components.items()):
        if data is not None:
            # Extract single feature if multi-dimensional
            if data.ndim == 2:
                data = data[:, feature_idx]

            time_steps = np.arange(len(data))

            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=data,
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=1.5),
                    hovertemplate=f'<b>{name}</b><br>Time: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>',
                    showlegend=False
                ),
                row=i+1,
                col=1
            )

    # Update axes
    fig.update_xaxes(title_text="Time Step", row=4, col=1)
    fig.update_yaxes(title_text="Value")

    _apply_dark_theme(fig, title=f"Time-Series Decomposition: {feature_name}")
    fig.update_layout(height=800)

    return fig


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]]
) -> go.Figure:
    """
    Compare detection metrics across multiple methods.

    Args:
        metrics_dict: Nested dict with format:
            {
                "Method Name": {
                    "precision": 0.85,
                    "recall": 0.90,
                    "f1": 0.87,
                    "accuracy": 0.88
                },
                ...
            }

    Returns:
        Plotly Figure object

    Example:
        >>> metrics = {
        ...     "Baseline IDS": {"precision": 0.65, "recall": 0.70, "f1": 0.67, "accuracy": 0.72},
        ...     "Moirai": {"precision": 0.92, "recall": 0.95, "f1": 0.93, "accuracy": 0.94}
        ... }
        >>> fig = plot_metrics_comparison(metrics)
    """
    metric_names = ['Precision', 'Recall', 'F1', 'Accuracy']
    metric_keys = ['precision', 'recall', 'f1', 'accuracy']

    colors_palette = [COLOR_BASELINE, COLOR_DETECTED, '#00CC96', '#AB63FA', '#FFA15A']

    fig = go.Figure()

    for idx, (method_name, metrics) in enumerate(metrics_dict.items()):
        values = [metrics.get(key.lower(), 0) for key in metric_keys]

        fig.add_trace(go.Bar(
            name=method_name,
            x=metric_names,
            y=values,
            marker_color=colors_palette[idx % len(colors_palette)],
            text=[f"{v:.3f}" for v in values],
            textposition='auto',
            hovertemplate=f'<b>{method_name}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>'
        ))

    # Add reference line at 0.5
    fig.add_hline(
        y=0.5,
        line=dict(color='gray', width=1, dash='dot'),
        annotation_text="Baseline = 0.5",
        annotation_position="right"
    )

    # Update layout
    fig.update_layout(
        barmode='group',
        xaxis_title="Metric",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    _apply_dark_theme(fig, title="Detection Metrics Comparison")

    return fig


def plot_diffusion_animation(
    diffusion_steps: List[np.ndarray],
    feature_idx: int = 0,
    feature_name: Optional[str] = None
) -> go.Figure:
    """
    Create an animated visualization of the diffusion denoising process (optional).

    Args:
        diffusion_steps: List of arrays showing progression from noise to data
        feature_idx: Which feature to visualize
        feature_name: Name of the feature

    Returns:
        Plotly Figure object with animation

    Example:
        >>> steps = [np.random.randn(128, 12) * (1 - i/50) for i in range(50)]
        >>> fig = plot_diffusion_animation(steps, feature_idx=0)
    """
    if feature_name is None:
        feature_name = DEFAULT_FEATURE_NAMES[feature_idx] if feature_idx < len(DEFAULT_FEATURE_NAMES) else f"Feature {feature_idx}"

    n_steps = len(diffusion_steps)

    # Create frames
    frames = []
    for step_idx, step_data in enumerate(diffusion_steps):
        if step_data.ndim == 2:
            data = step_data[:, feature_idx]
        else:
            data = step_data

        frame = go.Frame(
            data=[go.Scatter(
                x=np.arange(len(data)),
                y=data,
                mode='lines',
                line=dict(color=COLOR_DETECTED, width=2),
                name=f"Step {step_idx}"
            )],
            name=str(step_idx)
        )
        frames.append(frame)

    # Create initial figure
    initial_data = diffusion_steps[0]
    if initial_data.ndim == 2:
        initial_data = initial_data[:, feature_idx]

    fig = go.Figure(
        data=[go.Scatter(
            x=np.arange(len(initial_data)),
            y=initial_data,
            mode='lines',
            line=dict(color=COLOR_DETECTED, width=2)
        )],
        frames=frames
    )

    # Add animation controls
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 100, "redraw": True},
                                     "fromcurrent": True}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate",
                                       "transition": {"duration": 0}}])
                ],
                x=0.1,
                y=1.15,
                xanchor="left",
                yanchor="top"
            )
        ],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "y": -0.1,
            "xanchor": "left",
            "currentvalue": {
                "prefix": "Diffusion Step: ",
                "visible": True,
                "xanchor": "right"
            },
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "steps": [
                {
                    "args": [[f.name], {"frame": {"duration": 100, "redraw": True},
                                       "mode": "immediate"}],
                    "label": str(k),
                    "method": "animate"
                }
                for k, f in enumerate(frames)
            ]
        }],
        xaxis_title="Time Step",
        yaxis_title=f"{feature_name} Value"
    )

    _apply_dark_theme(fig, title=f"Diffusion Process: {feature_name}")
    fig.update_layout(height=600)

    return fig


# ============================================================================
# NOTE: Moirai-Specific Functions from Main Branch
# ============================================================================
# The main branch contains additional Moirai-specific visualization functions
# that work with AnomalyResult objects:
# - plot_prediction_vs_actual(result: AnomalyResult, ...) - renamed to avoid conflict
# - plot_anomaly_scores(result: AnomalyResult, ...) - renamed to avoid conflict  
# - plot_detection_metrics(result: AnomalyResult, ...)
# - plot_feature_contributions(result: AnomalyResult, ...)
#
# These functions are specific to the Moirai detection pipeline and work with
# the AnomalyResult data structure. They are wrapper functions that call into
# src.visualization module.
#
# The general-purpose visualization functions in this PR (plot_time_series,
# plot_comparison_grid, etc.) work with raw numpy arrays and are more flexible
# for use across different pages and contexts.
#
# If you need the Moirai-specific functions, they can be imported from:
# from src.visualization import moirai_plots
# ============================================================================
