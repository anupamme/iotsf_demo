"""Detection Results Page - Moirai anomaly detection"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np

# Add src to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models import MoiraiAnomalyDetector
from src.utils.config import Config
from app.components.plots import (
    plot_prediction_vs_actual,
    plot_anomaly_scores,
    plot_detection_metrics,
    plot_feature_contributions
)

st.set_page_config(
    page_title="Detection - IoT Security Demo",
    page_icon="🔒",
    layout="wide",
)

st.title("🤖 Moirai Detection Results")
st.markdown(
    "Moirai is a time-series foundation model that uses probabilistic forecasting "
    "to detect anomalies. If observed values fall outside the predicted confidence "
    "interval, they are flagged as potential attacks."
)

# Initialize config
if "config" not in st.session_state:
    st.session_state.config = Config()

config = st.session_state.config

# Feature names for IoT traffic
FEATURE_NAMES = [
    'flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot',
    'fwd_data_pkts_tot', 'bwd_data_pkts_tot',
    'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'flow_pkts_per_sec',
    'fwd_byts_b_avg', 'bwd_byts_b_avg',
    'fwd_iat_mean', 'bwd_iat_mean'
]

# Get color scheme from config
COLORS = {
    'benign': config.get('visualization.colors.benign', '#00CC96'),
    'attack': config.get('visualization.colors.attack', '#EF553B'),
    'detected': config.get('visualization.colors.detected', '#636EFA')
}

THEME = config.get('visualization.theme', 'plotly_dark')


@st.cache_resource
def load_detector(model_size: str):
    """Load and initialize the Moirai detector (cached)."""
    detector = MoiraiAnomalyDetector(
        model_size=model_size,
        context_length=config.get('models.moirai.context_length', 512),
        prediction_length=config.get('models.moirai.prediction_length', 64),
        confidence_level=config.get('models.moirai.confidence_level', 0.95)
    )
    detector.initialize()
    return detector


@st.cache_data
def load_synthetic_samples():
    """Load synthetic attack samples."""
    synthetic_dir = Path(config.get('data.synthetic_dir', 'data/synthetic'))

    samples = {}
    sample_types = []

    # Load benign
    benign_path = synthetic_dir / 'benign_samples.npy'
    if benign_path.exists():
        samples['benign'] = np.load(benign_path)
        sample_types.append('benign')

    # Load attacks
    attack_patterns = ['slow_exfiltration', 'lotl_mimicry', 'protocol_anomaly', 'beacon']
    stealth_levels = [85, 90, 95]

    for pattern in attack_patterns:
        for stealth in stealth_levels:
            key = f"{pattern}_stealth_{stealth}"
            path = synthetic_dir / f"{key}.npy"
            if path.exists():
                samples[key] = np.load(path)
                sample_types.append(key)

    return samples, sample_types


# Sidebar controls
with st.sidebar:
    st.header("Detection Settings")

    # Model selection
    model_size = st.selectbox(
        "Model Size",
        options=['small', 'base', 'large'],
        index=0,
        help="Small: faster, Base: balanced, Large: best accuracy"
    )

    # Threshold slider
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.80,
        max_value=0.99,
        value=float(config.get('models.moirai.anomaly_threshold', 0.95)),
        step=0.01,
        help="Higher threshold = fewer false positives, more false negatives"
    )

    st.divider()

    # Sample selection
    st.subheader("Sample Selection")

    # Load samples
    try:
        samples, sample_types = load_synthetic_samples()

        if len(samples) == 0:
            st.warning("No synthetic samples found. Run: `python scripts/precompute_attacks.py`")
            st.stop()

        selected_sample = st.selectbox(
            "Choose Sample",
            options=sample_types,
            format_func=lambda x: x.replace('_', ' ').title()
        )

        # Sample info
        sample_data = samples[selected_sample]
        st.info(f"""
        **Sample Info:**
        - Type: {selected_sample}
        - Sequences: {len(sample_data)}
        - Length: {sample_data.shape[1]}
        - Features: {sample_data.shape[2]}
        """)

        # Select specific sequence from batch
        seq_idx = st.slider(
            "Sequence Index",
            min_value=0,
            max_value=len(sample_data) - 1,
            value=0,
            help="Select which sequence from the batch to analyze"
        )

    except Exception as e:
        st.error(f"Error loading samples: {e}")
        st.stop()

# Load detector
try:
    with st.spinner(f"Loading Moirai {model_size} model..."):
        detector = load_detector(model_size)

    is_mock = detector._mock_mode
    if is_mock:
        st.info("ℹ️ Running in **mock mode** (uni2ts not available). Install with Python 3.12 for full functionality.")
    else:
        st.success(f"✅ Moirai {model_size} model loaded successfully!")

except Exception as e:
    st.error(f"Failed to load detector: {e}")
    st.stop()

# Main content
tabs = st.tabs([
    "📈 Prediction vs Actual",
    "🎯 Anomaly Scores",
    "📊 Metrics Summary",
    "🔍 Feature Analysis"
])

# Get selected sequence
traffic = sample_data[seq_idx]

# Run detection
with st.spinner("Running anomaly detection..."):
    try:
        result = detector.detect_anomalies(
            traffic,
            threshold=threshold,
            return_feature_contributions=True
        )
    except Exception as e:
        st.error(f"Detection failed: {e}")
        st.stop()

# Display results summary
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Anomalies Detected", result.n_anomalies)
with col2:
    st.metric("Anomaly Rate", f"{result.anomaly_rate:.1%}")
with col3:
    st.metric("Mean Score", f"{result.anomaly_scores.mean():.3f}")
with col4:
    st.metric("Max Score", f"{result.anomaly_scores.max():.3f}")

st.divider()

# Tab 1: Prediction vs Actual
with tabs[0]:
    st.subheader("Predictions vs Actual Values")
    st.markdown(
        "Moirai forecasts future values based on historical context. "
        "Shaded areas show confidence intervals. Red highlights indicate detected anomalies."
    )

    # Feature selector
    view_mode = st.radio(
        "View Mode",
        options=["Single Feature", "All Features"],
        horizontal=True
    )

    if view_mode == "Single Feature":
        feature_idx = st.selectbox(
            "Select Feature",
            options=list(range(len(FEATURE_NAMES))),
            format_func=lambda i: FEATURE_NAMES[i]
        )
        fig = plot_prediction_vs_actual(
            result,
            feature_idx=feature_idx,
            feature_names=FEATURE_NAMES,
            theme=THEME,
            colors=COLORS
        )
    else:
        fig = plot_prediction_vs_actual(
            result,
            feature_names=FEATURE_NAMES,
            theme=THEME,
            colors=COLORS
        )

    st.plotly_chart(fig, use_container_width=True)

    # Show anomalous timesteps
    if result.n_anomalies > 0:
        st.subheader("Detected Anomalous Timesteps")
        anomalous_timesteps = result.get_anomalous_timesteps()
        st.write(f"Timesteps: {anomalous_timesteps[:20].tolist()}" +
                 ("..." if len(anomalous_timesteps) > 20 else ""))

# Tab 2: Anomaly Scores
with tabs[1]:
    st.subheader("Anomaly Scores Over Time")
    st.markdown(
        "Anomaly score indicates how much the observed value deviates from the "
        "predicted confidence interval. Scores above the threshold are flagged as anomalies."
    )

    fig = plot_anomaly_scores(
        result,
        theme=THEME,
        colors=COLORS
    )
    st.plotly_chart(fig, use_container_width=True)

    # Distribution of scores
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Score Distribution")
        hist_fig = {
            'data': [{'type': 'histogram', 'x': result.anomaly_scores, 'nbinsx': 30}],
            'layout': {
                'template': THEME,
                'xaxis': {'title': 'Anomaly Score'},
                'yaxis': {'title': 'Frequency'},
                'height': 300
            }
        }
        st.plotly_chart(hist_fig, use_container_width=True)

    with col2:
        st.subheader("Score Statistics")
        st.dataframe({
            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '95th Percentile'],
            'Value': [
                f"{result.anomaly_scores.mean():.4f}",
                f"{np.median(result.anomaly_scores):.4f}",
                f"{result.anomaly_scores.std():.4f}",
                f"{result.anomaly_scores.min():.4f}",
                f"{result.anomaly_scores.max():.4f}",
                f"{np.percentile(result.anomaly_scores, 95):.4f}"
            ]
        }, use_container_width=True, hide_index=True)

# Tab 3: Metrics Summary
with tabs[2]:
    st.subheader("Detection Metrics")

    # Compare multiple samples if available
    if st.button("Compare All Samples"):
        with st.spinner("Running detection on all samples..."):
            results = []
            labels = []

            for sample_type in sample_types[:5]:  # Limit to 5 for performance
                sample = samples[sample_type][0]  # First sequence from each
                r = detector.detect_anomalies(sample, threshold=threshold, return_feature_contributions=False)
                results.append(r)
                labels.append(sample_type.replace('_', ' ').title())

            fig = plot_detection_metrics(
                results,
                labels=labels,
                theme=THEME,
                colors=COLORS
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click 'Compare All Samples' to see detection metrics across different attack types")

    # Current sample summary
    st.subheader("Current Sample Summary")
    summary_data = result.summary()
    col1, col2 = st.columns(2)

    with col1:
        st.json({
            'Sequence Length': summary_data['seq_length'],
            'Number of Features': summary_data['n_features'],
            'Anomalies Detected': summary_data['n_anomalies'],
            'Anomaly Rate': f"{summary_data['anomaly_rate']:.2%}"
        })

    with col2:
        st.json({
            'Detection Threshold': summary_data['threshold'],
            'Mean Anomaly Score': f"{summary_data['mean_anomaly_score']:.4f}",
            'Max Anomaly Score': f"{summary_data['max_anomaly_score']:.4f}",
            'Inference Time': f"{summary_data['metadata'].get('inference_time', 0):.2f}s"
        })

# Tab 4: Feature Analysis
with tabs[3]:
    st.subheader("Feature-Level Anomaly Contributions")
    st.markdown(
        "This heatmap shows which features contributed most to anomaly detection "
        "at each timestep. Darker red = higher contribution."
    )

    if result.feature_contributions is not None:
        try:
            fig = plot_feature_contributions(
                result,
                feature_names=FEATURE_NAMES,
                theme=THEME,
                colors=COLORS
            )
            st.plotly_chart(fig, use_container_width=True)

            # Top contributing features
            st.subheader("Top Contributing Features")
            if result.n_anomalies > 0:
                anomalous_timesteps = result.get_anomalous_timesteps()
                timestep = anomalous_timesteps[0]  # First anomalous timestep

                top_features = result.get_top_anomalous_features(timestep, top_k=5)
                contributions = result.feature_contributions[timestep, top_features]

                st.write(f"At timestep {timestep} (first detected anomaly):")
                for i, (feat_idx, contrib) in enumerate(zip(top_features, contributions)):
                    st.write(f"{i+1}. **{FEATURE_NAMES[feat_idx]}**: {contrib:.3f}")
            else:
                st.info("No anomalies detected in this sequence")

        except Exception as e:
            st.error(f"Error plotting feature contributions: {e}")
    else:
        st.warning("Feature contributions not available")

# Export results
st.divider()
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("📥 Export Results"):
        # Export as JSON
        export_data = {
            'sample_type': selected_sample,
            'sequence_index': seq_idx,
            'model_size': model_size,
            'threshold': threshold,
            'summary': result.summary(),
            'anomalous_timesteps': result.get_anomalous_timesteps().tolist()
        }
        st.json(export_data)

with col2:
    if st.button("🔄 Rerun Detection"):
        st.rerun()

with col3:
    st.caption(f"Model: Moirai {model_size} | Mock Mode: {is_mock} | Threshold: {threshold}")
