"""Traditional IDS Page - Baseline detection results"""

import streamlit as st
import sys
import numpy as np
from pathlib import Path

# Add src to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.baseline import (
    ThresholdIDS,
    StatisticalIDS,
    SignatureIDS,
    MLBasedIDS,
    CombinedBaselineIDS
)
from src.evaluation.metrics import IDSMetrics
from app.components.metrics import display_metrics_row, display_confusion_matrix_stats, display_fpr_metric
from app.components.plots import (
    display_confusion_matrix,
    display_roc_curves,
    display_detection_results,
    display_method_comparison,
    display_score_distribution
)

st.set_page_config(
    page_title="Traditional IDS - IoT Security Demo",
    page_icon="🔒",
    layout="wide",
)

st.title("📊 Traditional IDS Detection")
st.markdown("""
This page demonstrates **traditional intrusion detection** methods and their limitations.
These baseline approaches are effective for well-known attacks but struggle with hard-negative attacks.
""")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

method_choice = st.sidebar.selectbox(
    "Select IDS Method",
    ["Combined (Ensemble)", "Threshold", "Signature", "Statistical", "ML-Based (Isolation Forest)"],
    help="Choose which traditional IDS method to evaluate"
)

# Sample size configuration
n_samples = st.sidebar.slider(
    "Number of Test Samples",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
    help="Number of samples to use for testing"
)

# Access shared session state
if "config" not in st.session_state:
    st.warning("⚠️ Please return to the main page to initialize the app")
    st.stop()

config = st.session_state.config


# Helper function to get IDS method
def get_ids_method(method_name):
    """Get IDS instance based on method name."""
    if method_name == "Combined (Ensemble)":
        return CombinedBaselineIDS()
    elif method_name == "Threshold":
        return ThresholdIDS()
    elif method_name == "Signature":
        return SignatureIDS()
    elif method_name == "Statistical":
        return StatisticalIDS()
    elif method_name == "ML-Based (Isolation Forest)":
        return MLBasedIDS()
    else:
        return CombinedBaselineIDS()


# Generate synthetic test data
st.header("1️⃣ Test Data Generation")

if st.button("🎲 Generate Synthetic Test Data", type="primary"):
    with st.spinner("Generating synthetic data..."):
        # Generate synthetic benign and attack data
        # For demo purposes, create random sequences
        np.random.seed(42)

        # Benign sequences (normal distribution)
        n_benign = n_samples // 2
        X_benign = np.random.randn(n_benign, 128, 12) * 0.5 + 0.5
        y_benign = np.zeros(n_benign)

        # Attack sequences (anomalous patterns)
        n_attack = n_samples - n_benign
        X_attack = np.random.randn(n_attack, 128, 12) * 0.5 + 0.5

        # Inject attack patterns
        # Increase packet rates (feature 7)
        X_attack[:, :, 7] *= 3.0
        # Increase bytes (features 8, 9)
        X_attack[:, :, 8] *= 2.0
        X_attack[:, :, 9] *= 0.5  # Asymmetry

        y_attack = np.ones(n_attack)

        # Combine and shuffle
        X_test = np.concatenate([X_benign, X_attack], axis=0)
        y_test = np.concatenate([y_benign, y_attack])

        # Shuffle
        indices = np.random.permutation(len(X_test))
        X_test = X_test[indices]
        y_test = y_test[indices]

        # Store in session state
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        st.success(f"✅ Generated {n_samples} test samples ({n_benign} benign, {n_attack} attack)")

        # Display sample statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Benign Samples", n_benign)
        with col2:
            st.metric("Attack Samples", n_attack)


# Training and Detection
st.header("2️⃣ IDS Training & Detection")

if "X_test" not in st.session_state:
    st.info("👆 Please generate test data first")
else:
    if st.button("🔍 Run IDS Detection", type="primary"):
        with st.spinner(f"Training {method_choice} and detecting attacks..."):
            # Initialize IDS
            ids = get_ids_method(method_choice)

            # Generate training data (benign only)
            np.random.seed(123)
            X_train_benign = np.random.randn(100, 128, 12) * 0.5 + 0.5

            # Train IDS
            ids.fit(X_train_benign)

            # Predict on test data
            y_pred = ids.predict(st.session_state.X_test)
            y_scores = ids.predict_proba(st.session_state.X_test)

            # Store predictions
            st.session_state.y_pred = y_pred
            st.session_state.y_scores = y_scores
            st.session_state.current_method = method_choice

            st.success(f"✅ Detection complete using {method_choice}")


# Results Display
if "y_pred" in st.session_state and "y_test" in st.session_state:
    st.header("3️⃣ Detection Results")

    # Compute metrics
    metrics = IDSMetrics.compute_all_metrics(
        st.session_state.y_test,
        st.session_state.y_pred,
        st.session_state.y_scores
    )

    # Display key metrics
    st.subheader("📈 Performance Metrics")
    display_metrics_row(metrics)

    # Display confusion matrix stats
    st.subheader("🔢 Confusion Matrix Breakdown")
    display_confusion_matrix_stats(metrics)

    # Display false positive rate
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        display_fpr_metric(metrics)
    with col2:
        if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    # Visualizations
    st.subheader("📊 Visualizations")

    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Detection Results", "Score Distribution"])

    with tab1:
        display_confusion_matrix(
            st.session_state.y_test,
            st.session_state.y_pred,
            st.session_state.current_method
        )

    with tab2:
        display_detection_results(
            st.session_state.X_test,
            st.session_state.y_test,
            st.session_state.y_pred,
            feature_idx=7,
            n_show=5
        )

    with tab3:
        display_score_distribution(
            st.session_state.y_scores,
            st.session_state.y_test,
            st.session_state.current_method
        )


# Method Comparison
st.header("4️⃣ Compare All Methods")

if "X_test" in st.session_state:
    if st.button("🔬 Compare All IDS Methods"):
        with st.spinner("Evaluating all methods..."):
            # Initialize all methods
            methods = {
                'Threshold': ThresholdIDS(),
                'Signature': SignatureIDS(),
                'Statistical': StatisticalIDS(),
                'ML-Based': MLBasedIDS(),
                'Combined': CombinedBaselineIDS()
            }

            # Generate training data
            np.random.seed(123)
            X_train_benign = np.random.randn(100, 128, 12) * 0.5 + 0.5

            # Train all methods and get predictions
            results = {}
            for name, method in methods.items():
                method.fit(X_train_benign)
                y_pred = method.predict(st.session_state.X_test)
                y_scores = method.predict_proba(st.session_state.X_test)

                metrics = IDSMetrics.compute_all_metrics(
                    st.session_state.y_test,
                    y_pred,
                    y_scores
                )
                metrics['scores'] = y_scores  # Store scores for ROC curve
                results[name] = metrics

            st.session_state.comparison_results = results
            st.success("✅ Comparison complete!")

    # Display comparison results
    if "comparison_results" in st.session_state:
        results = st.session_state.comparison_results

        # Summary table
        st.subheader("📋 Summary Table")
        summary_text = IDSMetrics.compare_methods_summary(results)
        st.code(summary_text, language=None)

        # Comparison plots
        st.subheader("📊 Comparison Visualizations")

        col1, col2 = st.columns(2)
        with col1:
            display_method_comparison(results)
        with col2:
            display_roc_curves(results, st.session_state.y_test)

else:
    st.info("👆 Generate test data first to enable method comparison")


# Information section
with st.expander("ℹ️ About Traditional IDS Methods"):
    st.markdown("""
    ### Traditional IDS Approaches

    **1. Threshold-Based IDS**
    - Simple percentile-based thresholds
    - Flags traffic exceeding pre-computed bounds
    - Fast but limited to simple anomalies

    **2. Signature-Based IDS**
    - Pattern matching for known attacks (Mirai, DDoS)
    - High precision for recognized patterns
    - Fails on novel or sophisticated attacks

    **3. Statistical IDS**
    - Z-score and IQR outlier detection
    - Assumes normal traffic follows statistical distributions
    - Good for detecting statistical anomalies

    **4. ML-Based IDS (Isolation Forest)**
    - Machine learning for anomaly detection
    - Learns patterns from benign traffic
    - Better generalization than rule-based methods

    **5. Combined (Ensemble)**
    - Weighted voting across all methods
    - Balances strengths of individual approaches
    - Generally provides best overall performance

    ### Limitations
    Traditional IDS methods struggle with:
    - **Hard-negative attacks**: Subtle, stealthy attacks that mimic benign traffic
    - **Zero-day attacks**: Novel attack patterns never seen before
    - **Adversarial evasion**: Attacks specifically designed to evade detection
    """)
