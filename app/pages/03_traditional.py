"""Traditional IDS Page - Baseline detection results"""

import streamlit as st
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add src to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.utils.model_loaders import load_baseline_ids
from app.components.plots import plot_anomaly_scores
from app.components.presenter import render_presenter_notes
from app.utils.navigation import render_navigation_buttons

st.set_page_config(
    page_title="Traditional IDS - IoT Security Demo",
    page_icon="🔒",
    layout="wide",
)

# Check initialization
if "initialized" not in st.session_state:
    st.warning("⚠️ Please return to the main page to initialize the app")
    st.stop()

st.title("📊 Traditional IDS Detection")

st.markdown("""
Let's see how **traditional Intrusion Detection Systems** perform on our sophisticated attacks.

We're testing a combination of three common IDS approaches:
- **Threshold-based**: Statistical thresholds on traffic features
- **Signature-based**: Pattern matching against known attack signatures
- **Statistical**: Anomaly detection using statistical models

These methods are effective against well-known attacks, but how do they handle our stealthy samples?
""")

# Load data from session state
demo_data = st.session_state.demo_samples
samples = demo_data['samples']
true_labels = demo_data['labels']
attack_types = demo_data['attack_types']

# Prepare data for IDS
X = np.stack(samples)  # (6, 128, 12)
y_true = np.array([1 if label else 0 for label in true_labels])

# Run detection
st.markdown("---")
st.subheader("🔍 Running Traditional IDS Detection...")

with st.spinner("Loading and training traditional IDS methods..."):
    try:
        # Load baseline IDS
        ids = load_baseline_ids()

        # Train on benign samples only
        benign_indices = [i for i, label in enumerate(true_labels) if not label]
        X_benign = X[benign_indices]

        # Fit the IDS on benign traffic
        ids.fit(X_benign)

        # Predict on all samples
        y_pred = ids.predict(X)
        y_scores = ids.predict_proba(X)

        detection_successful = True

    except Exception as e:
        st.error(f"⚠️ Error loading baseline IDS: {e}")
        st.info("Using mock detection results for demonstration...")
        # Mock results showing poor detection
        y_pred = np.array([0, 0, 0, 0, 0, 0])  # Predicts all as benign
        y_scores = np.array([0.1, 0.15, 0.12, 0.2, 0.18, 0.22])  # Low scores
        detection_successful = False

# Display results
st.markdown("---")
st.subheader("📈 Detection Results")

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# Confusion matrix values
tp = int(np.sum((y_pred == 1) & (y_true == 1)))
fp = int(np.sum((y_pred == 1) & (y_true == 0)))
tn = int(np.sum((y_pred == 0) & (y_true == 0)))
fn = int(np.sum((y_pred == 0) & (y_true == 1)))

# Display metrics in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", f"{accuracy:.1%}")
with col2:
    st.metric("Precision", f"{precision:.1%}")
with col3:
    st.metric("Recall", f"{recall:.1%}", help="Percentage of attacks detected")
with col4:
    attacks_detected = tp
    total_attacks = int(np.sum(y_true))
    st.metric("Attacks Detected", f"{attacks_detected}/{total_attacks}")

# Confusion matrix
st.markdown("---")
st.subheader("Confusion Matrix")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("✅ True Positives", tp, help="Attacks correctly detected")
with col2:
    st.metric("❌ False Positives", fp, help="Benign traffic flagged as attacks")
with col3:
    st.metric("✅ True Negatives", tn, help="Benign traffic correctly identified")
with col4:
    st.metric("❌ False Negatives", fn, help="Attacks missed (not detected)")

# Visualize anomaly scores
st.markdown("---")
st.subheader("📊 Anomaly Scores")

fig_scores = plot_anomaly_scores(
    scores=y_scores,
    threshold=0.5,
    labels=[f"Sample {i+1}" for i in range(6)],
    ground_truth=y_true
)
st.plotly_chart(fig_scores, use_container_width=True)

st.caption("""
**How to read this chart**: Samples with scores above the threshold (dashed line) are classified as attacks.
Green bars = classified as benign, Red bars = classified as attacks.
""")

# Per-sample breakdown
st.markdown("---")
st.subheader("🔬 Per-Sample Analysis")

for i in range(6):
    is_attack = true_labels[i]
    predicted_attack = y_pred[i] == 1
    score = y_scores[i]

    # Determine detection status
    if is_attack and predicted_attack:
        status = "✅ True Positive"
        emoji = "🟢"
    elif not is_attack and not predicted_attack:
        status = "✅ True Negative"
        emoji = "🟢"
    elif not is_attack and predicted_attack:
        status = "❌ False Positive"
        emoji = "🔴"
    else:
        status = "❌ False Negative (MISSED!)"
        emoji = "🔴"

    with st.expander(f"{emoji} Sample {i+1}: {attack_types[i]} - {status}"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown(f"**Ground Truth:** {'🔴 Attack' if is_attack else '🟢 Benign'}")
        with col_b:
            st.markdown(f"**Prediction:** {'🔴 Attack' if predicted_attack else '🟢 Benign'}")
        with col_c:
            st.markdown(f"**Anomaly Score:** {score:.3f}")

        # Explanation for missed attacks
        if is_attack and not predicted_attack:
            st.warning(f"""
            **Why was this attack missed?**

            This is a **{attack_types[i]}** attack with high stealth level. Traditional IDS failed to detect it because:
            - The attack operates within normal statistical ranges
            - No matching signatures in the signature database
            - Threshold-based detection requires larger deviations
            - Statistical models trained on benign data can't recognize subtle anomalies
            """)

# Failure analysis
st.markdown("---")
st.error(f"""
### 🚫 Why Traditional IDS Fail on Stealthy Attacks

Traditional IDS methods struggle with sophisticated, high-stealth attacks for several reasons:

**1. Threshold-Based Detection**
- Relies on hard thresholds (e.g., "flag if packet rate > 100 pps")
- Stealthy attacks stay just below these thresholds
- Cannot adapt to subtle, gradual changes

**2. Signature-Based Detection**
- Requires exact pattern matching against known attacks
- Zero-day attacks have no signatures
- Cannot generalize to novel attack patterns

**3. Statistical Methods**
- Trained on benign traffic only
- Struggle with attacks that mimic benign behavior
- High false positive rates when tuned sensitively

**4. Lack of Context**
- Analyze individual features independently
- Miss correlations across multiple features
- Cannot capture temporal dependencies

**Result:** Detection rate of **{recall:.1%}** on our sophisticated attacks.
This demonstrates the critical need for advanced, ML-based detection methods.
""")

# Store results for later comparison
st.session_state.detection_results['baseline'] = {
    'predictions': y_pred,
    'scores': y_scores,
    'metrics': {
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
}
st.session_state.detection_results['computed'] = True

# Navigation buttons
render_navigation_buttons(current_page=2)

# Presenter notes
render_presenter_notes(
    timing="4-5 minutes",
    key_points=[
        f"Emphasize the low detection rate: {recall:.1%} recall on sophisticated attacks",
        "Traditional methods work well for known, obvious attacks",
        "Stealthy attacks specifically designed to evade threshold-based detection",
        "No signatures exist for zero-day attacks",
        "This motivates the need for our advanced ML-based approach",
        "Show per-sample analysis to highlight specific missed attacks"
    ],
    transition="Our pipeline addresses these limitations. Let me explain how...",
    qa_prep=[
        "Q: Can we tune thresholds better? A: Yes, but creates high false positives",
        "Q: What about ensemble methods? A: Still rely on same weak base classifiers",
        "Q: Why not add more signatures? A: Requires knowing attack patterns in advance",
        "Q: What's a realistic detection rate? A: Against APTs, often <30% for traditional IDS"
    ]
)
