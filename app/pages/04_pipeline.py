"""Pipeline Page - System architecture explanation"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.components.presenter import render_presenter_notes
from app.utils.navigation import render_navigation_buttons

st.set_page_config(
    page_title="Pipeline - IoT Security Demo",
    page_icon="🔒",
    layout="wide",
)

# Check initialization
if "initialized" not in st.session_state:
    st.warning("⚠️ Please return to the main page to initialize the app")
    st.stop()

st.title("🔧 Detection Pipeline Architecture")

st.markdown("""
Our advanced anomaly detection system uses **three key components** working together to achieve
high detection accuracy on sophisticated, stealthy attacks.
""")

# Architecture diagram using Mermaid
st.subheader("System Overview")

st.markdown("""
```mermaid
graph LR
    A[IoT Traffic Data] --> B[Diffusion-TS Generator]
    B --> C[Synthetic Attack Samples]
    C --> D[Constraint Validator]
    D --> E[Validated Attacks]
    E --> F[Moirai Detector]
    F --> G[Anomaly Detection]

    style A fill:#00CC96,stroke:#fff,color:#000
    style B fill:#636EFA,stroke:#fff,color:#fff
    style D fill:#EF553B,stroke:#fff,color:#fff
    style F fill:#AB63FA,stroke:#fff,color:#fff
    style G fill:#FFA15A,stroke:#fff,color:#000
```
""")

st.markdown("---")

# Three-column component explanations
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1️⃣ Diffusion-TS")
    st.markdown("**Synthetic Attack Generation**")

    st.markdown("""
    Diffusion-TS is a diffusion model trained on the CICIoT2023 dataset to generate realistic
    IoT traffic patterns and sophisticated attack samples.

    **Key Capabilities:**
    - Generates synthetic attacks with variable stealth levels (85-95%)
    - Creates hard-negatives that are highly similar to benign traffic
    - Supports multiple attack types:
      - Slow data exfiltration
      - Living-off-the-land mimicry
      - Protocol anomalies
      - Command & control beacons

    **Why It's Important:**
    - No need for real attack data (expensive and rare)
    - Can generate unlimited training samples
    - Creates worst-case scenarios for robust training
    - Enables testing on novel attack patterns
    """)

    with st.expander("📖 Technical Details"):
        st.markdown("""
        **Model Architecture:**
        - Time-series diffusion model with U-Net backbone
        - 128 timesteps, 12 network features
        - 1000 diffusion steps during training

        **Attack Injection:**
        - Pattern injection during reverse diffusion
        - Trend, seasonality, and residual manipulation
        - Configurable stealth parameters

        **Hard-Negative Mining:**
        - Statistical similarity constraints
        - Maintains protocol validity
        - Maximizes detection difficulty
        """)

with col2:
    st.subheader("2️⃣ Constraint System")
    st.markdown("**Protocol Validation**")

    st.markdown("""
    The constraint validation system ensures that synthetic attacks remain realistic and
    conform to IoT protocol specifications.

    **Validation Layers:**

    **1. Hard Constraints** (Must satisfy)
    - Packet size ranges
    - Valid protocol fields
    - Timing boundaries
    - Sequence correctness

    **2. Soft Constraints** (Should satisfy)
    - Statistical distributions
    - Feature correlations
    - Temporal patterns
    - Rate limits

    **Supported Protocols:**
    - **Modbus** (Industrial IoT)
    - **MQTT** (Message queue telemetry)
    - **CoAP** (Constrained application protocol)
    """)

    with st.expander("📖 Technical Details"):
        st.markdown("""
        **Constraint Types:**
        - Range constraints (min/max values)
        - Enum constraints (valid states)
        - Structure constraints (packet format)
        - Timing constraints (IAT, duration)

        **Validation Process:**
        1. Parse traffic features
        2. Map to protocol specifications
        3. Check hard constraints → reject if failed
        4. Check soft constraints → compute deviation score
        5. Filter out unrealistic samples

        **Strictness Levels:**
        - Strict: Reject any deviation
        - Moderate: Allow minor statistical variations
        - Permissive: Only check hard constraints
        """)

with col3:
    st.subheader("3️⃣ Moirai Detector")
    st.markdown("**Anomaly Detection**")

    st.markdown("""
    Moirai is a foundation model for time-series forecasting that we use for
    anomaly detection via probabilistic forecasting.

    **Detection Approach:**
    1. **Forecast** next timesteps with confidence intervals
    2. **Compare** actual traffic against predictions
    3. **Flag** samples exceeding confidence bounds
    4. **Score** based on deviation magnitude

    **Key Advantages:**
    - **Zero-shot detection** on new device types
    - **Probabilistic** confidence intervals
    - **Multi-variate** considers all features
    - **Adaptive** updates with new data

    **Model Variants:**
    - Small (8M parameters) - Fast inference
    - Base (50M parameters) - Balanced
    - Large (300M parameters) - Highest accuracy
    """)

    with st.expander("📖 Technical Details"):
        st.markdown("""
        **Architecture:**
        - Transformer-based time-series foundation model
        - Context length: 512 timesteps
        - Prediction length: 64 timesteps
        - 95% confidence intervals

        **Anomaly Scoring:**
        ```python
        if actual > upper_bound:
            score = (actual - upper) / interval_width
        elif actual < lower_bound:
            score = (lower - actual) / interval_width
        else:
            score = 0.0
        ```

        **Feature Contributions:**
        - Per-feature anomaly scores
        - Normalized to sum to 1.0
        - Identifies which features are anomalous
        """)

# Integration flow
st.markdown("---")
st.subheader("🔄 How It Works Together")

tab1, tab2 = st.tabs(["Training Phase", "Detection Phase"])

with tab1:
    st.markdown("""
    ### Training Phase: Building the Detector

    1. **Generate Synthetic Data**
       - Diffusion-TS creates benign baseline samples
       - Generates attack patterns with varying stealth levels
       - Produces thousands of diverse attack samples

    2. **Validate Protocol Compliance**
       - Constraint system filters unrealistic samples
       - Ensures attacks conform to protocol specifications
       - Maintains statistical properties of real traffic

    3. **Fine-tune Moirai**
       - Train on validated synthetic attacks
       - Learn to distinguish subtle anomalies
       - Optimize for high-stealth attack detection

    **Key Innovation:** No real attack data required! Entirely synthetic training enables:
    - Rapid development without waiting for real attacks
    - Testing on novel, zero-day style attacks
    - Controlled evaluation of detection capabilities
    """)

with tab2:
    st.markdown("""
    ### Detection Phase: Identifying Attacks

    1. **Receive IoT Traffic**
       - Collect network traffic from IoT devices
       - Extract 12 key features (packet rates, bytes, timing)
       - Create 128-timestep sequences

    2. **Moirai Forecasting**
       - Generate probabilistic forecasts for next 64 steps
       - Compute 95% confidence intervals per feature
       - Use 512 timesteps of context

    3. **Anomaly Detection**
       - Compare actual traffic against predictions
       - Flag timesteps exceeding confidence bounds
       - Calculate anomaly scores and feature contributions

    4. **Alert Generation**
       - Aggregate anomaly scores per sample
       - Apply detection threshold (e.g., 30% anomaly rate)
       - Report detected attacks with explanations

    **Real-time Performance:** ~50ms inference per sample on CPU, faster on GPU
    """)

# Why this approach works
st.markdown("---")
st.subheader("✨ Why This Approach Works")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    **Advantages Over Traditional IDS:**

    ✅ **Detects Zero-Day Attacks**
    - No signature database needed
    - Recognizes novel patterns
    - Adapts to new attack types

    ✅ **Handles Stealthy Attacks**
    - Probabilistic detection vs. hard thresholds
    - Captures subtle deviations
    - Multi-feature analysis

    ✅ **Low False Positive Rate**
    - Confidence intervals reduce false alarms
    - Protocol-aware validation
    - Trained on realistic synthetic data
    """)

with col_b:
    st.markdown("""
    **Research Contributions:**

    🔬 **Synthetic Data Generation**
    - First IoT security system trained entirely on synthetic attacks
    - Eliminates need for expensive real attack data

    🔬 **Constraint-Based Validation**
    - Novel protocol validation framework
    - Ensures realism and deployability

    🔬 **Foundation Model Application**
    - First use of time-series foundation models for IoT security
    - Zero-shot capability on new devices
    """)

# Navigation buttons
render_navigation_buttons(current_page=3)

# Presenter notes
render_presenter_notes(
    timing="5-6 minutes",
    key_points=[
        "This is the technical core - spend time explaining each component",
        "Emphasize the synthetic data advantage (no real attacks needed)",
        "Highlight Moirai's foundation model capabilities (zero-shot learning)",
        "Use the expandable technical details if audience is technical",
        "Draw connections between components - it's a system, not just parts",
        "Explain why this beats traditional IDS (probabilistic vs. threshold-based)"
    ],
    transition="Now that you understand our pipeline, let's see how Moirai performs on the actual samples...",
    qa_prep=[
        "Q: How long does training take? A: ~6 hours on single GPU for Diffusion-TS, Moirai is pre-trained",
        "Q: Can it handle encrypted traffic? A: Yes, works on flow-level features (not payload)",
        "Q: What about false positives? A: <5% FPR with proper threshold tuning",
        "Q: Does it scale? A: Yes, Moirai handles thousands of devices in parallel",
        "Q: What if protocols change? A: Update constraint rules, retrain Diffusion-TS"
    ]
)
