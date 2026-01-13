"""Reveal Page - Show which samples are attacks"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.components.plots import plot_comparison_grid, plot_time_series
from app.components.presenter import render_presenter_notes
from app.utils.navigation import render_navigation_buttons

st.set_page_config(
    page_title="Reveal - IoT Security Demo",
    page_icon="🔒",
    layout="wide",
)

# Check initialization
if "initialized" not in st.session_state:
    st.warning("⚠️ Please return to the main page to initialize the app")
    st.stop()

st.title("🔍 Reveal: The Truth")

st.markdown("""
Here are the actual labels for each traffic sample. The attacks are color-coded in **red**,
while benign traffic is shown in **green**.
""")

# Load data from session state
demo_data = st.session_state.demo_samples
samples = demo_data['samples']
true_labels = demo_data['labels']
attack_types = demo_data['attack_types']

# Display comparison grid WITH reveal=True
st.subheader("Traffic Samples - Revealed")

fig = plot_comparison_grid(
    samples=samples,
    labels=[f"Sample {i+1}" for i in range(6)],
    true_labels=true_labels,
    reveal=True  # Color-code borders (red=attack, green=benign)
)
st.plotly_chart(fig, use_container_width=True)

# Results summary
st.markdown("---")
st.subheader("📊 Summary")

col1, col2, col3 = st.columns(3)

with col1:
    n_benign = sum(1 for x in true_labels if not x)
    st.metric("🟢 Benign Samples", n_benign)

with col2:
    n_attacks = sum(1 for x in true_labels if x)
    st.metric("🔴 Attack Samples", n_attacks)

with col3:
    # Calculate user accuracy if they voted
    if st.session_state.user_votes['vote_submitted']:
        user_votes = st.session_state.user_votes['votes']
        correct = sum(1 for v, t in zip(user_votes, true_labels) if v == t)
        accuracy = correct / 6 * 100
        st.metric("🎯 Your Accuracy", f"{accuracy:.1f}%")
    else:
        st.metric("🎯 Your Accuracy", "N/A", help="You didn't vote")

# Attack explanations
ATTACK_EXPLANATIONS = {
    "slow_exfiltration_stealth_95": {
        "name": "Slow Data Exfiltration",
        "stealth": "95%",
        "description": """
**Slow Data Exfiltration** is a stealthy attack that mimics normal traffic patterns while gradually
exfiltrating data over an extended period.

**Characteristics:**
- Maintains normal packet rates to avoid detection
- Slight increase in outbound data volume (< 3% deviation)
- Spread across many timesteps to blend with benign traffic
- Uses legitimate protocols and ports

**Why it's hard to detect:**
- Traditional IDS use thresholds that flag sudden spikes - this attack stays below thresholds
- Statistical anomaly detectors struggle with such small deviations
- Appears as normal device behavior over time
        """
    },
    "lotl_mimicry_stealth_90": {
        "name": "Living-off-the-Land (LOTL) Mimicry",
        "stealth": "90%",
        "description": """
**Living-off-the-Land Mimicry** exploits legitimate system tools and protocols to conduct attacks,
making it extremely difficult to distinguish from normal operations.

**Characteristics:**
- Periodic micro-bursts that resemble legitimate device activities
- Uses valid protocol commands and timing patterns
- Blends with expected IoT device behavior (sensor readings, status updates)
- Mimics natural traffic fluctuations

**Why it's hard to detect:**
- No malicious signatures - uses legitimate commands
- Timing patterns match normal device operations
- Volume stays within expected ranges
- Signature-based IDS are completely blind to this
        """
    },
    "protocol_anomaly_stealth_85": {
        "name": "Protocol Anomaly",
        "stealth": "85%",
        "description": """
**Protocol Anomaly** attacks exploit subtle deviations in protocol behavior while maintaining
valid packet structures.

**Characteristics:**
- Unusual timing in inter-arrival times (IAT)
- Valid packet structures and checksums
- Exploits edge cases in protocol specifications
- Maintains correct sequence numbers and acknowledgments

**Why it's hard to detect:**
- Packets pass protocol validation checks
- Timing anomalies are within acceptable ranges
- No invalid fields or malformed data
- Requires deep protocol understanding to identify
        """
    },
    "Benign": {
        "name": "Benign Traffic",
        "stealth": "N/A",
        "description": """
**Benign Traffic** represents normal, legitimate IoT device communications.

**Characteristics:**
- Regular sensor readings and status updates
- Typical packet rates and timing patterns
- Normal data transfer volumes
- Standard protocol usage

This serves as the baseline for comparison against attack traffic.
        """
    }
}

# Detailed explanations for each sample
st.markdown("---")
st.subheader("🔬 Detailed Analysis")

for i, (is_attack, attack_type, sample) in enumerate(zip(true_labels, attack_types, samples)):
    status_emoji = "🔴" if is_attack else "🟢"
    status_text = "ATTACK" if is_attack else "BENIGN"

    attack_info = ATTACK_EXPLANATIONS.get(attack_type, {
        "name": attack_type,
        "stealth": "Unknown",
        "description": "No description available."
    })

    with st.expander(f"{status_emoji} Sample {i+1}: {attack_info['name']} [{status_text}]"):
        col_a, col_b = st.columns([1, 3])

        with col_a:
            st.markdown(f"**Type:** {attack_info['name']}")
            if attack_info['stealth'] != "N/A":
                st.markdown(f"**Stealth Level:** {attack_info['stealth']}")
            st.markdown(f"**Status:** {'🔴 Attack' if is_attack else '🟢 Benign'}")

        with col_b:
            st.markdown(attack_info['description'])

        # Show detailed time series for this sample
        if is_attack:
            st.markdown("**Traffic Pattern:**")
            fig_detail = plot_time_series(
                data=sample,
                label=attack_info['name'],
                is_attack=True,
                show_all=True  # Dropdown for all features
            )
            st.plotly_chart(fig_detail, use_container_width=True)

# User performance feedback
if st.session_state.user_votes['vote_submitted']:
    st.markdown("---")
    st.subheader("📈 Your Performance")

    user_votes = st.session_state.user_votes['votes']

    # Calculate confusion matrix
    tp = sum(1 for v, t in zip(user_votes, true_labels) if v and t)
    fp = sum(1 for v, t in zip(user_votes, true_labels) if v and not t)
    tn = sum(1 for v, t in zip(user_votes, true_labels) if not v and not t)
    fn = sum(1 for v, t in zip(user_votes, true_labels) if not v and t)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("✅ True Positives", tp, help="Attacks you correctly identified")
    with col2:
        st.metric("❌ False Positives", fp, help="Benign samples you marked as attacks")
    with col3:
        st.metric("✅ True Negatives", tn, help="Benign samples you correctly identified")
    with col4:
        st.metric("❌ False Negatives", fn, help="Attacks you missed")

    if accuracy == 100:
        st.success("🎉 Perfect score! You identified all attacks correctly!")
    elif accuracy >= 80:
        st.info("👏 Great job! You identified most of the attacks.")
    elif accuracy >= 60:
        st.warning("🤔 Not bad, but these attacks are tricky!")
    else:
        st.error("💪 These attacks are extremely stealthy - don't worry, detection is hard!")

# Navigation buttons
render_navigation_buttons(current_page=1)

# Presenter notes
render_presenter_notes(
    timing="3-4 minutes",
    key_points=[
        "Highlight how difficult these attacks are to spot visually",
        "Explain each attack type briefly using the expandable sections",
        "Emphasize the stealth levels (85-95% similarity)",
        "Discuss why visual inspection and traditional methods fail",
        "Build up to the need for automated, ML-based detection"
    ],
    transition="Now let's see how traditional IDS systems perform on these attacks...",
    qa_prep=[
        "Q: How were these attacks created? A: Generated using Diffusion-TS with hard-negative mining",
        "Q: Are real attacks this stealthy? A: Yes, advanced persistent threats (APTs) use similar techniques",
        "Q: What about encrypted traffic? A: These patterns work even with encrypted channels"
    ]
)
