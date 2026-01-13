"""Spot the Attack - Interactive Challenge Page"""

import streamlit as st
import sys
from pathlib import Path
import plotly.graph_objects as go

# Add src to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.components.plots import plot_comparison_grid
from app.components.presenter import render_presenter_notes
from app.utils.navigation import render_navigation_buttons

st.set_page_config(
    page_title="Challenge - IoT Security Demo",
    page_icon="🔒",
    layout="wide",
)

# Check initialization
if "initialized" not in st.session_state:
    st.warning("⚠️ Please return to the main page to initialize the app")
    st.stop()

st.title("🎯 Challenge: Spot the Attack")

st.markdown("""
Below are **6 IoT network traffic samples**. Three are benign, three are sophisticated attacks.

**Can you identify which ones are malicious?**

These attacks are highly stealthy (85-95% similarity to benign traffic), making them
extremely difficult to detect with visual inspection alone.
""")

# Load samples from session state
demo_data = st.session_state.demo_samples
samples = demo_data['samples']

# Display comparison grid (without revealing answers)
st.subheader("Traffic Samples")

fig = plot_comparison_grid(
    samples=samples,
    labels=[f"Sample {i+1}" for i in range(6)],
    reveal=False,  # Don't show answers yet
    feature_indices=[5, 6, 8]  # packet rates and bytes
)
st.plotly_chart(fig, use_container_width=True)

st.info("💡 **Hint**: Look for unusual patterns in packet rates, timing, or data transfer volumes")

# Interactive voting section
st.markdown("---")
st.subheader("📊 Cast Your Vote")

st.markdown("Select which samples you think are attacks:")

# Create 6 columns for checkboxes
cols = st.columns(6)
votes = []

for i, col in enumerate(cols):
    with col:
        vote = st.checkbox(
            f"**Sample {i+1}**",
            key=f"vote_{i}",
            help="Check if you think this is an attack",
            value=st.session_state.user_votes['votes'][i]
        )
        votes.append(vote)

# Update votes in session state
st.session_state.user_votes['votes'] = votes

# Submit and reset buttons
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("✅ Submit Vote", type="primary", use_container_width=True):
        st.session_state.user_votes['vote_submitted'] = True
        # Update aggregate vote counts
        for i, vote in enumerate(votes):
            if vote:
                st.session_state.user_votes['vote_counts'][i] = \
                    st.session_state.user_votes['vote_counts'].get(i, 0) + 1

with col2:
    if st.button("🔄 Reset Votes", use_container_width=True):
        st.session_state.user_votes = {
            'votes': [False] * 6,
            'vote_counts': {},
            'vote_submitted': False
        }
        st.rerun()

# Display vote results if submitted
if st.session_state.user_votes['vote_submitted']:
    st.success("✅ Vote recorded!")

    # Show aggregate tally
    st.subheader("📈 Current Vote Tally")

    vote_counts = st.session_state.user_votes['vote_counts']
    samples_labels = [f"Sample {i+1}" for i in range(6)]
    counts = [vote_counts.get(i, 0) for i in range(6)]

    # Create bar chart
    fig_tally = go.Figure(data=[
        go.Bar(
            x=samples_labels,
            y=counts,
            marker_color='#636EFA',
            text=counts,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Votes: %{y}<extra></extra>'
        )
    ])

    fig_tally.update_layout(
        template="plotly_dark",
        title="Votes per Sample",
        xaxis_title="Sample",
        yaxis_title="Number of Votes",
        plot_bgcolor='#111111',
        paper_bgcolor='#111111',
        showlegend=False,
        height=400
    )

    st.plotly_chart(fig_tally, use_container_width=True)

    st.info("👉 Navigate to the **Reveal** page to see the actual answers!")

# Navigation buttons
render_navigation_buttons(current_page=0)

# Presenter notes
render_presenter_notes(
    timing="3-4 minutes",
    key_points=[
        "Emphasize the difficulty of detecting sophisticated attacks visually",
        "These are stealth-level attacks at 85-95% similarity to benign traffic",
        "Even security experts struggle with visual inspection alone",
        "Build suspense - don't give hints about which are attacks",
        "Encourage audience participation and discussion"
    ],
    transition="Let's reveal which samples are actually attacks...",
    qa_prep=[
        "Q: Are these real attacks? A: Based on real attack patterns from CICIoT2023 dataset",
        "Q: What makes them stealthy? A: They operate within normal statistical ranges",
        "Q: Can signature-based IDS detect these? A: No, these are zero-day style attacks"
    ]
)
