"""Spot the Attack - Interactive Challenge Page"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(
    page_title="Challenge - IoT Security Demo",
    page_icon="🔒",
    layout="wide",
)

st.title("🎯 Challenge: Spot the Attack")
st.markdown("Content for this page will be implemented in future issues.")

# Access shared session state
if "config" in st.session_state:
    config = st.session_state.config
    st.info("App initialized successfully")
else:
    st.warning("Please return to the main page to initialize the app")
