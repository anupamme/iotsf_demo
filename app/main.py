"""Main entry point for the IoT Security Demo Streamlit application."""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import Config
from src.utils.device import get_device, get_device_info
from src.utils.logging import setup_logging
from app.utils.data_loader import load_demo_samples

# Page configuration
st.set_page_config(
    page_title="IoT Security Demo",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_app():
    """Initialize application state and configuration."""
    if "initialized" not in st.session_state:
        # Load configuration
        config = Config(ROOT_DIR / "config" / "config.yaml")
        st.session_state.config = config

        # Setup logging
        setup_logging(log_level="INFO")

        # Initialize device
        device = get_device(
            use_gpu=config.get("device.use_gpu", True),
            gpu_id=config.get("device.gpu_id", 0),
        )
        st.session_state.device = device
        st.session_state.device_info = get_device_info()

        # Load demo samples
        st.session_state.demo_samples = load_demo_samples(
            data_dir=ROOT_DIR / "data" / "synthetic",
            config=config
        )

        # Initialize voting state
        st.session_state.user_votes = {
            'votes': [False] * 6,
            'vote_counts': {},
            'vote_submitted': False
        }

        # Initialize detection results storage
        st.session_state.detection_results = {
            'baseline': {},
            'moirai': [],
            'computed': False
        }

        # Model loading state
        st.session_state.models = {
            'loaded': set()
        }

        st.session_state.initialized = True


def main():
    """Main application."""
    initialize_app()

    # Header
    st.title("🔒 IoT Security Anomaly Detection Demo")

    st.markdown(
        """
    This interactive demo showcases advanced IoT security anomaly detection using:

    - **Diffusion-TS**: Time series generation for synthetic attack patterns
    - **Moirai**: Foundation model for anomaly detection
    - **Traditional IDS**: Baseline comparison with rule-based detection

    ### How It Works
    1. **Challenge**: Try to spot the attacks in IoT network traffic
    2. **Reveal**: See which samples are actually attacks
    3. **Traditional IDS**: Compare with traditional intrusion detection
    4. **Pipeline**: Understand our detection system
    5. **Detection**: See Moirai's anomaly detection results

    Navigate using the sidebar to explore each step.
    """
    )

    # Display device info in sidebar
    with st.sidebar:
        st.subheader("System Info")
        device_info = st.session_state.device_info
        if device_info["cuda_available"]:
            st.success(f"🚀 GPU: {device_info['devices'][0]}")
        else:
            st.info("💻 Running on CPU")

        st.caption(f"PyTorch: {device_info['pytorch_version']}")

        # Check which generator mode is currently available
        st.markdown("---")
        st.subheader("Generator Status")
        try:
            from src.models import IoTDiffusionGenerator
            test_gen = IoTDiffusionGenerator()
            test_gen.initialize()
            if not test_gen._mock_mode:
                st.success("✅ Real Diffusion-TS Available")
            else:
                st.info("📊 Using Mock Generator")
        except Exception as e:
            st.warning(f"⚠️ Generator check failed")
            st.caption(f"Error: {str(e)[:50]}...")


if __name__ == "__main__":
    main()
