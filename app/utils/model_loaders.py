"""Model loading utilities with caching."""

import streamlit as st
from typing import Optional
from pathlib import Path


# Default checkpoint path for fine-tuned model
FINETUNED_CHECKPOINT = "models/moirai_finetuned/best_moirai.pt"


@st.cache_resource
def load_moirai_detector(use_finetuned: bool = False):
    """
    Lazy load Moirai detector with caching.

    Uses @st.cache_resource to ensure the model is loaded only once
    and shared across all users/sessions.

    Args:
        use_finetuned: If True and checkpoint exists, load fine-tuned weights.
                       Otherwise load base model from HuggingFace.

    Returns:
        MoiraiAnomalyDetector: Initialized Moirai detector

    Example:
        >>> detector = load_moirai_detector()
        >>> result = detector.detect_anomalies(traffic_sample, method='nll')
    """
    from src.models import MoiraiAnomalyDetector

    # Get config from session state
    config = st.session_state.get('config')
    device = st.session_state.get('device', 'cpu')

    if config:
        model_size = config.get('models.moirai.model_size', 'small')
        # Use shorter context/prediction for 128-timestep samples
        # Our samples are 128 timesteps, so context + prediction must be <= 128
        context_length = config.get('models.moirai.context_length', 96)
        prediction_length = config.get('models.moirai.prediction_length', 32)
    else:
        # Fallback defaults sized for 128-timestep sequences
        model_size = 'small'
        context_length = 96   # Use 96 timesteps as context
        prediction_length = 32  # Predict next 32 timesteps (total = 128)

    detector = MoiraiAnomalyDetector(
        model_size=model_size,
        context_length=context_length,
        prediction_length=prediction_length,
        confidence_level=0.95,
        device=device
    )

    # Check for fine-tuned checkpoint
    checkpoint_path = None
    if use_finetuned:
        checkpoint_file = Path(FINETUNED_CHECKPOINT)
        if checkpoint_file.exists():
            checkpoint_path = str(checkpoint_file)
            st.sidebar.success("✓ Using fine-tuned Moirai model")
        else:
            st.sidebar.info("ℹ️ Fine-tuned checkpoint not found, using base model")

    # Initialize the model (with or without fine-tuned weights)
    detector.initialize(checkpoint_path=checkpoint_path)

    return detector


@st.cache_resource
def load_baseline_ids():
    """
    Lazy load baseline IDS with caching.

    Uses @st.cache_resource to ensure the model is loaded only once
    and shared across all users/sessions.

    Returns:
        CombinedBaselineIDS: Initialized combined IDS

    Example:
        >>> ids = load_baseline_ids()
        >>> ids.fit(benign_samples)
        >>> predictions = ids.predict(test_samples)
    """
    from src.models.baseline import CombinedBaselineIDS

    # Get config from session state
    config = st.session_state.get('config')

    # Get weights from config if available
    if config:
        weights = config.get('models.baseline.combined.weights', None)
    else:
        weights = None

    ids = CombinedBaselineIDS(
        seq_length=128,
        feature_dim=12,
        weights=weights
    )

    return ids
