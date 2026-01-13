"""Model loading utilities with caching."""

import streamlit as st
from typing import Optional


@st.cache_resource
def load_moirai_detector():
    """
    Lazy load Moirai detector with caching.

    Uses @st.cache_resource to ensure the model is loaded only once
    and shared across all users/sessions.

    Returns:
        MoiraiAnomalyDetector: Initialized Moirai detector

    Example:
        >>> detector = load_moirai_detector()
        >>> result = detector.detect(traffic_sample)
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

    # Initialize the model
    detector.initialize()

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
