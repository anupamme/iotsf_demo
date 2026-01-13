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
        context_length = config.get('models.moirai.context_length', 512)
        prediction_length = config.get('models.moirai.prediction_length', 64)
    else:
        # Fallback defaults
        model_size = 'small'
        context_length = 512
        prediction_length = 64

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

    if config:
        methods = config.get('models.baseline.methods', ['threshold', 'statistical', 'signature'])
    else:
        # Use all three baseline methods
        methods = ['threshold', 'statistical', 'signature']

    ids = CombinedBaselineIDS(
        seq_length=128,
        feature_dim=12,
        methods=methods
    )

    return ids
