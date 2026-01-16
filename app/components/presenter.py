"""Presenter mode utilities."""

import streamlit as st
from typing import List, Optional


def render_presenter_notes(
    timing: str,
    key_points: List[str],
    transition: str,
    qa_prep: Optional[List[str]] = None
):
    """
    Render expandable presenter notes.

    Displays timing, key talking points, transition guidance, and optional Q&A prep
    in an expandable section at the bottom of the page.

    Args:
        timing: Expected duration (e.g., "3-4 minutes")
        key_points: List of key talking points
        transition: Text for transitioning to next page
        qa_prep: Optional list of Q&A preparation items

    Example:
        >>> render_presenter_notes(
        ...     timing="3-4 minutes",
        ...     key_points=[
        ...         "Emphasize the difficulty of visual detection",
        ...         "Mention stealth levels (85-95%)"
        ...     ],
        ...     transition="Let's reveal the answers...",
        ...     qa_prep=["Q: How realistic are these attacks? A: Based on real patterns"]
        ... )
    """
    # Check if presenter mode is enabled
    config = st.session_state.get('config')
    if config:
        presenter_enabled = config.get('presenter_mode.enabled', True)
    else:
        presenter_enabled = True

    if not presenter_enabled:
        return

    with st.expander("🎤 Presenter Notes", expanded=False):
        # Timing
        st.markdown(f"**⏱️ Timing**: {timing}")

        # Key points
        st.markdown("**🎯 Key Points**:")
        for point in key_points:
            st.markdown(f"- {point}")

        # Transition
        st.markdown(f"**➡️ Transition**: {transition}")

        # Q&A preparation (optional)
        if qa_prep:
            st.markdown("**❓ Q&A Preparation**:")
            for qa in qa_prep:
                st.markdown(f"- {qa}")
