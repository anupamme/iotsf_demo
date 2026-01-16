"""Navigation helpers."""

import streamlit as st


def get_page_name(page_num: int) -> str:
    """
    Get page name from number.

    Args:
        page_num: Page number (0-4)

    Returns:
        Page name string

    Example:
        >>> get_page_name(0)
        'Challenge'
        >>> get_page_name(4)
        'Detection'
    """
    names = ["Challenge", "Reveal", "Traditional IDS", "Pipeline", "Detection"]
    return names[page_num] if 0 <= page_num < len(names) else ""


def render_navigation_buttons(current_page: int, total_pages: int = 5):
    """
    Render prev/next navigation buttons.

    Shows previous and next buttons with instructions to use sidebar for navigation.
    Streamlit doesn't support programmatic navigation, so buttons are informational.

    Args:
        current_page: Current page number (0-indexed)
        total_pages: Total number of pages (default: 5)

    Example:
        >>> render_navigation_buttons(current_page=0)  # Shows only "Next" button
        >>> render_navigation_buttons(current_page=2)  # Shows both buttons
    """
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if current_page > 0:
            prev_page_name = get_page_name(current_page - 1)
            st.button(f"← {prev_page_name}", disabled=True, help="Use sidebar to navigate")

    with col2:
        st.markdown(
            f"<center><small>Page {current_page + 1} of {total_pages}</small></center>",
            unsafe_allow_html=True
        )

    with col3:
        if current_page < total_pages - 1:
            next_page_name = get_page_name(current_page + 1)
            st.button(f"{next_page_name} →", disabled=True, help="Use sidebar to navigate")
