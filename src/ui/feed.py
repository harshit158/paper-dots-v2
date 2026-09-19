"""The Feed view: a placeholder for the discovery surface."""

import streamlit as st


class FeedView:
    """Shows what the feed will become, without pretending it exists yet."""

    def render(self) -> None:
        st.subheader("Feed")
        st.info(
            "Coming soon — a feed of papers worth reading, built from what you "
            "already have in your library."
        )
