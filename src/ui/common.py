"""Helpers shared by the views.

Everything here is view-agnostic: the page names, the small formatting
functions and the two widgets that more than one view draws. Keeping them out
of the views means a change to the flash message or the download button cannot
drift between pages.
"""

from pathlib import Path

import streamlit as st
from streamlit.navigation.page import StreamlitPage

from models import Paper

PAGE_FEED = "feed"
PAGE_READER = "reader"
PAGE_LIBRARY = "library"

PDF_VIEWER_HEIGHT = 800
ABSTRACT_PREVIEW_CHARS = 220


class Navigator:
    """Cross-page navigation for pages defined by callables.

    ``st.switch_page`` needs a ``StreamlitPage`` object when a page is defined
    by a callable rather than a file, so the pages are registered here once
    they are built and the views ask for one by name. The pages cannot be
    created here: ``st.navigation`` must be called from the entrypoint, so
    ``app.py`` owns their construction.
    """

    def __init__(self) -> None:
        self._pages: dict[str, StreamlitPage] = {}

    def register(self, name: str, page: StreamlitPage) -> None:
        """Make ``page`` reachable under ``name``."""
        self._pages[name] = page

    def open_reader(self, paper_id: int) -> None:
        """Open ``paper_id`` in the reader, switching pages if necessary.

        The id is parked in session state rather than passed along because
        ``st.switch_page`` stops the current run: the reader picks it up on the
        next one, after its paper picker widget already exists.
        """
        st.session_state["pending_paper_id"] = paper_id
        st.switch_page(self._pages[PAGE_READER])


def truncate(text: str, limit: int = ABSTRACT_PREVIEW_CHARS) -> str:
    """Collapse whitespace and shorten ``text`` to at most ``limit`` characters."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 1].rstrip()}…"


def paper_label(papers: list[Paper], paper_id: int) -> str:
    """A short, human-readable label for the paper picker."""
    paper = next((item for item in papers if item.id == paper_id), None)
    if paper is None:
        return str(paper_id)
    return paper.title if len(paper.title) <= 42 else f"{paper.title[:39]}…"


def render_flash() -> None:
    """Show and clear the message left behind by the previous run."""
    message = st.session_state.pop("flash_message", None)
    level = st.session_state.pop("flash_level", "success")
    if message is None:
        return
    if level == "info":
        st.info(message)
    else:
        st.success(message)


def render_download_button(paper: Paper) -> None:
    """Offer the stored PDF, if there is one on disk."""
    if not paper.pdf_path or not Path(paper.pdf_path).is_file():
        return
    with open(paper.pdf_path, "rb") as pdf:
        st.download_button(
            "Download PDF",
            pdf,
            file_name=Path(paper.pdf_path).name,
            key=f"download-{paper.id}",
            use_container_width=True,
        )
