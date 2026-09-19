"""The Library view: every paper that has been ingested."""

import streamlit as st
from foundry.observability import get_logger, get_tracer

from models import Paper
from services import PaperService
from ui.common import Navigator, render_download_button, render_flash, truncate

logger = get_logger(__name__)
tracer = get_tracer(__name__)

CARDS_PER_ROW = 3

# Streamlit renders a keyed container with a ``st-key-<key>`` class, so the
# cards can be tinted without touching any other container on the page. The
# gradient is deliberately faint: it has to sit behind body text in both the
# light and the dark theme.
CARD_STYLE = """
<style>
[class*="st-key-paper-card-"] {
    height: 100%;
    padding: 1.1rem 1.2rem;
    border: 1px solid rgba(129, 140, 248, 0.35);
    border-radius: 14px;
    background: linear-gradient(160deg, rgba(99, 102, 241, 0.16), rgba(56, 189, 248, 0.07));
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.16);
    transition: transform 150ms ease, box-shadow 150ms ease, border-color 150ms ease;
}

[class*="st-key-paper-card-"]:hover {
    transform: translateY(-2px);
    border-color: rgba(129, 140, 248, 0.7);
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.26);
}
</style>
"""


class LibraryView:
    """Show the library as a grid of cards."""

    def __init__(self, papers: PaperService, navigator: Navigator) -> None:
        self._papers = papers
        self._navigator = navigator

    def render(self) -> None:
        # Read from the database on every rerun. Caching this list in
        # st.session_state would hide whether persistence actually works.
        papers = self._papers.list_papers()
        render_flash()
        st.subheader("Library")

        if not papers:
            st.info(
                "No papers yet. Import one from the Reader — it will still be here after a restart."
            )
            return

        st.caption(f"{len(papers)} paper{'s' if len(papers) != 1 else ''}")
        st.markdown(CARD_STYLE, unsafe_allow_html=True)
        for row in self._rows(papers):
            columns = st.columns(CARDS_PER_ROW, gap="medium")
            for column, paper in zip(columns, row, strict=False):
                with column:
                    self._render_card(paper)

    @staticmethod
    def _rows(papers: list[Paper]) -> list[list[Paper]]:
        return [
            papers[index : index + CARDS_PER_ROW] for index in range(0, len(papers), CARDS_PER_ROW)
        ]

    def _render_card(self, paper: Paper) -> None:
        with st.container(border=False, key=f"paper-card-{paper.id}"):
            st.markdown(f"**{paper.title}**")
            if paper.abstract:
                st.caption(truncate(paper.abstract))

            st.button(
                "Read",
                key=f"read-{paper.id}",
                on_click=self._navigator.open_reader,
                args=(paper.id,),
                use_container_width=True,
            )
            if paper.url:
                st.link_button("Open on arXiv", paper.url, use_container_width=True)
            render_download_button(paper)
