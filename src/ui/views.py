"""Streamlit pages."""

from pathlib import Path

import streamlit as st
from foundry.observability import get_logger, get_tracer

from errors import ValidationError
from services import PaperService

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class MainPage:
    """The main PaperDots page.

    Keep the first page in one class. Additional page classes can be added
    later without introducing a navigation framework now.
    """

    def __init__(self, papers: PaperService, papers_path: Path) -> None:
        self._papers = papers
        self._papers_path = papers_path

    def render(self, title: str, tagline: str) -> None:
        st.title(title)
        st.caption(tagline)
        self._render_add_form()

        st.subheader("Library")

        # Read from the database on every rerun. Caching this list in
        # st.session_state would hide whether persistence actually works.
        papers = self._papers.list_papers()

        if not papers:
            st.info("No papers yet. Add one above — it will still be here after a restart.")
            return

        for position, paper in enumerate(papers, start=1):
            st.markdown(f"**{position}.** {paper.title}")
            if paper.authors:
                st.caption(paper.authors)
            if paper.abstract:
                st.write(paper.abstract)
            if paper.url:
                st.link_button("Open on arXiv", paper.url)
            if paper.pdf_path:
                with open(paper.pdf_path, "rb") as pdf:
                    st.download_button(
                        "Download PDF",
                        pdf,
                        file_name=Path(paper.pdf_path).name,
                        key=f"pdf-{paper.id}",
                    )

    def _render_add_form(self) -> None:
        """Draw the form and act on a submit in the same rerun."""
        with st.form("add_paper", clear_on_submit=True):
            url = st.text_input("arXiv URL", placeholder="https://arxiv.org/abs/1706.03762")
            submitted = st.form_submit_button(
                "Import paper",
                type="primary",
                use_container_width=True,
            )

        if not submitted:
            return

        logger.info("paper import submitted")
        try:
            with st.spinner("Fetching paper details and PDF…"):
                paper, already_existed = self._papers.import_arxiv_paper(url, self._papers_path)
        except ValidationError as error:
            logger.warning("paper import validation failed", extra={"error": str(error)})
            st.warning(str(error))
        else:
            if already_existed:
                st.info(f"“{paper.title}” is already in your library.")
            else:
                st.success(f"Imported “{paper.title}”.")
