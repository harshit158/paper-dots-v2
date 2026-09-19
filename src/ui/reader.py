"""The Reader view: import a paper and read it."""

from pathlib import Path

import streamlit as st
from foundry.observability import get_logger, get_tracer

from errors import ValidationError
from models import Paper
from services import PaperService
from ui.common import (
    PDF_VIEWER_HEIGHT,
    Navigator,
    paper_label,
    render_download_button,
    render_flash,
)

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class ReaderView:
    """Import a paper by URL and read the selected one."""

    def __init__(
        self,
        papers: PaperService,
        papers_path: Path,
        navigator: Navigator,
    ) -> None:
        self._papers = papers
        self._papers_path = papers_path
        self._navigator = navigator

    def render(self) -> None:
        # Read from the database on every rerun. Caching this list in
        # st.session_state would hide whether persistence actually works.
        papers = self._papers.list_papers()
        self._consume_pending_paper()
        self._reset_stale_selection(papers)
        self._render_paper_picker(papers)
        render_flash()

        paper = self._selected_paper(papers)
        self._render_import_form(collapsed=paper is not None)

        if paper is None:
            self._render_empty_state(papers)
            return

        left, right = st.columns([2, 1], gap="large")
        with left:
            st.subheader(paper.title)
            if paper.pdf_path and Path(paper.pdf_path).is_file():
                st.pdf(paper.pdf_path, height=PDF_VIEWER_HEIGHT, key=f"pdf-{paper.id}")
            else:
                st.warning("The PDF file is not available for this paper.")

        with right:
            st.subheader("Insights")
            if paper.authors:
                st.caption(paper.authors)
            if paper.abstract:
                with st.expander("Abstract"):
                    st.write(paper.abstract)
            st.info("No insights yet — capturing arrives next.")
            if paper.url:
                st.link_button("Open on arXiv", paper.url, use_container_width=True)
            render_download_button(paper)

    def _render_paper_picker(self, papers: list[Paper]) -> None:
        if not papers:
            return

        paper_ids = [paper.id for paper in papers if paper.id is not None]
        st.sidebar.selectbox(
            "Paper",
            paper_ids,
            index=None,
            format_func=lambda paper_id: paper_label(papers, paper_id),
            key="selected_paper_id",
            placeholder="Select a paper",
        )

    @staticmethod
    def _selected_paper(papers: list[Paper]) -> Paper | None:
        selected_id = st.session_state.get("selected_paper_id")
        return next((paper for paper in papers if paper.id == selected_id), None)

    @staticmethod
    def _reset_stale_selection(papers: list[Paper]) -> None:
        selected_id = st.session_state.get("selected_paper_id")
        paper_ids = {paper.id for paper in papers}
        if selected_id is not None and selected_id not in paper_ids:
            st.session_state.pop("selected_paper_id", None)

    @staticmethod
    def _consume_pending_paper() -> None:
        """Adopt a paper chosen on another page.

        Runs before the picker is drawn: a widget's value cannot be assigned
        after the widget exists in the same run.
        """
        pending_id = st.session_state.pop("pending_paper_id", None)
        if pending_id is not None:
            st.session_state["selected_paper_id"] = pending_id

    @staticmethod
    def _render_empty_state(papers: list[Paper]) -> None:
        if papers:
            st.info("No paper open. Pick one from the sidebar to start reading.")
        else:
            st.info("No papers yet. Import one above to start reading.")

    def _render_import_form(self, collapsed: bool) -> None:
        """Draw the import form and act on a submit in the same rerun.

        Collapsed into an expander once a paper is open, so the reading surface
        stays the focus, and expanded when the library is empty because then it
        is the only thing to do.
        """
        if collapsed:
            with st.expander("Import a paper"):
                self._render_import_fields()
        else:
            st.subheader("Import a paper")
            self._render_import_fields()

    def _render_import_fields(self) -> None:
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
            st.session_state["pending_paper_id"] = paper.id
            st.session_state["flash_message"] = (
                f"“{paper.title}” is already in your library."
                if already_existed
                else f"Imported “{paper.title}”."
            )
            st.session_state["flash_level"] = "info" if already_existed else "success"
            st.rerun()
