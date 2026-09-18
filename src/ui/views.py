"""Streamlit pages."""

from pathlib import Path

import streamlit as st
from foundry.observability import get_logger, get_tracer

from errors import ValidationError
from models import Paper
from services import PaperService

logger = get_logger(__name__)
tracer = get_tracer(__name__)

PDF_VIEWER_HEIGHT = 800
VIEW_LIBRARY = "Library"
VIEW_READER = "Reader"


class MainPage:
    """The main PaperDots page with library and reader views."""

    def __init__(self, papers: PaperService, papers_path: Path) -> None:
        self._papers = papers
        self._papers_path = papers_path

    def render(self, title: str, tagline: str) -> None:
        st.title(title)
        st.caption(tagline)

        # Read from the database on every rerun. Caching this list in
        # st.session_state would hide whether persistence actually works.
        papers = self._papers.list_papers()
        self._consume_pending_paper()
        self._reset_stale_selection(papers)

        view = (
            st.sidebar.segmented_control(
                "View",
                [VIEW_LIBRARY, VIEW_READER],
                default=VIEW_LIBRARY,
                key="view",
                width="stretch",
                label_visibility="collapsed",
            )
            or VIEW_LIBRARY
        )
        self._render_paper_picker(papers)

        if view == VIEW_READER:
            paper = self._selected_paper(papers)
            self._render_reader(paper)
        else:
            self._render_library(papers)

    def _render_paper_picker(self, papers: list[Paper]) -> None:
        if not papers:
            return

        paper_ids = [paper.id for paper in papers if paper.id is not None]
        st.sidebar.selectbox(
            "Paper",
            paper_ids,
            index=None,
            format_func=lambda paper_id: self._paper_label(papers, paper_id),
            key="selected_paper_id",
            placeholder="Select a paper",
            on_change=self._open_reader,
        )

    @staticmethod
    def _paper_label(papers: list[Paper], paper_id: int) -> str:
        paper = next((item for item in papers if item.id == paper_id), None)
        if paper is None:
            return str(paper_id)
        return paper.title if len(paper.title) <= 42 else f"{paper.title[:39]}…"

    @staticmethod
    def _selected_paper(papers: list[Paper]) -> Paper | None:
        selected_id = st.session_state.get("selected_paper_id")
        return next((paper for paper in papers if paper.id == selected_id), None)

    @staticmethod
    def _open_reader() -> None:
        st.session_state["view"] = VIEW_READER

    @staticmethod
    def _open_paper(paper_id: int) -> None:
        st.session_state["selected_paper_id"] = paper_id
        st.session_state["view"] = VIEW_READER

    @staticmethod
    def _reset_stale_selection(papers: list[Paper]) -> None:
        selected_id = st.session_state.get("selected_paper_id")
        paper_ids = {paper.id for paper in papers}
        if selected_id is not None and selected_id not in paper_ids:
            st.session_state.pop("selected_paper_id", None)

    @staticmethod
    def _consume_pending_paper() -> None:
        pending_id = st.session_state.pop("pending_paper_id", None)
        if pending_id is not None:
            st.session_state["selected_paper_id"] = pending_id
            st.session_state["view"] = VIEW_READER

    def _render_library(self, papers: list[Paper]) -> None:
        self._render_flash()
        self._render_add_form()

        st.subheader("Library")
        if not papers:
            st.info("No papers yet. Add one above — it will still be here after a restart.")
            return

        for position, paper in enumerate(papers, start=1):
            st.markdown(f"**{position}.** {paper.title}")
            if paper.authors:
                st.caption(paper.authors)
            if paper.abstract:
                st.write(paper.abstract)
            actions = st.columns([1, 1, 1])
            with actions[0]:
                st.button(
                    "Read",
                    key=f"read-{paper.id}",
                    on_click=self._open_paper,
                    args=(paper.id,),
                    use_container_width=True,
                )
            with actions[1]:
                if paper.url:
                    st.link_button("Open on arXiv", paper.url, use_container_width=True)
            with actions[2]:
                self._render_download_button(paper)

    def _render_reader(self, paper: Paper | None) -> None:
        self._render_flash()
        if paper is None:
            if self._papers.list_papers():
                st.info("No paper open. Pick one from the sidebar to start reading.")
            else:
                st.info("No paper open yet. Import a paper from the Library view.")
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
            self._render_download_button(paper)

    def _render_download_button(self, paper: Paper) -> None:
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

    @staticmethod
    def _render_flash() -> None:
        message = st.session_state.pop("flash_message", None)
        level = st.session_state.pop("flash_level", "success")
        if message is None:
            return
        if level == "info":
            st.info(message)
        else:
            st.success(message)

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
            st.session_state["pending_paper_id"] = paper.id
            st.session_state["flash_message"] = (
                f"“{paper.title}” is already in your library."
                if already_existed
                else f"Imported “{paper.title}”."
            )
            st.session_state["flash_level"] = "info" if already_existed else "success"
            st.rerun()
