"""The rules the app enforces, independent of Streamlit.

Views validate nothing themselves: they hand raw user input to a service and
turn a :class:`ValidationError` into a message. One service method is one use
case, which is the seam to grow into later features (deduplication, PDF
handling, ownership) without the UI learning about any of it.
"""

from pathlib import Path

from foundry.observability import get_logger, get_tracer

from arxiv import ArxivClient
from errors import ValidationError
from models import Paper
from repository import PaperRepository

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class PaperService:
    """Use cases for the paper library."""

    def __init__(
        self,
        repository: PaperRepository,
        arxiv_client: ArxivClient | None = None,
    ) -> None:
        self._repository = repository
        self._arxiv = arxiv_client or ArxivClient()

    def add_paper(self, title: str) -> Paper:
        """Create a paper from raw user input.

        The single place a blank title is rejected: previously the UI and the
        persistence helper both checked.
        """
        title = title.strip()
        if not title:
            raise ValidationError("Enter a title before saving.")
        paper = self._repository.add(Paper(title=title))
        logger.info("paper added", extra={"paper_id": paper.id})
        return paper

    def list_papers(self) -> list[Paper]:
        """Return the library, newest first."""
        papers = self._repository.list_all()
        logger.info("paper library loaded", extra={"count": len(papers)})
        return papers

    def import_arxiv_paper(self, url: str, papers_path: Path) -> tuple[Paper, bool]:
        """Import an arXiv paper and return ``(paper, already_existed)``."""
        normalized_url, arxiv_id = self._arxiv.normalize_url(url)
        logger.info("importing arXiv paper", extra={"arxiv_id": arxiv_id})
        existing = self._repository.find_by_url(normalized_url)
        if existing is not None:
            logger.info("arXiv paper already imported", extra={"arxiv_id": arxiv_id})
            return existing, True

        metadata = self._arxiv.fetch_metadata(arxiv_id)
        pdf_path = papers_path / f"{arxiv_id.replace('/', '_')}.pdf"
        try:
            self._arxiv.download_pdf(arxiv_id, pdf_path)
            paper = self._repository.add(
                Paper(
                    title=metadata.title,
                    authors=metadata.authors,
                    abstract=metadata.abstract,
                    url=normalized_url,
                    pdf_path=str(pdf_path),
                )
            )
        except Exception:
            pdf_path.unlink(missing_ok=True)
            raise
        return paper, False
