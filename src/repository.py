"""Paper persistence.

One class per aggregate, and every method opens and closes its own session:
nothing here holds a long-lived session or returns an object that still needs
one, so callers cannot accidentally keep an ORM instance alive across a
Streamlit rerun.
"""

from foundry.observability import get_logger, get_tracer
from sqlmodel import select

from db import Database
from models import Paper

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class PaperRepository:
    """Stores and retrieves :class:`Paper` rows.

    Takes the :class:`Database` rather than a raw engine so the session
    lifecycle stays in one place and a test can hand over a throwaway file.
    """

    def __init__(self, database: Database) -> None:
        self._database = database

    def add(self, paper: Paper) -> Paper:
        """Persist ``paper`` and return the stored row."""
        with self._database.session() as session:
            session.add(paper)
            session.commit()
            # Reload before the session closes so the returned object is fully
            # populated and safe to read after it is detached.
            session.refresh(paper)
        logger.info("paper persisted", extra={"paper_id": paper.id})
        return paper

    def list_all(self) -> list[Paper]:
        """Return all saved papers, newest first."""
        statement = select(Paper).order_by(Paper.created_at.desc(), Paper.id.desc())
        with self._database.session() as session:
            papers = list(session.exec(statement).all())
        logger.info("papers listed", extra={"count": len(papers)})
        return papers

    def find_by_url(self, url: str) -> Paper | None:
        """Return the paper imported from ``url``, if it exists."""
        statement = select(Paper).where(Paper.url == url)
        with self._database.session() as session:
            paper = session.exec(statement).first()
        logger.info("paper lookup completed", extra={"found": paper is not None})
        return paper
