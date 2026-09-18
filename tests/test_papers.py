"""Persistence tests for the paper library.

These use a real file-backed SQLite database under ``tmp_path`` rather than
``:memory:``. An in-memory SQLite database lives inside a single connection
(SQLAlchemy gives it ``SingletonThreadPool``), so a table created on one
checkout is invisible on the next.

The wired-up fixtures live in ``conftest.py``; these tests only talk to the
service.
"""

from db import Database
from repository import PaperRepository
from services import PaperService


def test_list_is_empty_on_a_fresh_database(papers):
    assert papers.list_papers() == []


def test_add_paper_then_list(papers):
    papers.add_paper("Attention Is All You Need")

    assert [paper.title for paper in papers.list_papers()] == ["Attention Is All You Need"]


def test_added_paper_gets_an_id_and_timestamp(papers):
    paper = papers.add_paper("Deep Residual Learning")

    assert paper.id is not None
    assert paper.created_at is not None


def test_title_is_stripped(papers):
    papers.add_paper("   Neural Machine Translation   ")

    assert papers.list_papers()[0].title == "Neural Machine Translation"


def test_papers_are_listed_newest_first(papers):
    papers.add_paper("First")
    papers.add_paper("Second")

    assert [paper.title for paper in papers.list_papers()] == ["Second", "First"]


def test_returned_paper_is_readable_after_its_session_closes(papers):
    """Guards the repository's refresh-before-close: no live session needed."""
    paper = papers.add_paper("Detached but readable")

    assert paper.title == "Detached but readable"
    assert paper.id is not None


def test_paper_survives_a_new_engine(tmp_path):
    """The automated twin of the real check: restart the app, data is still there."""
    db_path = tmp_path / "persist.db"

    first = Database(db_path)
    first.init_schema()
    PaperService(PaperRepository(first)).add_paper("Survives a restart")
    first.dispose()

    second = Database(db_path)
    try:
        titles = [paper.title for paper in PaperService(PaperRepository(second)).list_papers()]
        assert titles == ["Survives a restart"]
    finally:
        second.dispose()
