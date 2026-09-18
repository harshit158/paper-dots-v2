"""Shared fixtures.

The tests build the real object graph (database → repository → service) rather
than mocking it. The pieces are small enough that the real thing is faster to
reason about, and it keeps the wiring in ``app.py`` honest.
"""

from pathlib import Path

import pytest

from db import Database
from repository import PaperRepository
from services import PaperService


@pytest.fixture
def database(tmp_path: Path) -> Database:
    """A fresh, schema-initialised database for each test."""
    database = Database(tmp_path / "test.db")
    database.init_schema()
    yield database
    database.dispose()


@pytest.fixture
def papers(database: Database) -> PaperService:
    """A service wired to the throwaway database."""
    return PaperService(PaperRepository(database))
