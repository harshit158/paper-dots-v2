"""SQLite engine and schema setup.

``Database`` owns one engine and hands out short-lived sessions. The engine is
created once per process and shared: SQLAlchemy's engine is thread-safe and
pools its connections, which is exactly what Streamlit's
rerun-per-interaction model needs. Note that SQLAlchemy already passes
``check_same_thread=False`` for file-based SQLite databases, so it is not
set here.

This module deliberately does not import Streamlit. Caching the instance with
``@st.cache_resource`` is the entrypoint's job, which keeps ``Database``
usable from tests, scripts and a future CLI.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from foundry.observability import get_logger, get_tracer
from sqlmodel import Session, SQLModel, create_engine

logger = get_logger(__name__)
tracer = get_tracer(__name__)


def db_url(path: Path | str) -> str:
    """Build a SQLAlchemy URL for a SQLite file.

    An absolute path needs four slashes: ``sqlite:////abs/path.db``.
    """
    return f"sqlite:///{Path(path).resolve()}"


class Database:
    """A SQLite database: one engine, many short-lived sessions.

    The path is required rather than defaulted so there is exactly one place
    that decides where the database lives (``settings.DEFAULT_DB_PATH``).
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(db_url(self.path))
        logger.info("database engine created", extra={"path": str(self.path)})

    def init_schema(self) -> None:
        """Create any missing tables.

        Importing the models registers them on SQLModel's metadata; without
        that import ``create_all`` would have nothing to create. The import is
        local so this module does not depend on the domain models unless
        somebody actually wants a schema.

        This is not a migration tool: it never alters an existing table.
        """
        import models  # noqa: F401

        SQLModel.metadata.create_all(self.engine)
        logger.info("database schema initialized", extra={"path": str(self.path)})

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Yield a session that is always closed again.

        Sessions are per operation on purpose: nothing holding one may outlive
        a Streamlit rerun.
        """
        with Session(self.engine) as session:
            yield session

    def dispose(self) -> None:
        """Close the connection pool. Used by tests and shutdown hooks."""
        self.engine.dispose()
