"""PaperDots — Streamlit entrypoint and composition root.

Run with: uv run streamlit run src/app.py

This is the only module that knows about every other one: it reads settings,
builds the database, wires repositories into services, hands the services to
views and passes the finished shell to Streamlit. Everything below it depends
inwards (view → service → repository → database), never the other way round.
"""

import streamlit as st
from foundry.observability import ObservabilityConfig, get_logger, get_tracer, init_observability

from db import Database
from repository import PaperRepository
from services import PaperService
from settings import Settings
from ui.views import MainPage

settings = Settings()
config = ObservabilityConfig(app_name=settings.app_name)
init_observability(config)

logger = get_logger(__name__)
tracer = get_tracer(__name__)


@st.cache_resource
def get_database(path: str) -> Database:
    """Return the shared database, creating the schema on first call.

    Cached for the lifetime of the server so every rerun and every user session
    reuses one connection pool - and so the schema is created once per server
    start, not once per rerun.
    """
    logger.info("initializing database", extra={"path": path})
    database = Database(path)
    database.init_schema()
    logger.info("database initialized", extra={"path": path})
    return database


def build_page(settings: Settings) -> MainPage:
    """Compose the object graph for one rerun.

    Cheap to call every time: the expensive part (the engine) comes from the
    cache, and everything else is a thin wrapper around it.
    """
    database = get_database(str(settings.db_path))
    return MainPage(PaperService(PaperRepository(database)), settings.papers_path)


def main() -> None:
    """Configure the page, build the app and render it."""
    logger.info("rendering PaperDots")

    # set_page_config must be the first Streamlit call in the script, so it
    # comes before anything that touches the runtime - including the cached
    # database above, which registers itself with Streamlit's cache.
    st.set_page_config(
        page_title=settings.page_title,
        page_icon=settings.page_icon,
        layout=settings.layout,
    )

    build_page(settings).render(settings.page_title, settings.tagline)


# Streamlit re-executes this module on every interaction, so the call is
# unconditional rather than guarded by ``if __name__ == "__main__"``.
main()
