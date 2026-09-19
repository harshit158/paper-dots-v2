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
from ui.common import PAGE_FEED, PAGE_LIBRARY, PAGE_READER, Navigator
from ui.feed import FeedView
from ui.library import LibraryView
from ui.reader import ReaderView

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


def build_pages(settings: Settings) -> list[st.Page]:
    """Compose the object graph and the navigation for one rerun.

    Cheap to call every time: the expensive part (the engine) comes from the
    cache, and everything else is a thin wrapper around it.

    ``st.navigation`` must be called from the entrypoint, so the pages are
    built here and handed back rather than assembled inside a view. The
    navigator is registered with them so a view can switch pages by name.
    """
    database = get_database(str(settings.db_path))
    papers = PaperService(PaperRepository(database))
    navigator = Navigator()

    feed_page = st.Page(
        FeedView().render,
        title="Feed",
        icon=":material/dynamic_feed:",
        url_path=PAGE_FEED,
        default=True,
    )
    reader_page = st.Page(
        ReaderView(papers, settings.papers_path, navigator).render,
        title="Reader",
        icon=":material/menu_book:",
        url_path=PAGE_READER,
    )
    library_page = st.Page(
        LibraryView(papers, navigator).render,
        title="Library",
        icon=":material/library_books:",
        url_path=PAGE_LIBRARY,
    )

    navigator.register(PAGE_READER, reader_page)
    navigator.register(PAGE_LIBRARY, library_page)
    return [feed_page, reader_page, library_page]


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

    st.title(settings.page_title)
    st.caption(settings.tagline)
    st.navigation(build_pages(settings)).run()


# Streamlit re-executes this module on every interaction, so the call is
# unconditional rather than guarded by ``if __name__ == "__main__"``.
main()
