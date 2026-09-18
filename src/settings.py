"""Runtime configuration.

Settings are resolved once at startup and then passed down explicitly, so no
module has to reach for a global to find the database or the page title. The
default database location lives here and nowhere else.

``BaseSettings`` *is* the "read it from the environment" behaviour, so
constructing ``Settings()`` is the whole API - no hand-written ``from_env``.
Every field can be overridden by name, which is handy for a scratch run or a
second deployment::

    PAPERDOTS_DB_PATH=/tmp/scratch.db uv run streamlit run src/app.py
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "paperdots.db"
DEFAULT_PAPERS_PATH = PROJECT_ROOT / "data" / "papers"


class Settings(BaseSettings):
    """Configuration for one run of the app.

    Frozen so it is hashable and safe to hand across a ``@st.cache_resource``
    boundary: no rerun can mutate what the next one sees.
    """

    model_config = SettingsConfigDict(env_prefix="PAPERDOTS_", frozen=True)

    db_path: Path = DEFAULT_DB_PATH
    papers_path: Path = DEFAULT_PAPERS_PATH
    page_title: str = "PaperDots"
    page_icon: str = "📄"
    layout: Literal["centered", "wide"] = "wide"
    tagline: str = "Read papers. Capture ideas. Connect knowledge."
    app_name: str = "paperdots"
