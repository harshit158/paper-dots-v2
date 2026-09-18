"""SQLModel table definitions."""

from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    """Naive UTC timestamp.

    SQLite's DATETIME storage format has no timezone offset, so a tz-aware
    value would be written as UTC and read back naive anyway. Returning naive
    UTC up front keeps comparisons consistent instead of mixing the two.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Paper(SQLModel, table=True):
    """A paper saved to the library.

    Only ``title`` is required, and only ``title`` is exposed in the UI yet.
    The optional metadata fields mirror the data model in ``PLAN.md`` because
    ``create_all`` only ever creates missing *tables* - it will not add a new
    column to an existing one. Declaring them now is cheaper than a migration.

    There is deliberately no ``owner_id``: there is no ``User`` table yet.
    """

    # Streamlit can reload application modules while retaining SQLModel's
    # process-wide metadata. Reusing the existing table prevents a rerun from
    # failing with ``Table 'paper' is already defined``.
    __table_args__ = {"extend_existing": True}

    id: int | None = Field(default=None, primary_key=True)
    title: str = Field(index=True)
    authors: str | None = None
    abstract: str | None = None
    url: str | None = Field(default=None, index=True)
    pdf_path: str | None = None
    public: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utcnow)
