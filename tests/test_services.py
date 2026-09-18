"""Validation tests: the rules enforced before the database is touched.

These live against the service because that is where the rules moved to. The
UI no longer decides what a valid title is.
"""

import pytest

from services import ValidationError


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_blank_title_is_rejected(papers, blank):
    with pytest.raises(ValidationError):
        papers.add_paper(blank)


def test_a_rejected_title_creates_no_row(papers):
    with pytest.raises(ValidationError):
        papers.add_paper("   ")

    assert papers.list_papers() == []


def test_validation_error_is_a_value_error():
    """Callers written against the old ``ValueError`` contract keep working."""
    assert issubclass(ValidationError, ValueError)
