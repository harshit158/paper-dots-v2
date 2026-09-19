"""Tests for the view helpers that do not need a Streamlit runtime."""

from ui.common import ABSTRACT_PREVIEW_CHARS, truncate


def test_short_text_is_returned_unchanged():
    assert truncate("A short abstract.") == "A short abstract."


def test_whitespace_is_collapsed():
    assert truncate("line one\n\n  line two") == "line one line two"


def test_long_text_is_shortened_to_the_limit():
    result = truncate("word " * 200)

    assert len(result) <= ABSTRACT_PREVIEW_CHARS
    assert result.endswith("…")


def test_truncation_does_not_leave_a_trailing_space():
    result = truncate("alpha beta gamma delta", limit=12)

    assert result == "alpha beta…"


def test_text_exactly_at_the_limit_is_untouched():
    text = "x" * ABSTRACT_PREVIEW_CHARS

    assert truncate(text) == text
