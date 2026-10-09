"""Unit tests for wiki_page_filename."""

from wildfrost_rag.domain.wiki_files import wiki_page_filename


def test_wiki_page_filename_appends_html_extension() -> None:
    """A plain name just gets ".html"."""
    assert wiki_page_filename("Bombom") == "Bombom.html"


def test_wiki_page_filename_keeps_spaces_and_apostrophes() -> None:
    """Real entity names like "Lil' Gazi" are valid filenames as-is."""
    assert wiki_page_filename("Lil' Gazi") == "Lil' Gazi.html"


def test_wiki_page_filename_strips_characters_filesystems_reject() -> None:
    """Characters Windows rejects in filenames are removed, not replaced."""
    assert wiki_page_filename('What: Is/This? "A*B"<C>|D\\') == "What IsThis ABCD.html"
