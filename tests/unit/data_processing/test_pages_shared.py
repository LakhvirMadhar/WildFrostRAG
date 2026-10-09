"""Unit tests for the helpers shared by page parsers."""

from bs4 import BeautifulSoup, Tag

from wildfrost_rag.data_processing.pages.shared import extract_page_url

_BASE_URL = "https://wildfrostwiki.com"


def _name_cell(inner_html: str) -> Tag:
    cell = BeautifulSoup(f"<td>{inner_html}</td>", "html.parser").td
    assert cell is not None
    return cell


def test_extract_page_url_joins_base_url_and_href() -> None:
    """A normal wiki link becomes an absolute URL."""
    cell = _name_cell('<a href="/Sun_Bell">Sun Bell</a>')

    assert extract_page_url(cell, _BASE_URL) == "https://wildfrostwiki.com/Sun_Bell"


def test_extract_page_url_skips_red_links() -> None:
    """Links to wiki pages that don't exist yet (class="new") give no URL."""
    cell = _name_cell('<a href="/Missing?action=edit&redlink=1" class="new">Missing</a>')

    assert extract_page_url(cell, _BASE_URL) is None


def test_extract_page_url_returns_none_without_a_link() -> None:
    """A name cell with plain text has no page to link to."""
    assert extract_page_url(_name_cell("Plain name"), _BASE_URL) is None
