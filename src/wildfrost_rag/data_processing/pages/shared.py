"""HTML helpers shared by more than one page parser under pages/."""

from bs4 import Tag


def extract_page_url(name_cell: Tag, base_url: str) -> str | None:
    """Return the wiki URL linked from a table's name cell, or None.

    Skips red links: the wiki marks links to pages that don't exist yet with
    class="new", and following one would only fetch an empty edit page.
    """
    link = name_cell.find("a")
    if link and link.get("href") and "new" not in (link.get("class") or []):
        return f"{base_url}{link['href']}"
    return None
