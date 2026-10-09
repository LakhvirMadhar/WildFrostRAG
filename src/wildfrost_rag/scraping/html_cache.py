"""Where a wiki entity's individual page HTML is cached, and how it's written.

One copy of the logic every model used to carry its own version of
(save_path/save_html on BellInfo, CharmInfo, StatInfo, CardInfo).
"""

from pathlib import Path

from bs4 import BeautifulSoup, Comment

from wildfrost_rag.core.config import get_settings
from wildfrost_rag.domain.wiki_files import wiki_page_filename


def cache_path(name: str, subdir: str) -> Path:
    """Return where the cached HTML for the entity `name` lives, under structured_outputs/`subdir`."""
    return get_settings().paths.structured_outputs_dir / subdir / wiki_page_filename(name)


def _clean_html(html: str) -> str:
    """Remove HTML comments and pretty-print, matching what the models' save_html() wrote."""
    soup = BeautifulSoup(html, "html.parser")
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    return soup.prettify()


def write_html(html: str, path: Path) -> None:
    """Write cleaned HTML to `path`, creating parent directories as needed.

    Raises:
        OSError: if the file can't be written. Callers decide whether one
            failed page should stop a scrape.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_clean_html(html), encoding="utf-8")
