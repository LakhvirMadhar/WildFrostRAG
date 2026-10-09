"""Unit tests for html_cache: where cached wiki HTML lives and how it's written."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from wildfrost_rag.core.config import get_settings
from wildfrost_rag.scraping.html_cache import cache_path, write_html


@pytest.fixture
def project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point PATH_PROJECT_ROOT at a temp dir.

    get_settings() is lru_cached process-wide, so the cache is cleared before
    and after - otherwise the real project root would stay cached, or this
    test's temp root would leak into later tests.
    """
    monkeypatch.setenv("PATH_PROJECT_ROOT", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_cache_path_is_under_structured_outputs_subdir(project_root: Path) -> None:
    """Cached HTML lives at <project_root>/data/structured_outputs/<subdir>/<name>.html."""
    path = cache_path("Bombom", "charms")

    assert path == project_root / "data" / "structured_outputs" / "charms" / "Bombom.html"


def test_cache_path_sanitizes_the_entity_name(project_root: Path) -> None:
    """The filename part goes through the same rule as wiki_page_filename."""
    assert cache_path("What?", "stats").name == "What.html"


def test_write_html_creates_missing_parent_directories(tmp_path: Path) -> None:
    """Writing into a folder that doesn't exist yet creates it."""
    path = tmp_path / "bells" / "nested" / "Sun Bell.html"

    write_html("<p>hi</p>", path)

    assert path.exists()


def test_write_html_removes_html_comments(tmp_path: Path) -> None:
    """Wiki pages carry HTML comments (parser reports etc.) that shouldn't be cached."""
    path = tmp_path / "page.html"

    write_html("<div><!-- NewPP limit report --><p>kept</p></div>", path)

    written = path.read_text(encoding="utf-8")
    assert "NewPP limit report" not in written
    assert "kept" in written
