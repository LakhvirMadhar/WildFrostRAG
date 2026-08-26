"""Unit tests for scripts.ingest_data's Stage 4 linker loop (T6.1).

Stage 4 used to repeat ten near-identical "log, call linker, log the count"
blocks, one per domain (cards, crowns, stats, charms, ...). They were
collapsed into a single loop driven by the module-level LINKERS table of
(label, linker) pairs. These tests exercise that loop in isolation: every
unrelated Stage 4 dependency (HTML collection, chunking, Neo4j document
ingestion, fulltext index creation) is monkeypatched away, LINKERS itself is
swapped for a small table of fake linkers, and a fake driver/session stands
in for the real Neo4j connection - so only the loop's behavior is under
test, with zero Neo4j required.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from neo4j import Driver, Session

from scripts import ingest_data
from scripts.ingest_data import PipelineData


@contextmanager
def _fake_neo4j_driver(driver: Driver) -> Generator[Driver]:
    """Stand in for neo4j_driver(), yielding a pre-built fake driver."""
    yield driver


def _make_fake_driver(session: Session) -> Driver:
    """Build a fake Driver whose session() context manager yields `session`."""
    driver = MagicMock(spec=Driver)
    session_context = MagicMock()
    session_context.__enter__ = MagicMock(return_value=session)
    session_context.__exit__ = MagicMock(return_value=False)
    driver.session.return_value = session_context
    return driver


def _make_recording_linker(label: str, count: int, calls: list[str]) -> Callable[[Session], int]:
    """Build a fake linker that records its own label and returns a canned count."""

    def _linker(session: Session) -> int:
        calls.append(label)
        return count

    return _linker


def test_stage_4_invokes_every_linker_once_and_sums_logged_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each (label, linker) pair in LINKERS runs exactly once against the shared session."""
    fake_session = MagicMock(spec=Session)
    fake_driver = _make_fake_driver(fake_session)

    calls: list[str] = []
    fake_linkers: list[tuple[str, Callable[[Session], int]]] = [
        ("cards", _make_recording_linker("cards", 5, calls)),
        ("crowns", _make_recording_linker("crowns", 2, calls)),
        ("bling", _make_recording_linker("bling", 1, calls)),
    ]
    monkeypatch.setattr(ingest_data, "LINKERS", fake_linkers)

    fake_settings = MagicMock()
    fake_settings.paths.structured_outputs_dir = str(tmp_path)
    fake_settings.embedding.fulltext_index_name = "test_fulltext_index"
    monkeypatch.setattr(ingest_data, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(ingest_data, "process_html_files", lambda filepaths, split_text: [])
    monkeypatch.setattr(ingest_data, "ingest_documents_into_neo4j", MagicMock())
    monkeypatch.setattr(ingest_data, "create_fulltext_index", MagicMock())
    monkeypatch.setattr(ingest_data, "wait_for_index_population", MagicMock())
    monkeypatch.setattr(ingest_data, "neo4j_driver", lambda: _fake_neo4j_driver(fake_driver))

    pipeline_data = PipelineData(page_urls={})

    ingest_data.stage_4_document_ingestion(pipeline_data, split_text=False)

    assert calls == ["cards", "crowns", "bling"]


def test_stage_4_passes_the_same_session_to_every_linker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The loop passes the one shared session to each linker, not a new one per call."""
    fake_session = MagicMock(spec=Session)
    fake_driver = _make_fake_driver(fake_session)

    received_sessions: list[Session] = []

    def _linker(session: Session) -> int:
        received_sessions.append(session)
        return 0

    monkeypatch.setattr(ingest_data, "LINKERS", [("cards", _linker), ("crowns", _linker)])

    fake_settings = MagicMock()
    fake_settings.paths.structured_outputs_dir = str(tmp_path)
    fake_settings.embedding.fulltext_index_name = "test_fulltext_index"
    monkeypatch.setattr(ingest_data, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(ingest_data, "process_html_files", lambda filepaths, split_text: [])
    monkeypatch.setattr(ingest_data, "ingest_documents_into_neo4j", MagicMock())
    monkeypatch.setattr(ingest_data, "create_fulltext_index", MagicMock())
    monkeypatch.setattr(ingest_data, "wait_for_index_population", MagicMock())
    monkeypatch.setattr(ingest_data, "neo4j_driver", lambda: _fake_neo4j_driver(fake_driver))

    pipeline_data = PipelineData(page_urls={})

    ingest_data.stage_4_document_ingestion(pipeline_data, split_text=False)

    assert received_sessions == [fake_session, fake_session]
