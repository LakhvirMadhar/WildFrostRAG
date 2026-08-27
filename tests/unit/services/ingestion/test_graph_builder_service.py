"""Unit tests for GraphBuilderService's run() orchestration (T6.2).

GraphBuilderService.run() reproduces what scripts/ingest_data.py's main() used to do
directly: clear the database when asked, always run Stage 1 and Stage 2, and run
Stages 3/4 unless their skip flags are set. These tests exercise that orchestration
in isolation - every stage method is monkeypatched away on the service instance, and
a fake Driver/Session stands in for Neo4j - so only run()'s control flow is under
test, with zero Neo4j or network access required.

pytest-asyncio isn't a project dependency, so each async run() call is driven with
asyncio.run() from an ordinary (synchronous) test function.
"""

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from neo4j import Driver, Session

from wildfrost_rag.services.ingestion.graph_builder_service import (
    GraphBuilderService,
    PipelineData,
)


def _make_fake_driver(session: Session) -> Driver:
    """Build a fake Driver whose session() context manager yields `session`."""
    driver = MagicMock(spec=Driver)
    session_context = MagicMock()
    session_context.__enter__ = MagicMock(return_value=session)
    session_context.__exit__ = MagicMock(return_value=False)
    driver.session.return_value = session_context
    return driver


@dataclass
class _ServiceWithMocks:
    """A GraphBuilderService plus typed handles to its stubbed-out stage methods.

    Asserting via these fields (not service.stage_X) keeps mypy happy: accessed
    through the service, stage_X's static type is the real bound method, which
    has no assert_* attributes.
    """

    service: GraphBuilderService
    stage_1: AsyncMock
    stage_2: MagicMock
    stage_3: MagicMock
    stage_4: MagicMock


def _make_service(monkeypatch: pytest.MonkeyPatch, session: Session) -> _ServiceWithMocks:
    """Build a GraphBuilderService with every stage method stubbed out."""
    driver = _make_fake_driver(session)
    service = GraphBuilderService(driver=driver)

    stage_1 = AsyncMock(return_value=PipelineData(page_urls={}))
    stage_2 = MagicMock()
    stage_3 = MagicMock()
    stage_4 = MagicMock()
    monkeypatch.setattr(service, "stage_1_scrape_cards", stage_1)
    monkeypatch.setattr(service, "stage_2_enrich_data", stage_2)
    monkeypatch.setattr(service, "stage_3_populate_graph", stage_3)
    monkeypatch.setattr(service, "stage_4_document_ingestion", stage_4)

    fake_settings = MagicMock()
    monkeypatch.setattr(
        "wildfrost_rag.services.ingestion.graph_builder_service.get_settings",
        lambda: fake_settings,
    )

    return _ServiceWithMocks(service, stage_1, stage_2, stage_3, stage_4)


def _run(service: GraphBuilderService, **kwargs: bool) -> None:
    """Drive service.run() to completion via asyncio.run()."""
    asyncio.run(service.run(**kwargs))


def test_constructor_stores_the_injected_driver_without_constructing_one() -> None:
    """The service never builds its own driver - it only holds the one it's given."""
    fake_driver = MagicMock(spec=Driver)

    service = GraphBuilderService(driver=fake_driver)

    assert service.driver is fake_driver


def test_run_always_calls_stage_1_and_stage_2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage 1 (scrape) and Stage 2 (enrich) run regardless of the skip flags."""
    fake_session = MagicMock(spec=Session)
    built = _make_service(monkeypatch, fake_session)

    _run(
        built.service,
        skip_scrape=True,
        skip_graph=True,
        skip_vectors=True,
        clear_db=False,
        no_chunking=False,
    )

    built.stage_1.assert_awaited_once_with(skip_scrape=True)
    built.stage_2.assert_called_once()


def test_run_calls_stage_3_when_skip_graph_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage 3 (graph population) runs when --skip-graph is not set."""
    fake_session = MagicMock(spec=Session)
    built = _make_service(monkeypatch, fake_session)

    _run(
        built.service,
        skip_scrape=False,
        skip_graph=False,
        skip_vectors=True,
        clear_db=False,
        no_chunking=False,
    )

    built.stage_3.assert_called_once()


def test_run_skips_stage_3_when_skip_graph_is_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage 3 (graph population) is skipped when --skip-graph is set."""
    fake_session = MagicMock(spec=Session)
    built = _make_service(monkeypatch, fake_session)

    _run(
        built.service,
        skip_scrape=False,
        skip_graph=True,
        skip_vectors=True,
        clear_db=False,
        no_chunking=False,
    )

    built.stage_3.assert_not_called()


@pytest.mark.parametrize(
    ("no_chunking", "expected_split_text"),
    [(False, True), (True, False)],
)
def test_run_calls_stage_4_with_split_text_derived_from_no_chunking(
    monkeypatch: pytest.MonkeyPatch, no_chunking: bool, expected_split_text: bool
) -> None:
    """--no-chunking flips split_text to False; its absence keeps chunking on."""
    fake_session = MagicMock(spec=Session)
    built = _make_service(monkeypatch, fake_session)

    _run(
        built.service,
        skip_scrape=False,
        skip_graph=True,
        skip_vectors=False,
        clear_db=False,
        no_chunking=no_chunking,
    )

    args, kwargs = built.stage_4.call_args
    assert kwargs.get("split_text", args[-1] if args else None) == expected_split_text


def test_run_skips_stage_4_when_skip_vectors_is_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage 4 (document ingestion) is skipped when --skip-vectors is set."""
    fake_session = MagicMock(spec=Session)
    built = _make_service(monkeypatch, fake_session)

    _run(
        built.service,
        skip_scrape=False,
        skip_graph=True,
        skip_vectors=True,
        clear_db=False,
        no_chunking=False,
    )

    built.stage_4.assert_not_called()


def test_run_clears_the_database_when_clear_db_is_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """--clear-db runs clear_database against the injected driver's session."""
    fake_session = MagicMock(spec=Session)
    built = _make_service(monkeypatch, fake_session)

    _run(
        built.service,
        skip_scrape=True,
        skip_graph=True,
        skip_vectors=True,
        clear_db=True,
        no_chunking=False,
    )

    fake_session.execute_write.assert_called_once()
    called_fn: Any = fake_session.execute_write.call_args.args[0]
    assert called_fn.__name__ == "clear_database"


def test_run_does_not_touch_the_database_when_clear_db_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without --clear-db, run() never opens a session to clear anything."""
    fake_session = MagicMock(spec=Session)
    built = _make_service(monkeypatch, fake_session)

    _run(
        built.service,
        skip_scrape=True,
        skip_graph=True,
        skip_vectors=True,
        clear_db=False,
        no_chunking=False,
    )

    fake_session.execute_write.assert_not_called()
