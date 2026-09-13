"""Materializes retrieval_results with fakes at the I/O boundary.

RetrievalService is patched entirely (its own unit tests cover its real
behavior); Neo4jResource.get_driver yields a MagicMock(spec=Driver) instead
of a live connection.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from dagster import materialize
from neo4j import Driver

from wildfrost_rag.core.config import get_settings
from wildfrost_rag.defs.resources import Neo4jResource
from wildfrost_rag.defs.retrieval.assets import RetrievalRunConfig, retrieval_results
from wildfrost_rag.domain.retriever_type import RetrieverType
from wildfrost_rag.core.embedder_type import EmbedderType
from wildfrost_rag.services.retrieval.retrieval_service import RetrievalExperimentResult


@pytest.fixture(autouse=True)
def _fake_neo4j_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stub NEO4J_* env vars so get_settings() validates without a real Neo4j instance."""
    monkeypatch.setenv("NEO4J_URI", "bolt://fake-host:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "fake")
    monkeypatch.setenv("NEO4J_PASSWORD", "fake")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _FakeNeo4jResource(Neo4jResource):
    """A Neo4jResource whose driver is a MagicMock(spec=Driver), never a live connection."""

    def get_driver(self) -> AbstractContextManager[Driver]:
        @contextmanager
        def _fake_driver_context() -> Iterator[Driver]:
            yield MagicMock(spec=Driver)

        return _fake_driver_context()


def test_retrieval_results_materializes_and_returns_the_experiment_dir() -> None:
    """The asset materializes, passes config through, and returns the experiment dir path."""
    fake_experiment = RetrievalExperimentResult(
        results=[MagicMock(), MagicMock()],
        experiment_dir=Path("outputs/run_1/retrievals/bm25/001"),
    )
    with (
        patch(
            "wildfrost_rag.defs.retrieval.assets.RetrievalService.load_and_filter_queries",
            return_value=pd.DataFrame({"query_id": [1, 2], "query": ["a", "b"]}),
        ),
        patch(
            "wildfrost_rag.defs.retrieval.assets.RetrievalService.run_experiment",
            return_value=fake_experiment,
        ) as fake_run_experiment,
    ):
        result = materialize(
            [retrieval_results],
            resources={"neo4j": _FakeNeo4jResource()},
            run_config={
                "ops": {"retrieval_results": {"config": {"retriever_type": "BM25", "run_num": 1}}}
            },
        )

    assert result.success
    assert result.output_for_node("retrieval_results") == str(fake_experiment.experiment_dir)
    called_kwargs = fake_run_experiment.call_args.kwargs
    assert called_kwargs["retriever_type"] is RetrieverType.BM25
    assert called_kwargs["run_num"] == 1


def test_retrieval_run_config_defaults() -> None:
    """Only retriever_type and run_num are required; everything else has a sane default."""
    config = RetrievalRunConfig(retriever_type=RetrieverType.VECTOR, run_num=1)

    assert config.embedder is EmbedderType.HF
    assert config.k == 10
    assert config.chunking is False
    assert config.query_ids is None
