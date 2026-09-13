"""Materializes retrieval_metrics and generation_taxonomy with fakes at the I/O boundary."""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from dagster import materialize
from neo4j import Driver

from wildfrost_rag.core.config import get_settings
from wildfrost_rag.defs.evaluation.assets import generation_taxonomy, retrieval_metrics
from wildfrost_rag.defs.resources import Neo4jResource
from wildfrost_rag.defs.retrieval.assets import retrieval_results
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


def test_retrieval_metrics_reads_the_path_retrieval_results_produced() -> None:
    """retrieval_metrics receives retrieval_results' actual output path, not a guessed one."""
    fake_experiment = RetrievalExperimentResult(
        results=[MagicMock()], experiment_dir=Path("outputs/run_1/retrievals/bm25/001")
    )
    fake_metrics_data = {"aggregate_metrics": {"avg_hit@1": 1.0}}

    with (
        patch(
            "wildfrost_rag.defs.retrieval.assets.RetrievalService.load_and_filter_queries",
            return_value=pd.DataFrame({"query_id": [1], "query": ["q"]}),
        ),
        patch(
            "wildfrost_rag.defs.retrieval.assets.RetrievalService.run_experiment",
            return_value=fake_experiment,
        ),
        patch(
            "wildfrost_rag.defs.evaluation.assets.calculate_metrics",
            return_value=fake_metrics_data,
        ) as fake_calculate,
        patch(
            "wildfrost_rag.defs.evaluation.assets.save_metrics",
            return_value="outputs/run_1/retrievals/bm25/001/metrics.json",
        ) as fake_save,
    ):
        result = materialize(
            [retrieval_results, retrieval_metrics],
            resources={"neo4j": _FakeNeo4jResource()},
            run_config={
                "ops": {
                    "retrieval_results": {"config": {"retriever_type": "BM25", "run_num": 1}},
                    "retrieval_metrics": {"config": {"k_values": [1, 5]}},
                }
            },
        )

    assert result.success
    assert result.output_for_node("retrieval_metrics") == (
        "outputs/run_1/retrievals/bm25/001/metrics.json"
    )
    called_path = fake_calculate.call_args[0][0]
    assert Path(called_path) == fake_experiment.experiment_dir
    assert fake_calculate.call_args[0][1] == [1, 5]
    fake_save.assert_called_once_with(called_path, fake_metrics_data)


def test_generation_taxonomy_calls_the_service_with_configured_path() -> None:
    """generation_taxonomy is parameterized entirely by its own config, no upstream asset."""
    with patch(
        "wildfrost_rag.defs.evaluation.assets.generate_taxonomy_from_annotations",
    ) as fake_generate:
        result = materialize(
            [generation_taxonomy],
            run_config={
                "ops": {
                    "generation_taxonomy": {
                        "config": {"generation_experiment_path": "outputs/run_1/generation/001"}
                    }
                }
            },
        )

    assert result.success
    fake_generate.assert_called_once()
    assert Path(fake_generate.call_args[0][0]) == Path("outputs/run_1/generation/001")
