"""Unit tests for RetrievalService.

Retriever classes are real (their constructors do no I/O - only Neo4j calls
during search()), but get_query_embed_fn and the experiment-tracking/save
functions are faked so these tests never touch a real embedding model, Neo4j,
or the filesystem beyond a tmp_path CSV.
"""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlflow
import pandas as pd
import pytest
from neo4j import Driver

from wildfrost_rag.domain.retriever_type import RetrieverType
from wildfrost_rag.models.experiment_config import EmbeddingConfig, QueryStats, RetrievalConfig
from wildfrost_rag.services.embeddings.embedder_type import EmbedderType
from wildfrost_rag.services.evaluation import mlflow_tracking
from wildfrost_rag.services.retrieval.bm25_retriever import BM25Retriever
from wildfrost_rag.services.retrieval.neo4j_vector_search import Neo4jVectorSearch
from wildfrost_rag.services.retrieval.retrieval_service import RetrievalService
from wildfrost_rag.services.retrieval.text2cypher_retriever import Text2CypherRetriever

_MODULE = "wildfrost_rag.services.retrieval.retrieval_service"


def _write_queries_csv(tmp_path: Path, rows: list[tuple[int, str]]) -> Path:
    path = tmp_path / "queries.csv"
    pd.DataFrame(rows, columns=["query_id", "query"]).to_csv(path, index=False)
    return path


def test_load_and_filter_queries_applies_include_filter(tmp_path: Path) -> None:
    """query_ids keeps only the listed rows."""
    csv_path = _write_queries_csv(tmp_path, [(1, "a"), (2, "b"), (3, "c")])

    df = RetrievalService().load_and_filter_queries(str(csv_path), query_ids=[1, 3])

    assert sorted(df["query_id"].tolist()) == [1, 3]


def test_load_and_filter_queries_applies_exclude_filter(tmp_path: Path) -> None:
    """exclude_ids drops the listed rows."""
    csv_path = _write_queries_csv(tmp_path, [(1, "a"), (2, "b"), (3, "c")])

    df = RetrievalService().load_and_filter_queries(str(csv_path), exclude_ids=[2])

    assert sorted(df["query_id"].tolist()) == [1, 3]


def test_load_and_filter_queries_raises_when_file_missing(tmp_path: Path) -> None:
    """A missing CSV path raises FileNotFoundError instead of a confusing pandas error."""
    with pytest.raises(FileNotFoundError):
        RetrievalService().load_and_filter_queries(str(tmp_path / "missing.csv"))


def test_load_and_filter_queries_raises_when_nothing_left_after_filter(tmp_path: Path) -> None:
    """Filtering out every row raises rather than silently running zero queries."""
    csv_path = _write_queries_csv(tmp_path, [(1, "a")])

    with pytest.raises(ValueError, match="No queries"):
        RetrievalService().load_and_filter_queries(str(csv_path), exclude_ids=[1])


def test_get_retriever_builds_bm25_without_an_embed_fn() -> None:
    """BM25 isn't vector-based - get_query_embed_fn must never be called for it."""
    driver = MagicMock(spec=Driver)

    with patch(f"{_MODULE}.get_query_embed_fn") as fake_get_embed_fn:
        retriever = RetrievalService().get_retriever(RetrieverType.BM25, driver)

    fake_get_embed_fn.assert_not_called()
    assert isinstance(retriever, BM25Retriever)


def test_get_retriever_builds_vector_with_an_embed_fn() -> None:
    """Vector-based retrievers must be given the embed function for the configured provider."""
    driver = MagicMock(spec=Driver)
    fake_embed_fn = MagicMock()

    with patch(f"{_MODULE}.get_query_embed_fn", return_value=fake_embed_fn) as fake_get_embed_fn:
        retriever = RetrievalService().get_retriever(
            RetrieverType.VECTOR, driver, embedder=EmbedderType.OPENAI
        )

    fake_get_embed_fn.assert_called_once_with("openai")
    assert isinstance(retriever, Neo4jVectorSearch)


def test_get_retriever_passes_text2cypher_prompt_kwarg() -> None:
    """text2cypher_prompt reaches the Text2CypherRetriever constructor via **kwargs."""
    driver = MagicMock(spec=Driver)
    fake_prompt = MagicMock()

    retriever = RetrievalService().get_retriever(
        RetrieverType.TEXT2CYPHER, driver, text2cypher_prompt=fake_prompt
    )

    assert isinstance(retriever, Text2CypherRetriever)


def test_run_experiment_resolves_text2cypher_prompt_and_records_its_version() -> None:
    """text2cypher retrievers load their prompt and record its version in saved metadata."""
    fake_prompt = MagicMock(prompt_version_name="TEXT2CYPHER_PROMPT_V1")
    service = RetrievalService()

    with (
        patch.object(service, "load_text2cypher_prompt", return_value=fake_prompt) as fake_load,
        patch.object(service, "get_retriever", return_value=MagicMock()) as fake_get_retriever,
        patch.object(service, "_run_with_retriever", return_value=[]) as fake_run_with_retriever,
    ):
        asyncio.run(
            service.run_experiment(
                driver=MagicMock(spec=Driver),
                df=pd.DataFrame({"query_id": [1], "query": ["q"]}),
                retriever_type=RetrieverType.TEXT2CYPHER,
                run_num=1,
                chunking=False,
            )
        )

    fake_load.assert_called_once_with("TEXT2CYPHER_PROMPT_V1")
    fake_get_retriever.assert_called_once()
    assert fake_get_retriever.call_args.kwargs["text2cypher_prompt"] is fake_prompt
    assert (
        fake_run_with_retriever.call_args.kwargs["text2cypher_prompt_version"]
        == "TEXT2CYPHER_PROMPT_V1"
    )


def test_run_experiment_records_stop_word_settings_only_for_relevant_retrievers() -> None:
    """BM25 cares about sw_query/sw_docs; vector search doesn't - config shouldn't claim it does."""
    service = RetrievalService()

    with (
        patch.object(service, "get_retriever", return_value=MagicMock()),
        patch.object(service, "_run_with_retriever", return_value=[]) as fake_run_with_retriever,
    ):
        asyncio.run(
            service.run_experiment(
                driver=MagicMock(spec=Driver),
                df=pd.DataFrame({"query_id": [1], "query": ["q"]}),
                retriever_type=RetrieverType.VECTOR,
                run_num=1,
                chunking=False,
            )
        )

    assert "sw_query" not in fake_run_with_retriever.call_args.kwargs
    assert "sw_docs" not in fake_run_with_retriever.call_args.kwargs


def test_save_experiment_artifacts_logs_to_real_mlflow_with_none_embedding_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BM25 has no embedder - embedding_provider/vector_index_name are None.

    Uses a real, isolated MLflow backend rather than mocking mlflow.log_params -
    a mock would hide whether MLflow's own param validation actually accepts None
    values (it does), the same class of bug the metric-name "@" case caught.
    """
    monkeypatch.setattr(mlflow_tracking, "TRACKING_URI", f"sqlite:///{tmp_path}/test_mlflow.db")
    config = RetrievalConfig(
        retrieval_id="bm25/001",
        run_number=1,
        timestamp="2026-01-01T00:00:00",
        retriever_type="bm25",
        chunking=False,
        k=10,
        description="test run",
        query_stats=QueryStats(total=1, successful=1, failed=0),
        embedding=EmbeddingConfig(model="all-MiniLM-L6-v2", provider=None, vector_index_name=None),
    )
    service = RetrievalService()

    service._save_experiment_artifacts(
        experiment_dir=tmp_path,
        config=config,
        results=[],
        individual_results=[],
        retriever=MagicMock(),
        retriever_type=RetrieverType.BM25,
        experiment_id="001",
    )

    with mlflow_tracking.get_or_create_run("bm25/001") as run:
        run_id = run.info.run_id
    logged_params = mlflow.get_run(run_id).data.params
    assert logged_params["embedding_provider"] == "None"
    assert logged_params["description"] == "test run"
