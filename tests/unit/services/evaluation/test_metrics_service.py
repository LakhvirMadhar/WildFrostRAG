"""Unit tests for calculate_metrics - a pure function over on-disk experiment files."""

import json
from pathlib import Path
from typing import Any

import pytest

from wildfrost_rag.services.evaluation.metrics_service import calculate_metrics


def _write_experiment(
    tmp_path: Path,
    results: list[dict[str, Any]],
    annotations: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> Path:
    (tmp_path / "results.json").write_text(json.dumps(results), encoding="utf-8")
    (tmp_path / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")
    if config is not None:
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def test_calculate_metrics_raises_when_results_missing(tmp_path: Path) -> None:
    """No results.json -> a clear FileNotFoundError, not a confusing downstream failure."""
    with pytest.raises(FileNotFoundError, match="results.json"):
        calculate_metrics(tmp_path)


def test_calculate_metrics_raises_when_annotations_missing(tmp_path: Path) -> None:
    """No annotations.json -> a clear FileNotFoundError telling the user to annotate first."""
    (tmp_path / "results.json").write_text("[]", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="annotations.json"):
        calculate_metrics(tmp_path)


def test_calculate_metrics_scores_a_relevant_top_chunk_as_a_hit(tmp_path: Path) -> None:
    """A query whose top-ranked chunk is annotated relevant scores hit@1 = 1 and mrr = 1."""
    results = [
        {
            "query_id": 1,
            "query": "what does bombom do",
            "retrieved_chunks": [{"source_url": "a"}, {"source_url": "b"}],
        }
    ]
    annotations = {"1": {"relevance_annotations": [{"chunk_index": 0, "is_relevant": True}]}}
    experiment_path = _write_experiment(tmp_path, results, annotations)

    metrics_data = calculate_metrics(experiment_path, k_values=[1])

    query_metrics = metrics_data["per_query_metrics"][0]["metrics"]
    assert query_metrics["hit@1"] == 1
    assert query_metrics["mrr"] == 1
    assert metrics_data["aggregate_metrics"]["avg_hit@1"] == 1
    assert metrics_data["unannotated_queries"] == 0


def test_calculate_metrics_treats_unannotated_queries_as_a_miss(tmp_path: Path) -> None:
    """A query with no matching annotation entry gets zeroed metrics, not skipped silently."""
    results = [
        {"query_id": 1, "query": "q", "retrieved_chunks": [{"source_url": "a"}]},
    ]
    experiment_path = _write_experiment(tmp_path, results, annotations={})

    metrics_data = calculate_metrics(experiment_path, k_values=[1])

    assert metrics_data["unannotated_queries"] == 1
    assert metrics_data["per_query_metrics"][0]["metrics"]["hit@1"] == 0


def test_calculate_metrics_reads_retriever_type_from_config(tmp_path: Path) -> None:
    """retriever_type in the output comes from config.json, defaulting to 'unknown' without it."""
    experiment_path = _write_experiment(
        tmp_path, results=[], annotations={}, config={"retriever_type": "bm25"}
    )

    metrics_data = calculate_metrics(experiment_path)

    assert metrics_data["retriever_type"] == "bm25"
