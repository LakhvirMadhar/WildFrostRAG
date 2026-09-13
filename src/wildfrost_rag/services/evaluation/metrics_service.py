"""Computes retrieval metrics (Hit@k, Precision@k, Recall@k, MRR).

Reads an experiment's results.json and annotations.json.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow

from wildfrost_rag.core.logger import logger
from wildfrost_rag.services.evaluation.mlflow_tracking import get_or_create_run
from wildfrost_rag.services.evaluation.retrieval_metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    hit_at_k,
    mrr,
)

DEFAULT_K_VALUES = [1, 3, 5, 10]


def _build_relevance_map(annotations: dict[str, Any], query_id: int) -> dict[int, bool]:
    """Build chunk_index -> is_relevant map from annotations for a single query."""
    query_ann = annotations.get(str(query_id), {})
    relevance_list = query_ann.get("relevance_annotations", [])
    return {
        ann["chunk_index"]: ann.get("is_relevant", False)
        for ann in relevance_list
        if "chunk_index" in ann
    }


def calculate_metrics(  # noqa: C901
    experiment_path: Path, k_values: list[int] | None = None
) -> dict[str, Any]:
    """Calculate retrieval metrics for an experiment.

    Args:
        experiment_path: Path to experiment directory containing results.json and annotations.json
        k_values: List of k values for hit/precision/recall metrics

    Returns:
        Metrics dict with aggregate and per-query metrics

    Raises:
        FileNotFoundError: results.json or annotations.json is missing.
    """
    results_path = experiment_path / "results.json"
    annotations_path = experiment_path / "annotations.json"
    config_path = experiment_path / "config.json"

    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found at {experiment_path}")

    if not annotations_path.exists():
        raise FileNotFoundError(
            f"annotations.json not found at {experiment_path} - "
            "run auto-annotation or manual annotation first"
        )

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    with open(annotations_path, encoding="utf-8") as f:
        annotations = json.load(f)

    config = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

    if k_values is None:
        k_values = DEFAULT_K_VALUES

    per_query_metrics = []
    unannotated_queries = []

    for result in results:
        query_id = result.get("query_id")
        query = result.get("query", "")
        chunks = result.get("retrieved_chunks", [])
        n_chunks = len(chunks)

        relevance_map = _build_relevance_map(annotations, query_id)

        if not relevance_map:
            unannotated_queries.append(query_id)

        # retrieved_ids = chunk indices in rank order
        retrieved_ids = list(range(n_chunks))
        # relevant_ids = indices of chunks annotated as relevant.
        # Unannotated queries get an empty list -> all metrics = 0 (treated as failure).
        relevant_ids = [idx for idx in range(n_chunks) if relevance_map.get(idx, False)]

        query_metrics = {
            "query_id": query_id,
            "query": query,
            "n_chunks": n_chunks,
            "n_relevant": len(relevant_ids),
            "metrics": {},
        }

        for k in k_values:
            query_metrics["metrics"][f"hit@{k}"] = hit_at_k(retrieved_ids, relevant_ids, k)
            query_metrics["metrics"][f"precision@{k}"] = calculate_precision_at_k(
                retrieved_ids, relevant_ids, k
            )
            query_metrics["metrics"][f"recall@{k}"] = calculate_recall_at_k(
                retrieved_ids, relevant_ids, k
            )

        query_metrics["metrics"]["mrr"] = mrr(retrieved_ids, relevant_ids)
        per_query_metrics.append(query_metrics)

    if unannotated_queries:
        logger.warning(
            f"Skipped {len(unannotated_queries)} unannotated queries: {unannotated_queries}"
        )

    aggregate_metrics = {}
    if per_query_metrics:
        metric_names = []
        for k in k_values:
            metric_names.extend([f"hit@{k}", f"precision@{k}", f"recall@{k}"])
        metric_names.append("mrr")

        for name in metric_names:
            values = [qm["metrics"][name] for qm in per_query_metrics]
            aggregate_metrics[f"avg_{name}"] = sum(values) / len(values)

    metrics_data = {
        "experiment_path": str(experiment_path),
        "retriever_type": config.get("retriever_type", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(results),
        "annotated_queries": len(per_query_metrics),
        "unannotated_queries": len(unannotated_queries),
        "aggregate_metrics": aggregate_metrics,
        "per_query_metrics": per_query_metrics,
    }

    return metrics_data


def save_metrics(experiment_path: Path, metrics_data: dict[str, Any]) -> Path:
    """Save metrics.json to the experiment directory, and log the aggregate metrics to MLflow.

    Reuses the MLflow run the same experiment's retrieval_results already
    created (matched by run name, e.g. "bm25/007") rather than starting a new one.

    Returns the path written.
    """
    output_path = experiment_path / "metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, default=str)
    logger.info(f"Metrics saved to {output_path}")

    run_name = "/".join(experiment_path.parts[-2:])
    with get_or_create_run(run_name):
        # MLflow metric names allow only alphanumerics/_/-/./space//, not "@" -
        # "avg_hit@1" -> "avg_hit_at_1". The JSON file keeps the original "@k" keys.
        mlflow_metrics = {
            name.replace("@", "_at_"): value
            for name, value in metrics_data["aggregate_metrics"].items()
        }
        mlflow.log_metrics(mlflow_metrics)
        mlflow.log_artifact(str(output_path))

    return output_path
