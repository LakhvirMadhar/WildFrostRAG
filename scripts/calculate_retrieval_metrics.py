#!/usr/bin/env python3
"""
Calculate retrieval metrics from annotations for WildFrostRAG.

Reads results.json (flat array) and annotations.json (separate file)
from an experiment directory, joins on (query_id, chunk_index), and
computes standard IR metrics (Hit@k, Precision@k, Recall@k, MRR).

Usage:
    python -m scripts.calculate_retrieval_metrics --experiment-path outputs/run_1/retrievals/bm25/001
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from rag.evaluation.retrieval_metrics import (
    hit_at_k,
    mrr,
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from utils.logger import logger


DEFAULT_K_VALUES = [1, 3, 5, 10]


def _build_relevance_map(annotations: dict, query_id: int) -> dict[int, bool]:
    """
    Build chunk_index -> is_relevant map from annotations for a single query.

    Returns:
        Dict mapping chunk_index to is_relevant bool
    """
    query_ann = annotations.get(str(query_id), {})
    relevance_list = query_ann.get('relevance_annotations', [])
    return {
        ann['chunk_index']: ann.get('is_relevant', False)
        for ann in relevance_list
        if 'chunk_index' in ann
    }


def calculate_metrics(experiment_path: Path, k_values: list[int] | None = None) -> dict:
    """
    Calculate retrieval metrics for an experiment.

    Args:
        experiment_path: Path to experiment directory containing results.json and annotations.json

    Returns:
        Metrics dict with aggregate and per-query metrics
    """
    results_path = experiment_path / 'results.json'
    annotations_path = experiment_path / 'annotations.json'
    config_path = experiment_path / 'config.json'

    if not results_path.exists():
        logger.error(f"results.json not found at {experiment_path}")
        sys.exit(1)

    if not annotations_path.exists():
        logger.error(f"annotations.json not found at {experiment_path}")
        logger.error("Run auto-annotation or manual annotation first.")
        sys.exit(1)

    # Load flat results array
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Load annotations
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    # Load config for metadata
    config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

    if k_values is None:
        k_values = DEFAULT_K_VALUES

    # Calculate per-query metrics
    per_query_metrics = []
    unannotated_queries = []

    for result in results:
        query_id = result.get('query_id')
        query = result.get('query', '')
        chunks = result.get('retrieved_chunks', [])
        n_chunks = len(chunks)

        # Build relevance map from annotations
        relevance_map = _build_relevance_map(annotations, query_id)

        if not relevance_map:
            unannotated_queries.append(query_id)

        # retrieved_ids = chunk indices in rank order
        retrieved_ids = list(range(n_chunks))
        # relevant_ids = indices of chunks annotated as relevant
        # Unannotated queries get an empty list -> all metrics = 0 (treated as failure)
        relevant_ids = [idx for idx in range(n_chunks) if relevance_map.get(idx, False)]

        query_metrics = {
            'query_id': query_id,
            'query': query,
            'n_chunks': n_chunks,
            'n_relevant': len(relevant_ids),
            'metrics': {},
        }

        for k in k_values:
            query_metrics['metrics'][f'hit@{k}'] = hit_at_k(retrieved_ids, relevant_ids, k)
            query_metrics['metrics'][f'precision@{k}'] = calculate_precision_at_k(retrieved_ids, relevant_ids, k)
            query_metrics['metrics'][f'recall@{k}'] = calculate_recall_at_k(retrieved_ids, relevant_ids, k)

        query_metrics['metrics']['mrr'] = mrr(retrieved_ids, relevant_ids)
        per_query_metrics.append(query_metrics)

    if unannotated_queries:
        logger.warning(f"Skipped {len(unannotated_queries)} unannotated queries: {unannotated_queries}")

    # Calculate aggregate metrics
    aggregate_metrics = {}
    if per_query_metrics:
        metric_names = []
        for k in k_values:
            metric_names.extend([f'hit@{k}', f'precision@{k}', f'recall@{k}'])
        metric_names.append('mrr')

        for name in metric_names:
            values = [qm['metrics'][name] for qm in per_query_metrics]
            aggregate_metrics[f'avg_{name}'] = sum(values) / len(values)

    # Build output
    metrics_data = {
        'experiment_path': str(experiment_path),
        'retriever_type': config.get('retriever_type', 'unknown'),
        'timestamp': datetime.now().isoformat(),
        'total_queries': len(results),
        'annotated_queries': len(per_query_metrics),
        'unannotated_queries': len(unannotated_queries),
        'aggregate_metrics': aggregate_metrics,
        'per_query_metrics': per_query_metrics,
    }

    return metrics_data


def main():
    parser = argparse.ArgumentParser(
        description="Calculate retrieval metrics from annotations"
    )
    parser.add_argument(
        "--experiment-path", type=str, required=True,
        help="Path to experiment directory (e.g., outputs/run_1/retrievals/bm25/001)"
    )
    parser.add_argument(
        "--k-values", type=str, default=None,
        help="Comma-separated k values for Hit@k, Precision@k, Recall@k (default: 1,3,5,10)"
    )

    args = parser.parse_args()
    experiment_path = Path(args.experiment_path)
    k_values = [int(k.strip()) for k in args.k_values.split(',')] if args.k_values else None

    if not experiment_path.exists():
        logger.error(f"Experiment path does not exist: {experiment_path}")
        sys.exit(1)

    metrics_data = calculate_metrics(experiment_path, k_values)

    # Save metrics
    output_path = experiment_path / 'metrics.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, default=str)

    logger.info(f"Metrics saved to {output_path}")

    # Print summary
    agg = metrics_data['aggregate_metrics']
    print(f"\nMetrics for {experiment_path}")
    print(f"  Queries: {metrics_data['annotated_queries']}/{metrics_data['total_queries']} annotated")
    if agg:
        used_k = k_values or DEFAULT_K_VALUES
        for k in used_k:
            print(f"  Hit@{k}:{' ' * (8 - len(str(k)))}{agg.get(f'avg_hit@{k}', 0):.3f}")
        print(f"  MRR:         {agg.get('avg_mrr', 0):.3f}")
        print(f"  Precision@1: {agg.get('avg_precision@1', 0):.3f}")
        max_k = max(used_k)
        print(f"  Recall@{max_k}:{' ' * (5 - len(str(max_k)))}{agg.get(f'avg_recall@{max_k}', 0):.3f}")
    else:
        print("  No metrics calculated (no annotated queries)")


if __name__ == "__main__":
    main()
