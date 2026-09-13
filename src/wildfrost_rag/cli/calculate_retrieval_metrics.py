#!/usr/bin/env python3
"""Calculate retrieval metrics from annotations for WildFrostRAG.

Reads results.json (flat array) and annotations.json (separate file)
from an experiment directory, joins on (query_id, chunk_index), and
computes standard IR metrics (Hit@k, Precision@k, Recall@k, MRR).

Usage:
    python -m wildfrost_rag.cli.calculate_retrieval_metrics --experiment-path outputs/run_1/retrievals/bm25/001
"""

import argparse
import sys
from pathlib import Path

from wildfrost_rag.core.logger import logger
from wildfrost_rag.services.evaluation.metrics_service import (
    DEFAULT_K_VALUES,
    calculate_metrics,
    save_metrics,
)


def main() -> None:
    """Calculate retrieval metrics from experiment annotations."""
    parser = argparse.ArgumentParser(description="Calculate retrieval metrics from annotations")
    parser.add_argument(
        "--experiment-path",
        type=str,
        required=True,
        help="Path to experiment directory (e.g., outputs/run_1/retrievals/bm25/001)",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default=None,
        help="Comma-separated k values for Hit@k, Precision@k, Recall@k (default: 1,3,5,10)",
    )

    args = parser.parse_args()
    experiment_path = Path(args.experiment_path)
    k_values = [int(k.strip()) for k in args.k_values.split(",")] if args.k_values else None

    if not experiment_path.exists():
        logger.error(f"Experiment path does not exist: {experiment_path}")
        sys.exit(1)

    try:
        metrics_data = calculate_metrics(experiment_path, k_values)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    save_metrics(experiment_path, metrics_data)

    agg = metrics_data["aggregate_metrics"]
    print(f"\nMetrics for {experiment_path}")
    print(
        f"  Queries: {metrics_data['annotated_queries']}/{metrics_data['total_queries']} annotated"
    )
    if agg:
        used_k = k_values or DEFAULT_K_VALUES
        for k in used_k:
            print(f"  Hit@{k}:{' ' * (8 - len(str(k)))}{agg.get(f'avg_hit@{k}', 0):.3f}")
        print(f"  MRR:         {agg.get('avg_mrr', 0):.3f}")
        print(f"  Precision@1: {agg.get('avg_precision@1', 0):.3f}")
        max_k = max(used_k)
        print(
            f"  Recall@{max_k}:{' ' * (5 - len(str(max_k)))}{agg.get(f'avg_recall@{max_k}', 0):.3f}"
        )
    else:
        print("  No metrics calculated (no annotated queries)")


if __name__ == "__main__":
    main()
