#!/usr/bin/env python3
"""
Calculate Retrieval Metrics from Manual Evaluation for WildFrostRAG.

This script calculates retrieval metrics based on manually annotated relevance judgments:
1. Loads manually evaluated results from JSON files
2. Calculates retrieval metrics (Hit@k, MRR, etc.) based on manual relevance judgments
3. Saves calculated metrics to output files

Usage:
    python -m scripts.calculate_retrieval_metrics --input-path outputs/retrievers/run_1/vector_no/results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from src.rag.evaluation.retrieval_metrics import (
    hit_at_k,
    mrr,
    calculate_precision_at_k,
    calculate_recall_at_k
)
from src.utils.logger import logger


def calculate_metrics_from_manual_annotations(results_file_path: str, output_path: str = None):
    """
    Calculate retrieval metrics based on manually annotated relevance judgments.

    Args:
        results_file_path: Path to the JSON file containing retrieval results with manual annotations
        output_path: Path to save the calculated metrics (optional, defaults to input path + _metrics)
    """
    # Load the results file
    with open(results_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get('results', [])
    metadata = data.get('metadata', {})

    # Validate that the file contains relevance annotations
    if not results or 'relevance_annotations' not in results[0]:
        logger.error(f"The file {results_file_path} does not contain manual relevance annotations.")
        logger.error("Please ensure the results were manually evaluated first.")
        return

    # Calculate metrics for each query result
    all_query_metrics = []
    for result in results:
        query = result['query']
        retrieved_chunks = result['retrieved_chunks']
        relevance_annotations = result.get('relevance_annotations', [])

        # Validate that we have relevance annotations for this query
        if not relevance_annotations:
            logger.warning(f"No relevance annotations found for query: {query}")
            continue

        # Extract relevance judgments (assuming binary relevance: 0 for non-relevant, 1 for relevant)
        # The relevance_annotations should contain relevance scores for each retrieved chunk
        # Format: [{'chunk_id': ..., 'relevance': 0 or 1}, ...]
        relevance_judgments = []
        for chunk, annotation in zip(retrieved_chunks, relevance_annotations):
            relevance = annotation.get('relevance', 0)  # Default to 0 if not specified
            relevance_judgments.append(relevance)

        # Calculate metrics for this query
        query_metrics = {
            'query_id': result.get('query_id'),
            'query': query,
            'metrics': {}
        }

        # Calculate metrics at different k values
        k_values = [1, 3, 5, 10]
        for k in k_values:
            # Create lists of relevant IDs for metric calculation
            retrieved_ids = list(range(len(relevance_judgments[:k])))  # Use indices as IDs
            relevant_ids = [i for i, rel in enumerate(relevance_judgments) if rel > 0]

            # Calculate metrics
            hit_k = hit_at_k(retrieved_ids, relevant_ids, k)
            precision_k = calculate_precision_at_k(retrieved_ids, relevant_ids, k)
            recall_k = calculate_recall_at_k(retrieved_ids, relevant_ids, k)

            query_metrics['metrics'][f'hit@{k}'] = hit_k
            query_metrics['metrics'][f'precision@{k}'] = precision_k
            query_metrics['metrics'][f'recall@{k}'] = recall_k

        # Calculate MRR for this query (using all results)
        retrieved_ids = list(range(len(relevance_judgments)))
        relevant_ids = [i for i, rel in enumerate(relevance_judgments) if rel > 0]
        query_metrics['metrics']['mrr'] = mrr(retrieved_ids, relevant_ids)

        all_query_metrics.append(query_metrics)

    # Calculate aggregate metrics
    aggregate_metrics = {}
    if all_query_metrics:
        for metric_name in ['hit@1', 'hit@3', 'hit@5', 'hit@10', 'precision@1', 'precision@3', 'precision@5', 'precision@10', 'recall@1', 'recall@3', 'recall@5', 'recall@10', 'mrr']:
            values = [qm['metrics'].get(metric_name, 0) for qm in all_query_metrics]
            if values:
                aggregate_metrics[f'avg_{metric_name}'] = sum(values) / len(values)

    # Prepare final metrics data
    metrics_data = {
        'input_file': results_file_path,
        'metadata': metadata,
        'aggregate_metrics': aggregate_metrics,
        'per_query_metrics': all_query_metrics
    }

    # Determine output path
    if output_path is None:
        input_path = Path(results_file_path)
        output_path = input_path.parent / f"{input_path.stem}_metrics.json"

    # Save metrics to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=4, default=str)

    logger.info(f"Calculated metrics saved to {output_path}")
    logger.info(f"Aggregate metrics: {aggregate_metrics}")


def main():
    parser = argparse.ArgumentParser(description="Calculate retrieval metrics from manually annotated results")
    parser.add_argument("--input-path", type=str, required=True,
                       help="Path to the JSON file containing retrieval results with manual annotations")
    parser.add_argument("--output-path", type=str,
                       help="Path to save calculated metrics (optional, defaults to input path + _metrics)")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_path):
        logger.error(f"Input file {args.input_path} not found.")
        sys.exit(1)

    # Calculate metrics
    calculate_metrics_from_manual_annotations(args.input_path, args.output_path)


if __name__ == "__main__":
    main()