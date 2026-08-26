"""Retrieval metrics for evaluating RAG systems in WildFrostRAG.

This module provides standard information retrieval metrics to evaluate
the effectiveness of different retrieval strategies (vector search, BM25, hybrid, etc.).
"""

from typing import Any


def hit_at_k(retrieved_ids: list[Any], relevant_ids: list[Any], k: int) -> int:
    """Calculate Hit@k metric.

    Hit@k measures whether at least one relevant document appears in the top-k
    retrieved results. It's a boolean measure (1 if hit, 0 if miss).

    Args:
        retrieved_ids: List of IDs of retrieved documents in ranked order
        relevant_ids: List of IDs of relevant documents (ground truth)
        k: Number of top results to consider

    Returns:
        1 if at least one relevant ID is in the top k retrieved IDs, else 0
    """
    top_k = retrieved_ids[:k]
    for rid in top_k:
        if rid in relevant_ids:
            return 1
    return 0


def mrr(retrieved_ids: list[Any], relevant_ids: list[Any]) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    MRR measures the ranking quality by considering the rank of the first
    relevant document. Higher scores indicate better ranking.

    Args:
        retrieved_ids: List of IDs of retrieved documents in ranked order
        relevant_ids: List of IDs of relevant documents (ground truth)

    Returns:
        Reciprocal rank of the first relevant item (1/rank), or 0.0 if none found
    """
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def calculate_precision_at_k(retrieved_ids: list[Any], relevant_ids: list[Any], k: int) -> float:
    """Calculate Precision@k metric.

    Precision@k measures the proportion of relevant documents among the top-k
    retrieved results. It focuses on the accuracy of the top results.

    Args:
        retrieved_ids: List of IDs of retrieved documents in ranked order
        relevant_ids: List of IDs of relevant documents (ground truth)
        k: Number of top results to consider

    Returns:
        Proportion of relevant items in the top k retrieved items
    """
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_count = sum(1 for rid in top_k if rid in relevant_ids)
    return relevant_count / k


def calculate_recall_at_k(retrieved_ids: list[Any], relevant_ids: list[Any], k: int) -> float:
    """Calculate Recall@k metric.

    Recall@k measures the proportion of total relevant documents that are
    found in the top-k retrieved results. It focuses on the completeness
    of the retrieval.

    Args:
        retrieved_ids: List of IDs of retrieved documents in ranked order
        relevant_ids: List of IDs of relevant documents (ground truth)
        k: Number of top results to consider

    Returns:
        Proportion of total relevant items found in the top k retrieved items
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_count = sum(1 for rid in top_k if rid in relevant_ids)
    return relevant_count / len(relevant_ids)
