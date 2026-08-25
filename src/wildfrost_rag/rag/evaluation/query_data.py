"""Query JSON data access for WildFrostRAG evaluation and annotation.

These functions read/write the queries JSON file (queries + doc_references +
metadata). They're consumed by both the annotation GUI and evaluation
scripts, so they live here rather than in scripts/process_queries.py, which
is a CLI entry point and should not be imported from src/ (see
docs/migration_plan — "depend inward").
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from wildfrost_rag.utils.logger import logger


def load_queries_json(json_path: Path) -> dict[str, Any]:
    """Load queries from JSON file.

    Args:
        json_path: Path to queries JSON file

    Returns:
        Dict with 'queries' list and 'metadata'
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        logger.error(f"Expected dict in {json_path}, got {type(data)}")
        raise TypeError(f"Expected dict in {json_path}, got {type(data)}")

    return data


def save_queries_json(data: dict[str, Any], json_path: Path) -> None:
    """Save queries to JSON file.

    Args:
        data: Dict with 'queries' list and 'metadata'
        json_path: Path to save to
    """
    # Update metadata
    data["metadata"]["updated_at"] = datetime.now().isoformat()

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def add_doc_reference(json_path: Path, query_id: int, url: str) -> bool:
    """Add a doc_reference URL to a query (if not already present).

    This is called by the annotation GUI when a user marks a chunk
    as relevant and its source URL isn't already in doc_references.

    Args:
        json_path: Path to queries JSON file
        query_id: Query ID to update
        url: URL to add to doc_references

    Returns:
        True if URL was added, False if already present
    """
    data = load_queries_json(json_path)

    for query in data["queries"]:
        if query["query_id"] == query_id:
            if url in query["doc_references"]:
                logger.debug(f"URL {url} already in query {query_id} doc_references")
                return False

            query["doc_references"].append(url)
            save_queries_json(data, json_path)
            logger.info(f"Added {url} to query {query_id} doc_references")
            return True

    logger.warning(f"Query {query_id} not found in {json_path}")
    return False


def get_query_ground_truth(json_path: Path, query_id: int) -> dict[str, Any] | None:
    """Get ground truth data for a specific query.

    Args:
        json_path: Path to queries JSON file
        query_id: Query ID to look up

    Returns:
        Dict with 'ground_truth' and 'doc_references', or None if not found
    """
    data = load_queries_json(json_path)

    for query in data["queries"]:
        if query["query_id"] == query_id:
            return {
                "ground_truth": query.get("ground_truth", ""),
                "doc_references": query.get("doc_references", []),
            }

    return None
