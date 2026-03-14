"""
Batch auto-annotation for retrieval experiments.

Matches retrieved chunks against ground truth doc_references using
bidirectional URL substring matching, then saves relevance annotations
via the experiment adapter. This decouples auto-annotation from the
GUI so it can run as a batch step after retrieval.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from src.gui.experiment_adapters import get_adapter
from src.utils.logger import logger


def _load_ground_truth(queries_json_path: Path) -> Dict[int, list[str]]:
    """
    Load doc_references for each query from the queries JSON.

    Returns:
        Dict mapping query_id -> list of doc_reference URLs
    """
    import json

    with open(queries_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ground_truth = {}
    for query in data.get('queries', []):
        query_id = query['query_id']
        doc_refs = query.get('doc_references', [])
        ground_truth[query_id] = doc_refs

    return ground_truth


def _url_matches(source_url: str, doc_refs: list[str]) -> bool:
    """
    Check if source_url matches any doc_reference via bidirectional substring.

    This handles cases like:
    - source_url = "https://wildfrostwiki.com/Mimik"
    - doc_ref    = "https://wildfrostwiki.com/Mimik"  (exact match)

    And edge cases where one URL might be a substring of the other
    (e.g., trailing slashes, anchors, etc.).
    """
    if not source_url:
        return False
    for ref in doc_refs:
        if source_url in ref or ref in source_url:
            return True
    return False


def run_auto_annotation(
    experiment_path: Path,
    queries_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run batch auto-annotation on a retrieval experiment.

    For each query's retrieved chunks, checks if the chunk's source_url
    matches any ground truth doc_reference. If so, saves is_relevant=True
    with auto_populated=True flag.

    Skips chunks that already have an annotation (manual or auto).

    Args:
        experiment_path: Path to experiment directory (e.g., outputs/run_1/retrievals/bm25/001)
        queries_json_path: Path to queries JSON with doc_references.
                          If None, auto-annotation is skipped.

    Returns:
        Summary dict with counts: total_checked, auto_annotated, skipped_existing, queries_processed
    """
    experiment_path = Path(experiment_path)
    if queries_json_path is None:
        logger.info("No queries JSON path provided, skipping auto-annotation")
        return {'total_checked': 0, 'auto_annotated': 0, 'skipped_existing': 0, 'queries_processed': 0}

    # Load ground truth
    ground_truth = _load_ground_truth(queries_json_path)
    logger.info(f"Loaded ground truth for {len(ground_truth)} queries")

    # Get adapter (handles results + annotations loading)
    adapter = get_adapter(experiment_path)
    queries = adapter.get_queries()

    # Load existing annotations to check what's already annotated
    existing_annotations = adapter.load_annotations()

    total_checked = 0
    auto_annotated = 0
    skipped_existing = 0
    queries_processed = 0

    for query_result in queries:
        query_id = query_result.query_id
        doc_refs = ground_truth.get(query_id, [])

        if not doc_refs:
            continue

        queries_processed += 1

        # Get existing relevance annotations for this query
        query_annotations = existing_annotations.get(str(query_id), {})
        existing_relevance = query_annotations.get('relevance_annotations', [])

        # Build set of already-annotated chunk indices
        annotated_indices = {
            ann.get('chunk_index') for ann in existing_relevance
            if ann.get('chunk_index') is not None
        }

        for chunk_idx, chunk in enumerate(query_result.retrieved_chunks):
            total_checked += 1

            # Skip if already annotated
            if chunk_idx in annotated_indices:
                skipped_existing += 1
                continue

            source_url = chunk.get('source_url', '')
            if _url_matches(source_url, doc_refs):
                adapter.save_chunk_relevance(
                    query_id, chunk_idx, True, auto_populated=True
                )
                auto_annotated += 1

    summary = {
        'total_checked': total_checked,
        'auto_annotated': auto_annotated,
        'skipped_existing': skipped_existing,
        'queries_processed': queries_processed,
    }

    logger.info(
        f"Auto-annotation complete: {auto_annotated} annotated, "
        f"{skipped_existing} skipped (existing), "
        f"{total_checked} total chunks checked across {queries_processed} queries"
    )

    return summary
