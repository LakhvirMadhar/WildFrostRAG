#!/usr/bin/env python3
"""
Standalone CLI for batch auto-annotation of retrieval experiments.

Matches retrieved chunks against ground truth doc_references
and saves relevance annotations. Useful for re-running auto-annotation
after updating doc_references in the queries JSON.

Usage:
    python -m scripts.auto_annotate --experiment-path outputs/run_1/retrievals/bm25/001
    python -m scripts.auto_annotate --experiment-path outputs/run_1/retrievals/bm25/001 --queries-json queries/custom.json
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from gui.auto_annotator import run_auto_annotation
from utils.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="Batch auto-annotate retrieval results using ground truth URL matching"
    )
    parser.add_argument(
        "--experiment-path", type=str, required=True,
        help="Path to experiment directory (e.g., outputs/run_1/retrievals/bm25/001)"
    )
    parser.add_argument(
        "--queries-json", type=str, default=None,
        help="Path to queries JSON with doc_references (required for auto-annotation)"
    )

    args = parser.parse_args()

    experiment_path = Path(args.experiment_path)
    if not experiment_path.exists():
        logger.error(f"Experiment path does not exist: {experiment_path}")
        sys.exit(1)

    queries_json = Path(args.queries_json) if args.queries_json else None

    summary = run_auto_annotation(experiment_path, queries_json)

    print("\nAuto-annotation summary:")
    print(f"  Queries processed:  {summary['queries_processed']}")
    print(f"  Total chunks:       {summary['total_checked']}")
    print(f"  Auto-annotated:     {summary['auto_annotated']}")
    print(f"  Skipped (existing): {summary['skipped_existing']}")


if __name__ == "__main__":
    main()
