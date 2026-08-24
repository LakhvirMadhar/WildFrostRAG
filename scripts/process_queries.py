#!/usr/bin/env python3
"""Process queries CSV into JSON format for the annotation GUI.

This script converts the queries CSV file into a structured JSON format that:
- Supports multiple doc_references per query (as a list)
- Includes metadata about the source file
- Can be updated by the annotation GUI when new relevant sources are found

Usage:
    python -m scripts.process_queries                           # Use default paths
    python -m scripts.process_queries --input queries/my.csv    # Custom input
    python -m scripts.process_queries --output queries/out.json # Custom output
"""

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Any

from utils.logger import logger


def load_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Load queries from CSV file.

    Expected columns: query_id, query, ground_truth, doc_reference
    """
    queries = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            query: dict[str, Any] = {
                "query_id": int(row.get("query_id", 0)),
                "query": row.get("query", "").strip(),
                "ground_truth": row.get("ground_truth", "").strip(),
                "doc_references": [],
            }

            # Handle doc_reference - could be single URL or comma-separated
            doc_ref = row.get("doc_reference", "").strip()
            if doc_ref:
                # Split by comma in case multiple URLs
                refs = [r.strip() for r in doc_ref.split(",") if r.strip()]
                query["doc_references"] = refs

            queries.append(query)

    return queries


def convert_csv_to_json(csv_path: Path, output_path: Path | None = None) -> Path:
    """Convert queries CSV to JSON format.

    Args:
        csv_path: Path to input CSV file
        output_path: Path for output JSON (default: same name with .json extension)

    Returns:
        Path to the created JSON file
    """
    csv_path = Path(csv_path)

    if output_path is None:
        output_path = csv_path.with_suffix(".json")
    else:
        output_path = Path(output_path)

    logger.info(f"Converting {csv_path} to {output_path}")

    # Load queries from CSV
    queries = load_csv(csv_path)

    # Build JSON structure
    data = {
        "queries": queries,
        "metadata": {
            "source_csv": str(csv_path),
            "created_at": datetime.now().isoformat(),
            "total_queries": len(queries),
            "version": 1,
        },
    }

    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Created {output_path} with {len(queries)} queries")

    return output_path


def main() -> int:
    """Convert queries CSV to JSON format with ground truth references."""
    parser = argparse.ArgumentParser(description="Convert queries CSV to JSON format")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="queries/simple_reference_based_queries.csv",
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: same name with .json)",
    )

    args = parser.parse_args()

    csv_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    if not csv_path.exists():
        logger.error(f"Input file not found: {csv_path}")
        return 1

    output = convert_csv_to_json(csv_path, output_path)
    print(f"Created: {output}")

    return 0


if __name__ == "__main__":
    exit(main())
