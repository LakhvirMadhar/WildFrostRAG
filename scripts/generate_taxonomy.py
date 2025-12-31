#!/usr/bin/env python3
"""
Taxonomy Generation Script for WildFrostRAG.

This script generates axial codes (higher-level categories) from open codes
using LLMs. It loads qualitative coding results from CSV files and creates
structured taxonomies for understanding failure modes and patterns.

Usage:
    python -m scripts.generate_taxonomy                                    # Run with defaults
    python -m scripts.generate_taxonomy --filepath queries/my_queries.csv   # Custom input file
    python -m scripts.generate_taxonomy --column "my_coding_column"        # Custom column name
    python -m scripts.generate_taxonomy --output taxonomy.md               # Custom output file
    python -m scripts.generate_taxonomy --model gpt-4o                     # Different model
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.evaluation.taxonomy import generate_taxonomy_from_csv
from src.utils.logger import logger


async def main():
    parser = argparse.ArgumentParser(description="Generate taxonomy from qualitative codes")
    parser.add_argument(
        "--filepath",
        type=str,
        default="queries/simple_reference_based_queries.csv",
        help="Path to input CSV file containing the open codes (default: queries/simple_reference_based_queries.csv)"
    )
    parser.add_argument(
        "--column",
        type=str,
        default="openAI_zero_shot Open Coding",
        help="Name of the column containing open codes (default: openAI_zero_shot Open Coding)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="queries/simple_reference_based_failure_mode_taxonomy.md",
        help="Path to save the generated taxonomy (default: queries/simple_reference_based_failure_mode_taxonomy.md)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for generation (default: gpt-4o-mini)"
    )

    args = parser.parse_args()

    logger.info(f"Generating taxonomy from: {args.filepath}")
    logger.info(f"Using column: {args.column}")
    logger.info(f"Saving to: {args.output}")
    logger.info(f"Using model: {args.model}")

    # Generate the taxonomy
    await generate_taxonomy_from_csv(
        filepath=args.filepath,
        column_name=args.column,
        output_path=args.output,
        model=args.model
    )


if __name__ == "__main__":
    asyncio.run(main())