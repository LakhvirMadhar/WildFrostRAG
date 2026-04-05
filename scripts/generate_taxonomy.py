#!/usr/bin/env python3
"""
Taxonomy Generation Script for WildFrostRAG.

This script generates axial codes (higher-level categories) from open codes
using LLMs. It loads qualitative coding results from generation experiment
annotations and creates structured taxonomies for understanding failure modes.

Usage:
    python -m scripts.generate_taxonomy --experiment outputs/run_1/generation/001
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from rag.evaluation.taxonomy import generate_taxonomy_from_annotations
from utils.logger import logger


async def main():
    parser = argparse.ArgumentParser(description="Generate taxonomy from qualitative codes")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to generation experiment directory (e.g., outputs/run_1/generation/001)"
    )

    args = parser.parse_args()

    experiment_path = Path(args.experiment)
    if not experiment_path.exists():
        logger.error(f"Experiment path does not exist: {experiment_path}")
        sys.exit(1)

    logger.info(f"Generating taxonomy from experiment: {experiment_path}")
    await generate_taxonomy_from_annotations(experiment_path)


if __name__ == "__main__":
    asyncio.run(main())
