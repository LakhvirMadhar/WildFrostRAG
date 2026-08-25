"""Synthetic Taxonomy Generator for WildFrostRAG.

This module provides functionality to generate axial codes from open codes
using LLMs. It analyzes qualitative coding results from the evaluation process
and creates higher-level categorical structures for better understanding
of failure modes and patterns.
"""

import json
from pathlib import Path

from utils.config import get_settings
from utils.logger import logger
from prompts.prompt_utils import format_prompt_tuple
from rag.augmented_generation.openai_client import call_openai_api
from prompts.taxonomy_prompts import TAXONOMY_SYSTEM_PROMPT_V1, TAXONOMY_USER_PROMPT_V1


async def generate_taxonomy(open_codes: list[str]) -> str:
    """Generate axial codes from open codes.

    Args:
        open_codes: List of open code strings

    Returns:
        Markdown formatted taxonomy with axial codes
    """
    # Remove duplicates and filter out empty values
    unique_codes = list({code for code in open_codes if code and code.strip() != ""})

    # Sort for consistency
    sorted_codes = sorted(unique_codes)

    # Create a numbered list of codes
    codes_text = "\n".join([f"{i + 1}. {code}" for i, code in enumerate(sorted_codes)])

    # Format user message using the versioned prompt
    user_message = format_prompt_tuple(TAXONOMY_USER_PROMPT_V1.prompt_tuple, codes_text=codes_text)

    try:
        settings = get_settings()
        return await call_openai_api(
            messages=[
                {
                    "role": "system",
                    "content": TAXONOMY_SYSTEM_PROMPT_V1.prompt_tuple[0],
                },
                {"role": "user", "content": user_message},
            ],
            model=settings.openai.taxonomy_model,
            temperature=settings.openai.taxonomy_temperature,
        )
    except Exception as e:
        logger.error(f"Error generating taxonomy: {str(e)}")
        return f"ERROR: {str(e)}"


async def generate_taxonomy_from_annotations(experiment_path: Path) -> None:
    """Generate axial codes taxonomy from a generation experiment's annotations.json.

    Args:
        experiment_path: Path to the generation experiment directory
                        (e.g., outputs/run_1/generation/001/)
    """
    annotations_path = experiment_path / "annotations.json"
    output_path = experiment_path / "taxonomy.md"

    if not annotations_path.exists():
        logger.error(f"No annotations.json found at {annotations_path}")
        return

    with open(annotations_path, encoding="utf-8") as f:
        annotations = json.load(f)

    # Extract open_coding values from all annotated queries
    open_codes = []
    for _query_id, annotation in annotations.items():
        if "open_coding" in annotation and annotation["open_coding"]:
            open_codes.append(annotation["open_coding"])

    logger.info(f"Found {len(open_codes)} open codes")
    logger.info(f"Unique codes: {len(set(open_codes))}")

    if len(open_codes) == 0:
        logger.warning("No open codes found in annotations!")
        return

    if output_path.exists():
        logger.warning(f"File already exists at {output_path}")
        logger.info("Skipping taxonomy generation to avoid overwriting existing file.")
        logger.info("Delete or rename the existing file if you want to generate a new taxonomy.")
        return

    logger.info("Generating axial codes from open codes...")
    taxonomy = await generate_taxonomy(open_codes)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(taxonomy)

    logger.info(f"Axial codes taxonomy saved to {output_path}")
