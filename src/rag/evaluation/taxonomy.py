"""
Synthetic Taxonomy Generator for WildFrostRAG.

This module provides functionality to generate axial codes from open codes
using LLMs. It analyzes qualitative coding results from the evaluation process
and creates higher-level categorical structures for better understanding
of failure modes and patterns.
"""

import pandas as pd
import asyncio
from openai import AsyncOpenAI
import os
from src.utils.config import settings
from src.utils.logger import logger


TAXONOMY_SYSTEM_PROMPT = """You are an expert at qualitative coding analysis, specifically creating axial codes from open codes.

You will be given a numbered list of open codes that describe various failure modes from an LLM evaluation.

Your task is to perform axial coding:
1. Analyze all the open codes
2. Identify common themes and patterns across the codes
3. Create higher-level axial codes (categories) that group related open codes
4. Each axial code should represent a broader conceptual category
5. Provide clear definitions for each axial code
6. Reference which open codes (by number) fall under each axial code

Format your response as a well-structured markdown document with:
- A title: "Failure Mode Taxonomy - Axial Codes"
- A brief introduction explaining the coding approach
- Axial codes as H2 headers (##)
- Sub-categories as H3 headers (###) if needed
- Clear descriptions of each axial code category
- List the relevant open code numbers that fall under each axial code
- A summary section with key insights

Be comprehensive but concise. Make the taxonomy useful for understanding and addressing these failure modes."""


async def generate_taxonomy(open_codes: list[str], model: str = "gpt-4o-mini") -> str:
    """
    Generate axial codes from open codes.

    Args:
        open_codes: List of open code strings
        model: The OpenAI model to use

    Returns:
        Markdown formatted taxonomy with axial codes
    """
    # Remove duplicates and filter out empty/NA values
    unique_codes = list(set([code for code in open_codes if pd.notna(code) and code.strip() != '']))

    # Sort for consistency
    sorted_codes = sorted(unique_codes)

    # Create a numbered list of codes
    codes_text = "\n".join([f"{i+1}. {code}" for i, code in enumerate(sorted_codes)])

    user_message = f"""Here are the open codes from the failure analysis:\n\n{codes_text}\n\n
Please create axial codes that group these open codes into higher-level categories. Reference the open codes by their numbers."""

    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TAXONOMY_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,  # Slightly higher for more creative categorization
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating taxonomy: {str(e)}")
        return f"ERROR: {str(e)}"


async def generate_taxonomy_from_csv(
    filepath: str,
    column_name: str,
    output_path: str
) -> None:
    """
    Generate axial codes taxonomy from a CSV file containing open codes.

    Args:
        filepath: Path to the CSV file containing the open codes
        column_name: Name of the column containing open codes
        output_path: Path to save the generated taxonomy markdown file
    """
    # Load the CSV
    df = pd.read_csv(filepath)

    logger.info(f"Loaded {len(df)} total rows")

    # Check if Open Coding column exists
    if column_name not in df.columns:
        logger.error(f"'{column_name}' column not found in the CSV")
        logger.info(f"Available columns: {df.columns.tolist()}")
        return

    # Get all open codes
    open_codes = df[column_name].tolist()
    valid_codes = [code for code in open_codes if pd.notna(code) and str(code).strip() != '']

    logger.info(f"Found {len(valid_codes)} open codes")
    logger.info(f"Unique codes: {len(set(valid_codes))}")

    if len(valid_codes) == 0:
        logger.warning("No open codes found to process!")
        return

    # Check if file already exists BEFORE making API call
    if os.path.exists(output_path):
        logger.warning(f"File already exists at {output_path}")
        logger.info("Skipping taxonomy generation to avoid overwriting existing file.")
        logger.info("Delete or rename the existing file if you want to generate a new taxonomy.")
        return

    logger.info("Generating axial codes from open codes...")
    taxonomy = await generate_taxonomy(open_codes)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(taxonomy)

    logger.info(f"Axial codes taxonomy saved to {output_path}")


async def main():
    """Main function to run the taxonomy generation."""
    await generate_taxonomy_from_csv()


if __name__ == "__main__":
    asyncio.run(main())