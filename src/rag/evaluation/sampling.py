"""
Query sampling module for WildFrostRAG.

This module provides functionality to generate query datasets for evaluation.
It samples random card URLs from the Wildfrost Wiki and creates a structured
CSV file ready for LLM evaluation.
"""

import os
import json
import random
import pandas as pd
import re
from data_processing.cards import CardType, CardInfo
from data_processing.generate_schemas import generate_card_type_html_schema
from utils.config import settings


def clean_name_for_url(name: str) -> str:
    """
    Clean card name for use in URLs by replacing spaces with underscores.

    Args:
        name: The card name to clean

    Returns:
        Cleaned name with spaces replaced by underscores
    """
    return re.sub(r'\s+', '_', name)


def sample_queries(
    k: int = 100,
    output_filepath: str = 'queries/simple_reference_based_queries.csv',
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Sample k random URLs from the Wildfrost Wiki card set and generate a query CSV.
    Queries are manually generated upon viewing URLs.

    Args:
        k: Number of URLs to sample.
        output_filepath: Path to save the CSV.
        overwrite: If True, overwrite existing file. If False, return existing df if file exists.

    Returns:
        pd.DataFrame containing the query data with columns:
        - query_id: Sequential ID for each query
        - query: Empty string (to be filled by user)
        - ground_truth: Empty string (to be filled by user)
        - doc_reference: URL reference for the card
        - openAI_zero_shot: Empty string (to be filled by pipeline)
        - openAI_RAG_response: Empty string (to be filled by pipeline)
    """

    if os.path.exists(output_filepath) and not overwrite:
        print(f"File already exists at {output_filepath}")
        print("Loading existing data...")
        return pd.read_csv(output_filepath)

    # Generate card type schema
    card_type_schema = generate_card_type_html_schema()

    # Save schema to file (using settings for path)
    schema_filename = settings.schemas_dir / 'card_type_schema.json'
    os.makedirs(settings.schemas_dir, exist_ok=True)

    with open(schema_filename, 'w', encoding='utf-8') as f:
        json.dump(card_type_schema, f, indent=4)

    # Set base URL
    base_url = settings.wildfrost_wiki_base_url

    # Create card infos
    card_infos = []
    for card_type, cards in card_type_schema.items():
        # TODO: Leader page is currently not setup for scraping
        if card_type == 'leaders':
            continue

        for card_name in cards:
            cleaned_name = clean_name_for_url(card_name)
            # Use CardInfo (assuming it's a Pydantic model or dataclass)
            # Note: CardType(card_type) might fail if card_type string doesn't match enum exact value
            # We trust the existing logic for now, but might need error handling
            try:
                c_type = CardType(card_type)
            except ValueError:
                # Fallback or skip if enum doesn't match
                continue

            card_info = CardInfo(
                card_name=card_name,
                card_type=c_type,
                url=f'{base_url}/{cleaned_name}'
            )
            card_infos.append(card_info)

    # Extract URLs
    urls = [card.url for card in card_infos]

    # Get random URLs (without replacement)
    random_urls = random.sample(urls, k=min(k, len(urls)))

    # Create new DataFrame
    df = pd.DataFrame({
        'query_id': range(1, len(random_urls) + 1),
        'query': [''] * len(random_urls),  # Empty strings for now
        'ground_truth': [''] * len(random_urls),  # Empty strings for now
        'doc_reference': random_urls,  # Raw URLs
        'openAI_zero_shot': [''] * len(random_urls),  # Empty strings for now
        'openAI_RAG_response': [''] * len(random_urls)  # Empty strings for now
    })

    # Save to CSV
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)

    print(f"Sampled {len(random_urls)} URLs and saved to {output_filepath}")

    return df
