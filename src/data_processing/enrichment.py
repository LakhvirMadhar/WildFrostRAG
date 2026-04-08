"""Data enrichment module for WildFrostRAG.

This module handles enrichment of CardInfo objects with additional data
that isn't available in the individual card pages, such as tribe exclusivity
information that must be scraped from aggregate pages.
"""

import requests
from bs4 import BeautifulSoup, Comment
from data_processing.cards import CardInfo
from data_processing.tribes import TribeExclusivity
from utils.logger import logger


# ===== Tribe Exclusivity Scraping =====


def scrape_tribe_exclusivity_table(url: str, table_index: int = 1) -> dict[str, str]:
    """Scrape tribe exclusivity information from a wiki page table.

    Args:
        url: URL of the wiki page containing the tribe table
        table_index: Index of the table to parse (0-based). Default is 1 for
                    the second table, which is typical for Companions page.

    Returns:
        Dictionary mapping card names to tribe names (e.g., "Snowdwellers", "All")

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails
        ValueError: If the expected table structure is not found
    """
    logger.info(f"Scraping tribe exclusivity table from: {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove HTML comments
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()

        # Find all sortable wiki tables
        tables = soup.find_all("table", {"class": "wikitable sortable"})

        if not tables:
            raise ValueError(f"No sortable wikitable found at {url}")

        if table_index >= len(tables):
            raise ValueError(
                f"Table index {table_index} out of range. "
                f"Found {len(tables)} tables at {url}"
            )

        target_table = tables[table_index]

        # Extract headers
        first_row = target_table.find("tr")
        if first_row is None:
            raise ValueError(f"No rows found in table at {url}")
        headers = [th.text.strip() for th in first_row.find_all("th")]
        logger.debug(f"Found table headers: {headers}")

        # Find the indices of required columns
        try:
            card_name_index = headers.index("Card Name")
            tribe_exclusive_index = headers.index("Tribe-exclusive?")
        except ValueError as e:
            raise ValueError(f"Required column not found in table headers: {e}") from e

        # Parse table rows
        tribe_lookup = {}

        for row in target_table.find_all("tr")[1:]:  # Skip header row
            cells = row.find_all(["th", "td"])

            if len(cells) > max(card_name_index, tribe_exclusive_index):
                card_name = cells[card_name_index].get_text(strip=True)
                tribe_name = cells[tribe_exclusive_index].get_text(strip=True)

                tribe_lookup[card_name] = tribe_name

        logger.info(
            f"Successfully scraped {len(tribe_lookup)} card-tribe mappings from {url}"
        )
        return tribe_lookup

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error parsing tribe table from {url}: {e}")
        raise


def enrich_card_with_tribe(card_info: CardInfo, tribe_lookup: dict[str, str]) -> bool:
    """Enrich a single CardInfo object with tribe exclusivity data.

    Args:
        card_info: CardInfo object to enrich
        tribe_lookup: Dictionary mapping card names to tribe strings

    Returns:
        True if enrichment was successful, False if no tribe data found
    """
    tribe_name_str = tribe_lookup.get(card_info.card_name)

    if not tribe_name_str:
        return False

    try:
        # Find matching TribeExclusivity enum
        matching_enum = next(t for t in TribeExclusivity if t.value == tribe_name_str)
        card_info.tribe_exclusivity = matching_enum
        return True

    except StopIteration:
        logger.warning(
            f"No matching TribeExclusivity enum found for '{tribe_name_str}' "
            f"for card '{card_info.card_name}'"
        )
        return False


def enrich_cards_with_tribes(
    card_infos: list[CardInfo], companions_url: str, items_url: str
) -> None:
    """Enrich a list of CardInfo objects with tribe exclusivity information.

    This function scrapes tribe data from the Companions and Items aggregate
    pages and enriches the CardInfo objects in-place.

    Args:
        card_infos: List of CardInfo objects to enrich
        companions_url: URL of the Companions wiki page
        items_url: URL of the Items wiki page

    Note:
        This function modifies the card_infos list in-place by setting
        the tribe_exclusivity field on matching cards.
    """
    logger.info(f"Starting tribe enrichment for {len(card_infos)} cards")

    # Scrape tribe data from both pages
    try:
        # Companions page uses the second table (index 1)
        companions_tribe_lookup = scrape_tribe_exclusivity_table(
            companions_url, table_index=1
        )

        # Items page uses the first (and only) table (index 0)
        items_tribe_lookup = scrape_tribe_exclusivity_table(items_url, table_index=0)

    except Exception as e:
        logger.error(f"Failed to scrape tribe data: {e}")
        raise

    # Merge the lookups
    combined_tribe_lookup = {**companions_tribe_lookup, **items_tribe_lookup}
    logger.info(f"Combined tribe lookup has {len(combined_tribe_lookup)} entries")

    # Enrich cards
    enriched_count = 0
    for card_info in card_infos:
        if enrich_card_with_tribe(card_info, combined_tribe_lookup):
            enriched_count += 1

    logger.info(
        f"Tribe enrichment complete: {enriched_count}/{len(card_infos)} "
        f"cards enriched with tribe data"
    )
