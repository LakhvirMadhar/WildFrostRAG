"""Enriches CardInfo objects with data not on their own card pages.

Currently: tribe exclusivity, looked up from the Companions/Items pages'
tribe tables (parsed by data_processing/pages/tribes/parser.py). Pure, no
network access - fetching those pages is EnrichmentService's job
(services/ingestion/), since data_processing/ sits below scraping/ in the
import-linter layers contract.
"""

from wildfrost_rag.core.logger import logger
from wildfrost_rag.data_processing.cards import CardInfo
from wildfrost_rag.data_processing.pages.tribes.parser import parse_tribe_exclusivity_table
from wildfrost_rag.data_processing.tribes import TribeExclusivity


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
    card_infos: list[CardInfo], companions_html: str, items_html: str
) -> None:
    """Enrich a list of CardInfo objects with tribe exclusivity information.

    Args:
        card_infos: List of CardInfo objects to enrich
        companions_html: Raw HTML of the Companions wiki page
        items_html: Raw HTML of the Items wiki page

    Note:
        This function modifies the card_infos list in-place by setting
        the tribe_exclusivity field on matching cards.
    """
    logger.info(f"Starting tribe enrichment for {len(card_infos)} cards")

    # Companions page uses the second table (index 1)
    companions_tribe_lookup = parse_tribe_exclusivity_table(companions_html, table_index=1)

    # Items page uses the first (and only) table (index 0)
    items_tribe_lookup = parse_tribe_exclusivity_table(items_html, table_index=0)

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
