from typing import Any

import neo4j

from wildfrost_rag.data_processing.tribes import TribeExclusivity
from wildfrost_rag.neo4j_kg.query_utils import single_value
from wildfrost_rag.utils.logger import logger


def create_tribes(tx: neo4j.ManagedTransaction) -> int:
    """Create the three exclusive tribe nodes."""
    # Get the names of the exclusive tribes directly from the enum
    exclusive_tribe_names = [t.value for t in TribeExclusivity if t.is_exclusive]

    query = """
    UNWIND $tribe_names AS name
    MERGE (t:Tribe {name: name})
    RETURN count(t) AS tribesCreated
    """
    result = tx.run(query, tribe_names=exclusive_tribe_names)
    return single_value(result, "tribesCreated")


def _build_card_tribe_pairs(cards_data: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build card-tribe pairs from card data for Neo4j ingestion."""
    card_tribes: list[dict[str, str]] = []

    for card_dict in cards_data:
        exclusivity_value = card_dict.get("tribe_exclusivity")
        if exclusivity_value is None:
            continue

        try:
            exclusivity = TribeExclusivity(exclusivity_value)
        except ValueError:
            logger.error(f"Unknown tribe exclusivity value: {exclusivity_value}")
            continue

        for tribe_name in exclusivity.get_tribes():
            card_tribes.append({"card_name": card_dict["card_name"], "tribe_name": tribe_name})

    return card_tribes


def create_card_tribe_relationships(
    tx: neo4j.ManagedTransaction, cards_data: list[dict[str, Any]]
) -> int:
    """Create relationships between cards and tribes based on their exclusivity."""
    card_tribes = _build_card_tribe_pairs(cards_data)

    logger.info(f"Creating {len(card_tribes)} card-tribe relationships")

    if not card_tribes:
        return 0

    query = """
    UNWIND $card_tribes AS card_tribe
    MATCH (c:Card {card_name: card_tribe.card_name})
    MATCH (t:Tribe {name: card_tribe.tribe_name})
    MERGE (c)-[:BELONGS_TO_TRIBE]->(t)
    RETURN count(DISTINCT c) AS processedCards
    """
    result = tx.run(query, card_tribes=card_tribes)
    return single_value(result, "processedCards")
