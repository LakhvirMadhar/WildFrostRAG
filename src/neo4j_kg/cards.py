from typing import Any

import neo4j

from data_processing.cards import CardType
from data_processing.phase_config import RECRUITABLE_ENEMIES
from neo4j_kg.query_utils import single_value


def _get_all_ancestors(card_type_value: str) -> list[str]:
    """Recursively resolve all ancestor CardTypes for a given type."""
    type_to_parents = {ct.value: ct.parents for ct in CardType}
    ancestors = []
    queue = list(type_to_parents.get(card_type_value, []))
    while queue:
        parent = queue.pop(0)
        if parent not in ancestors:
            ancestors.append(parent)
            queue.extend(type_to_parents.get(parent, []))
    return ancestors


def _enrich_cards_with_parent_types(
    cards_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add all ancestor CardType names to each card dict so the Cypher can link them."""
    ancestor_cache = {ct.value: _get_all_ancestors(ct.value) for ct in CardType}
    for card in cards_data:
        card["parent_types"] = ancestor_cache.get(card.get("card_type"), [])
    return cards_data


def create_cards(tx: neo4j.ManagedTransaction, cards_data: list[dict[str, Any]]) -> int:
    """Bulk create card nodes in neo4j.

    For multi-phase cards (like Infernoko Phase 1 and Phase 2), we use
    card_name + phase as the unique identifier. For single-phase cards,
    phase is null and we merge by card_name only.

    Cards are linked to both their own CardType and all parent CardTypes
    (e.g., a pet card gets HAS_CARD_TYPE edges to both 'pets' and 'companions').
    """
    cards_data = _enrich_cards_with_parent_types(cards_data)

    # Strip None values so Neo4j doesn't create properties with null
    # (e.g., Clunkers have scrap instead of health — health shouldn't exist on the node)
    cards_data = [
        {k: v for k, v in card.items() if v is not None} for card in cards_data
    ]

    # Separate phased and non-phased cards
    phased_cards = [c for c in cards_data if c.get("phase") is not None]
    non_phased_cards = [c for c in cards_data if c.get("phase") is None]

    total_created = 0

    # Create non-phased cards (MERGE by card_name only)
    if non_phased_cards:
        query = """
        UNWIND $cards AS card
        MERGE (c:Card {card_name: card.card_name})
        SET c += card
        MERGE (t:CardType {name: card.card_type})
        MERGE (c)-[:HAS_CARD_TYPE]->(t)
        FOREACH (parent IN card.parent_types |
            MERGE (pt:CardType {name: parent})
            MERGE (c)-[:HAS_CARD_TYPE]->(pt)
        )
        RETURN count(c) AS createdCount
        """
        result = tx.run(query, cards=non_phased_cards)
        total_created += single_value(result, "createdCount")

    # Create phased cards (MERGE by card_name + phase)
    if phased_cards:
        query = """
        UNWIND $cards AS card
        MERGE (c:Card {card_name: card.card_name, phase: card.phase})
        SET c += card
        MERGE (t:CardType {name: card.card_type})
        MERGE (c)-[:HAS_CARD_TYPE]->(t)
        FOREACH (parent IN card.parent_types |
            MERGE (pt:CardType {name: parent})
            MERGE (c)-[:HAS_CARD_TYPE]->(pt)
        )
        RETURN count(c) AS createdCount
        """
        result = tx.run(query, cards=phased_cards)
        total_created += single_value(result, "createdCount")

    return total_created


def create_phase_relationships(tx: neo4j.ManagedTransaction) -> int:
    """Create TRANSFORMS_INTO relationships between card phases.

    Links Phase 1 -> Phase 2 -> Phase 3, etc. for multi-phase cards.
    Uses base_name for matching since phased cards may have different card_names
    (e.g., "Truffle" -> "Truffle (medium)" -> "Truffle (small)").
    """
    query = """
    MATCH (c1:Card)
    WHERE c1.phase IS NOT NULL AND c1.phase > 0 AND c1.phase < c1.total_phases
    MATCH (c2:Card {base_name: c1.base_name, phase: c1.phase + 1})
    MERGE (c1)-[:TRANSFORMS_INTO]->(c2)
    RETURN count(*) AS relationshipsCreated
    """
    result = tx.run(query)
    return single_value(result, "relationshipsCreated")


def create_recruitment_relationships(tx: neo4j.ManagedTransaction) -> int:
    """Create CAN_BE_RECRUITED_AS relationships for enemy cards that can become companions.

    Some enemies (like Naked Gnome) can be recruited as companions if kept alive.
    This links the enemy variant to its companion variant.
    """
    if not RECRUITABLE_ENEMIES:
        return 0

    query = """
    UNWIND $recruitables AS pair
    MATCH (enemy:Card {card_name: pair.enemy_name})
    MATCH (companion:Card {card_name: pair.companion_name})
    MERGE (enemy)-[:CAN_BE_RECRUITED_AS]->(companion)
    RETURN count(*) AS relationshipsCreated
    """

    recruitables = [
        {"enemy_name": enemy, "companion_name": companion}
        for enemy, companion in RECRUITABLE_ENEMIES.items()
    ]

    result = tx.run(query, recruitables=recruitables)
    return single_value(result, "relationshipsCreated")


def create_card_type_hierarchy(tx: neo4j.ManagedTransaction) -> int:
    """Create hierarchy relationships between card types."""
    query = """
    UNWIND $hierarchies AS hierarchy
    MERGE (child:CardType {name: hierarchy.child})
    MERGE (parent:CardType {name: hierarchy.parent})
    MERGE (child)-[:SUBTYPE_OF]->(parent)
    """

    # Build hierarchy data from enum
    hierarchies = []
    for card_type in CardType:
        for parent in card_type.parents:
            hierarchies.append({"child": card_type.value, "parent": parent})

    if hierarchies:
        tx.run(query, hierarchies=hierarchies)
    return len(hierarchies)
