import neo4j

from wildfrost_rag.neo4j_kg.query_utils import single_value
from wildfrost_rag.utils.logger import logger


def create_fight_enemy_relationships(
    tx: neo4j.ManagedTransaction,
    fight_enemies: dict[str, list[str]],
    fight_page_mapping: dict[str, str],
) -> int:
    """Link Fight nodes to their enemy Card nodes via FEATURES_ENEMY.

    Handles multi-phase cards (e.g., "Infernoko" matches both
    "Infernoko Phase 1" and "Infernoko Phase 2").

    Args:
        tx: Neo4j managed transaction
        fight_enemies: page_name -> list of enemy card names from the fight page
        fight_page_mapping: display_name -> page_name (to map fight names to page names)

    Returns:
        Number of relationships created
    """
    # Build reverse mapping: page_name -> display_name(s)
    page_to_display: dict[str, list[str]] = {}
    for display_name, page_name in fight_page_mapping.items():
        page_to_display.setdefault(page_name, []).append(display_name)

    pairs = []
    for page_name, enemy_names in fight_enemies.items():
        # Get the display name(s) for this fight
        display_names = page_to_display.get(page_name, [])
        for fight_display_name in display_names:
            for enemy_name in enemy_names:
                pairs.append(
                    {
                        "fight_name": fight_display_name,
                        "enemy_name": enemy_name,
                    }
                )

    if not pairs:
        return 0

    # Match cards by:
    #   - Exact name: "Beeberry"
    #   - Multi-phase: "Infernoko" -> "Infernoko Phase 1", "Infernoko Phase 2"
    #   - Parenthetical variant: "Naked Gnome" -> "Naked Gnome (Enemy)",
    #     "Frost Guardian" -> "Frost Guardian (Frost Wizard)"
    query = """
    UNWIND $pairs AS pair
    MATCH (f:Fight {name: pair.fight_name})
    MATCH (c:Card)
    WHERE c.card_name = pair.enemy_name
       OR c.card_name STARTS WITH (pair.enemy_name + ' Phase')
       OR c.card_name STARTS WITH (pair.enemy_name + ' (')
    MERGE (f)-[:FEATURES_ENEMY]->(c)
    RETURN count(*) AS created
    """
    result = tx.run(query, pairs=pairs)
    count = single_value(result, "created")
    logger.info(f"Created {count} FEATURES_ENEMY relationships")
    return count
