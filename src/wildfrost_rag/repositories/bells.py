"""Bell node creation and relationship linking for Neo4j.

What each bell does (the game facts) lives in data_processing/bell_effects.py;
this module only turns those facts into Cypher.
"""

import neo4j

from wildfrost_rag.core.logger import logger
from wildfrost_rag.data_processing.bell_effects import (
    BELL_ADDS_CARD_TO_FIGHT,
    BELL_AFFECTS_KEYWORD,
    BELL_AFFECTS_MAP_EVENTS,
    BELL_GRANTS_KEYWORD,
    BELL_INTRODUCES_CROWN,
    BELL_MODIFIES_STAT,
    BELL_TARGETS_CARD_TYPE,
    BELLS_AFFECTING_BLING,
    BELLS_APPLYING_ALL_CURSED_CHARMS,
)
from wildfrost_rag.data_processing.bells import BellCategory, BellInfo
from wildfrost_rag.domain.wiki_files import wiki_page_filename
from wildfrost_rag.repositories.query_utils import single_value

# Maps BellCategory enum values to BellType node names (how the graph models categories)
_CATEGORY_TO_BELL_TYPE = {
    BellCategory.SUN.value: "Sun Bell",
    BellCategory.STORM.value: "Storm Bell",
    BellCategory.MODIFIER.value: "Modifier Bell",
}


def create_bells_from_parsed(tx: neo4j.ManagedTransaction, bells: list[BellInfo]) -> int:
    """Create Bell nodes, BellType nodes, and HAS_BELL_TYPE relationships.

    Bells with individual wiki pages get their own URL; others get None
    (they'll still link to the summary Bells.html Document).

    Args:
        tx: Neo4j transaction
        bells: List of BellInfo objects from pages/bells/parser.parse_bells_page()

    Returns:
        Number of Bell nodes created
    """
    bell_data = [
        {
            "name": b.name,
            "category": b.category.value,
            "bell_type": _CATEGORY_TO_BELL_TYPE[b.category.value],
            "description": b.description,
            "notes": b.notes,
            "storm_strength": b.storm_strength,
            "url": b.url,
            "filename": wiki_page_filename(b.name) if b.url else None,
        }
        for b in bells
    ]

    query = """
    UNWIND $bells AS b
    MERGE (bell:Bell {name: b.name})
    SET bell.category = b.category,
        bell.description = b.description,
        bell.notes = b.notes,
        bell.storm_strength = b.storm_strength,
        bell.url = b.url,
        bell.filename = b.filename
    MERGE (bt:BellType {name: b.bell_type})
    MERGE (bell)-[:HAS_BELL_TYPE]->(bt)
    RETURN count(bell) AS created
    """
    result = tx.run(query, bells=bell_data)
    count = single_value(result, "created")
    logger.info(f"Created {count} Bell nodes with BellType relationships")
    return count


def _create_bell_charm_text_matches(tx: neo4j.ManagedTransaction) -> int:
    """Create APPLIES_CHARM relationships by text-matching Charm names in bell descriptions/notes.

    Returns:
        Number of relationships created
    """
    charm_result = tx.run("MATCH (ch:Charm) RETURN ch.name AS name")
    all_charm_names = sorted([r["name"] for r in charm_result], key=len, reverse=True)

    bell_result = tx.run(
        "MATCH (b:Bell) RETURN b.name AS name, b.description AS description, b.notes AS notes"
    )
    charm_pairs = []
    for bell in bell_result:
        text = (bell["description"] or "") + " " + (bell["notes"] or "")
        text_lower = text.lower()
        for charm_name in all_charm_names:
            if charm_name.lower() in text_lower:
                charm_pairs.append({"bell_name": bell["name"], "charm_name": charm_name})

    if charm_pairs:
        tx.run(
            """
            UNWIND $pairs AS p
            MATCH (b:Bell {name: p.bell_name})
            MATCH (ch:Charm {name: p.charm_name})
            MERGE (b)-[:APPLIES_CHARM]->(ch)
        """,
            pairs=charm_pairs,
        )
    logger.info(f"Created {len(charm_pairs)} APPLIES_CHARM relationships (text-matched)")
    return len(charm_pairs)


def _create_bell_all_cursed_charms_relationships(tx: neo4j.ManagedTransaction) -> int:
    """Link bells that can apply every cursed charm (Gloom Bell) to all cursed charms.

    Returns:
        Number of relationships created
    """
    result = tx.run(
        """
        UNWIND $bells AS bell_name
        MATCH (b:Bell {name: bell_name})
        MATCH (ch:Charm {is_cursed: true})
        MERGE (b)-[:APPLIES_CHARM]->(ch)
        RETURN count(*) AS created
    """,
        bells=BELLS_APPLYING_ALL_CURSED_CHARMS,
    )
    count = single_value(result, "created")
    logger.info(f"Created {count} APPLIES_CHARM relationships (bells -> all cursed charms)")
    return count


def _create_bell_crown_relationships(tx: neo4j.ManagedTransaction) -> int:
    """Link bells to the crown they let appear in a run (Tyrant Bell -> Cursed Crown) via INTRODUCES.

    Distinct from APPLIES_CHARM: the bell enables the crown, it doesn't apply it.

    Returns:
        Number of relationships created
    """
    pairs = [
        {"bell_name": bell, "crown_name": crown} for bell, crown in BELL_INTRODUCES_CROWN.items()
    ]
    result = tx.run(
        """
        UNWIND $pairs AS p
        MATCH (b:Bell {name: p.bell_name})
        MATCH (cr:Crown {name: p.crown_name})
        MERGE (b)-[:INTRODUCES]->(cr)
        RETURN count(*) AS created
    """,
        pairs=pairs,
    )
    count = single_value(result, "created")
    logger.info(f"Created {count} INTRODUCES relationships (bell -> crown)")
    return count


def _create_bell_keyword_relationships(tx: neo4j.ManagedTransaction) -> int:
    """Create GRANTS_KEYWORD and AFFECTS_KEYWORD relationships.

    GRANTS_KEYWORD: bell grants this keyword to cards (e.g. Noomlin Sun Bell adds Noomlin)
    AFFECTS_KEYWORD: bell modifies behavior of cards with this keyword (e.g. Breakfast Sun Bell)

    Returns:
        Number of relationships created
    """
    total = 0

    # GRANTS_KEYWORD
    grants_pairs = [
        {"bell_name": bell, "keyword_name": kw} for bell, kw in BELL_GRANTS_KEYWORD.items()
    ]
    if grants_pairs:
        result = tx.run(
            """
            UNWIND $pairs AS p
            MATCH (b:Bell {name: p.bell_name})
            MATCH (k:Keyword {name: p.keyword_name})
            MERGE (b)-[:GRANTS_KEYWORD]->(k)
            RETURN count(*) AS created
        """,
            pairs=grants_pairs,
        )
        count = single_value(result, "created")
        total += count
        logger.info(f"Created {count} GRANTS_KEYWORD relationships")

    # AFFECTS_KEYWORD
    affects_pairs = [
        {"bell_name": bell, "keyword_name": kw} for bell, kw in BELL_AFFECTS_KEYWORD.items()
    ]
    if affects_pairs:
        result = tx.run(
            """
            UNWIND $pairs AS p
            MATCH (b:Bell {name: p.bell_name})
            MATCH (k:Keyword {name: p.keyword_name})
            MERGE (b)-[:AFFECTS_KEYWORD]->(k)
            RETURN count(*) AS created
        """,
            pairs=affects_pairs,
        )
        count = single_value(result, "created")
        total += count
        logger.info(f"Created {count} AFFECTS_KEYWORD relationships")

    return total


def _create_bell_stat_relationships(tx: neo4j.ManagedTransaction) -> int:
    """Create MODIFIES_STAT relationships from curated bell-stat mappings.

    Returns:
        Number of relationships created
    """
    pairs = [{"bell_name": bell, "stat_name": stat} for bell, stat in BELL_MODIFIES_STAT.items()]
    if not pairs:
        return 0

    result = tx.run(
        """
        UNWIND $pairs AS p
        MATCH (b:Bell {name: p.bell_name})
        MATCH (s:Stat {name: p.stat_name})
        MERGE (b)-[:MODIFIES_STAT]->(s)
        RETURN count(*) AS created
    """,
        pairs=pairs,
    )
    count = single_value(result, "created")
    logger.info(f"Created {count} MODIFIES_STAT relationships")
    return count


def _create_bell_bling_relationships(tx: neo4j.ManagedTransaction) -> int:
    """Create AFFECTS_BLING relationships for bells that modify Bling economy.

    Returns:
        Number of relationships created
    """
    if not BELLS_AFFECTING_BLING:
        return 0

    result = tx.run(
        """
        UNWIND $bells AS bell_name
        MATCH (b:Bell {name: bell_name})
        MATCH (bl:Bling {name: "Bling"})
        MERGE (b)-[:AFFECTS_BLING]->(bl)
        RETURN count(*) AS created
    """,
        bells=BELLS_AFFECTING_BLING,
    )
    count = single_value(result, "created")
    logger.info(f"Created {count} AFFECTS_BLING relationships")
    return count


def _create_bell_card_relationships(tx: neo4j.ManagedTransaction) -> int:
    """Create ADDS_TO_FIGHT relationships for bells that add cards to fights.

    Returns:
        Number of relationships created
    """
    pairs = [
        {"bell_name": bell, "card_name": card} for bell, card in BELL_ADDS_CARD_TO_FIGHT.items()
    ]
    if not pairs:
        return 0

    result = tx.run(
        """
        UNWIND $pairs AS p
        MATCH (b:Bell {name: p.bell_name})
        MATCH (c:Card {card_name: p.card_name})
        MERGE (b)-[:ADDS_TO_FIGHT]->(c)
        RETURN count(*) AS created
    """,
        pairs=pairs,
    )
    count = single_value(result, "created")
    logger.info(f"Created {count} ADDS_TO_FIGHT relationships")
    return count


def _create_bell_target_relationships(tx: neo4j.ManagedTransaction) -> int:
    """Create TARGETS_CARD_TYPE relationships linking bells to the CardType nodes they affect.

    Returns:
        Number of relationships created
    """
    pairs = [
        {"bell_name": bell, "card_type": ct}
        for bell, card_types in BELL_TARGETS_CARD_TYPE.items()
        for ct in card_types
    ]
    if not pairs:
        return 0

    result = tx.run(
        """
        UNWIND $pairs AS p
        MATCH (b:Bell {name: p.bell_name})
        MATCH (ct:CardType {name: p.card_type})
        MERGE (b)-[:TARGETS_CARD_TYPE]->(ct)
        RETURN count(*) AS created
    """,
        pairs=pairs,
    )
    count = single_value(result, "created")
    logger.info(f"Created {count} TARGETS_CARD_TYPE relationships")
    return count


def _create_bell_map_event_relationships(tx: neo4j.ManagedTransaction) -> int:
    """Create AFFECTS_MAP_EVENT relationships from bells to the map events they affect.

    Returns:
        Number of relationships created
    """
    pairs = [
        {"bell_name": bell, "event_name": event}
        for bell, events in BELL_AFFECTS_MAP_EVENTS.items()
        for event in events
    ]
    if not pairs:
        return 0

    result = tx.run(
        """
        UNWIND $pairs AS p
        MATCH (b:Bell {name: p.bell_name})
        MATCH (me:MapEvent {name: p.event_name})
        MERGE (b)-[:AFFECTS_MAP_EVENT]->(me)
        RETURN count(*) AS created
    """,
        pairs=pairs,
    )
    count = single_value(result, "created")
    logger.info(f"Created {count} AFFECTS_MAP_EVENT relationships")
    return count


def create_bell_relationships(tx: neo4j.ManagedTransaction) -> int:
    """Create all bell linking relationships.

    Orchestrates the creation of APPLIES_CHARM, GRANTS_KEYWORD,
    AFFECTS_KEYWORD, MODIFIES_STAT, AFFECTS_BLING, ADDS_TO_FIGHT,
    TARGETS_CARD_TYPE, and AFFECTS_MAP_EVENT relationships.
    Bell nodes must already exist.

    Args:
        tx: Neo4j transaction

    Returns:
        Total number of relationships created
    """
    total = 0
    total += _create_bell_charm_text_matches(tx)
    total += _create_bell_all_cursed_charms_relationships(tx)
    total += _create_bell_crown_relationships(tx)
    total += _create_bell_keyword_relationships(tx)
    total += _create_bell_stat_relationships(tx)
    total += _create_bell_bling_relationships(tx)
    total += _create_bell_card_relationships(tx)
    total += _create_bell_target_relationships(tx)
    total += _create_bell_map_event_relationships(tx)
    logger.info(f"Created {total} total bell linking relationships")
    return total
