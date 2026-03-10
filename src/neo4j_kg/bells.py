"""
Bell node creation and relationship linking for Neo4j.
"""

from typing import List

from src.data_processing.bells import BellCategory, BellInfo
from src.utils.logger import logger

# Maps BellCategory enum values to BellType node names
_CATEGORY_TO_BELL_TYPE = {
    BellCategory.SUN.value: "Sun Bell",
    BellCategory.STORM.value: "Storm Bell",
    BellCategory.MODIFIER.value: "Modifier Bell",
}

# Curated mappings: bell name -> keyword name (bell GRANTS this keyword to cards)
_BELL_GRANTS_KEYWORD_MAP = {
    "Bell of Death": "Injured",
    "Noomlin Sun Bell": "Noomlin",
}

# Curated mappings: bell name -> keyword name (bell AFFECTS cards with this keyword)
_BELL_AFFECTS_KEYWORD_MAP = {
    "Breakfast Sun Bell": "Consume",
}

# Curated mappings: bell name -> stat name
_BELL_STAT_MAP = {
    "Battle Bell": "Attack",
    "Frenzy Bell": "Frenzy",
    "Heart Bell": "Health",
    "Blood Bell": "Health",
    "Sun Bell of Health": "Health",
    "Sun Bell of Strength": "Attack",
}

# Bells that affect the Bling economy
_BELL_BLING = ["Blingsack Bell", "Gold Blade Bell", "Blingsnail Bell"]

# Bells that add specific cards to fights
_BELL_CARD_MAP = {
    "Gobbler Bell": "Gobbler",
}

# Curated mappings: bell name -> list of CardType names the bell targets
_BELL_TARGETS_CARD_TYPE = {
    "Bombskull Bell": ["clunkers"],
    "Dread Bell": ["non_boss_enemies", "enemy_clunkers"],
    "Fog Bell": ["non_boss_enemies"],
    "Goat Bell": ["non_boss_enemies"],
    "Frostbourne Bell": ["non_boss_enemies"],
    "Frosthand Bell": ["non_boss_enemies"],
    "Icebourne Bell": ["non_boss_enemies"],
    "Gloom Bell": ["companions", "items"],
    "Battle Bell": ["companions"],
    "Blood Bell": ["companions", "leaders"],
    "Sun Bell of Health": ["leaders"],
    "Sun Bell of Strength": ["items"],
    "Frenzy Bell": ["items"],
}

# Gloom Bell affects these map events (card rewards can have cursed charms)
_GLOOM_BELL_MAP_EVENTS = [
    "Frozen Travellers",
    "Treasure Chest",
    "Gnome Traveller",
    "The Woolly Snail",
    "Charm Merchant",
]


def create_bells_from_parsed(tx, bells: List[BellInfo], url: str = None):
    """
    Create Bell nodes, BellType nodes, and HAS_BELL_TYPE relationships.

    Args:
        tx: Neo4j transaction
        bells: List of BellInfo objects from parse_bells_page()
        url: Wiki page URL for all bells (shared /Bells page)

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
        bell.url = $url
    MERGE (bt:BellType {name: b.bell_type})
    MERGE (bell)-[:HAS_BELL_TYPE]->(bt)
    RETURN count(bell) AS created
    """
    result = tx.run(query, bells=bell_data, url=url)
    count = result.single()["created"]
    logger.info(f"Created {count} Bell nodes with BellType relationships")
    return count


def _create_bell_charm_text_matches(tx):
    """
    Create APPLIES_CHARM relationships by text-matching Charm names
    in bell descriptions/notes.

    Returns:
        Number of relationships created
    """
    charm_result = tx.run("MATCH (ch:Charm) RETURN ch.name AS name")
    all_charm_names = sorted(
        [r["name"] for r in charm_result], key=len, reverse=True
    )

    bell_result = tx.run(
        "MATCH (b:Bell) RETURN b.name AS name, b.description AS description, b.notes AS notes"
    )
    charm_pairs = []
    for bell in bell_result:
        text = (bell["description"] or "") + " " + (bell["notes"] or "")
        text_lower = text.lower()
        for charm_name in all_charm_names:
            if charm_name.lower() in text_lower:
                charm_pairs.append(
                    {"bell_name": bell["name"], "charm_name": charm_name}
                )

    if charm_pairs:
        tx.run("""
            UNWIND $pairs AS p
            MATCH (b:Bell {name: p.bell_name})
            MATCH (ch:Charm {name: p.charm_name})
            MERGE (b)-[:APPLIES_CHARM]->(ch)
        """, pairs=charm_pairs)
    logger.info(f"Created {len(charm_pairs)} APPLIES_CHARM relationships (text-matched)")
    return len(charm_pairs)


def _create_gloom_bell_cursed_charms(tx):
    """
    Link Gloom Bell to all cursed charms via APPLIES_CHARM.

    Returns:
        Number of relationships created
    """
    result = tx.run("""
        MATCH (b:Bell {name: "Gloom Bell"})
        MATCH (ch:Charm {is_cursed: true})
        MERGE (b)-[:APPLIES_CHARM]->(ch)
        RETURN count(*) AS created
    """)
    count = result.single()["created"]
    logger.info(f"Created {count} APPLIES_CHARM relationships (Gloom Bell -> cursed charms)")
    return count


def _create_tyrant_bell_crown(tx):
    """
    Link Tyrant Bell to Cursed Crown via INTRODUCES.

    Tyrant Bell enables Cursed Crowns to appear in a run —
    distinct from APPLIES_CHARM.

    Returns:
        Number of relationships created
    """
    result = tx.run("""
        MATCH (b:Bell {name: "Tyrant Bell"})
        MATCH (cr:Crown {name: "Cursed Crown"})
        MERGE (b)-[:INTRODUCES]->(cr)
        RETURN count(*) AS created
    """)
    count = result.single()["created"]
    logger.info(f"Created {count} INTRODUCES relationships (Tyrant Bell -> Cursed Crown)")
    return count


def _create_bell_keyword_relationships(tx):
    """
    Create GRANTS_KEYWORD and AFFECTS_KEYWORD relationships.

    GRANTS_KEYWORD: bell grants this keyword to cards (e.g. Noomlin Sun Bell adds Noomlin)
    AFFECTS_KEYWORD: bell modifies behavior of cards with this keyword (e.g. Breakfast Sun Bell)

    Returns:
        Number of relationships created
    """
    total = 0

    # GRANTS_KEYWORD
    grants_pairs = [
        {"bell_name": bell, "keyword_name": kw}
        for bell, kw in _BELL_GRANTS_KEYWORD_MAP.items()
    ]
    if grants_pairs:
        result = tx.run("""
            UNWIND $pairs AS p
            MATCH (b:Bell {name: p.bell_name})
            MATCH (k:Keyword {name: p.keyword_name})
            MERGE (b)-[:GRANTS_KEYWORD]->(k)
            RETURN count(*) AS created
        """, pairs=grants_pairs)
        count = result.single()["created"]
        total += count
        logger.info(f"Created {count} GRANTS_KEYWORD relationships")

    # AFFECTS_KEYWORD
    affects_pairs = [
        {"bell_name": bell, "keyword_name": kw}
        for bell, kw in _BELL_AFFECTS_KEYWORD_MAP.items()
    ]
    if affects_pairs:
        result = tx.run("""
            UNWIND $pairs AS p
            MATCH (b:Bell {name: p.bell_name})
            MATCH (k:Keyword {name: p.keyword_name})
            MERGE (b)-[:AFFECTS_KEYWORD]->(k)
            RETURN count(*) AS created
        """, pairs=affects_pairs)
        count = result.single()["created"]
        total += count
        logger.info(f"Created {count} AFFECTS_KEYWORD relationships")

    return total


def _create_bell_stat_relationships(tx):
    """
    Create MODIFIES_STAT relationships from curated bell-stat mappings.

    Returns:
        Number of relationships created
    """
    pairs = [
        {"bell_name": bell, "stat_name": stat}
        for bell, stat in _BELL_STAT_MAP.items()
    ]
    if not pairs:
        return 0

    result = tx.run("""
        UNWIND $pairs AS p
        MATCH (b:Bell {name: p.bell_name})
        MATCH (s:Stat {name: p.stat_name})
        MERGE (b)-[:MODIFIES_STAT]->(s)
        RETURN count(*) AS created
    """, pairs=pairs)
    count = result.single()["created"]
    logger.info(f"Created {count} MODIFIES_STAT relationships")
    return count


def _create_bell_bling_relationships(tx):
    """
    Create AFFECTS_BLING relationships for bells that modify Bling economy.

    Returns:
        Number of relationships created
    """
    if not _BELL_BLING:
        return 0

    result = tx.run("""
        UNWIND $bells AS bell_name
        MATCH (b:Bell {name: bell_name})
        MATCH (bl:Bling {name: "Bling"})
        MERGE (b)-[:AFFECTS_BLING]->(bl)
        RETURN count(*) AS created
    """, bells=_BELL_BLING)
    count = result.single()["created"]
    logger.info(f"Created {count} AFFECTS_BLING relationships")
    return count


def _create_bell_card_relationships(tx):
    """
    Create ADDS_TO_FIGHT relationships for bells that add cards to fights.

    Returns:
        Number of relationships created
    """
    pairs = [
        {"bell_name": bell, "card_name": card}
        for bell, card in _BELL_CARD_MAP.items()
    ]
    if not pairs:
        return 0

    result = tx.run("""
        UNWIND $pairs AS p
        MATCH (b:Bell {name: p.bell_name})
        MATCH (c:Card {card_name: p.card_name})
        MERGE (b)-[:ADDS_TO_FIGHT]->(c)
        RETURN count(*) AS created
    """, pairs=pairs)
    count = result.single()["created"]
    logger.info(f"Created {count} ADDS_TO_FIGHT relationships")
    return count


def _create_bell_target_relationships(tx):
    """
    Create TARGETS_CARD_TYPE relationships linking bells to the
    CardType nodes they affect.

    Returns:
        Number of relationships created
    """
    pairs = [
        {"bell_name": bell, "card_type": ct}
        for bell, card_types in _BELL_TARGETS_CARD_TYPE.items()
        for ct in card_types
    ]
    if not pairs:
        return 0

    result = tx.run("""
        UNWIND $pairs AS p
        MATCH (b:Bell {name: p.bell_name})
        MATCH (ct:CardType {name: p.card_type})
        MERGE (b)-[:TARGETS_CARD_TYPE]->(ct)
        RETURN count(*) AS created
    """, pairs=pairs)
    count = result.single()["created"]
    logger.info(f"Created {count} TARGETS_CARD_TYPE relationships")
    return count


def _create_bell_map_event_relationships(tx):
    """
    Create AFFECTS_MAP_EVENT relationships. Currently only Gloom Bell
    affects specific map events (card rewards can have cursed charms).

    Returns:
        Number of relationships created
    """
    if not _GLOOM_BELL_MAP_EVENTS:
        return 0

    result = tx.run("""
        UNWIND $events AS event_name
        MATCH (b:Bell {name: "Gloom Bell"})
        MATCH (me:MapEvent {name: event_name})
        MERGE (b)-[:AFFECTS_MAP_EVENT]->(me)
        RETURN count(*) AS created
    """, events=_GLOOM_BELL_MAP_EVENTS)
    count = result.single()["created"]
    logger.info(f"Created {count} AFFECTS_MAP_EVENT relationships (Gloom Bell)")
    return count


def create_bell_relationships(tx):
    """
    Create all bell linking relationships.

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
    total += _create_gloom_bell_cursed_charms(tx)
    total += _create_tyrant_bell_crown(tx)
    total += _create_bell_keyword_relationships(tx)
    total += _create_bell_stat_relationships(tx)
    total += _create_bell_bling_relationships(tx)
    total += _create_bell_card_relationships(tx)
    total += _create_bell_target_relationships(tx)
    total += _create_bell_map_event_relationships(tx)
    logger.info(f"Created {total} total bell linking relationships")
    return total
