from typing import List
from neo4j import GraphDatabase
from src.data_processing.cards import CardType
from src.data_processing.tribes import TribeExclusivity
from src.data_processing.phase_config import RECRUITABLE_ENEMIES
from src.data_processing.crowns import CROWNS, CROWNABLE_CARD_TYPES
from src.data_processing.stats import StatInfo
from src.data_processing.charms import CharmInfo
from src.data_processing.map import ZoneInfo, MapEventInfo, FightSlotInfo
from src.data_processing.shades import SummonInfo
from src.utils.config import settings
from src.utils.logger import logger


def create_cards(tx, cards_data):
    """
    Bulk create card nodes in neo4j.

    For multi-phase cards (like Infernoko Phase 1 and Phase 2), we use
    card_name + phase as the unique identifier. For single-phase cards,
    phase is null and we merge by card_name only.
    """
    # Separate phased and non-phased cards
    phased_cards = [c for c in cards_data if c.get('phase') is not None]
    non_phased_cards = [c for c in cards_data if c.get('phase') is None]

    total_created = 0

    # Create non-phased cards (MERGE by card_name only)
    if non_phased_cards:
        query = """
        UNWIND $cards AS card
        MERGE (c:Card {card_name: card.card_name})
        SET c += card
        MERGE (t:CardType {name: card.card_type})
        MERGE (c)-[:HAS_CARD_TYPE]->(t)
        RETURN count(c) AS createdCount
        """
        result = tx.run(query, cards=non_phased_cards)
        total_created += result.single()["createdCount"]

    # Create phased cards (MERGE by card_name + phase)
    if phased_cards:
        query = """
        UNWIND $cards AS card
        MERGE (c:Card {card_name: card.card_name, phase: card.phase})
        SET c += card
        MERGE (t:CardType {name: card.card_type})
        MERGE (c)-[:HAS_CARD_TYPE]->(t)
        RETURN count(c) AS createdCount
        """
        result = tx.run(query, cards=phased_cards)
        total_created += result.single()["createdCount"]

    return total_created


def create_phase_relationships(tx):
    """
    Create TRANSFORMS_INTO relationships between card phases.

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
    return result.single()["relationshipsCreated"]


def create_recruitment_relationships(tx):
    """
    Create CAN_BE_RECRUITED_AS relationships for enemy cards that can become companions.

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
    return result.single()["relationshipsCreated"]


def create_tribes(tx):
    """
    Create the three exclusive tribe nodes
    """
    # Get the names of the exclusive tribes directly from the enum
    exclusive_tribe_names = [t.value for t in TribeExclusivity if t.is_exclusive]

    query = """
    UNWIND $tribe_names AS name
    MERGE (t:Tribe {name: name})
    RETURN count(t) AS tribesCreated
    """
    result = tx.run(query, tribe_names=exclusive_tribe_names)
    return result.single()["tribesCreated"]


def create_card_tribe_relationships(tx, cards_data):
    """
    Create relationships between cards and tribes based on their exclusivity
    """
    query = """
    UNWIND $card_tribes AS card_tribe
    MATCH (c:Card {card_name: card_tribe.card_name})
    MATCH (t:Tribe {name: card_tribe.tribe_name})
    MERGE (c)-[:BELONGS_TO_TRIBE]->(t)
    RETURN count(DISTINCT c) AS processedCards
    """

    card_tribes = []

    # Debug counters
    total_cards_with_tribes = 0
    cards_by_exclusivity = {}

    for card_dict in cards_data:
        # Check if the card has a tribe exclusivity value
        if 'tribe_exclusivity' in card_dict and card_dict['tribe_exclusivity'] is not None:
            total_cards_with_tribes += 1
            exclusivity_value = card_dict['tribe_exclusivity']

            # Count by exclusivity type
            cards_by_exclusivity[exclusivity_value] = cards_by_exclusivity.get(exclusivity_value, 0) + 1

            # Recreate the enum member from its value
            try:
                exclusivity_enum_member = next(t for t in TribeExclusivity if t.value == exclusivity_value)

                # Use the get_tribes() method to get the list of tribe names
                tribes_to_link = exclusivity_enum_member.get_tribes()

                # Debug for first few cards
                if total_cards_with_tribes <= 5:
                    logger.debug(f"Card: {card_dict['card_name']}")
                    logger.debug(f"  Exclusivity: {exclusivity_value}")
                    logger.debug(f"  Is Universal: {exclusivity_enum_member.is_universal}")
                    logger.debug(f"  Tribes to link: {tribes_to_link}")

                # Handle both string and list returns from get_tribes()
                if isinstance(tribes_to_link, str):
                    tribes_to_link = [tribes_to_link]

                for tribe_name in tribes_to_link:
                    card_tribes.append({
                        'card_name': card_dict['card_name'],
                        'tribe_name': tribe_name
                    })

            except StopIteration:
                logger.error(f"Could not find enum member for value: {exclusivity_value}")
                continue

    logger.info("=== TRIBE RELATIONSHIP SUMMARY ===")
    logger.info(f"Total cards with tribe data: {total_cards_with_tribes}")
    logger.info(f"Cards by exclusivity: {cards_by_exclusivity}")
    logger.info(f"Total card-tribe relationships to create: {len(card_tribes)}")

    # Show sample relationships
    if card_tribes:
        logger.debug("Sample relationships:")
        for i, rel in enumerate(card_tribes[:10]):  # Show first 10
            logger.debug(f"  {rel['card_name']} -> {rel['tribe_name']}")
        if len(card_tribes) > 10:
            logger.debug(f"  ... and {len(card_tribes) - 10} more")

    if card_tribes:
        result = tx.run(query, card_tribes=card_tribes)
        return result.single()["processedCards"]
    return 0


def create_card_type_hierarchy(tx):
    """
    Create hierarchy relationships between card types
    """
    query = """
    UNWIND $hierarchies AS hierarchy
    MATCH (child:CardType {name: hierarchy.child})
    MATCH (parent:CardType {name: hierarchy.parent})
    MERGE (child)-[:SUBTYPE_OF]->(parent)
    """

    # Build hierarchy data from enum
    hierarchies = []
    for card_type in CardType:
        for parent in card_type.parents:
            hierarchies.append({
                'child': card_type.value,
                'parent': parent
            })

    if hierarchies:
        tx.run(query, hierarchies=hierarchies)
    return len(hierarchies)


def parse_other_stats(other_stats_str: str) -> List[tuple]:
    """
    Parse the other_stats string into (stat_name, value) tuples.

    Examples:
        "2 Teeth" -> [("Teeth", 2)]
        "x3 Frenzy" -> [("Frenzy", 3)]
        "Reaction" -> [("Reaction", 1)]
        "Reaction , 1 Block" -> [("Reaction", 1), ("Block", 1)]
        "x5 Frenzy Resist Snow" -> [("Frenzy", 5), ("Resist Snow", 1)]
    """
    import re

    if not other_stats_str:
        return []

    # Known multi-word stat names (order matters - check longer names first)
    multi_word_stats = ['Resist Snow']

    results = []

    # Split by comma
    parts = [p.strip() for p in other_stats_str.split(',')]

    for part in parts:
        if not part:
            continue

        # Handle multi-word stats first
        remaining = part
        for multi_stat in multi_word_stats:
            if multi_stat in remaining:
                results.append((multi_stat, 1))
                remaining = remaining.replace(multi_stat, '').strip()

        if not remaining:
            continue

        # Pattern: optional "x" + optional number + stat name
        # e.g., "2 Teeth", "x3 Frenzy", "Reaction"
        match = re.match(r'^x?(\d+)?\s*(.+)$', remaining)
        if match:
            value_str, stat_name = match.groups()
            stat_name = stat_name.strip()
            if stat_name:
                value = int(value_str) if value_str else 1
                results.append((stat_name, value))

    return results


# Mapping from card field names to Stat node names
STAT_FIELD_MAPPING = {
    'health': 'Health',
    'attack': 'Attack',
    'scrap': 'Scrap',
    'counter': 'Counter',
}


def _extract_primary_stats(card: dict) -> List[dict]:
    """Extract primary stats (health, attack, scrap, counter) from card fields."""
    stats = []
    card_name = card['card_name']
    for field_name, stat_name in STAT_FIELD_MAPPING.items():
        if field_name in card and card[field_name] is not None:
            stats.append({'card_name': card_name, 'stat_name': stat_name, 'value': card[field_name]})
    return stats


def _extract_other_stats(card: dict) -> List[dict]:
    """Extract stats from other_stats field (buffs/debuffs like '2 Teeth', 'x3 Frenzy')."""
    stats = []
    card_name = card['card_name']
    if 'other_stats' in card and card['other_stats']:
        for stat_name, value in parse_other_stats(card['other_stats']):
            stats.append({'card_name': card_name, 'stat_name': stat_name, 'value': value})
    return stats


def _extract_ability_stats(card: dict, all_stat_names: List[str]) -> List[dict]:
    """Extract stats mentioned in abilities (simple string match, no value)."""
    stats = []
    ability = card.get('abilities_specific', '')
    if not ability:
        return stats

    card_name = card['card_name']
    ability_lower = ability.lower()
    for stat_name in all_stat_names:
        if stat_name.lower() in ability_lower:
            stats.append({'card_name': card_name, 'stat_name': stat_name})
    return stats


def create_card_stat_relationships(tx, cards_data):
    """
    Create HAS_STAT relationships between Cards and existing Stat nodes.

    Note: Stat nodes must be created first via create_stats_from_parsed().
    Sources: primary stats (with value), other_stats (with value), abilities (no value).

    TODO: Consider using different relationship types for abilities vs inherent stats.
    """
    # Get stat names from DB, sorted by length (match "Resist Snow" before "Snow")
    stat_result = tx.run("MATCH (s:Stat) RETURN s.name AS name")
    all_stat_names = sorted([r["name"] for r in stat_result], key=len, reverse=True)

    # Collect stats from all sources
    stats_with_value = []
    stats_no_value = []

    for card in cards_data:
        stats_with_value.extend(_extract_primary_stats(card))
        stats_with_value.extend(_extract_other_stats(card))
        stats_no_value.extend(_extract_ability_stats(card, all_stat_names))

    # Run queries
    if stats_with_value:
        tx.run("""
            UNWIND $card_stats AS cs
            MATCH (c:Card {card_name: cs.card_name})
            MATCH (stat:Stat {name: cs.stat_name})
            MERGE (c)-[:HAS_STAT {value: cs.value}]->(stat)
        """, card_stats=stats_with_value)

    if stats_no_value:
        tx.run("""
            UNWIND $card_stats AS cs
            MATCH (c:Card {card_name: cs.card_name})
            MATCH (stat:Stat {name: cs.stat_name})
            MERGE (c)-[:HAS_STAT]->(stat)
        """, card_stats=stats_no_value)

    total = len(stats_with_value) + len(stats_no_value)
    logger.info(f"Creating {total} card-stat relationships ({len(stats_with_value)} with value, {len(stats_no_value)} from abilities)")
    return len(set(c['card_name'] for c in stats_with_value + stats_no_value))


def create_crowns(tx):
    """
    Create Crown nodes from hardcoded crown definitions.
    """
    crown_data = [
        {
            'name': crown.name,
            'removable': crown.removable,
            'description': crown.description,
            'max_per_card': crown.max_per_card,
        }
        for crown in CROWNS
    ]

    query = """
    UNWIND $crowns AS crown
    MERGE (c:Crown {name: crown.name})
    SET c.removable = crown.removable,
        c.description = crown.description,
        c.max_per_card = crown.max_per_card
    RETURN count(c) AS crownsCreated
    """
    result = tx.run(query, crowns=crown_data)
    return result.single()["crownsCreated"]


def create_stats_from_parsed(tx, stats: List[StatInfo]):
    """
    Create Stat nodes from parsed StatInfo objects.

    Args:
        tx: Neo4j transaction
        stats: List of StatInfo objects from parse_stats_page()

    Returns:
        Number of Stat nodes created
    """
    stat_data = [
        {
            'name': stat.name,
            'category': stat.category.value,
            'description': stat.description,
            'additional_info': stat.additional_info,
        }
        for stat in stats
    ]

    query = """
    UNWIND $stats AS stat
    MERGE (s:Stat {name: stat.name})
    SET s.category = stat.category,
        s.description = stat.description,
        s.additional_info = stat.additional_info
    RETURN count(s) AS statsCreated
    """
    result = tx.run(query, stats=stat_data)
    return result.single()["statsCreated"]


def create_charms_from_parsed(tx, charms: List[CharmInfo]):
    """Create Charm nodes from parsed CharmInfo objects."""
    charm_data = [
        {
            'name': charm.name,
            'description': charm.description,
            'is_cursed': charm.is_cursed,
            'unlock': charm.unlock,
            'challenge': charm.challenge,
        }
        for charm in charms
    ]

    query = """
    UNWIND $charms AS charm
    MERGE (c:Charm {name: charm.name})
    SET c.description = charm.description,
        c.is_cursed = charm.is_cursed,
        c.unlock = charm.unlock,
        c.challenge = charm.challenge
    RETURN count(c) AS charmsCreated
    """
    result = tx.run(query, charms=charm_data)
    return result.single()["charmsCreated"]


def create_charm_tribe_relationships(tx, charms: List[CharmInfo]):
    """
    Create EXCLUSIVE_TO relationships between Charms and Tribes.

    "All" expands to all 3 tribes (same pattern as cards).
    Handles multi-tribe like "Snowdwellers,Clunkmasters".
    Cursed charms have no tribe (no relationship created).
    """
    charm_tribe_pairs = []
    for charm in charms:
        if not charm.tribe_exclusive:
            continue

        try:
            exclusivity = TribeExclusivity(charm.tribe_exclusive)
            tribes = exclusivity.get_tribes()
        except ValueError:
            # Handle multi-tribe like "Snowdwellers,Clunkmasters"
            tribes = [t.strip() for t in charm.tribe_exclusive.split(',')]

        for tribe_name in tribes:
            charm_tribe_pairs.append({
                'charm_name': charm.name,
                'tribe_name': tribe_name,
            })

    if not charm_tribe_pairs:
        return 0

    query = """
    UNWIND $pairs AS pair
    MATCH (c:Charm {name: pair.charm_name})
    MATCH (t:Tribe {name: pair.tribe_name})
    MERGE (c)-[:BELONGS_TO_TRIBE]->(t)
    RETURN count(*) AS created
    """
    result = tx.run(query, pairs=charm_tribe_pairs)
    return result.single()["created"]


def create_crown_relationships(tx):
    """
    Create all Crown relationships:
    - IS_CURSED_VERSION_OF: Cursed Crown -> Crown
    - CAN_BE_PLACED_ON: Crown -> CardType (companions, items, clunkers, pets)
    - REDUCES: Cursed Crown -> Stat (Health, Attack)
    - STARTS_WITH_PERMANENT: CardType:leaders -> Crown
    """
    relationships_created = 0

    # IS_CURSED_VERSION_OF: Cursed Crown -> Crown
    query_cursed = """
    MATCH (cursed:Crown {name: "Cursed Crown"})
    MATCH (regular:Crown {name: "Crown"})
    MERGE (cursed)-[:IS_CURSED_VERSION_OF]->(regular)
    RETURN count(*) AS created
    """
    result = tx.run(query_cursed)
    relationships_created += result.single()["created"]

    # CAN_BE_PLACED_ON: Crown -> CardType
    query_placeable = """
    UNWIND $card_types AS card_type_name
    MATCH (crown:Crown {name: "Crown"})
    MATCH (ct:CardType {name: card_type_name})
    MERGE (crown)-[:CAN_BE_PLACED_ON]->(ct)
    RETURN count(*) AS created
    """
    result = tx.run(query_placeable, card_types=CROWNABLE_CARD_TYPES)
    relationships_created += result.single()["created"]

    # Cursed Crown can also be placed on same card types
    query_cursed_placeable = """
    UNWIND $card_types AS card_type_name
    MATCH (crown:Crown {name: "Cursed Crown"})
    MATCH (ct:CardType {name: card_type_name})
    MERGE (crown)-[:CAN_BE_PLACED_ON]->(ct)
    RETURN count(*) AS created
    """
    result = tx.run(query_cursed_placeable, card_types=CROWNABLE_CARD_TYPES)
    relationships_created += result.single()["created"]

    # REDUCES: Cursed Crown -> Stat (Health, Attack)
    # Find the cursed crown and create REDUCES relationships
    for crown in CROWNS:
        if crown.reduces_stats:
            query_reduces = """
            UNWIND $stats AS stat_name
            MATCH (cursed:Crown {name: $crown_name})
            MATCH (stat:Stat {name: stat_name})
            MERGE (cursed)-[:REDUCES {amount: $amount}]->(stat)
            RETURN count(*) AS created
            """
            result = tx.run(
                query_reduces,
                crown_name=crown.name,
                stats=crown.reduces_stats,
                amount=crown.reduces_amount
            )
            relationships_created += result.single()["created"]

    # STARTS_WITH_PERMANENT: CardType:leaders -> Crown
    query_leader_crown = """
    MATCH (leader_type:CardType {name: "leaders"})
    MATCH (crown:Crown {name: "Crown"})
    MERGE (leader_type)-[:STARTS_WITH_PERMANENT]->(crown)
    RETURN count(*) AS created
    """
    result = tx.run(query_leader_crown)
    relationships_created += result.single()["created"]

    return relationships_created


def create_map_graph(tx, zones: List[ZoneInfo], map_events: List[MapEventInfo], fight_slots: List[FightSlotInfo], fight_page_mapping: dict[str, str] = None):
    """
    Create the full map graph: Map node, Zones, MapEvents, Fights, and all relationships.

    Graph structure:
        (Map)-[:HAS_ZONE]->(Zone)-[:HAS_FIGHT {fight_number}]->(Fight)
        (Map)-[:HAS_MAP_EVENT]->(MapEvent)

    Args:
        fight_page_mapping: Display name → wiki page slug (e.g., {"Infernoko": "Infernoko_Fight"})
    """
    # Create Map node
    tx.run("MERGE (m:Map {name: 'Map'})")

    # Create Zone nodes + Map->Zone
    zone_count = _create_zones(tx, zones)

    # Create MapEvent nodes + Map->MapEvent
    event_count = _create_map_events(tx, map_events)

    # Create Fight nodes + Zone->Fight
    fight_count = _create_fights_from_slots(tx, fight_slots, fight_page_mapping or {})

    logger.info(f"Map graph: 1 Map, {zone_count} Zones, {event_count} MapEvents, {fight_count} Fights")
    return {'zones': zone_count, 'map_events': event_count, 'fights': fight_count}


def _create_zones(tx, zones: List[ZoneInfo]):
    """Create Zone nodes and Map->Zone relationships."""
    zone_data = [
        {
            'name': zone.name,
            'zone_order': zone.zone_order,
            'description': zone.description,
        }
        for zone in zones
    ]

    query = """
    UNWIND $zones AS zone
    MERGE (z:Zone {name: zone.name})
    SET z.zone_order = zone.zone_order,
        z.description = zone.description
    WITH z
    MATCH (m:Map {name: 'Map'})
    MERGE (m)-[:HAS_ZONE]->(z)
    RETURN count(z) AS created
    """
    result = tx.run(query, zones=zone_data)
    return result.single()["created"]


def _create_map_events(tx, map_events: List[MapEventInfo]):
    """Create MapEvent nodes and Map->MapEvent relationships."""
    event_data = [
        {
            'name': event.name,
            'description': event.description,
            'notes': event.notes,
        }
        for event in map_events
    ]

    query = """
    UNWIND $events AS event
    MERGE (e:MapEvent {name: event.name})
    SET e.description = event.description,
        e.notes = event.notes
    WITH e
    MATCH (m:Map {name: 'Map'})
    MERGE (m)-[:HAS_MAP_EVENT]->(e)
    RETURN count(e) AS created
    """
    result = tx.run(query, events=event_data)
    return result.single()["created"]


def _create_fights_from_slots(tx, fight_slots: List[FightSlotInfo], fight_page_mapping: dict[str, str]):
    """
    Create Fight nodes and link them to Zones via HAS_FIGHT.

    The fight_number goes on the relationship so we know which fights
    are alternatives at the same position in the run.

    Args:
        fight_page_mapping: Display name → wiki page slug for document linking
    """
    fight_data = []
    for slot in fight_slots:
        for fight_name in slot.possible_fights:
            page_name = fight_page_mapping.get(fight_name, fight_name.replace(' ', '_'))
            fight_data.append({
                'name': fight_name,
                'page_name': page_name,
            })

    fight_query = """
    UNWIND $fights AS fight
    MERGE (f:Fight {name: fight.name})
    SET f.page_name = fight.page_name
    RETURN count(f) AS created
    """
    result = tx.run(fight_query, fights=fight_data)
    fights_created = result.single()["created"]

    zone_fight_pairs = []
    for slot in fight_slots:
        for fight_name in slot.possible_fights:
            zone_fight_pairs.append({
                'zone_name': slot.zone,
                'fight_name': fight_name,
                'fight_number': slot.fight_number,
            })

    rel_query = """
    UNWIND $pairs AS pair
    MATCH (z:Zone {name: pair.zone_name})
    MATCH (f:Fight {name: pair.fight_name})
    MERGE (z)-[:HAS_FIGHT {fight_number: pair.fight_number}]->(f)
    RETURN count(*) AS created
    """
    result = tx.run(rel_query, pairs=zone_fight_pairs)

    return fights_created


def create_fight_enemy_relationships(tx, fight_enemies: dict[str, list[str]], fight_page_mapping: dict[str, str]) -> int:
    """
    Link Fight nodes to their enemy Card nodes via FEATURES_ENEMY.

    Handles multi-phase cards (e.g., "Infernoko" matches both
    "Infernoko Phase 1" and "Infernoko Phase 2").

    Args:
        fight_enemies: page_name → list of enemy card names from the fight page
        fight_page_mapping: display_name → page_name (to map fight names to page names)

    Returns:
        Number of relationships created
    """
    # Build reverse mapping: page_name → display_name(s)
    page_to_display = {}
    for display_name, page_name in fight_page_mapping.items():
        page_to_display.setdefault(page_name, []).append(display_name)

    pairs = []
    for page_name, enemy_names in fight_enemies.items():
        # Get the display name(s) for this fight
        display_names = page_to_display.get(page_name, [])
        for fight_display_name in display_names:
            for enemy_name in enemy_names:
                pairs.append({
                    'fight_name': fight_display_name,
                    'enemy_name': enemy_name,
                })

    if not pairs:
        return 0

    # Match cards by:
    #   - Exact name: "Beeberry"
    #   - Multi-phase: "Infernoko" → "Infernoko Phase 1", "Infernoko Phase 2"
    #   - Parenthetical variant: "Naked Gnome" → "Naked Gnome (Enemy)",
    #     "Frost Guardian" → "Frost Guardian (Frost Wizard)"
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
    count = result.single()["created"]
    logger.info(f"Created {count} FEATURES_ENEMY relationships")
    return count


def create_summon_relationships(tx, summons: List[SummonInfo]) -> int:
    """
    Create SUMMONS relationships between cards.

    Links summoner cards (items, companions, or shades) to the shades they summon.
    e.g., (Beepop Mask)-[:SUMMONS]->(Beepop)

    Args:
        summons: List of SummonInfo with summoner_name and shade_name

    Returns:
        Number of relationships created
    """
    if not summons:
        return 0

    pairs = [{'summoner_name': s.summoner_name, 'shade_name': s.shade_name} for s in summons]

    query = """
    UNWIND $pairs AS pair
    MATCH (summoner:Card {card_name: pair.summoner_name})
    MATCH (shade:Card {card_name: pair.shade_name})
    MERGE (summoner)-[:SUMMONS]->(shade)
    RETURN count(*) AS created
    """
    result = tx.run(query, pairs=pairs)
    count = result.single()["created"]
    logger.info(f"Created {count} SUMMONS relationships")
    return count


def clear_database(tx) -> None:
    """
    Optional: Clear all nodes and relationships (use with caution!)
    """
    query = "MATCH (n) DETACH DELETE n"
    tx.run(query)


def create_neo4j_data(cards_data):
    """
    Main function to run the card import process into neo4j

    cards_data = List of cardInfo objects containing the cards to import
    """

    # Define the URI and authentication
    uri = settings.neo4j_uri.get_secret_value()              # Use URI from config
    username = settings.neo4j_username    # Your Neo4j username from config
    password = settings.neo4j_password.get_secret_value()    # Your Neo4j password from config

    driver = GraphDatabase.driver(uri, auth=(username, password))

    try:
        driver.verify_authentication()
        logger.info('Connected to Neo4j!')

        with driver.session() as session:
            # Create tribes first
            tribes_created = session.execute_write(create_tribes)
            logger.info(f"Created {tribes_created} tribes")

            # Create cards first
            created_count = session.execute_write(create_cards, cards_data)
            logger.info(f"Created/updated {created_count} cards")

            # Create hierarchy relationships
            hierarchy_count = session.execute_write(create_card_type_hierarchy)
            logger.info(f"Created {hierarchy_count} hierarchy relationships")

            # Create tribe relationships
            tribe_relationships = session.execute_write(create_card_tribe_relationships, cards_data)
            logger.info(f"Created tribe relationships for {tribe_relationships} cards")

            # Create card-stat relationships (Stat nodes created separately in ingest_data.py)
            stats_processed = session.execute_write(create_card_stat_relationships, cards_data)
            logger.info(f"Processed stats for {stats_processed} cards")

            # Create phase relationships (TRANSFORMS_INTO)
            phase_relationships = session.execute_write(create_phase_relationships)
            logger.info(f"Created {phase_relationships} phase relationships (TRANSFORMS_INTO)")

            # Create recruitment relationships (CAN_BE_RECRUITED_AS)
            recruitment_relationships = session.execute_write(create_recruitment_relationships)
            logger.info(f"Created {recruitment_relationships} recruitment relationships (CAN_BE_RECRUITED_AS)")

            # Create crowns
            crowns_created = session.execute_write(create_crowns)
            logger.info(f"Created {crowns_created} crown nodes")

            # Create crown relationships
            crown_relationships = session.execute_write(create_crown_relationships)
            logger.info(f"Created {crown_relationships} crown relationships")

        logger.info("Import completed successfully")
    except Exception as e:
        logger.error(f'Connection failed: {e}')

    finally:
        driver.close()