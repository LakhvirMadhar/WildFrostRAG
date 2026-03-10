import re
from typing import List

from src.data_processing.stats import StatInfo
from src.utils.logger import logger


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


# Stats that are passive traits only — NOT game mechanics that cards "apply"
STATS_NOT_KEYWORDS = {'Reaction', 'Resist Snow'}


def add_keyword_label_to_stats(tx):
    """
    Add :Keyword label to Stat nodes that also function as keywords.

    Stats like Frost, Shroom, Bom etc. are both numeric stats AND game mechanics
    that cards actively "apply". Adding the :Keyword label lets them be found
    by MATCH (k:Keyword) alongside traditional keywords like Barrage or Consume.

    Excludes passive-only stats (Reaction, Resist Snow) that are never "applied".
    """
    result = tx.run("""
        MATCH (s:Stat)
        WHERE NOT s.name IN $exclude
        SET s:Keyword
        RETURN count(s) AS labeled
    """, exclude=list(STATS_NOT_KEYWORDS))
    count = result.single()["labeled"]
    logger.info(f"Added :Keyword label to {count} Stat nodes")
    return count


def create_stats_from_parsed(tx, stats: List[StatInfo], url: str = None):
    """
    Create Stat nodes from parsed StatInfo objects.

    Args:
        tx: Neo4j transaction
        stats: List of StatInfo objects from parse_stats_page()
        url: Wiki page URL for all stats (shared /Stats page)

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
        s.additional_info = stat.additional_info,
        s.url = $url
    RETURN count(s) AS statsCreated
    """
    result = tx.run(query, stats=stat_data, url=url)
    return result.single()["statsCreated"]
