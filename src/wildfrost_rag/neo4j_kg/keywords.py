"""Keyword node creation and card/charm-keyword relationship management for Neo4j."""

import re
from typing import Any

import neo4j

from wildfrost_rag.data_processing.keywords import KeywordInfo
from wildfrost_rag.neo4j_kg.query_utils import single_value
from wildfrost_rag.utils.logger import logger


# Suffixes to check when matching keyword roots against card/charm text.
_INFLECTION_SUFFIXES = r"(?:ing|ed|es|s|ied|ies|er|ers|ly)?"


def _keyword_matches_text(keyword_lower: str, text_lower: str) -> bool:
    """Check if a keyword or any morphological variant appears in text.

    Two-pass approach:
      1. Substring match (existing behavior, handles most cases).
      2. For single-word keywords, generate alternate roots based on English
         inflection rules and regex-match with common suffixes. This catches
         cases like sacrifice→sacrificing (e-drop) and ally→allies (y→i).
    """
    if keyword_lower in text_lower:
        return True

    # Only apply morphological matching to single-word keywords
    if " " in keyword_lower:
        return False

    # Build alternate roots from English inflection patterns
    roots = []
    if keyword_lower.endswith("e"):
        # e-drop before -ing: sacrifice → sacrific(+ing)
        roots.append(keyword_lower[:-1])
    if keyword_lower.endswith("y"):
        # y→i before -ed/-es: frenzy → frenzi(+ed), apply → appli(+ed)
        roots.append(keyword_lower[:-1] + "i")

    for root in roots:
        pattern = r"\b" + re.escape(root) + _INFLECTION_SUFFIXES + r"\b"
        if re.search(pattern, text_lower):
            return True

    return False


def create_keywords_from_parsed(
    tx: neo4j.ManagedTransaction, keywords: list[KeywordInfo], url: str | None = None
) -> int:
    """Create Keyword nodes from parsed KeywordInfo objects.

    Args:
        tx: Neo4j transaction
        keywords: List of KeywordInfo objects from parse_keywords_page()
        url: Wiki page URL for all keywords (shared /Keywords page)

    Returns:
        Number of Keyword nodes created
    """
    keyword_data = [
        {
            "name": kw.name,
            "category": kw.category.value,
            "description_field": kw.description_field,
            "description_items": kw.description_items,
        }
        for kw in keywords
    ]

    query = """
    UNWIND $keywords AS kw
    MERGE (k:Keyword {name: kw.name})
    SET k.category = kw.category,
        k.description_field = kw.description_field,
        k.description_items = kw.description_items,
        k.url = $url
    RETURN count(k) AS created
    """
    result = tx.run(query, keywords=keyword_data, url=url)
    count = single_value(result, "created")
    logger.info(f"Created {count} Keyword nodes")
    return count


def create_card_keyword_relationships(
    tx: neo4j.ManagedTransaction, cards_data: list[dict[str, Any]]
) -> int:
    """Create HAS_KEYWORD relationships between Cards and Keyword nodes.

    Scans each card's abilities_specific text for keyword name matches.
    Keyword nodes must be created first via create_keywords_from_parsed().

    Args:
        tx: Neo4j transaction
        cards_data: List of card dicts (must have card_name, abilities_specific)

    Returns:
        Number of HAS_KEYWORD relationships created
    """
    # Get keyword names from DB, sorted by length (match longer names first)
    keyword_result = tx.run("MATCH (k:Keyword) RETURN k.name AS name")
    all_keyword_names = sorted([r["name"] for r in keyword_result], key=len, reverse=True)

    card_keywords = []
    for card in cards_data:
        ability = card.get("abilities_specific", "")
        if not ability:
            continue

        card_name = card["card_name"]
        ability_lower = ability.lower()
        for keyword_name in all_keyword_names:
            if _keyword_matches_text(keyword_name.lower(), ability_lower):
                card_keywords.append({"card_name": card_name, "keyword_name": keyword_name})

    if card_keywords:
        tx.run(
            """
            UNWIND $card_keywords AS ck
            MATCH (c:Card {card_name: ck.card_name})
            MATCH (k:Keyword {name: ck.keyword_name})
            MERGE (c)-[:HAS_KEYWORD]->(k)
            """,
            card_keywords=card_keywords,
        )

    logger.info(f"Created {len(card_keywords)} card-keyword relationships")
    return len(card_keywords)


def create_charm_keyword_relationships(tx: neo4j.ManagedTransaction) -> int:
    """Create HAS_KEYWORD relationships between Charms and Keyword nodes.

    Scans each charm's description text for keyword name matches.
    Keyword and Charm nodes must be created first.

    Args:
        tx: Neo4j transaction

    Returns:
        Number of HAS_KEYWORD relationships created
    """
    keyword_result = tx.run("MATCH (k:Keyword) RETURN k.name AS name")
    all_keyword_names = sorted([r["name"] for r in keyword_result], key=len, reverse=True)

    charm_result = tx.run("MATCH (ch:Charm) RETURN ch.name AS name, ch.description AS description")
    charm_keywords = []
    for charm in charm_result:
        description = charm["description"] or ""
        if not description:
            continue

        charm_name = charm["name"]
        desc_lower = description.lower()
        for keyword_name in all_keyword_names:
            if _keyword_matches_text(keyword_name.lower(), desc_lower):
                charm_keywords.append({"charm_name": charm_name, "keyword_name": keyword_name})

    if charm_keywords:
        tx.run(
            """
            UNWIND $charm_keywords AS ck
            MATCH (ch:Charm {name: ck.charm_name})
            MATCH (k:Keyword {name: ck.keyword_name})
            MERGE (ch)-[:HAS_KEYWORD]->(k)
            """,
            charm_keywords=charm_keywords,
        )

    logger.info(f"Created {len(charm_keywords)} charm-keyword relationships")
    return len(charm_keywords)
