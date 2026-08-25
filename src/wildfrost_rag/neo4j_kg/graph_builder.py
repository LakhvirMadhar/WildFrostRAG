from typing import Any

import neo4j

from wildfrost_rag.neo4j_kg.query_utils import single_value
from wildfrost_rag.neo4j_kg.cards import (
    create_cards,
    create_phase_relationships,
    create_recruitment_relationships,
    create_card_type_hierarchy,
)
from wildfrost_rag.neo4j_kg.tribes import create_tribes, create_card_tribe_relationships
from wildfrost_rag.neo4j_kg.stats import create_card_stat_relationships
from wildfrost_rag.neo4j_kg.crowns import create_crowns, create_crown_relationships
from wildfrost_rag.utils.logger import logger


def clear_database(tx: neo4j.ManagedTransaction) -> None:
    """Optional: Clear all nodes and relationships (use with caution!)."""
    query = "MATCH (n) DETACH DELETE n"
    tx.run(query)


def create_neo4j_data(
    session: neo4j.Session,
    cards_data: list[dict[str, Any]],
    crowns_url: str | None = None,
) -> None:
    """Create card-related nodes and relationships in Neo4j.

    Args:
        session: Neo4j session (caller manages driver lifecycle)
        cards_data: List of card dicts to import
        crowns_url: Wiki page URL for Crown nodes
    """
    # Create tribes first
    tribes_created = session.execute_write(create_tribes)
    logger.info(f"Created {tribes_created} tribes")

    # Create cards
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
    logger.info(
        f"Created {recruitment_relationships} recruitment relationships (CAN_BE_RECRUITED_AS)"
    )

    # Create crowns
    crowns_created = session.execute_write(create_crowns, crowns_url)
    logger.info(f"Created {crowns_created} crown nodes")

    # Create crown relationships
    crown_relationships = session.execute_write(create_crown_relationships)
    logger.info(f"Created {crown_relationships} crown relationships")

    logger.info("Card import completed successfully")


def create_url_nodes(tx: neo4j.ManagedTransaction) -> int:
    """Create URL nodes and HAS_LINK relationships for all entities with a url property.

    Queries all nodes that have a non-null url property, creates a URL node for each
    unique URL, and links each entity to its URL node via HAS_LINK.

    This runs after all entity nodes are created so it picks up everything in one pass.
    """
    result = tx.run("""
        MATCH (n)
        WHERE n.url IS NOT NULL
        WITH DISTINCT n.url AS url
        MERGE (u:URL {url: url})
        RETURN count(u) AS created
    """)
    url_count = single_value(result, "created")

    result = tx.run("""
        MATCH (n)
        WHERE n.url IS NOT NULL
        MATCH (u:URL {url: n.url})
        MERGE (n)-[:HAS_LINK]->(u)
        RETURN count(*) AS linked
    """)
    link_count = single_value(result, "linked")

    logger.info(f"Created {url_count} URL nodes and {link_count} HAS_LINK relationships")
    return link_count
