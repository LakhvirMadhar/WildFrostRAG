from neo4j import GraphDatabase

from src.neo4j_kg.cards import (
    create_cards,
    create_phase_relationships,
    create_recruitment_relationships,
    create_card_type_hierarchy,
)
from src.neo4j_kg.tribes import create_tribes, create_card_tribe_relationships
from src.neo4j_kg.stats import create_card_stat_relationships
from src.neo4j_kg.crowns import create_crowns, create_crown_relationships
from src.utils.config import settings
from src.utils.logger import logger


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
