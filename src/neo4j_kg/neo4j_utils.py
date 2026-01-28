from neo4j import GraphDatabase
from src.data_processing.cards import CardType, TribeExclusivity
from src.data_processing.phase_config import RECRUITABLE_ENEMIES
from src.data_processing.crowns import CROWNS, CROWNABLE_CARD_TYPES
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


def create_card_stats_flexible(tx, cards_data):
    """
    Flexible stat creation that can handle any stat type
    """
    query = """
    UNWIND $card_stats AS card_stat
    MATCH (c:Card {card_name: card_stat.card_name})

    MERGE (stat:Stat {name: card_stat.stat_name, category: card_stat.category})
    MERGE (c)-[:HAS_STAT {
        value: card_stat.value,
        category: card_stat.category
    }]->(stat)

    RETURN count(DISTINCT c) AS processedCards
    """

    # Build card-stat combinations only for stats that exist
    card_stats = []
    stat_definitions = [
        {'field_name': 'health', 'stat_name': 'Health', 'category': 'primary'},
        {'field_name': 'attack', 'stat_name': 'Attack', 'category': 'primary'},
        {'field_name': 'scrap', 'stat_name': 'Scrap', 'category': 'primary'},
        {'field_name': 'counter', 'stat_name': 'Counter', 'category': 'primary'},
    ]

    for card in cards_data:
        for stat_def in stat_definitions:
            field_name = stat_def['field_name']
            # Only add if the field exists in the card AND has a non-None value
            if field_name in card and card[field_name] is not None:
                card_stats.append({
                    'card_name': card['card_name'],
                    'stat_name': stat_def['stat_name'],
                    'category': stat_def['category'],
                    'value': card[field_name]
                })

    result = tx.run(query, card_stats=card_stats)
    return result.single()["processedCards"]


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

            # Create stats
            stats_processed = session.execute_write(create_card_stats_flexible, cards_data)
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