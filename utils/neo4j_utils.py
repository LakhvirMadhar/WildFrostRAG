import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from typing import List, Dict, Any
from utils.cards import CardType, TribeExclusivity
load_dotenv()


def create_cards(tx, cards_data):
    """
    Bulk create card nodes in neo4j
    """
    query = """
    UNWIND $cards AS card
    MERGE (c:Card {card_name: card.card_name})
    SET c += card
    MERGE (t:CardType {name: card.card_type})
    MERGE (c)-[:HAS_CARD_TYPE]->(t)
    RETURN count(c) AS createdCount
    """
    result = tx.run(query, cards=cards_data)
    return result.single()["createdCount"]


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
                    print(f"Card: {card_dict['card_name']}")
                    print(f"  Exclusivity: {exclusivity_value}")
                    print(f"  Is Universal: {exclusivity_enum_member.is_universal}")
                    print(f"  Tribes to link: {tribes_to_link}")
                    print(f"  Type of tribes_to_link: {type(tribes_to_link)}")

                # Handle both string and list returns from get_tribes()
                if isinstance(tribes_to_link, str):
                    tribes_to_link = [tribes_to_link]
                
                for tribe_name in tribes_to_link:
                    card_tribes.append({
                        'card_name': card_dict['card_name'],
                        'tribe_name': tribe_name
                    })
                    
            except StopIteration:
                print(f"ERROR: Could not find enum member for value: {exclusivity_value}")
                continue
    
    print(f"\n=== TRIBE RELATIONSHIP SUMMARY ===")
    print(f"Total cards with tribe data: {total_cards_with_tribes}")
    print(f"Cards by exclusivity: {cards_by_exclusivity}")
    print(f"Total card-tribe relationships to create: {len(card_tribes)}")
    
    # Show sample relationships
    if card_tribes:
        print(f"Sample relationships:")
        for i, rel in enumerate(card_tribes[:10]):  # Show first 10
            print(f"  {rel['card_name']} -> {rel['tribe_name']}")
        if len(card_tribes) > 10:
            print(f"  ... and {len(card_tribes) - 10} more")
    
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
    uri = "neo4j://127.0.0.1:7687"              # Adjust if using a remote server or different port
    username = os.getenv('NEO4J_USERNAME')      # Your Neo4j username
    password = os.getenv('NEO4J_PASSWORD')      # Your Neo4j password

    driver = GraphDatabase.driver(uri, auth=(username,password))

    try:
        driver.verify_authentication()
        print('Connected to Neo4j!')

        with driver.session() as session:
            # TEMP - Clear the database as I'm still creating this
            session.execute_write(clear_database)
            print("Database cleared")

            # Create tribes first
            tribes_created = session.execute_write(create_tribes)
            print(f"Created {tribes_created} tribes")

            # Create cards first
            created_count = session.execute_write(create_cards, cards_data)
            print(f"Created/updated {created_count} cards")

            # Create hierarchy relationships
            hierarchy_count = session.execute_write(create_card_type_hierarchy)
            print(f"Created {hierarchy_count} hierarchy relationships")

            # Create tribe relationships
            tribe_relationships = session.execute_write(create_card_tribe_relationships, cards_data)
            print(f"Created tribe relationships for {tribe_relationships} cards")

            # Create stats
            stats_processed = session.execute_write(create_card_stats_flexible, cards_data)
            print(f"Processed stats for {stats_processed} cards")

        print("Import completed successfully")
    except Exception as e:
        print(f'Connection failed: {e}')

    finally:
        driver.close()