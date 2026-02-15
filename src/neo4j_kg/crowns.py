from src.data_processing.crowns import CROWNS, CROWNABLE_CARD_TYPES


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
