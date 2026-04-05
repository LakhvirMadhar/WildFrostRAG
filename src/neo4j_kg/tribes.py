from data_processing.tribes import TribeExclusivity
from utils.logger import logger


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
