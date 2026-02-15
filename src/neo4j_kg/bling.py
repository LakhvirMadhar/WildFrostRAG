"""
Bling node, Shop nodes, and economy relationships for Neo4j.
"""

from typing import List

from src.data_processing.bling import EnemyBlingDrop, ShopListing
from src.utils.logger import logger


def create_bling_and_shops(tx):
    """
    Create the Bling node and Shop nodes.

    Args:
        tx: Neo4j transaction

    Returns:
        Number of nodes created
    """
    query = """
    MERGE (b:Bling {name: "Bling"})
    SET b.description = "Currency used to purchase goods from shops during a run"

    MERGE (s1:Shop {name: "The Woolly Snail"})
    SET s1.description = "Shop for purchasing Items and Crowns",
        s1.stock = "3 non-Consumable Items (30-50 Blings), 1 Consumable Item (45-65 Blings), 1 Crown (75 Blings), Charm Dispenser (up to 3 uses: 45/65/85 Blings)",
        s1.discount = "1 of 4 Items discounted 50%",
        s1.item_price_formula = "base_price * random(0.8-1.2) - 5",
        s1.crown_price = 75,
        s1.charm_dispenser_prices = "45 / 65 / 85 Blings per use, resets each visit"

    MERGE (s2:Shop {name: "Charm Merchant"})
    SET s2.description = "Shop for purchasing Charms and an upgraded Item or Clunker with charms",
        s2.stock = "3 Charms, 1 Item or Clunker with 1-2 charms attached",
        s2.charm_price_formula = "base_price + random(-30, +30)",
        s2.upgraded_card_price_formula = "Item shop value or Clunker price + 10-20 Blings per charm - 5",
        s2.upgraded_card_two_charm_chance = "1% chance the Item or Clunker has 2 charms instead of 1"

    RETURN 3 AS created
    """
    result = tx.run(query)
    count = result.single()["created"]
    logger.info(f"Created Bling node and {count - 1} Shop nodes")
    return count


def create_drops_bling_relationships(tx, drops: List[EnemyBlingDrop]):
    """
    Create DROPS_BLING relationships between enemy Cards and the Bling node.

    For multi-phase bosses (TRANSFORMS_INTO), only the final phase drops bling.

    Args:
        tx: Neo4j transaction
        drops: List of EnemyBlingDrop objects

    Returns:
        Number of relationships created
    """
    drop_data = [{"card_name": d.card_name, "amount": d.amount} for d in drops]

    query = """
    UNWIND $drops AS d
    MATCH (c:Card {card_name: d.card_name})
    MATCH (b:Bling {name: "Bling"})
    MERGE (c)-[r:DROPS_BLING]->(b)
    SET r.amount = d.amount
    RETURN count(r) AS created
    """
    result = tx.run(query, drops=drop_data)
    count = result.single()["created"]

    # For multi-phase cards, propagate DROPS_BLING to the final phase
    # (the one with no outgoing TRANSFORMS_INTO)
    propagate = tx.run("""
    MATCH (c:Card)-[r:DROPS_BLING]->(b:Bling)
    WHERE (c)-[:TRANSFORMS_INTO]->()
    MATCH (c)-[:TRANSFORMS_INTO*]->(final:Card)
    WHERE NOT (final)-[:TRANSFORMS_INTO]->()
    MERGE (final)-[r2:DROPS_BLING]->(b)
    SET r2.amount = r.amount
    RETURN count(r2) AS propagated
    """)
    propagated = propagate.single()["propagated"]

    # Remove DROPS_BLING from non-final phases (cards that TRANSFORMS_INTO something else)
    cleanup = tx.run("""
    MATCH (c:Card)-[:TRANSFORMS_INTO]->(:Card)
    MATCH (c)-[r:DROPS_BLING]->(:Bling)
    DELETE r
    RETURN count(r) AS removed
    """)
    removed = cleanup.single()["removed"]

    final_count = count + propagated - removed
    logger.info(f"Created {final_count} DROPS_BLING relationships ({propagated} propagated to final phase, {removed} removed from non-final phases)")
    return final_count


def create_shop_sells_relationships(tx, shop_name: str, listings: List[ShopListing], target_label: str):
    """
    Create SELLS relationships between a Shop and Items/Charms.

    Args:
        tx: Neo4j transaction
        shop_name: Name of the shop node
        listings: List of ShopListing objects
        target_label: Node label to match ("Card" for items, "Charm" for charms)

    Returns:
        Number of relationships created
    """
    listing_data = [{"card_name": l.card_name, "base_price": l.base_price} for l in listings]

    # Use dynamic label matching via property name
    if target_label == "Charm":
        query = """
        UNWIND $listings AS l
        MATCH (s:Shop {name: $shop_name})
        MATCH (target:Charm {name: l.card_name})
        MERGE (s)-[r:SELLS]->(target)
        SET r.base_price = l.base_price
        RETURN count(r) AS created
        """
    else:
        query = """
        UNWIND $listings AS l
        MATCH (s:Shop {name: $shop_name})
        MATCH (target:Card {card_name: l.card_name})
        MERGE (s)-[r:SELLS]->(target)
        SET r.base_price = l.base_price
        RETURN count(r) AS created
        """

    result = tx.run(query, listings=listing_data, shop_name=shop_name)
    count = result.single()["created"]
    logger.info(f"Created {count} SELLS relationships for {shop_name} -> {target_label}")
    return count
