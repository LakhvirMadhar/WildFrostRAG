import neo4j

from data_processing.shades import SummonInfo
from neo4j_kg.query_utils import single_value
from utils.logger import logger


def create_summon_relationships(tx: neo4j.ManagedTransaction, summons: list[SummonInfo]) -> int:
    """Create SUMMONS relationships between cards.

    Links summoner cards (items, companions, or shades) to the shades they summon.
    e.g., (Beepop Mask)-[:SUMMONS]->(Beepop)

    Args:
        tx: Neo4j managed transaction
        summons: List of SummonInfo with summoner_name and shade_name

    Returns:
        Number of relationships created
    """
    if not summons:
        return 0

    pairs = [{"summoner_name": s.summoner_name, "shade_name": s.shade_name} for s in summons]

    query = """
    UNWIND $pairs AS pair
    MATCH (summoner:Card {card_name: pair.summoner_name})
    MATCH (shade:Card {card_name: pair.shade_name})
    MERGE (summoner)-[:SUMMONS]->(shade)
    RETURN count(*) AS created
    """
    result = tx.run(query, pairs=pairs)
    count = single_value(result, "created")
    logger.info(f"Created {count} SUMMONS relationships")
    return count
