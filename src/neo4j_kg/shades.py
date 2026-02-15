from typing import List

from src.data_processing.shades import SummonInfo
from src.utils.logger import logger


def create_summon_relationships(tx, summons: List[SummonInfo]) -> int:
    """
    Create SUMMONS relationships between cards.

    Links summoner cards (items, companions, or shades) to the shades they summon.
    e.g., (Beepop Mask)-[:SUMMONS]->(Beepop)

    Args:
        summons: List of SummonInfo with summoner_name and shade_name

    Returns:
        Number of relationships created
    """
    if not summons:
        return 0

    pairs = [{'summoner_name': s.summoner_name, 'shade_name': s.shade_name} for s in summons]

    query = """
    UNWIND $pairs AS pair
    MATCH (summoner:Card {card_name: pair.summoner_name})
    MATCH (shade:Card {card_name: pair.shade_name})
    MERGE (summoner)-[:SUMMONS]->(shade)
    RETURN count(*) AS created
    """
    result = tx.run(query, pairs=pairs)
    count = result.single()["created"]
    logger.info(f"Created {count} SUMMONS relationships")
    return count
