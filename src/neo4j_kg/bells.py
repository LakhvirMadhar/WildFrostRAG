"""
Bell node creation for Neo4j.
"""

from typing import List

from src.data_processing.bells import BellCategory, BellInfo
from src.utils.logger import logger

# Maps BellCategory enum values to BellType node names
_CATEGORY_TO_BELL_TYPE = {
    BellCategory.SUN.value: "Sun Bell",
    BellCategory.STORM.value: "Storm Bell",
    BellCategory.MODIFIER.value: "Modifier Bell",
}


def create_bells_from_parsed(tx, bells: List[BellInfo]):
    """
    Create Bell nodes, BellType nodes, and HAS_BELL_TYPE relationships.

    Args:
        tx: Neo4j transaction
        bells: List of BellInfo objects from parse_bells_page()

    Returns:
        Number of Bell nodes created
    """
    bell_data = [
        {
            "name": b.name,
            "category": b.category.value,
            "bell_type": _CATEGORY_TO_BELL_TYPE[b.category.value],
            "description": b.description,
            "notes": b.notes,
            "storm_strength": b.storm_strength,
        }
        for b in bells
    ]

    query = """
    UNWIND $bells AS b
    MERGE (bell:Bell {name: b.name})
    SET bell.category = b.category,
        bell.description = b.description,
        bell.notes = b.notes,
        bell.storm_strength = b.storm_strength
    MERGE (bt:BellType {name: b.bell_type})
    MERGE (bell)-[:HAS_BELL_TYPE]->(bt)
    RETURN count(bell) AS created
    """
    result = tx.run(query, bells=bell_data)
    count = result.single()["created"]
    logger.info(f"Created {count} Bell nodes with BellType relationships")
    return count
