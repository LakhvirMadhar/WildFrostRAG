"""Neo4j query utilities for WildFrostRAG."""

import neo4j


def single_value(result: neo4j.Result, key: str) -> int:
    """Extract a single integer value from a Neo4j query result.

    All neo4j_kg functions expect exactly one result record with a named field.
    This helper handles the None check for Result.single().

    Args:
        result: Neo4j query result.
        key: Name of the field to extract.

    Raises:
        RuntimeError: If the query returned no records.
    """
    record = result.single()
    if record is None:
        raise RuntimeError(f"Expected a single record with key '{key}', got None")
    count: int = record[key]
    return count
