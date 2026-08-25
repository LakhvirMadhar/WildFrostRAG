"""Shared Neo4j record-flattening helpers.

Extracted from BaseNeo4jRetriever so repositories (which now own query
execution, per T4.3) and the retriever base class share one implementation
instead of two copies.
"""

from typing import Any

from neo4j import Record
from neo4j.graph import Node, Path, Relationship

# Types for Neo4j values before and after serialization
Neo4jValue = Node | Relationship | Path | list[Any] | str | int | float | bool | None
SerializedValue = dict[str, Any] | list[Any] | str | int | float | bool | None


def serialize_value(value: Neo4jValue) -> SerializedValue:
    """Convert a Neo4j graph object to a JSON-serializable Python type.

    Handles Node, Relationship, and Path objects that raw or LLM-generated
    Cypher may return (e.g., `RETURN c` instead of `RETURN c.card_name`).
    """
    if isinstance(value, Node):
        props = {
            k: v for k, v in value.items() if k != "embedding" and not k.endswith("_embedding")
        }
        props["_labels"] = list(value.labels)
        return props
    if isinstance(value, Relationship):
        return {"_type": value.type, **dict(value.items())}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [serialize_value(v) for v in value]
    return value


def record_to_dict(record: Record) -> dict[str, Any]:
    """Convert any Neo4j record to a flat dictionary.

    Handles any Cypher query result structure:
    - Single nodes: RETURN node, score
    - Multiple nodes: RETURN d, c, t, score
    - Scalars: RETURN count(*), name
    - Mixed: Any combination
    - Raw Node/Relationship/Path objects from LLM-generated Cypher

    For backward compatibility:
    - Variable named 'node' has properties extracted WITHOUT prefix (text, source_file)
    - Other node variables are prefixed (d_text, c_card_name) to avoid collisions

    Args:
        record: Neo4j record from query result

    Returns:
        Flattened dictionary with all properties
    """
    result = {}

    for key in record.keys():
        value = record[key]
        if value is None:
            continue

        # Handle Neo4j Node/Relationship objects - extract their properties
        if isinstance(value, (Node, Relationship)):
            for prop_key, prop_value in value.items():
                if prop_key == "embedding" or prop_key.endswith("_embedding"):
                    continue  # Skip embedding vectors

                serialized = serialize_value(prop_value)

                # Backward compat: 'node' variable doesn't get prefixed
                # Other variables (d, c, t, etc.) get prefixed to avoid collisions
                if key == "node":
                    result[prop_key] = serialized
                else:
                    result[f"{key}_{prop_key}"] = serialized
        elif isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, list):
            result[key] = serialize_value(value)
        else:
            # Scalar values (score, strings, ints, etc.)
            result[key] = value

    return result
