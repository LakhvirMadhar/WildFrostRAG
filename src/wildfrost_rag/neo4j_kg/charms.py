import neo4j

from wildfrost_rag.data_processing.charms import CharmInfo
from wildfrost_rag.data_processing.tribes import TribeExclusivity
from wildfrost_rag.neo4j_kg.query_utils import single_value


def create_charms_from_parsed(tx: neo4j.ManagedTransaction, charms: list[CharmInfo]) -> int:
    """Create Charm nodes from parsed CharmInfo objects."""
    charm_data = [charm.to_dict() for charm in charms]

    query = """
    UNWIND $charms AS charm
    MERGE (c:Charm {name: charm.name})
    SET c.description = charm.description,
        c.is_cursed = charm.is_cursed,
        c.unlock = charm.unlock,
        c.challenge = charm.challenge,
        c.filename = charm.filename,
        c.url = charm.url
    RETURN count(c) AS charmsCreated
    """
    result = tx.run(query, charms=charm_data)
    return single_value(result, "charmsCreated")


def create_charm_tribe_relationships(tx: neo4j.ManagedTransaction, charms: list[CharmInfo]) -> int:
    """Create EXCLUSIVE_TO relationships between Charms and Tribes.

    "All" expands to all 3 tribes (same pattern as cards).
    Handles multi-tribe like "Snowdwellers,Clunkmasters".
    Cursed charms have no tribe (no relationship created).
    """
    charm_tribe_pairs = []
    for charm in charms:
        if not charm.tribe_exclusive:
            continue

        try:
            exclusivity = TribeExclusivity(charm.tribe_exclusive)
            tribes = exclusivity.get_tribes()
        except ValueError:
            # Handle multi-tribe like "Snowdwellers,Clunkmasters"
            tribes = [t.strip() for t in charm.tribe_exclusive.split(",")]

        for tribe_name in tribes:
            charm_tribe_pairs.append(
                {
                    "charm_name": charm.name,
                    "tribe_name": tribe_name,
                }
            )

    if not charm_tribe_pairs:
        return 0

    query = """
    UNWIND $pairs AS pair
    MATCH (c:Charm {name: pair.charm_name})
    MATCH (t:Tribe {name: pair.tribe_name})
    MERGE (c)-[:BELONGS_TO_TRIBE]->(t)
    RETURN count(*) AS created
    """
    result = tx.run(query, pairs=charm_tribe_pairs)
    return single_value(result, "created")
