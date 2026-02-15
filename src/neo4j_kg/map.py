from typing import List

from src.data_processing.map import ZoneInfo, MapEventInfo, FightSlotInfo
from src.utils.logger import logger


def create_map_graph(tx, zones: List[ZoneInfo], map_events: List[MapEventInfo], fight_slots: List[FightSlotInfo], fight_page_mapping: dict[str, str] = None):
    """
    Create the full map graph: Map node, Zones, MapEvents, Fights, and all relationships.

    Graph structure:
        (Map)-[:HAS_ZONE]->(Zone)-[:HAS_FIGHT {fight_number}]->(Fight)
        (Map)-[:HAS_MAP_EVENT]->(MapEvent)

    Args:
        fight_page_mapping: Display name -> wiki page slug (e.g., {"Infernoko": "Infernoko_Fight"})
    """
    # Create Map node
    tx.run("MERGE (m:Map {name: 'Map'})")

    # Create Zone nodes + Map->Zone
    zone_count = _create_zones(tx, zones)

    # Create MapEvent nodes + Map->MapEvent
    event_count = _create_map_events(tx, map_events)

    # Create Fight nodes + Zone->Fight
    fight_count = _create_fights_from_slots(tx, fight_slots, fight_page_mapping or {})

    logger.info(f"Map graph: 1 Map, {zone_count} Zones, {event_count} MapEvents, {fight_count} Fights")
    return {'zones': zone_count, 'map_events': event_count, 'fights': fight_count}


def _create_zones(tx, zones: List[ZoneInfo]):
    """Create Zone nodes and Map->Zone relationships."""
    zone_data = [
        {
            'name': zone.name,
            'zone_order': zone.zone_order,
            'description': zone.description,
        }
        for zone in zones
    ]

    query = """
    UNWIND $zones AS zone
    MERGE (z:Zone {name: zone.name})
    SET z.zone_order = zone.zone_order,
        z.description = zone.description
    WITH z
    MATCH (m:Map {name: 'Map'})
    MERGE (m)-[:HAS_ZONE]->(z)
    RETURN count(z) AS created
    """
    result = tx.run(query, zones=zone_data)
    return result.single()["created"]


def _create_map_events(tx, map_events: List[MapEventInfo]):
    """Create MapEvent nodes and Map->MapEvent relationships."""
    event_data = [
        {
            'name': event.name,
            'description': event.description,
            'notes': event.notes,
        }
        for event in map_events
    ]

    query = """
    UNWIND $events AS event
    MERGE (e:MapEvent {name: event.name})
    SET e.description = event.description,
        e.notes = event.notes
    WITH e
    MATCH (m:Map {name: 'Map'})
    MERGE (m)-[:HAS_MAP_EVENT]->(e)
    RETURN count(e) AS created
    """
    result = tx.run(query, events=event_data)
    return result.single()["created"]


def _create_fights_from_slots(tx, fight_slots: List[FightSlotInfo], fight_page_mapping: dict[str, str]):
    """
    Create Fight nodes and link them to Zones via HAS_FIGHT.

    The fight_number goes on the relationship so we know which fights
    are alternatives at the same position in the run.

    Args:
        fight_page_mapping: Display name -> wiki page slug for document linking
    """
    fight_data = []
    for slot in fight_slots:
        for fight_name in slot.possible_fights:
            page_name = fight_page_mapping.get(fight_name, fight_name.replace(' ', '_'))
            fight_data.append({
                'name': fight_name,
                'page_name': page_name,
            })

    fight_query = """
    UNWIND $fights AS fight
    MERGE (f:Fight {name: fight.name})
    SET f.page_name = fight.page_name
    RETURN count(f) AS created
    """
    result = tx.run(fight_query, fights=fight_data)
    fights_created = result.single()["created"]

    zone_fight_pairs = []
    for slot in fight_slots:
        for fight_name in slot.possible_fights:
            zone_fight_pairs.append({
                'zone_name': slot.zone,
                'fight_name': fight_name,
                'fight_number': slot.fight_number,
            })

    rel_query = """
    UNWIND $pairs AS pair
    MATCH (z:Zone {name: pair.zone_name})
    MATCH (f:Fight {name: pair.fight_name})
    MERGE (z)-[:HAS_FIGHT {fight_number: pair.fight_number}]->(f)
    RETURN count(*) AS created
    """
    result = tx.run(rel_query, pairs=zone_fight_pairs)

    return fights_created
