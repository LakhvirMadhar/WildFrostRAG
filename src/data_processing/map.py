"""
Map parsing for WildFrostRAG.

Parses the Map wiki page to extract Zones, Map Events, and Fight structure.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from bs4 import BeautifulSoup

from utils.logger import logger


# Maps wiki section heading names to canonical zone names
SECTION_TO_ZONE_NAME = {
    "Snowdwell and Tundra": "Tundra",
    "Ice Caves": "Ice Caves",
    "Frostlands": "Frostlands",
}


@dataclass
class ZoneInfo:
    """Represents a map zone."""
    name: str
    zone_order: int
    description: str


@dataclass
class MapEventInfo:
    """Represents a map event (non-combat encounter between fights)."""
    name: str
    description: str
    notes: Optional[str] = None


@dataclass
class FightSlotInfo:
    """Represents a numbered fight slot in a zone with its possible encounters."""
    fight_number: int
    zone: str
    possible_fights: List[str] = field(default_factory=list)


def _parse_zones(soup: BeautifulSoup) -> List[ZoneInfo]:
    """Parse zone information from the Zones section h3 headings."""
    zones = []

    zones_heading = soup.find('span', class_='mw-headline', id='Zones')
    if not zones_heading:
        return zones

    h2 = zones_heading.find_parent('h2')

    # Find h3 headings between Zones h2 and the next h2
    zone_order = 1
    for sibling in h2.find_next_siblings():
        if sibling.name == 'h2':
            break
        if sibling.name == 'h3':
            headline = sibling.find('span', class_='mw-headline')
            if not headline:
                continue
            name = headline.get_text(strip=True)

            next_p = sibling.find_next_sibling('p')
            description = next_p.get_text(strip=True) if next_p else ""

            zone_name = SECTION_TO_ZONE_NAME.get(name, name)
            zones.append(ZoneInfo(
                name=zone_name,
                zone_order=zone_order,
                description=description,
            ))
            zone_order += 1

    return zones


def _parse_map_events(soup: BeautifulSoup) -> List[MapEventInfo]:
    """Parse map events from the Map Events table."""
    events = []

    events_heading = soup.find('span', class_='mw-headline', id='Map_Events')
    if not events_heading:
        return events

    h2 = events_heading.find_parent('h2')
    table = h2.find_next('table', class_='wikitable')
    if not table:
        return events

    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        if len(cells) < 2:
            continue

        # Columns: Image | Event Name | Description | Notes
        name = cells[1].get_text(strip=True)
        description = cells[2].get_text(separator=' ', strip=True) if len(cells) > 2 else ""
        notes = cells[3].get_text(separator=' ', strip=True) if len(cells) > 3 else None
        if not notes:
            notes = None

        events.append(MapEventInfo(
            name=name,
            description=description,
            notes=notes,
        ))

    return events


def _parse_fight_structure(soup: BeautifulSoup) -> List[FightSlotInfo]:
    """
    Parse fight slots from the Fights table.

    The table uses rowspan for zones and fight numbers, with alternating
    image rows (skipped) and name rows for each fight slot.
    """
    fight_slots = []

    fights_heading = soup.find('span', class_='mw-headline', id='Fights')
    if not fights_heading:
        return fight_slots

    h2 = fights_heading.find_parent('h2')
    table = h2.find_next('table', class_='wikitable')
    if not table:
        return fight_slots

    current_zone = None
    current_fight_num = None

    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        if not cells:
            continue

        cell_idx = 0

        # Check for zone cell (large rowspan, no images)
        first_cell = cells[0]
        rowspan = first_cell.get('rowspan')
        if rowspan and not first_cell.find('img'):
            if int(rowspan) > 2:
                current_zone = first_cell.get_text(strip=True)
                cell_idx = 1

        # Check for fight number cell (rowspan=2)
        if cell_idx < len(cells):
            cell = cells[cell_idx]
            if cell.get('rowspan'):
                fight_text = cell.get_text(strip=True)
                try:
                    current_fight_num = int(fight_text)
                    cell_idx += 1
                except ValueError:
                    pass

        # Image rows contain <img> tags - skip them
        remaining = cells[cell_idx:]
        if any(c.find('img') for c in remaining):
            continue

        # Name row - extract fight names from links
        if current_zone and current_fight_num:
            names = []
            for cell in remaining:
                for link in cell.find_all('a'):
                    name = link.get_text(strip=True)
                    if name:
                        names.append(name)

            if names:
                fight_slots.append(FightSlotInfo(
                    fight_number=current_fight_num,
                    zone=current_zone,
                    possible_fights=names,
                ))

    return fight_slots


def get_fight_page_mapping(html: str) -> dict[str, str]:
    """
    Extract fight display name → wiki page slug mapping from the Map HTML.

    Parses <a> tags in the Fights table name rows to get the href (page slug)
    for each fight's display name. Boss fights have different page names than
    display names (e.g., "Infernoko" → "Infernoko_Fight").

    Args:
        html: Raw HTML content of the Map page

    Returns:
        Dict mapping display name to page slug
        (e.g., {"Infernoko": "Infernoko_Fight", "The Bog Berries": "The_Bog_Berries"})
    """
    soup = BeautifulSoup(html, 'html.parser')

    fights_heading = soup.find('span', class_='mw-headline', id='Fights')
    if not fights_heading:
        return {}

    h2 = fights_heading.find_parent('h2')
    table = h2.find_next('table', class_='wikitable')
    if not table:
        return {}

    mapping = {}
    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        if not cells:
            continue

        # Skip image rows
        if any(c.find('img') for c in cells):
            continue

        # Name rows - extract href and text from links
        for cell in cells:
            for link in cell.find_all('a'):
                name = link.get_text(strip=True)
                href = link.get('href', '')
                if name and href:
                    page_slug = href.lstrip('/')
                    mapping[name] = page_slug

    logger.info(f"Extracted {len(mapping)} fight page mappings")
    return mapping


def parse_map_page(html: str) -> tuple[List[ZoneInfo], List[MapEventInfo], List[FightSlotInfo]]:
    """
    Parse the Map wiki page HTML.

    Args:
        html: Raw HTML content of the Map page

    Returns:
        Tuple of (zones, map_events, fight_slots)
    """
    soup = BeautifulSoup(html, 'html.parser')

    zones = _parse_zones(soup)
    logger.info(f"Parsed {len(zones)} zones")

    map_events = _parse_map_events(soup)
    logger.info(f"Parsed {len(map_events)} map events")

    fight_slots = _parse_fight_structure(soup)
    logger.info(f"Parsed {len(fight_slots)} fight slots")

    return zones, map_events, fight_slots
