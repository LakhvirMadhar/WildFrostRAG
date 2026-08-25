"""Map parsing for WildFrostRAG.

Parses the Map wiki page to extract Zones, Map Events, and Fight structure.
"""

from dataclasses import dataclass, field
from bs4 import BeautifulSoup, Tag
from bs4.element import ResultSet

from wildfrost_rag.utils.logger import logger


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
    notes: str | None = None


@dataclass
class FightSlotInfo:
    """Represents a numbered fight slot in a zone with its possible encounters."""

    fight_number: int
    zone: str
    possible_fights: list[str] = field(default_factory=list)


def _parse_zones(soup: BeautifulSoup) -> list[ZoneInfo]:
    """Parse zone information from the Zones section h3 headings."""
    zones: list[ZoneInfo] = []

    zones_heading = soup.find("span", class_="mw-headline", id="Zones")
    if not zones_heading:
        return zones

    h2 = zones_heading.find_parent("h2")
    if h2 is None:
        return zones

    # Find h3 headings between Zones h2 and the next h2
    zone_order = 1
    for sibling in h2.find_next_siblings():
        if sibling.name == "h2":
            break
        if sibling.name == "h3":
            headline = sibling.find("span", class_="mw-headline")
            if not headline:
                continue
            name = headline.get_text(strip=True)

            next_p = sibling.find_next_sibling("p")
            description = next_p.get_text(strip=True) if next_p else ""

            zone_name = SECTION_TO_ZONE_NAME.get(name, name)
            zones.append(
                ZoneInfo(
                    name=zone_name,
                    zone_order=zone_order,
                    description=description,
                )
            )
            zone_order += 1

    return zones


def _parse_map_events(soup: BeautifulSoup) -> list[MapEventInfo]:
    """Parse map events from the Map Events table."""
    events: list[MapEventInfo] = []

    events_heading = soup.find("span", class_="mw-headline", id="Map_Events")
    if not events_heading:
        return events

    h2 = events_heading.find_parent("h2")
    if h2 is None:
        return events
    table = h2.find_next("table", class_="wikitable")
    if not table:
        return events

    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        # Columns: Image | Event Name | Description | Notes
        name = cells[1].get_text(strip=True)
        description = cells[2].get_text(separator=" ", strip=True) if len(cells) > 2 else ""
        notes = cells[3].get_text(separator=" ", strip=True) if len(cells) > 3 else None
        if not notes:
            notes = None

        events.append(
            MapEventInfo(
                name=name,
                description=description,
                notes=notes,
            )
        )

    return events


def _find_fights_table(soup: BeautifulSoup) -> Tag | None:
    """Find the Fights wikitable from the Map page."""
    fights_heading = soup.find("span", class_="mw-headline", id="Fights")
    if not fights_heading:
        return None
    h2 = fights_heading.find_parent("h2")
    if h2 is None:
        return None
    return h2.find_next("table", class_="wikitable")


def _extract_zone_from_row(cells: ResultSet[Tag], cell_idx: int) -> tuple[str | None, int]:
    """Check if the first cell is a zone header (large rowspan, no images).

    Returns:
        Tuple of (zone_name or None, updated cell_idx).
    """
    first_cell = cells[0]
    rowspan = first_cell.get("rowspan")
    if rowspan and not first_cell.find("img"):
        if isinstance(rowspan, str) and int(rowspan) > 2:
            return first_cell.get_text(strip=True), cell_idx + 1
    return None, cell_idx


def _extract_fight_number(cells: ResultSet[Tag], cell_idx: int) -> tuple[int | None, int]:
    """Check if the current cell is a fight number (has rowspan).

    Returns:
        Tuple of (fight_number or None, updated cell_idx).
    """
    if cell_idx >= len(cells):
        return None, cell_idx
    cell = cells[cell_idx]
    if cell.get("rowspan"):
        fight_text = cell.get_text(strip=True)
        try:
            return int(fight_text), cell_idx + 1
        except ValueError:
            pass
    return None, cell_idx


def _extract_fight_names(cells: list[Tag] | ResultSet[Tag]) -> list[str]:
    """Extract fight names from link elements in the remaining cells."""
    names = []
    for cell in cells:
        for link in cell.find_all("a"):
            name = link.get_text(strip=True)
            if name:
                names.append(name)
    return names


def _parse_fight_structure(soup: BeautifulSoup) -> list[FightSlotInfo]:
    """Parse fight slots from the Fights table.

    The table uses rowspan for zones and fight numbers, with alternating
    image rows (skipped) and name rows for each fight slot.
    """
    table = _find_fights_table(soup)
    if not table:
        return []

    fight_slots: list[FightSlotInfo] = []
    current_zone: str | None = None
    current_fight_num: int | None = None

    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if not cells:
            continue

        cell_idx = 0

        zone, cell_idx = _extract_zone_from_row(cells, cell_idx)
        if zone:
            current_zone = zone

        fight_num, cell_idx = _extract_fight_number(cells, cell_idx)
        if fight_num:
            current_fight_num = fight_num

        # Skip image rows
        remaining = cells[cell_idx:]
        if any(c.find("img") for c in remaining):
            continue

        # Name row — extract fight names
        if current_zone and current_fight_num:
            names = _extract_fight_names(remaining)
            if names:
                fight_slots.append(
                    FightSlotInfo(
                        fight_number=current_fight_num,
                        zone=current_zone,
                        possible_fights=names,
                    )
                )

    return fight_slots


def get_fight_page_mapping(html: str) -> dict[str, str]:
    """Extract fight display name → wiki page slug mapping from the Map HTML.

    Parses <a> tags in the Fights table name rows to get the href (page slug)
    for each fight's display name. Boss fights have different page names than
    display names (e.g., "Infernoko" → "Infernoko_Fight").

    Args:
        html: Raw HTML content of the Map page

    Returns:
        Dict mapping display name to page slug
        (e.g., {"Infernoko": "Infernoko_Fight", "The Bog Berries": "The_Bog_Berries"})
    """
    soup = BeautifulSoup(html, "html.parser")

    fights_heading = soup.find("span", class_="mw-headline", id="Fights")
    if not fights_heading:
        return {}

    h2 = fights_heading.find_parent("h2")
    if h2 is None:
        return {}
    table = h2.find_next("table", class_="wikitable")
    if not table:
        return {}

    mapping = {}
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if not cells:
            continue

        # Skip image rows
        if any(c.find("img") for c in cells):
            continue

        # Name rows - extract href and text from links
        for cell in cells:
            for link in cell.find_all("a"):
                name = link.get_text(strip=True)
                href = link.get("href", "")
                if name and href:
                    page_slug = href.lstrip("/") if isinstance(href, str) else ""
                    mapping[name] = page_slug

    logger.info(f"Extracted {len(mapping)} fight page mappings")
    return mapping


def parse_map_page(
    html: str,
) -> tuple[list[ZoneInfo], list[MapEventInfo], list[FightSlotInfo]]:
    """Parse the Map wiki page HTML.

    Args:
        html: Raw HTML content of the Map page

    Returns:
        Tuple of (zones, map_events, fight_slots)
    """
    soup = BeautifulSoup(html, "html.parser")

    zones = _parse_zones(soup)
    logger.info(f"Parsed {len(zones)} zones")

    map_events = _parse_map_events(soup)
    logger.info(f"Parsed {len(map_events)} map events")

    fight_slots = _parse_fight_structure(soup)
    logger.info(f"Parsed {len(fight_slots)} fight slots")

    return zones, map_events, fight_slots
