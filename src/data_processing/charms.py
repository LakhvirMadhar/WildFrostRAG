"""
Charms parsing for WildFrostRAG.

Parses the Charms wiki page to extract regular and cursed charms.
"""

from dataclasses import dataclass
from typing import List, Optional
from bs4 import BeautifulSoup

from src.utils.logger import logger


@dataclass
class CharmInfo:
    """Represents a Charm from the game."""
    name: str
    description: str
    is_cursed: bool
    unlock: Optional[str] = None
    challenge: Optional[str] = None
    tribe_exclusive: Optional[str] = None


def _parse_regular_charms(table) -> List[CharmInfo]:
    """Parse the regular charms table (Image, Card Name, Description, Unlock, Challenge, Tribe-exclusive?)."""
    charms = []
    for row in table.find_all('tr')[1:]:
        cells = row.find_all(['td', 'th'])
        if len(cells) < 6:
            continue

        name = cells[1].get_text(strip=True)
        description = cells[2].get_text(separator=' ', strip=True)
        unlock = cells[3].get_text(strip=True) or None
        challenge = cells[4].get_text(strip=True) or None
        tribe_raw = cells[5].get_text(strip=True)
        tribe_exclusive = tribe_raw if tribe_raw else None

        charms.append(CharmInfo(
            name=name,
            description=description,
            is_cursed=False,
            unlock=unlock,
            challenge=challenge,
            tribe_exclusive=tribe_exclusive,
        ))

    return charms


def _parse_cursed_charms(table) -> List[CharmInfo]:
    """Parse the cursed charms table (Image, Card Name, Description)."""
    charms = []
    for row in table.find_all('tr')[1:]:
        cells = row.find_all(['td', 'th'])
        if len(cells) < 3:
            continue

        name = cells[1].get_text(strip=True)
        description = cells[2].get_text(separator=' ', strip=True)

        charms.append(CharmInfo(
            name=name,
            description=description,
            is_cursed=True,
        ))

    return charms


def parse_charms_page(html: str) -> List[CharmInfo]:
    """
    Parse the Charms wiki page HTML to extract all charms.

    Args:
        html: Raw HTML content of the Charms page

    Returns:
        List of CharmInfo objects (regular + cursed)
    """
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table', class_='wikitable')

    charms: List[CharmInfo] = []

    if len(tables) >= 1:
        regular = _parse_regular_charms(tables[0])
        charms.extend(regular)
        logger.info(f"Parsed {len(regular)} regular charms")

    if len(tables) >= 2:
        cursed = _parse_cursed_charms(tables[1])
        charms.extend(cursed)
        logger.info(f"Parsed {len(cursed)} cursed charms")

    logger.info(f"Total charms parsed: {len(charms)}")
    return charms
