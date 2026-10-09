"""Parses the Charms wiki page into regular and cursed CharmInfo objects."""

from bs4 import BeautifulSoup, Tag

from wildfrost_rag.core.logger import logger
from wildfrost_rag.data_processing.charms import CharmInfo
from wildfrost_rag.data_processing.pages.shared import extract_page_url


def _parse_regular_charms(table: Tag, base_url: str) -> list[CharmInfo]:
    """Parse the regular charms table (Image, Card Name, Description, Unlock, Challenge, Tribe-exclusive?)."""
    charms = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) < 6:
            continue

        name = cells[1].get_text(strip=True)
        url = extract_page_url(cells[1], base_url)
        description = cells[2].get_text(separator=" ", strip=True)
        unlock = cells[3].get_text(strip=True) or None
        challenge = cells[4].get_text(strip=True) or None
        tribe_raw = cells[5].get_text(strip=True)
        tribe_exclusive = tribe_raw if tribe_raw else None

        charms.append(
            CharmInfo(
                name=name,
                description=description,
                is_cursed=False,
                url=url,
                unlock=unlock,
                challenge=challenge,
                tribe_exclusive=tribe_exclusive,
            )
        )

    return charms


def _parse_cursed_charms(table: Tag, base_url: str) -> list[CharmInfo]:
    """Parse the cursed charms table (Image, Card Name, Description)."""
    charms = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) < 3:
            continue

        name = cells[1].get_text(strip=True)
        url = extract_page_url(cells[1], base_url)
        description = cells[2].get_text(separator=" ", strip=True)

        charms.append(
            CharmInfo(
                name=name,
                description=description,
                is_cursed=True,
                url=url,
            )
        )

    return charms


def parse_charms_page(html: str, base_url: str = "") -> list[CharmInfo]:
    """Parse the Charms wiki page HTML to extract all charms.

    Args:
        html: Raw HTML content of the Charms page
        base_url: Base URL for constructing individual charm page URLs

    Returns:
        List of CharmInfo objects (regular + cursed)
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", class_="wikitable")

    charms: list[CharmInfo] = []

    if len(tables) >= 1:
        regular = _parse_regular_charms(tables[0], base_url)
        charms.extend(regular)
        logger.info(f"Parsed {len(regular)} regular charms")

    if len(tables) >= 2:
        cursed = _parse_cursed_charms(tables[1], base_url)
        charms.extend(cursed)
        logger.info(f"Parsed {len(cursed)} cursed charms")

    logger.info(f"Total charms parsed: {len(charms)}")
    return charms
