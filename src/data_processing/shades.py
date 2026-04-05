"""
Shades (summoning system) parsing for WildFrostRAG.

Parses the Shades wiki page to extract summoner → shade relationships
from the "Summon conditions" column.
"""

from dataclasses import dataclass
from typing import List
from bs4 import BeautifulSoup

from utils.logger import logger


@dataclass
class SummonInfo:
    """Represents a summoning relationship: summoner_card summons shade_card."""
    summoner_name: str
    shade_name: str


def parse_shades_page(html: str) -> List[SummonInfo]:
    """
    Parse the Shades wiki page to extract summoner → shade relationships.

    The page has a single wikitable with columns:
    Image | Card Name | Health | Attack | Counter | Other | Description | Summon conditions

    The "Summon conditions" column contains links to the summoning card(s).

    Args:
        html: Raw HTML content of the Shades page

    Returns:
        List of SummonInfo objects linking summoner cards to their shades
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Find the "Summoned Cards" section heading
    heading = soup.find('span', class_='mw-headline', id='Summoned_Cards')
    if not heading:
        logger.warning("Could not find 'Summoned Cards' heading on Shades page")
        return []

    parent_heading = heading.find_parent(['h2', 'h3'])
    if not parent_heading:
        logger.warning("Could not find parent heading for 'Summoned Cards'")
        return []

    table = parent_heading.find_next('table', class_='wikitable')
    if not table:
        logger.warning("Could not find wikitable after 'Summoned Cards' heading")
        return []

    summons: List[SummonInfo] = []

    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        if len(cells) < 8:
            continue

        # Column 1: Card Name (the shade)
        shade_name_cell = cells[1]
        shade_link = shade_name_cell.find('a')
        if not shade_link:
            continue
        shade_name = shade_link.get_text(strip=True)

        # Column 7: Summon conditions (contains links to summoner cards)
        summon_cell = cells[7]
        summoner_links = summon_cell.find_all('a')

        for link in summoner_links:
            # Skip keyword links (e.g., "Sacrificing", "Summoned")
            href = link.get('href', '')
            if '/Keywords' in href:
                continue

            # Use title attribute (matches actual card name) over display text
            # e.g., title="JunJun Mask" but display text is "Junjun Mask"
            summoner_name = link.get('title', '') or link.get_text(strip=True)
            if summoner_name:
                summons.append(SummonInfo(
                    summoner_name=summoner_name,
                    shade_name=shade_name,
                ))

    logger.info(f"Parsed {len(summons)} summoning relationships from Shades page")
    return summons
