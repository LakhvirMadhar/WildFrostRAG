"""
Stats parsing for WildFrostRAG.

Parses the Stats wiki page to extract Primary Stats, Buffs, and Debuffs.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional
from bs4 import BeautifulSoup

from src.utils.logger import logger


class StatCategory(Enum):
    """Categories of stats in the game."""
    PRIMARY = "primary"
    BUFF = "buff"
    DEBUFF = "debuff"


@dataclass
class StatInfo:
    """Represents a Stat from the game."""
    name: str
    category: StatCategory
    description: str
    additional_info: Optional[str] = None


def parse_stats_page(html: str) -> List[StatInfo]:
    """
    Parse the Stats wiki page HTML to extract all stats.

    Args:
        html: Raw HTML content of the Stats page

    Returns:
        List of StatInfo objects
    """
    soup = BeautifulSoup(html, 'html.parser')
    stats: List[StatInfo] = []

    # Find all wikitables
    tables = soup.find_all('table', class_='wikitable')

    # The page has 3 tables in order: Primary Stats, Buffs, Debuffs
    category_map = {
        0: StatCategory.PRIMARY,
        1: StatCategory.BUFF,
        2: StatCategory.DEBUFF
    }

    for table_idx, table in enumerate(tables):
        if table_idx > 2:
            break  # Only process first 3 tables

        category = category_map.get(table_idx)
        if category is None:
            continue  # Skip unknown tables
        rows = table.find_all('tr')

        # Skip header row
        for row in rows[1:]:
            cells = row.find_all('td')
            if len(cells) < 3:
                continue

            # Column 0: Icon (skip)
            # Column 1: Name (with link)
            # Column 2: Description
            # Column 3: Additional (for buffs/debuffs, optional)

            name_cell = cells[1]
            name_link = name_cell.find('a')
            name = name_link.get_text(strip=True) if name_link else name_cell.get_text(strip=True)

            description = cells[2].get_text(strip=True)

            additional_info = None
            if len(cells) > 3:
                additional_info = cells[3].get_text(strip=True)
                if not additional_info:
                    additional_info = None

            stat = StatInfo(
                name=name,
                category=category,
                description=description,
                additional_info=additional_info
            )
            stats.append(stat)

    logger.info(f"Parsed {len(stats)} stats from Stats page")
    return stats


def parse_stats_from_file(filepath: Path) -> List[StatInfo]:
    """
    Parse stats from a saved HTML file.

    Args:
        filepath: Path to the Stats.html file

    Returns:
        List of StatInfo objects
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        html = f.read()
    return parse_stats_page(html)
