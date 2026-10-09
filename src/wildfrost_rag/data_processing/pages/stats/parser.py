"""Parses the Stats wiki page into primary-stat, buff, and debuff StatInfo objects."""

from bs4 import BeautifulSoup

from wildfrost_rag.core.logger import logger
from wildfrost_rag.data_processing.pages.shared import extract_page_url
from wildfrost_rag.data_processing.stats import StatCategory, StatInfo

# The page has 3 tables in order: Primary Stats, Buffs, Debuffs
_TABLE_CATEGORY_MAP = {
    0: StatCategory.PRIMARY,
    1: StatCategory.BUFF,
    2: StatCategory.DEBUFF,
}


def parse_stats_page(html: str, base_url: str = "") -> list[StatInfo]:
    """Parse the Stats wiki page HTML to extract all stats.

    Args:
        html: Raw HTML content of the Stats page
        base_url: Base URL for constructing individual stat page URLs

    Returns:
        List of StatInfo objects
    """
    soup = BeautifulSoup(html, "html.parser")
    stats: list[StatInfo] = []

    tables = soup.find_all("table", class_="wikitable")

    for table_idx, table in enumerate(tables):
        category = _TABLE_CATEGORY_MAP.get(table_idx)
        if category is None:
            break  # Only the first 3 tables are stat tables
        rows = table.find_all("tr")

        # Skip header row
        for row in rows[1:]:
            cells = row.find_all("td")
            if len(cells) < 3:
                continue

            # Column 0: Icon (skip)
            # Column 1: Name (with link)
            # Column 2: Description
            # Column 3: Additional (for buffs/debuffs, optional)

            name_cell = cells[1]
            name_link = name_cell.find("a")
            name = name_link.get_text(strip=True) if name_link else name_cell.get_text(strip=True)
            url = extract_page_url(name_cell, base_url)

            description = cells[2].get_text(strip=True)

            additional_info = None
            if len(cells) > 3:
                additional_info = cells[3].get_text(strip=True)
                if not additional_info:
                    additional_info = None

            stats.append(
                StatInfo(
                    name=name,
                    category=category,
                    description=description,
                    additional_info=additional_info,
                    url=url,
                )
            )

    logger.info(f"Parsed {len(stats)} stats from Stats page")
    return stats
