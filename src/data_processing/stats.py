"""Stats parsing for WildFrostRAG.

Parses the Stats wiki page to extract Primary Stats, Buffs, and Debuffs.
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from bs4 import BeautifulSoup, Comment, Tag

from utils.logger import logger


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
    additional_info: str | None = None
    url: str | None = None
    stat_html: str | None = None

    def sanitized_name(self) -> str:
        """Get sanitized stat name safe for filenames."""
        return re.sub(r'[\\/:*?"<>|]', "", self.name)

    def save_path(self) -> str:
        """Generate the save path for this stat's HTML."""
        return f"data/structured_outputs/stats/{self.sanitized_name()}.html"

    def save_html(self) -> bool:
        """Save the stat's HTML to file."""
        if self.stat_html is None:
            logger.warning(f"No HTML content to save for {self.name}")
            return False

        try:
            save_path = Path(self.save_path())
            save_path.parent.mkdir(parents=True, exist_ok=True)

            soup = BeautifulSoup(self.stat_html, "html.parser")
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                comment.extract()

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(soup.prettify())
            return True

        except Exception as e:
            logger.error(f"Failed to save HTML for {self.name}: {e}")
            return False


def _extract_url(name_cell: Tag, base_url: str) -> str | None:
    """Extract stat URL from the name cell's <a href>."""
    link = name_cell.find("a")
    if link and link.get("href"):
        return f"{base_url}{link['href']}"
    return None


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

    # Find all wikitables
    tables = soup.find_all("table", class_="wikitable")

    # The page has 3 tables in order: Primary Stats, Buffs, Debuffs
    category_map = {
        0: StatCategory.PRIMARY,
        1: StatCategory.BUFF,
        2: StatCategory.DEBUFF,
    }

    for table_idx, table in enumerate(tables):
        if table_idx > 2:
            break  # Only process first 3 tables

        category = category_map.get(table_idx)
        if category is None:
            continue  # Skip unknown tables
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
            url = _extract_url(name_cell, base_url)

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
                additional_info=additional_info,
                url=url,
            )
            stats.append(stat)

    logger.info(f"Parsed {len(stats)} stats from Stats page")
    return stats


def parse_stats_from_file(filepath: Path) -> list[StatInfo]:
    """Parse stats from a saved HTML file.

    Args:
        filepath: Path to the Stats.html file

    Returns:
        List of StatInfo objects
    """
    with open(filepath, encoding="utf-8") as f:
        html = f.read()
    return parse_stats_page(html)
