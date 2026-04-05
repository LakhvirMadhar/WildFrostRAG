"""
Bells parsing for WildFrostRAG.

Parses the Bells wiki page to extract Sun Bells, Storm Bells, and Modifier Bells.
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup, Comment

from data_processing.text_utils import clean_element_text
from utils.logger import logger

_clean_text = clean_element_text


class BellCategory(Enum):
    """Categories of bells in the game."""
    SUN = "sun"
    STORM = "storm"
    MODIFIER = "modifier"


@dataclass
class BellInfo:
    """Represents a Bell from the game."""
    name: str
    category: BellCategory
    description: str
    notes: Optional[str] = None
    storm_strength: Optional[int] = None
    url: Optional[str] = None
    bell_html: Optional[str] = None

    def sanitized_name(self) -> str:
        """Get sanitized bell name safe for filenames."""
        return re.sub(r'[\\/:*?"<>|]', '', self.name)

    def save_path(self) -> str:
        """Generate the save path for this bell's HTML."""
        return f'data/structured_outputs/bells/{self.sanitized_name()}.html'

    def save_html(self) -> bool:
        """Save the bell's HTML to file."""
        if self.bell_html is None:
            logger.warning(f"No HTML content to save for {self.name}")
            return False

        try:
            save_path = Path(self.save_path())
            save_path.parent.mkdir(parents=True, exist_ok=True)

            soup = BeautifulSoup(self.bell_html, 'html.parser')
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                comment.extract()

            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(soup.prettify())
            return True

        except Exception as e:
            logger.error(f'Failed to save HTML for {self.name}: {e}')
            return False


# Table 0 = Sun Bells (4 cols: Image | Name | Description | Notes)
# Table 1 = Storm Bells (5 cols: Image | Name | Storm Strength | Description | Notes)
# Table 2 = Modifier Bells (4 cols: Image | Name | Description | Notes)
_TABLE_CATEGORY_MAP = {
    0: BellCategory.SUN,
    1: BellCategory.STORM,
    2: BellCategory.MODIFIER,
}


def _extract_url(name_cell, base_url: str) -> Optional[str]:
    """Extract bell URL from the name cell's <a href>. Skips red links (pages that don't exist)."""
    link = name_cell.find('a')
    if link and link.get('href') and 'new' not in (link.get('class') or []):
        return f"{base_url}{link['href']}"
    return None


def parse_bells_page(html: str, base_url: str = "") -> List[BellInfo]:
    """
    Parse the Bells wiki page HTML to extract all bells.

    Args:
        html: Raw HTML content of the Bells page
        base_url: Base URL for constructing individual bell page URLs

    Returns:
        List of BellInfo objects
    """
    soup = BeautifulSoup(html, "html.parser")
    bells: List[BellInfo] = []

    tables = soup.find_all("table", class_="wikitable")

    for table_idx, table in enumerate(tables):
        category = _TABLE_CATEGORY_MAP.get(table_idx)
        if category is None:
            break

        rows = table.find_all("tr")

        for row in rows[1:]:  # skip header
            cells = row.find_all("td")
            if not cells:
                continue

            if category == BellCategory.STORM:
                if len(cells) < 5:
                    continue
                name = cells[1].get_text(strip=True)
                url = _extract_url(cells[1], base_url)
                strength_text = cells[2].get_text(strip=True)
                description = _clean_text(cells[3])
                notes = _clean_text(cells[4]) or None

                try:
                    storm_strength = int(strength_text)
                except ValueError:
                    storm_strength = None

                bells.append(BellInfo(
                    name=name,
                    category=category,
                    description=description,
                    notes=notes,
                    storm_strength=storm_strength,
                    url=url,
                ))
            else:
                if len(cells) < 4:
                    continue
                name = cells[1].get_text(strip=True)
                url = _extract_url(cells[1], base_url)
                description = _clean_text(cells[2])
                notes = _clean_text(cells[3]) or None

                bells.append(BellInfo(
                    name=name,
                    category=category,
                    description=description,
                    notes=notes,
                    url=url,
                ))

    logger.info(f"Parsed {len(bells)} bells from Bells page")
    return bells
