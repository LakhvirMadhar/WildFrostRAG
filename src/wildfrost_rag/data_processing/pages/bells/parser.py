"""Parses the Bells wiki page into Sun, Storm, and Modifier BellInfo objects."""

from bs4 import BeautifulSoup

from wildfrost_rag.core.logger import logger
from wildfrost_rag.data_processing.bells import BellCategory, BellInfo
from wildfrost_rag.data_processing.pages.shared import extract_page_url
from wildfrost_rag.data_processing.text_utils import clean_element_text

_clean_text = clean_element_text

# Table 0 = Sun Bells (4 cols: Image | Name | Description | Notes)
# Table 1 = Storm Bells (5 cols: Image | Name | Storm Strength | Description | Notes)
# Table 2 = Modifier Bells (4 cols: Image | Name | Description | Notes)
_TABLE_CATEGORY_MAP = {
    0: BellCategory.SUN,
    1: BellCategory.STORM,
    2: BellCategory.MODIFIER,
}


def parse_bells_page(html: str, base_url: str = "") -> list[BellInfo]:
    """Parse the Bells wiki page HTML to extract all bells.

    Args:
        html: Raw HTML content of the Bells page
        base_url: Base URL for constructing individual bell page URLs

    Returns:
        List of BellInfo objects
    """
    soup = BeautifulSoup(html, "html.parser")
    bells: list[BellInfo] = []

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
                url = extract_page_url(cells[1], base_url)
                strength_text = cells[2].get_text(strip=True)
                description = _clean_text(cells[3])
                notes = _clean_text(cells[4]) or None

                try:
                    storm_strength = int(strength_text)
                except ValueError:
                    storm_strength = None

                bells.append(
                    BellInfo(
                        name=name,
                        category=category,
                        description=description,
                        notes=notes,
                        storm_strength=storm_strength,
                        url=url,
                    )
                )
            else:
                if len(cells) < 4:
                    continue
                name = cells[1].get_text(strip=True)
                url = extract_page_url(cells[1], base_url)
                description = _clean_text(cells[2])
                notes = _clean_text(cells[3]) or None

                bells.append(
                    BellInfo(
                        name=name,
                        category=category,
                        description=description,
                        notes=notes,
                        url=url,
                    )
                )

    logger.info(f"Parsed {len(bells)} bells from Bells page")
    return bells
