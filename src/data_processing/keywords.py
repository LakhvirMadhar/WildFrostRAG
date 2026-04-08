"""Keywords parsing for WildFrostRAG.

Parses the Keywords wiki page to extract keyword definitions across categories:
Targeting, Damaging/Attack, Restriction, Miscellaneous, Enemy-Specific, Special, Hidden.
"""

from dataclasses import dataclass, field
from enum import Enum

from bs4 import BeautifulSoup, Tag

from utils.logger import logger


class KeywordCategory(Enum):
    """Categories of keywords in the game."""

    TARGETING = "targeting"
    DAMAGING = "damaging"
    RESTRICTION = "restriction"
    MISCELLANEOUS = "miscellaneous"
    ENEMY_SPECIFIC = "enemy_specific"
    SPECIAL = "special"
    HIDDEN = "hidden"


@dataclass
class KeywordInfo:
    """Represents a Keyword from the game."""

    name: str
    category: KeywordCategory
    description_field: str | None = None
    description_items: str | None = None
    cards_with_keyword: list[str] = field(default_factory=list)


# Maps table index to category. The Keywords page has 7 wikitables in order.
_TABLE_CATEGORY_MAP = {
    0: KeywordCategory.TARGETING,
    1: KeywordCategory.DAMAGING,
    2: KeywordCategory.RESTRICTION,
    3: KeywordCategory.MISCELLANEOUS,
    4: KeywordCategory.ENEMY_SPECIFIC,
    5: KeywordCategory.SPECIAL,
    6: KeywordCategory.HIDDEN,
}

# Tables 0-3 have 5 columns: Name, Field, Items, Cards with Keyword, Sources
# Tables 4-5 have 3 columns: Name, Description, Enemies/Causes
# Table 6 has 2 columns: Name, Description
_FIVE_COL_TABLES = {0, 1, 2, 3}
_THREE_COL_TABLES = {4, 5}
_TWO_COL_TABLES = {6}


def _clean_keyword_name(raw_name: str) -> str:
    """Strip numeric suffixes like 'ExplodeX', 'FuryX', 'RecycleX' -> 'Explode', 'Fury', 'Recycle'."""
    if raw_name.endswith("X") and len(raw_name) > 1:
        return raw_name[:-1]
    return raw_name


def _parse_card_list(raw_text: str) -> list[str]:
    """Parse comma-separated card names from a table cell, filtering 'None'."""
    if not raw_text or raw_text.strip().lower() == "none":
        return []
    return [name.strip() for name in raw_text.split(",") if name.strip()]


def _parse_five_col_row(
    name: str, cells: list[Tag], category: KeywordCategory
) -> KeywordInfo | None:
    """Parse a 5-column keyword row (Name, Field, Items, Cards, Sources)."""
    if len(cells) < 4:
        return None
    return KeywordInfo(
        name=name,
        category=category,
        description_field=cells[1].get_text(strip=True),
        description_items=cells[2].get_text(strip=True),
        cards_with_keyword=_parse_card_list(cells[3].get_text(strip=True)),
    )


def _parse_three_col_row(
    name: str, cells: list[Tag], category: KeywordCategory
) -> KeywordInfo | None:
    """Parse a 3-column keyword row (Name, Description, Enemies/Causes)."""
    if len(cells) < 3:
        return None
    return KeywordInfo(
        name=name,
        category=category,
        description_field=cells[1].get_text(strip=True),
        cards_with_keyword=_parse_card_list(cells[2].get_text(strip=True)),
    )


def _parse_two_col_row(
    name: str, cells: list[Tag], category: KeywordCategory
) -> KeywordInfo | None:
    """Parse a 2-column keyword row (Name, Description)."""
    if len(cells) < 2:
        return None
    return KeywordInfo(
        name=name,
        category=category,
        description_field=cells[1].get_text(strip=True),
    )


def parse_keywords_page(html: str) -> list[KeywordInfo]:
    """Parse the Keywords wiki page HTML to extract all keywords.

    Args:
        html: Raw HTML content of the Keywords page

    Returns:
        List of KeywordInfo objects
    """
    soup = BeautifulSoup(html, "html.parser")
    keywords: list[KeywordInfo] = []

    tables = soup.find_all("table", class_="wikitable")

    for table_idx, table in enumerate(tables):
        category = _TABLE_CATEGORY_MAP.get(table_idx)
        if category is None:
            break

        # Select parser based on table column count
        if table_idx in _FIVE_COL_TABLES:
            row_parser = _parse_five_col_row
        elif table_idx in _THREE_COL_TABLES:
            row_parser = _parse_three_col_row
        elif table_idx in _TWO_COL_TABLES:
            row_parser = _parse_two_col_row
        else:
            continue

        for row in table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if not cells:
                continue
            name = _clean_keyword_name(cells[0].get_text(strip=True))
            keyword = row_parser(name, cells, category)
            if keyword:
                keywords.append(keyword)

    logger.info(f"Parsed {len(keywords)} keywords from Keywords page")
    return keywords
