"""Charms parsing for WildFrostRAG.

Parses the Charms wiki page to extract regular and cursed charms.
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Comment, Tag

from utils.logger import logger


@dataclass
class CharmInfo:
    """Represents a Charm from the game."""

    name: str
    description: str
    is_cursed: bool
    url: str | None = None
    charm_html: str | None = None
    unlock: str | None = None
    challenge: str | None = None
    tribe_exclusive: str | None = None

    def sanitized_name(self) -> str:
        """Get sanitized charm name safe for filenames."""
        return re.sub(r'[\\/:*?"<>|]', "", self.name)

    def save_path(self) -> str:
        """Generate the save path for this charm's HTML."""
        return f"data/structured_outputs/charms/{self.sanitized_name()}.html"

    def save_html(self) -> bool:
        """Save the charm's HTML to file (same pattern as CardInfo.save_html)."""
        if self.charm_html is None:
            logger.warning(f"No HTML content to save for {self.name}")
            return False

        try:
            save_path = Path(self.save_path())
            save_path.parent.mkdir(parents=True, exist_ok=True)

            soup = BeautifulSoup(self.charm_html, "html.parser")
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                comment.extract()

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(soup.prettify())
            return True

        except Exception as e:
            logger.error(f"Failed to save HTML for {self.name}: {e}")
            return False

    def to_dict(self) -> dict[str, Any]:
        """Dict for Neo4j consumption. Excludes None/HTML, adds filename."""
        result = {}
        for k, v in self.__dict__.items():
            if v is None or k == "charm_html":
                continue
            if isinstance(v, Enum):
                result[k] = v.value
            else:
                result[k] = v
        result["filename"] = f"{self.sanitized_name()}.html"
        return result


def _extract_url(name_cell: Tag, base_url: str) -> str | None:
    """Extract charm URL from the name cell's <a href>."""
    link = name_cell.find("a")
    if link and link.get("href"):
        return f"{base_url}{link['href']}"
    return None


def _parse_regular_charms(table: Tag, base_url: str) -> list[CharmInfo]:
    """Parse the regular charms table (Image, Card Name, Description, Unlock, Challenge, Tribe-exclusive?)."""
    charms = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) < 6:
            continue

        name = cells[1].get_text(strip=True)
        url = _extract_url(cells[1], base_url)
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
        url = _extract_url(cells[1], base_url)
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
