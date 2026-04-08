import re
from dataclasses import dataclass
from enum import Enum
from typing import Any
from pathlib import Path
from urllib.parse import unquote
from bs4 import BeautifulSoup, Comment, Tag
import logging
from collections import defaultdict

from data_processing.phase_config import VARIANT_CARDS, PHASE_ORDER_OVERRIDES
from data_processing.tribes import TribeExclusivity

logger = logging.getLogger(__name__)


def get_base_name(card_name: str) -> str:
    """Extract base name by removing parenthetical suffixes.

    Used for detecting multi-phase cards where phases have names like:
    - 'Truffle', 'Truffle (medium)', 'Truffle (small)' → all return 'Truffle'
    - 'Infernoko' → returns 'Infernoko'
    - 'Naked Gnome' → returns 'Naked Gnome'
    """
    return re.sub(r"\s*\([^)]*\)\s*$", "", card_name).strip()


# Wiki schema uses "enemies" for regular enemies; we split it into non_boss_enemies.
# Lives at module level because dicts inside an Enum body become enum members.
_SCHEMA_REMAP = {
    "enemies": "non_boss_enemies",
}


class CardType(Enum):
    """What type is the card."""

    _value_: str
    parents: list[str]

    LEADER = ("leaders", list[str]())
    PETS = ("pets", ["companions"])  # pets are a subtype of companions
    COMPANIONS = ("companions", list[str]())
    SHADES = ("shades", list[str]())
    CLUNKERS = ("clunkers", list[str]())
    ITEMS = ("items", list[str]())
    ENEMIES = ("enemies", list[str]())  # abstract parent — no direct cards
    NON_BOSS_ENEMIES = ("non_boss_enemies", ["enemies"])  # regular enemies
    ENEMY_CLUNKERS = (
        "enemy_clunkers",
        ["non_boss_enemies", "clunkers"],
    )  # inherits from both
    MINIBOSSES = ("minibosses", ["enemies"])  # minibosses are a subtype of enemies
    BOSSES = ("bosses", ["enemies"])  # bosses are a subtype of enemies

    def __new__(cls, value: str, parents: list[str] | None = None) -> "CardType":
        """Create a CardType enum member with optional parent types."""
        obj = object.__new__(cls)
        obj._value_ = value
        obj.parents = parents or []
        return obj

    # Wiki schema uses "enemies" for regular enemies; we split it into non_boss_enemies.
    # Remap lives outside the enum body (module-level) to avoid becoming a member.
    @classmethod
    def from_schema_key(cls, key: str) -> "CardType":
        """Resolve a wiki schema key to a CardType, applying remaps."""
        return cls(_SCHEMA_REMAP.get(key, key))

    @property
    def has_parents(self) -> bool:
        """Check if this card type has parent types."""
        return len(self.parents) > 0


@dataclass
class CardInfo:
    """Parsed card data from the Wildfrost Wiki."""

    card_name: str
    card_type: CardType
    url: str
    card_html: str | None = None

    # Card Stats
    card_description: str | None = None
    health: int | None = None
    attack: int | None = None
    scrap: int | None = None  # Alternative to Heatlh
    counter: int | None = None
    other_stats: str | None = None
    abilities_normalized: str | None = None
    abilities_specific: str | None = None
    flavor_text: str | None = None

    # Stat ranges (for Leaders with variable stats like "5-9")
    health_min: int | None = None
    health_max: int | None = None
    attack_min: int | None = None
    attack_max: int | None = None
    counter_min: int | None = None
    counter_max: int | None = None

    # Tribe Exclusivity
    tribe_exclusivity: TribeExclusivity | None = None

    # Phase information (for multi-phase cards like Infernoko, Truffle)
    phase: int | None = None  # 1, 2, 3... or None for non-phased cards
    total_phases: int | None = None  # Total phases or None for non-phased cards
    base_name: str | None = (
        None  # Base name for phase matching (e.g., "Truffle" for "Truffle (medium)")
    )

    def sanitized_name(self) -> str:
        """Get sanitized card name safe for filenames."""
        return re.sub(r'[\\/:*?"<>|]', "", self.card_name)

    def save_path(self) -> str:
        """Generate the save path for this card's HTML."""
        return f"data/structured_outputs/{self.card_type.value}/{self.sanitized_name()}.html"

    def save_html(self) -> bool:
        """Save the card's HTML to file with proper directory creation and cleaning.

        Returns:
            bool: True is saved sucessfully, False otherwise
        """
        if self.card_html is None:
            logger.warning(f"No HTML content to save for {self.card_name}")
            return False

        try:
            # Create directory if it doesn't exist
            save_path = Path(self.save_path())
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Get HTML file
            soup = BeautifulSoup(self.card_html, "html.parser")

            # Remove comments in HTML
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                comment.extract()

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(soup.prettify())

        except Exception as e:
            logger.error(f"Failed to save HTML for {self.card_name}: {e}")
        return False

    def _extract_description(self, soup: BeautifulSoup) -> None:
        """Extract card description from meta tag."""
        description_tag = soup.find("meta", attrs={"name": "description"})
        if description_tag:
            content = description_tag.get("content", "")
            self.card_description = content if isinstance(content, str) else ""

    def _set_stat_value(self, attr_name: str, value: str) -> None:
        """Set a stat attribute with appropriate type conversion."""
        if not hasattr(self, attr_name):
            return

        if value.strip() == "":
            setattr(self, attr_name, None)
        elif value.isdigit():
            setattr(self, attr_name, int(value))
        else:
            setattr(self, attr_name, value)

    def _extract_stats_from_rows(self, rows: list[Tag]) -> None:
        """Extract stats from infobox table rows."""
        if len(rows) < 4:
            return

        # Stats are in rows 2 (headers) and 3 (values)
        stats_headers = [th.text.strip() for th in rows[2].find_all("th")]
        stats_values = [td.text.strip() for td in rows[3].find_all("td")]
        stats = dict(zip(stats_headers, stats_values, strict=False))

        for stat_name, value in stats.items():
            self._set_stat_value(stat_name.lower(), value)

    def _extract_other_stats(self, rows: list[Tag]) -> None:
        """Extract 'Other Stats' section from infobox rows."""
        for i, row in enumerate(rows):
            th = row.find("th")
            if not th or th.text.strip() != "Other Stats":
                continue

            if i + 1 >= len(rows):
                break

            td = rows[i + 1].find("td")
            if td:
                other_stats_text = td.get_text(strip=True)
                self.other_stats = other_stats_text if other_stats_text else None
            break

    def parse_html(self) -> bool:
        """Parse the HTML and populate the card stats fields.

        Returns:
            bool: True if parsing succeeded, False otherwise
        """
        if not self.card_html:
            logger.warning(f"No HTML content to parse for {self.card_name}")
            return False

        try:
            soup = BeautifulSoup(self.card_html, "html.parser")
            self._extract_description(soup)

            infobox = soup.find("table", {"id": "infobox"})
            if infobox:
                rows = infobox.find_all("tr")
                self._extract_stats_from_rows(rows)
                self._extract_other_stats(rows)

            return True

        except Exception as e:
            logger.error(f"Failed to parse HTML for {self.card_name}: {e}")
            return False

    def __str__(self) -> str:
        """String representation of the card based on to_dict."""
        card_data = self.to_dict()
        lines = [
            f"{key.replace('_', ' ').title()}: {value}"
            for key, value in card_data.items()
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Dictionary representation excluding None values and card_html, for neo4j consumption.

        Handles CardType(Enum), which neo4j can't do on it's own.
        """
        result = {}
        for k, v in self.__dict__.items():
            if v is None or k == "card_html":
                continue
            # Convert enum to its value
            if isinstance(v, Enum):
                result[k] = v.value
            else:
                result[k] = v

        # Add derived fields useful for Neo4j
        # Leaders all share the same "Leaders.html" document
        if self.card_type == CardType.LEADER:
            result["filename"] = "Leaders.html"
        else:
            result["filename"] = f"{self.sanitized_name()}.html"

        return result

    @staticmethod
    def _extract_card_name_from_infobox(infobox: Tag) -> str | None:
        """Extract card name from infobox table.

        The card name is in the first row's <th> element:
        <tr><th colspan="3">Card Name</th></tr>
        """
        first_row = infobox.find("tr")
        if not first_row:
            return None

        th = first_row.find("th")
        if not th:
            return None

        return th.get_text(strip=True)

    @staticmethod
    def _extract_phase_from_infobox(infobox: Tag) -> int | None:
        """Extract phase number from image filename inside infobox.

        Wiki convention:
        - Infernoko_(FrenzyBoss2).png → Phase 2
        - Truffle_(SummonBoss3).png → Phase 3
        - No number in parentheses → Phase 1

        Handles URL-encoded parentheses: %28 = (, %29 = )

        Returns None if no image found or can't determine phase.
        """
        img = infobox.find("img")
        if not img:
            return None

        # Try src or alt attribute
        raw_ref = img.get("src") or img.get("alt") or ""
        img_ref = raw_ref if isinstance(raw_ref, str) else ""

        # URL decode %28 and %29 to ( and )
        img_ref_decoded = unquote(img_ref)

        # Look for number at end of parentheses before .png: (SomeName2).png → 2
        match = re.search(r"\([^)]*?(\d+)\)\.png", img_ref_decoded, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # No number in parentheses - this is Phase 1
        if ".png" in img_ref_decoded.lower():
            return 1

        return None

    @staticmethod
    def _extract_stats_from_infobox(infobox: Tag) -> dict[str, int | str | None]:  # noqa: C901
        """Extract stats dictionary from an infobox table.

        Returns dict with keys like 'health', 'attack', 'counter', 'scrap', 'other_stats'.
        """
        rows = infobox.find_all("tr")
        stats: dict[str, int | str | None] = {}

        if len(rows) >= 4:
            # Stats are in rows 2 (headers) and 3 (values)
            stats_headers = [th.text.strip().lower() for th in rows[2].find_all("th")]
            stats_values = [td.text.strip() for td in rows[3].find_all("td")]

            for header, value in zip(stats_headers, stats_values, strict=False):
                if value.strip() == "":
                    stats[header] = None
                elif value.isdigit():
                    stats[header] = int(value)
                else:
                    stats[header] = value

        # Extract 'Other Stats' and 'Card Description' sections
        for i, row in enumerate(rows):
            th = row.find("th")
            if not th:
                continue

            header_text = th.get_text(strip=True)

            if header_text == "Other Stats":
                if i + 1 < len(rows):
                    td = rows[i + 1].find("td")
                    if td:
                        other_stats_text = td.get_text(separator=" ", strip=True)
                        other_stats_text = re.sub(r"\s+", " ", other_stats_text).strip()
                        stats["other_stats"] = (
                            other_stats_text if other_stats_text else None
                        )

            elif header_text == "Card Description":
                if i + 1 < len(rows):
                    td = rows[i + 1].find("td")
                    if td:
                        # Check for flavor text: italic text in gray/grey span
                        gray_span = td.find(
                            "span",
                            style=lambda s: s
                            and ("color:gray" in s or "color:grey" in s),
                        )
                        italic = td.find("i")

                        if gray_span and italic:
                            # This is flavor text (like "Does absolutely nothing...")
                            flavor_text = italic.get_text(separator=" ", strip=True)
                            flavor_text = re.sub(r"\s+", " ", flavor_text).strip()
                            stats["flavor_text"] = flavor_text if flavor_text else None
                        else:
                            # This is ability text
                            ability_text = td.get_text(separator=" ", strip=True)
                            ability_text = re.sub(r"\s+", " ", ability_text).strip()
                            stats["abilities_specific"] = (
                                ability_text if ability_text else None
                            )

        return stats

    @classmethod
    def _create_card(
        cls,
        card_data: dict[str, Any],
        card_type: CardType,
        url: str,
        html: str,
        description: str | None,
        phase: int | None = None,
        total_phases: int | None = None,
        base_name: str | None = None,
    ) -> "CardInfo":
        """Create a CardInfo from parsed infobox data."""
        return cls(
            card_name=card_data["card_name"],
            card_type=card_type,
            url=url,
            card_html=html,
            card_description=description,
            health=card_data["stats"].get("health"),
            attack=card_data["stats"].get("attack"),
            scrap=card_data["stats"].get("scrap"),
            counter=card_data["stats"].get("counter"),
            other_stats=card_data["stats"].get("other_stats"),
            abilities_specific=card_data["stats"].get("abilities_specific"),
            flavor_text=card_data["stats"].get("flavor_text"),
            phase=phase,
            total_phases=total_phases,
            base_name=base_name,
        )

    @staticmethod
    def _detect_variant_card_type(
        card_name: str, default_type: CardType
    ) -> CardType:
        """Detect card type from name suffix for variant cards."""
        if card_name.endswith("(Enemy)"):
            return CardType.NON_BOSS_ENEMIES
        elif card_name.endswith("(Companion)"):
            return CardType.COMPANIONS
        return default_type

    @staticmethod
    def _sort_phases(
        group: list[dict[str, Any]], base_name: str
    ) -> list[dict[str, Any]]:
        """Sort cards in a group by phase order."""
        if base_name in PHASE_ORDER_OVERRIDES:
            order_list = PHASE_ORDER_OVERRIDES[base_name]
            return sorted(
                group,
                key=lambda x: order_list.index(x["card_name"])
                if x["card_name"] in order_list
                else 999,
            )
        # Sort by phase number extracted from image (Phase 1 has no number)
        return sorted(
            group, key=lambda x: x["phase_from_image"] if x["phase_from_image"] else 1
        )

    @classmethod
    def _parse_infoboxes(cls, soup: BeautifulSoup, url: str) -> list[dict[str, Any]]:
        """Parse all infoboxes from HTML into a list of card data dicts."""
        infoboxes = soup.find_all("table", {"id": "infobox"})
        parsed_cards: list[dict[str, Any]] = []

        for infobox in infoboxes:
            card_name = cls._extract_card_name_from_infobox(infobox)
            if not card_name:
                logger.warning(f"Could not extract card name from infobox in {url}")
                continue

            parsed_cards.append(
                {
                    "card_name": card_name,
                    "base_name": get_base_name(card_name),
                    "stats": cls._extract_stats_from_infobox(infobox),
                    "phase_from_image": cls._extract_phase_from_infobox(infobox),
                }
            )

        return parsed_cards

    @classmethod
    def parse_html_multi_phase(
        cls,
        html: str,
        card_type: CardType,
        url: str,
    ) -> list["CardInfo"]:
        """Parse HTML that may contain multiple infoboxes (multi-phase cards).

        Handles:
        - Single infobox: Returns list with one CardInfo (phase=None)
        - Multiple infoboxes, same base name, same type: PHASES (linked via phase numbers)
        - Multiple infoboxes, same base name, different type: VARIANTS (separate cards, no phase)
        - Multiple infoboxes, different base names: MULTIPLE CARDS (separate cards, no phase)

        Args:
            html: The HTML content of the page
            card_type: The CardType for all cards on this page
            url: The URL of the page

        Returns:
            List of CardInfo objects, one per infobox
        """
        soup = BeautifulSoup(html, "html.parser")

        if not soup.find("table", {"id": "infobox"}):
            logger.warning(f"No infoboxes found in HTML for {url}")
            return []

        # Extract description from meta tag (shared across all cards on page)
        description_tag = soup.find("meta", attrs={"name": "description"})
        raw_desc = description_tag.get("content", "") if description_tag else None
        description: str | None = raw_desc if isinstance(raw_desc, str) else None

        # Parse all infoboxes
        parsed_cards = cls._parse_infoboxes(soup, url)
        if not parsed_cards:
            return []

        # Single infobox - no phases
        if len(parsed_cards) == 1:
            return [
                cls._create_card(parsed_cards[0], card_type, url, html, description)
            ]

        # Multiple infoboxes - group by base name
        base_name_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for card_data in parsed_cards:
            base_name_groups[card_data["base_name"]].append(card_data)

        result: list[CardInfo] = []

        for base_name, group in base_name_groups.items():
            if len(group) == 1:
                # Single card with this base name - no phases
                result.append(
                    cls._create_card(group[0], card_type, url, html, description)
                )

            elif base_name in VARIANT_CARDS:
                # Variant cards (e.g., Naked Gnome enemy/companion) - no phase linking
                for card_data in group:
                    variant_type = cls._detect_variant_card_type(
                        card_data["card_name"], card_type
                    )
                    result.append(
                        cls._create_card(
                            card_data, variant_type, url, html, description
                        )
                    )

            else:
                # Multiple cards with same base name - these are PHASES
                group_sorted = cls._sort_phases(group, base_name)
                total_phases = len(group_sorted)
                for phase_num, card_data in enumerate(group_sorted, start=1):
                    result.append(
                        cls._create_card(
                            card_data,
                            card_type,
                            url,
                            html,
                            description,
                            phase=phase_num,
                            total_phases=total_phases,
                            base_name=base_name,
                        )
                    )

        return result
