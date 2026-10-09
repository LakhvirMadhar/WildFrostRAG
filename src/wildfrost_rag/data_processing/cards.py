"""Card domain models: CardType and CardInfo.

Parsing a card's wiki page lives in data_processing/pages/cards/parser.py.
CardInfo still carries its own HTML caching (save_path/save_html) and Neo4j
serialization (to_dict): those are left for the planned Dagster-native
ingestion work rather than moved onto the hand-rolled html_cache (see
docs/migration_plan/data_processing_split_plan.md, Task 9).
"""

import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Comment
from pydantic import BaseModel

from wildfrost_rag.data_processing.tribes import TribeExclusivity

logger = logging.getLogger(__name__)


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


class CardInfo(BaseModel):
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
            return True

        except Exception as e:
            logger.error(f"Failed to save HTML for {self.card_name}: {e}")
            return False

    def __str__(self) -> str:
        """String representation of the card based on to_dict."""
        card_data = self.to_dict()
        lines = [f"{key.replace('_', ' ').title()}: {value}" for key, value in card_data.items()]
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
