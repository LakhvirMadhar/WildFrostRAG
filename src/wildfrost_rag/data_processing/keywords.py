"""Keyword domain model: what a keyword is in Wildfrost, by category.

Parsing the Keywords wiki page lives in data_processing/pages/keywords/parser.py.
"""

from enum import Enum

from pydantic import BaseModel, Field


class KeywordCategory(Enum):
    """Categories of keywords in the game."""

    TARGETING = "targeting"
    DAMAGING = "damaging"
    RESTRICTION = "restriction"
    MISCELLANEOUS = "miscellaneous"
    ENEMY_SPECIFIC = "enemy_specific"
    SPECIAL = "special"
    HIDDEN = "hidden"


class KeywordInfo(BaseModel):
    """Represents a Keyword from the game."""

    name: str = Field(min_length=1)
    category: KeywordCategory
    description_field: str | None = None
    description_items: str | None = None
    cards_with_keyword: list[str] = Field(default_factory=list)
