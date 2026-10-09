"""Stat domain model: what a stat (primary stat, buff, debuff) is in Wildfrost.

Parsing the Stats wiki page lives in data_processing/pages/stats/parser.py.
"""

from enum import Enum

from pydantic import BaseModel, Field


class StatCategory(Enum):
    """Categories of stats in the game."""

    PRIMARY = "primary"
    BUFF = "buff"
    DEBUFF = "debuff"


class StatInfo(BaseModel):
    """Represents a Stat from the game."""

    name: str = Field(min_length=1)
    category: StatCategory
    description: str = Field(min_length=1)
    additional_info: str | None = None
    url: str | None = None
