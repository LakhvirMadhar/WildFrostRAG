"""Summon domain model: which card summons which shade.

Parsing the Shades wiki page lives in data_processing/pages/shades/parser.py.
"""

from pydantic import BaseModel, Field


class SummonInfo(BaseModel):
    """Represents a summoning relationship: summoner_card summons shade_card."""

    summoner_name: str = Field(min_length=1)
    shade_name: str = Field(min_length=1)
