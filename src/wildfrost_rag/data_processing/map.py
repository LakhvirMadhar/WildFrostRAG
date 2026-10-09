"""Map domain models: zones, map events, and fight slots.

Parsing the Map wiki page lives in data_processing/pages/map/parser.py.
"""

from pydantic import BaseModel, Field


class ZoneInfo(BaseModel):
    """Represents a map zone."""

    name: str = Field(min_length=1)
    zone_order: int = Field(ge=1)
    # description can legitimately be "" - the parser falls back to "" when
    # a zone has no following <p> element, which is real, valid page shape.
    description: str


class MapEventInfo(BaseModel):
    """Represents a map event (non-combat encounter between fights)."""

    name: str = Field(min_length=1)
    # description can legitimately be "" - the parser falls back to "" when
    # a row has fewer than 3 cells, which is real, valid page shape.
    description: str
    notes: str | None = None


class FightSlotInfo(BaseModel):
    """Represents a numbered fight slot in a zone with its possible encounters."""

    fight_number: int = Field(ge=1)
    zone: str = Field(min_length=1)
    possible_fights: list[str] = Field(default_factory=list)
