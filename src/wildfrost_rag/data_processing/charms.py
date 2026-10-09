"""Charm domain model: what a charm is in Wildfrost.

Parsing the Charms wiki page lives in data_processing/pages/charms/parser.py;
the Neo4j shape lives in repositories/charms.py.
"""

from pydantic import BaseModel, Field


class CharmInfo(BaseModel):
    """Represents a Charm from the game."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    is_cursed: bool
    url: str | None = None
    unlock: str | None = None
    challenge: str | None = None
    tribe_exclusive: str | None = None
