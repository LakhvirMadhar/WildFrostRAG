"""Bell domain model: what a bell is in Wildfrost.

Parsing the Bells wiki page lives in data_processing/pages/bells/parser.py;
caching a bell's own wiki page lives in scraping/html_cache.py.
"""

from enum import Enum

from pydantic import BaseModel, Field


class BellCategory(Enum):
    """Categories of bells in the game."""

    SUN = "sun"
    STORM = "storm"
    MODIFIER = "modifier"


class BellInfo(BaseModel):
    """Represents a Bell from the game."""

    name: str = Field(min_length=1)
    category: BellCategory
    description: str = Field(min_length=1)
    notes: str | None = None
    storm_strength: int | None = Field(default=None, ge=0)
    url: str | None = None
