"""
Crown data definitions.

Crowns are items that make cards deploy with the Leader at battle start.
There are two types: regular Crown (removable) and Cursed Crown (permanent, with stat penalties).
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class CrownInfo:
    """Represents a Crown type in the game."""
    name: str
    removable: bool
    description: str
    max_per_card: int = 1
    reduces_stats: Optional[list[str]] = None  # e.g., ["Health", "Attack"]
    reduces_amount: Optional[int] = None


# Hardcoded crown definitions
CROWNS = [
    CrownInfo(
        name="Crown",
        removable=True,
        description="Cards with Crowns are always played at the start of battle.",
    ),
    CrownInfo(
        name="Cursed Crown",
        removable=False,
        description="Cards with Crowns are always played at the start of battle. Reduces Attack and Health by 1 if possible. Cannot be removed.",
        reduces_stats=["Health", "Attack"],
        reduces_amount=1,
    ),
]

# CardTypes that can have crowns placed on them
CROWNABLE_CARD_TYPES = [
    "companions",
    "items",
    "clunkers",
    "pets",
]
