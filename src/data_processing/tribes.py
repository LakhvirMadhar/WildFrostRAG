"""
Tribe definitions for WildFrostRAG.

Shared enum used by cards, charms, and other game entities
that have tribe exclusivity.
"""

from enum import Enum


class TribeExclusivity(Enum):
    """Defines which tribe(s) an entity belongs to."""
    SNOWDWELLERS = 'Snowdwellers'
    SHADMANCERS = 'Shademancers'
    CLUNKMASTERS = 'Clunkmasters'
    ALL_TRIBES = 'All'

    @property
    def is_exclusive(self) -> bool:
        """Check if this is exclusive to a single tribe."""
        return self != TribeExclusivity.ALL_TRIBES

    @property
    def is_universal(self) -> bool:
        return self == TribeExclusivity.ALL_TRIBES

    def get_tribes(self) -> list[str]:
        """Get list of tribe names this exclusivity represents."""
        if self.is_universal:
            return [tribe.value for tribe in TribeExclusivity if tribe.is_exclusive]
        else:
            return [self.value]
