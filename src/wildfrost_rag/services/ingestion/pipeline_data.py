"""PipelineData - the data collected during ingestion's scraping stage.

Its own module (not part of graph_builder_service.py) because every one of
scraping_service.py, enrichment_service.py, graph_population_service.py,
document_ingestion_service.py, graph_builder_service.py, and
defs/ingestion/assets.py needs this type - defining it inside any one of
those files would make the others import back from it, creating a cycle.
"""

from dataclasses import dataclass, field

from wildfrost_rag.data_processing.bells import BellInfo
from wildfrost_rag.data_processing.bling import EnemyBlingDrop, ShopListing
from wildfrost_rag.data_processing.cards import CardInfo
from wildfrost_rag.data_processing.charms import CharmInfo
from wildfrost_rag.data_processing.keywords import KeywordInfo
from wildfrost_rag.data_processing.map import FightSlotInfo, MapEventInfo, ZoneInfo
from wildfrost_rag.data_processing.shades import SummonInfo
from wildfrost_rag.data_processing.stats import StatInfo
from wildfrost_rag.domain.scraping_types import FightEnemies, FightPageMapping, PageUrls


@dataclass
class PipelineData:
    """All data collected during Stage 1 (scraping).

    Replaces the positional tuple that previously grew with every new data source.
    New scraping targets just add a field here — no tuple unpacking to update.
    """

    cards: list[CardInfo] = field(default_factory=list)
    stats: list[StatInfo] = field(default_factory=list)
    keywords: list[KeywordInfo] = field(default_factory=list)
    bling_drops: list[EnemyBlingDrop] = field(default_factory=list)
    woolly_snail_listings: list[ShopListing] = field(default_factory=list)
    charm_merchant_listings: list[ShopListing] = field(default_factory=list)
    clunker_prices: list[ShopListing] = field(default_factory=list)
    bells: list[BellInfo] = field(default_factory=list)
    charms: list[CharmInfo] = field(default_factory=list)
    summons: list[SummonInfo] = field(default_factory=list)
    zones: list[ZoneInfo] = field(default_factory=list)
    map_events: list[MapEventInfo] = field(default_factory=list)
    fight_slots: list[FightSlotInfo] = field(default_factory=list)
    fight_page_mapping: FightPageMapping = field(default_factory=dict)
    fight_enemies: FightEnemies = field(default_factory=dict)
    page_urls: PageUrls = field(default_factory=dict)
