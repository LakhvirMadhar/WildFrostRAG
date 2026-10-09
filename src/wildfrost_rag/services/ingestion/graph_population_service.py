"""Stage 3 of the WildFrostRAG ingestion pipeline: Neo4j graph population.

Extracted from GraphBuilderService so graph population is unit-testable and
reusable (by both the CLI-facing GraphBuilderService and the Dagster asset
chain). Follows the same dependency-injection pattern as CardRepository,
DocumentRepository, and BaseNeo4jRetriever: the Neo4j Driver is constructed
externally and passed in, never created here.
"""

from typing import Any

from neo4j import Driver, Session

from wildfrost_rag.core.config import get_settings
from wildfrost_rag.core.logger import logger
from wildfrost_rag.repositories.bells import create_bell_relationships, create_bells_from_parsed
from wildfrost_rag.repositories.bling import (
    create_bling_and_shops,
    create_drops_bling_relationships,
    create_shop_sells_relationships,
)
from wildfrost_rag.repositories.charms import (
    create_charm_tribe_relationships,
    create_charms_from_parsed,
)
from wildfrost_rag.repositories.fights import create_fight_enemy_relationships
from wildfrost_rag.repositories.graph_builder import create_neo4j_data, create_url_nodes
from wildfrost_rag.repositories.keywords import (
    create_card_keyword_relationships,
    create_charm_keyword_relationships,
    create_keywords_from_parsed,
)
from wildfrost_rag.repositories.map import create_map_graph
from wildfrost_rag.repositories.shades import create_summon_relationships
from wildfrost_rag.repositories.stats import add_keyword_label_to_stats, create_stats_from_parsed
from wildfrost_rag.services.ingestion.pipeline_data import PipelineData


class GraphPopulationService:
    """Stage 3: create nodes and relationships in the Neo4j knowledge graph."""

    def __init__(self, driver: Driver) -> None:
        """Initialize the service.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
        """
        self.driver = driver

    def _populate_stats_and_keywords(
        self, session: Session, data: PipelineData, urls: dict[str, str]
    ) -> None:
        """Populate Stat and Keyword nodes."""
        if data.stats:
            count = session.execute_write(create_stats_from_parsed, data.stats)
            logger.info(f"Created {count} Stat nodes")
            session.execute_write(add_keyword_label_to_stats)

        if data.keywords:
            count = session.execute_write(
                create_keywords_from_parsed, data.keywords, urls.get("Keywords.html")
            )
            logger.info(f"Created {count} Keyword nodes")

    def _populate_charms(self, session: Session, data: PipelineData) -> None:
        """Populate Charm nodes."""
        if data.charms:
            charm_count = session.execute_write(create_charms_from_parsed, data.charms)
            logger.info(f"Created {charm_count} Charm nodes")

    def _populate_cards_and_core(
        self, session: Session, data: PipelineData, urls: dict[str, str]
    ) -> list[dict[str, Any]]:
        """Populate Cards, Tribes, Crowns, and core relationships.

        Returns:
            cards_dict_data for use by downstream relationship builders.
        """
        cards_dict_data = [card.to_dict() for card in data.cards]
        logger.info(f"Ingesting {len(cards_dict_data)} cards into Neo4j graph...")
        create_neo4j_data(session, cards_dict_data, crowns_url=urls.get("Crowns.html"))
        return cards_dict_data

    def _populate_keyword_relationships(
        self, session: Session, data: PipelineData, cards_dict_data: list[dict[str, Any]]
    ) -> None:
        """Create Card-Keyword and Charm-Keyword relationships."""
        if data.keywords:
            kw_rel_count = session.execute_write(create_card_keyword_relationships, cards_dict_data)
            logger.info(f"Created {kw_rel_count} Card-Keyword relationships")
            charm_kw_count = session.execute_write(create_charm_keyword_relationships)
            logger.info(f"Created {charm_kw_count} Charm-Keyword relationships")

        if data.charms:
            tribe_count = session.execute_write(create_charm_tribe_relationships, data.charms)
            logger.info(f"Created {tribe_count} Charm-Tribe relationships")

    def _populate_map_and_fights(
        self, session: Session, data: PipelineData, urls: dict[str, str]
    ) -> None:
        """Populate Map, Zone, Fight nodes and relationships."""
        if data.zones or data.map_events or data.fight_slots:
            counts = session.execute_write(
                create_map_graph,
                data.zones,
                data.map_events,
                data.fight_slots,
                data.fight_page_mapping,
                url=urls.get("Map.html"),
                base_url=get_settings().scraping.wildfrost_wiki_base_url,
            )
            logger.info(f"Map graph created: {counts}")

        if data.fight_enemies and data.fight_page_mapping:
            enemy_count = session.execute_write(
                create_fight_enemy_relationships,
                data.fight_enemies,
                data.fight_page_mapping,
            )
            logger.info(f"Created {enemy_count} Fight-Enemy relationships")

        if data.summons:
            summon_count = session.execute_write(create_summon_relationships, data.summons)
            logger.info(f"Created {summon_count} SUMMONS relationships")

    def _populate_bells(self, session: Session, data: PipelineData) -> None:
        """Populate Bell nodes and relationships."""
        if data.bells:
            bell_count = session.execute_write(create_bells_from_parsed, data.bells)
            logger.info(f"Created {bell_count} Bell nodes")
            bell_rel_count = session.execute_write(create_bell_relationships)
            logger.info(f"Created {bell_rel_count} bell linking relationships")

    def _populate_bling_economy(
        self, session: Session, data: PipelineData, urls: dict[str, str]
    ) -> None:
        """Populate Bling, Shop nodes, and economy relationships."""
        bling_shop_urls: dict[str, str] = {
            k: v
            for k, v in {
                "Bling": urls.get("Bling.html"),
                "The Woolly Snail": urls.get("The_Woolly_Snail.html"),
                "Charm Merchant": urls.get("Charm_Merchant.html"),
            }.items()
            if v is not None
        }
        session.execute_write(create_bling_and_shops, bling_shop_urls)

        if data.bling_drops:
            drop_count = session.execute_write(create_drops_bling_relationships, data.bling_drops)
            logger.info(f"Created {drop_count} DROPS_BLING relationships")
        if data.woolly_snail_listings:
            snail_count = session.execute_write(
                create_shop_sells_relationships,
                "The Woolly Snail",
                data.woolly_snail_listings,
                "Card",
            )
            logger.info(f"Created {snail_count} Woolly Snail SELLS relationships")
        if data.charm_merchant_listings:
            charm_shop_count = session.execute_write(
                create_shop_sells_relationships,
                "Charm Merchant",
                data.charm_merchant_listings,
                "Charm",
            )
            logger.info(f"Created {charm_shop_count} Charm Merchant SELLS Charm relationships")

        if data.woolly_snail_listings:
            cm_item_count = session.execute_write(
                create_shop_sells_relationships,
                "Charm Merchant",
                data.woolly_snail_listings,
                "Card",
            )
            logger.info(f"Created {cm_item_count} Charm Merchant SELLS Item relationships")
        if data.clunker_prices:
            cm_clunker_count = session.execute_write(
                create_shop_sells_relationships,
                "Charm Merchant",
                data.clunker_prices,
                "Card",
            )
            logger.info(f"Created {cm_clunker_count} Charm Merchant SELLS Clunker relationships")

    def populate(self, data: PipelineData) -> None:
        """Stage 3: Neo4j Graph Population.

        Creates nodes and relationships in Neo4j knowledge graph.

        Args:
            data: PipelineData containing all scraped data to populate the graph
        """
        logger.info("=" * 60)
        logger.info("STAGE 3: NEO4J GRAPH POPULATION")
        logger.info("=" * 60)

        with self.driver.session() as session:
            urls = data.page_urls

            self._populate_stats_and_keywords(session, data, urls)
            self._populate_charms(session, data)
            cards_dict_data = self._populate_cards_and_core(session, data, urls)
            self._populate_keyword_relationships(session, data, cards_dict_data)
            self._populate_map_and_fights(session, data, urls)
            # Bling before bells: bell linking MATCHes the Bling node, and MERGE-based
            # linking silently creates nothing if that node doesn't exist yet.
            self._populate_bling_economy(session, data, urls)
            self._populate_bells(session, data)

            url_link_count = session.execute_write(create_url_nodes)
            logger.info(f"Created URL nodes with {url_link_count} HAS_LINK relationships")

        logger.info("Graph population complete")
