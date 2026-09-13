"""Dagster asset chain for Document embeddings.

card_embeddings depends on neo4j_documents (not neo4j_graph) because
Document nodes - the things being embedded - are created in stage 4
(neo4j_documents), not stage 3 (neo4j_graph, which only creates
Card/Tribe/CardType/etc. nodes).
"""

from dagster import Backoff, Config, Jitter, RetryPolicy, asset

from wildfrost_rag.defs.ingestion.assets import neo4j_documents
from wildfrost_rag.defs.resources import Neo4jResource
from wildfrost_rag.services.embeddings.embedder_type import EmbedderType
from wildfrost_rag.services.embeddings.embedding_service import EmbeddingService

_EMBEDDING_RETRY_POLICY = RetryPolicy(
    max_retries=3, delay=5, backoff=Backoff.EXPONENTIAL, jitter=Jitter.FULL
)


class EmbeddingConfig(Config):
    """Which embedding provider to run - the Dagster-config equivalent of --embedder."""

    embedder: EmbedderType = EmbedderType.HF


@asset(deps=[neo4j_documents], retry_policy=_EMBEDDING_RETRY_POLICY)
async def card_embeddings(config: EmbeddingConfig, neo4j: Neo4jResource) -> int:
    """Generate and store embeddings for every Document node missing them.

    Resumable: only Documents still missing the configured provider's
    embedding property are processed, so a retry after partial failure
    finishes the remaining documents instead of re-skipping everything.
    """
    with neo4j.get_driver() as driver:
        return await EmbeddingService().add_embeddings(driver, config.embedder)


@asset(deps=[card_embeddings])
def vector_index(config: EmbeddingConfig, neo4j: Neo4jResource) -> None:
    """Create the vector index for the configured provider's embedding property.

    Must be materialized with the same `embedder` config value used for
    card_embeddings - Dagster does not share config across separate assets.
    """
    with neo4j.get_driver() as driver:
        EmbeddingService().create_vector_index(driver, config.embedder)
