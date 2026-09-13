"""Top-level Dagster Definitions object, wiring together defs/'s assets/resources.

Launch locally with:
    poetry run dagster dev -m wildfrost_rag.definitions
"""

from dagster import Definitions

from wildfrost_rag.defs.embeddings.assets import card_embeddings, vector_index
from wildfrost_rag.defs.ingestion.assets import (
    enriched_cards,
    neo4j_documents,
    neo4j_graph,
    scraped_cards,
)
from wildfrost_rag.defs.resources import Neo4jResource, OpenAIResource
from wildfrost_rag.defs.retrieval.assets import retrieval_results

defs = Definitions(
    assets=[
        scraped_cards,
        enriched_cards,
        neo4j_graph,
        neo4j_documents,
        card_embeddings,
        vector_index,
        retrieval_results,
    ],
    resources={
        "neo4j": Neo4jResource(),
        "openai": OpenAIResource(),
    },
)
