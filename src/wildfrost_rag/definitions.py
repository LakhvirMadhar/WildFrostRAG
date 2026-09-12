"""Top-level Dagster Definitions object, wiring together defs/'s assets/resources.

Launch locally with:
    poetry run dagster dev -m wildfrost_rag.definitions
"""

from dagster import Definitions

from wildfrost_rag.defs.resources import Neo4jResource, OpenAIResource

defs = Definitions(
    assets=[],
    resources={
        "neo4j": Neo4jResource(),
        "openai": OpenAIResource(),
    },
)
