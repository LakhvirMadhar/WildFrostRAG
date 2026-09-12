"""Dagster resources - shared, injected dependencies for assets."""

from contextlib import AbstractContextManager

from dagster import ConfigurableResource
from neo4j import Driver
from openai.types.chat import ChatCompletionMessageParam

from wildfrost_rag.clients.neo4j_driver import neo4j_driver
from wildfrost_rag.clients.openai_client import call_openai_api, call_openai_embeddings


class Neo4jResource(ConfigurableResource["Neo4jResource"]):
    """Injects the shared Neo4j driver into assets.

    Delegates to clients.neo4j_driver.neo4j_driver() - the same
    composition-root construction every script/service already uses -
    rather than building a second, competing way to get a driver.
    """

    def get_driver(self) -> AbstractContextManager[Driver]:
        """Return the driver context manager; use as `with resource.get_driver() as driver:`."""
        return neo4j_driver()


class OpenAIResource(ConfigurableResource["OpenAIResource"]):
    """Injects the OpenAI ACL (clients.openai_client) into assets.

    Thin passthrough to the existing module-level functions, which already
    manage their own client/semaphore singletons read from settings - this
    resource exists so assets depend on it through Dagster's DI (for
    lineage and swappability in tests), not so a second client gets built.
    """

    async def call_api(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> str:
        """Delegate to call_openai_api."""
        return await call_openai_api(messages, model=model, temperature=temperature, seed=seed)

    async def call_embeddings(
        self, texts: list[str], model: str | None = None
    ) -> list[list[float]]:
        """Delegate to call_openai_embeddings."""
        return await call_openai_embeddings(texts, model=model)
