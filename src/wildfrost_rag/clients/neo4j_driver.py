"""Composition-root helper for constructing the shared Neo4j driver.

The Driver returned here manages a connection pool, not a single connection
(see Neo4j's Python driver docs) - it's meant to be created once per process
and shared, never self-constructed by the classes that use it. Every script
was duplicating this construction by hand; this is the one place it happens.
"""

from collections.abc import Generator
from contextlib import contextmanager

from neo4j import Driver, GraphDatabase

from wildfrost_rag.utils.config import get_settings


@contextmanager
def neo4j_driver() -> Generator[Driver]:
    """Build the shared Neo4j driver from settings, closing it on exit."""
    settings = get_settings()
    driver = GraphDatabase.driver(
        settings.neo4j.uri.get_secret_value(),
        auth=(settings.neo4j.username, settings.neo4j.password.get_secret_value()),
    )
    try:
        yield driver
    finally:
        driver.close()
