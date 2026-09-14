"""Shared HTTP session for the scraping pipeline.

Alongside neo4j_driver.py and openai_client.py - the external-connection
clients this project depends on.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import aiohttp


@asynccontextmanager
async def scraping_session() -> AsyncGenerator[aiohttp.ClientSession]:
    """The one shared HTTP session for a full pipeline run.

    aiohttp.ClientSession is tied to the event loop that created it, so it
    can't be a lazily-created module-level singleton the way _get_client()
    is for OpenAI - that would break across separate asyncio.run() calls
    (CLI runs, tests, Dagster each start their own loop). Callers own the
    session for the duration of one run and pass it down explicitly, same
    as the Neo4j Driver dependency-injection pattern.
    """
    async with aiohttp.ClientSession() as session:
        yield session
