"""Unit tests for scrape_multiple_links against a real local HTTP server.

Uses aiohttp's own TestServer rather than mocking the session: the bug this
guards against (failed requests being dropped, shifting every later result
onto the wrong input) is about real request ordering, which a mock would
just restate.
"""

import asyncio

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from wildfrost_rag.scraping.sitemap_scraper import scrape_multiple_links


async def _page(request: web.Request) -> web.Response:
    name = request.match_info["name"]
    if name == "broken":
        return web.Response(status=500)
    return web.Response(text=f"<html>{name}</html>")


async def _scrape(paths: list[str]) -> list[str | None]:
    app = web.Application()
    app.router.add_get("/{name}", _page)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        urls = [str(server.make_url(f"/{path}")) for path in paths]
        return await scrape_multiple_links(session, urls, max_concurrent=2)


def test_scrape_multiple_links_keeps_a_failed_request_in_its_position() -> None:
    """A failure in the middle comes back as None in place, so later results stay aligned."""
    results = asyncio.run(_scrape(["first", "broken", "third"]))

    assert results == ["<html>first</html>", None, "<html>third</html>"]


def test_scrape_multiple_links_returns_one_result_per_url() -> None:
    """Callers zip results with their inputs, so the lengths must always match."""
    results = asyncio.run(_scrape(["broken", "a", "broken", "b"]))

    assert len(results) == 4
