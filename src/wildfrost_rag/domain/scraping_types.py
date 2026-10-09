"""Named type aliases and shapes for scraper result data.

The type aliases are zero runtime behavior, zero I/O - domain is their
correct home per the layered architecture. They exist so signatures like
`fight_page_mapping: dict[str, str]` say what the mapping actually means
instead of forcing the reader to go find out.

HasWikiPage is a structural Protocol (also zero runtime behavior) naming
the shape StatInfo, CharmInfo, and BellInfo already share, so scraping code
can handle "any entity with its own wiki page" generically instead of once
per entity type.
"""

from typing import Protocol

type PageUrls = dict[str, str]
"""Filename -> wiki URL, e.g. {"Bombom.html": "https://wildfrost.wiki.gg/Bombom"}."""

type FightPageMapping = dict[str, str]
"""Fight display name -> wiki page slug, e.g. {"Infernoko": "Infernoko_Fight"}."""

type FightEnemies = dict[str, list[str]]
"""Wiki page slug -> the enemy card names appearing on that fight's page."""


class HasWikiPage(Protocol):
    """Structural shape of an entity that has its own individual wiki page.

    StatInfo, CharmInfo, and BellInfo all satisfy this without inheriting
    from it - Protocol matching is structural, not nominal. Only identity is
    needed: where its HTML is cached and how it's written are decided by
    scraping/html_cache.py, not by the entity.
    """

    name: str
    url: str | None
