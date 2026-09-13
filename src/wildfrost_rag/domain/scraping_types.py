"""Named type aliases for scraper result shapes.

These are pure type aliases (zero runtime behavior, zero I/O) - domain is
their correct home per the layered architecture. They exist so signatures
like `fight_page_mapping: dict[str, str]` say what the mapping actually
means instead of forcing the reader to go find out.
"""

type PageUrls = dict[str, str]
"""Filename -> wiki URL, e.g. {"Bombom.html": "https://wildfrost.wiki.gg/Bombom"}."""

type FightPageMapping = dict[str, str]
"""Fight display name -> wiki page slug, e.g. {"Infernoko": "Infernoko_Fight"}."""

type FightEnemies = dict[str, list[str]]
"""Wiki page slug -> the enemy card names appearing on that fight's page."""
