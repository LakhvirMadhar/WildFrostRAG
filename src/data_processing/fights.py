"""Fight page parsing for WildFrostRAG.

Parses individual fight wiki pages to extract enemy card names
for linking Fight nodes to Card nodes in the knowledge graph.
"""

from bs4 import BeautifulSoup, Tag

from utils.logger import logger


def _extract_names_from_enemy_table(table: Tag) -> list[str]:
    """Extract card names from an enemies wikitable (Image | Card Name | ...)."""
    names = []
    for row in table.find_all("tr")[1:]:  # Skip header row
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        # Card Name is the 2nd column (index 1)
        name_cell = cells[1]
        link = name_cell.find("a")
        if link:
            name = link.get_text(strip=True)
            if name and name not in names:
                names.append(name)
    return names


def parse_fight_enemies(html: str) -> list[str]:
    """Extract enemy card names from a fight page's Enemies tables.

    Finds all headings containing "Enemies" (e.g., "Enemies", "Summoned Enemies")
    and extracts card names from each table. Some pages like Eye of the Storm
    have multiple enemy tables across phases.

    Args:
        html: Raw HTML content of a fight page

    Returns:
        Deduplicated list of enemy card names found across all enemy tables
    """
    soup = BeautifulSoup(html, "html.parser")

    # Find all headings that contain "Enemies" (covers "Enemies" and "Summoned Enemies")
    all_headlines = soup.find_all("span", class_="mw-headline")
    enemy_headings = [
        headline for headline in all_headlines if headline.string and "Enemies" in headline.string
    ]
    if not enemy_headings:
        return []

    enemy_names = []
    for span in enemy_headings:
        heading = span.find_parent(["h2", "h3"])
        if not heading:
            continue

        table = heading.find_next("table", class_="wikitable")
        if not table:
            continue

        for name in _extract_names_from_enemy_table(table):
            if name not in enemy_names:
                enemy_names.append(name)

    return enemy_names


def parse_all_fight_enemies(fight_htmls: dict[str, str]) -> dict[str, list[str]]:
    """Parse enemy names from multiple fight pages.

    Args:
        fight_htmls: Dict mapping fight page_name to HTML content

    Returns:
        Dict mapping fight page_name to list of enemy card names
    """
    results = {}
    for page_name, html in fight_htmls.items():
        enemies = parse_fight_enemies(html)
        results[page_name] = enemies
        logger.info(f"  {page_name}: {len(enemies)} enemies")

    total = sum(len(e) for e in results.values())
    logger.info(f"Parsed enemies from {len(results)} fight pages ({total} total enemy entries)")
    return results
