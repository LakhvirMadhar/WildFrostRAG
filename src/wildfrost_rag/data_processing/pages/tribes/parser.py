"""Parses a wiki page's "Tribe-exclusive?" table into a card name -> tribe lookup.

Used on the Companions page (second table) and the Items page (first table).
"""

from bs4 import BeautifulSoup, Comment

from wildfrost_rag.core.logger import logger


def parse_tribe_exclusivity_table(html: str, table_index: int = 1) -> dict[str, str]:
    """Parse tribe exclusivity information from a wiki page table.

    Args:
        html: Raw HTML of the wiki page containing the tribe table
        table_index: Index of the table to parse (0-based). Default is 1 for
                    the second table, which is typical for Companions page.

    Returns:
        Dictionary mapping card names to tribe names (e.g., "Snowdwellers", "All")

    Raises:
        ValueError: If the expected table structure is not found
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove HTML comments
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()

    # Find all sortable wiki tables
    tables = soup.find_all("table", {"class": "wikitable sortable"})

    if not tables:
        raise ValueError("No sortable wikitable found")

    if table_index >= len(tables):
        raise ValueError(f"Table index {table_index} out of range. Found {len(tables)} tables")

    target_table = tables[table_index]

    # Extract headers
    first_row = target_table.find("tr")
    if first_row is None:
        raise ValueError("No rows found in table")
    headers = [th.text.strip() for th in first_row.find_all("th")]
    logger.debug(f"Found table headers: {headers}")

    # Find the indices of required columns
    try:
        card_name_index = headers.index("Card Name")
        tribe_exclusive_index = headers.index("Tribe-exclusive?")
    except ValueError as e:
        raise ValueError(f"Required column not found in table headers: {e}") from e

    # Parse table rows
    tribe_lookup = {}

    for row in target_table.find_all("tr")[1:]:  # Skip header row
        cells = row.find_all(["th", "td"])

        if len(cells) > max(card_name_index, tribe_exclusive_index):
            card_name = cells[card_name_index].get_text(strip=True)
            tribe_name = cells[tribe_exclusive_index].get_text(strip=True)

            tribe_lookup[card_name] = tribe_name

    logger.info(f"Parsed {len(tribe_lookup)} card-tribe mappings")
    return tribe_lookup
