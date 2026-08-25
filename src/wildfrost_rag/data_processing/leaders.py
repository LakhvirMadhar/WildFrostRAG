"""Parser for the Leaders wiki page.

Leaders are special cards with:
- No fixed names (randomly generated in-game)
- Stat ranges instead of fixed values (e.g., health: 5-9)
- Abilities that define the leader archetype
- Single tribe exclusivity (not ALL_TRIBES)
"""

import re
from bs4 import BeautifulSoup, Tag

from wildfrost_rag.data_processing.cards import CardInfo, CardType
from wildfrost_rag.data_processing.tribes import TribeExclusivity
from wildfrost_rag.utils.logger import logger


# Map tribe section IDs to TribeExclusivity enum
TRIBE_SECTION_MAP = {
    "Snowdwellers": TribeExclusivity.SNOWDWELLERS,
    "Shademancers": TribeExclusivity.SHADMANCERS,
    "Clunkmasters": TribeExclusivity.CLUNKMASTERS,
}


def parse_stat_range(stat_text: str) -> tuple[int | None, int | None]:
    """Parse a stat value that may be a range or single value.

    Examples:
        "5-9" -> (5, 9)
        "5" -> (5, 5)
        "" -> (None, None)
        "x3" -> (None, None)  # Not a stat, it's a modifier

    Returns:
        Tuple of (min_value, max_value) or (None, None) if not parseable
    """
    stat_text = stat_text.strip()

    if not stat_text:
        return (None, None)

    # Check for range pattern: "5-9"
    range_match = re.match(r"^(\d+)-(\d+)$", stat_text)
    if range_match:
        return (int(range_match.group(1)), int(range_match.group(2)))

    # Check for single value: "5"
    if stat_text.isdigit():
        val = int(stat_text)
        return (val, val)

    # Not a parseable stat (e.g., "x3 Frenzy" in Other column)
    return (None, None)


def extract_ability_text(description_cell: Tag) -> str:
    """Extract clean ability text from the description cell.

    Handles:
    - Keyword links (e.g., <a href="/Keywords">Aimless</a>)
    - Status effect icons and links
    - Line breaks (<br>)
    """
    # Get text, replacing <br> with newlines
    for br in description_cell.find_all("br"):
        br.replace_with("\n")

    text = description_cell.get_text(separator=" ", strip=True)

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_other_stats(other_cell: Tag) -> str | None:
    """Extract the 'Other' column content (Frenzy, Shell, Reaction, etc.)."""
    text = other_cell.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else None


def parse_leader_table(
    table: Tag, tribe: TribeExclusivity, start_num: int, url: str = ""
) -> list[CardInfo]:
    """Parse a single tribe's leader table.

    Args:
        table: The BeautifulSoup table element
        tribe: The tribe this table belongs to
        start_num: Starting number for leader naming
        url: Wiki URL for this leader page

    Returns:
        List of CardInfo objects for leaders in this table
    """
    leaders = []
    rows = table.find_all("tr")

    # Skip header row
    data_rows = rows[1:] if rows else []

    tribe_name = tribe.value  # e.g., "Snowdwellers"

    for i, row in enumerate(data_rows, start=start_num):
        cells = row.find_all("td")

        if len(cells) < 5:
            logger.warning(f"Skipping row with {len(cells)} cells (expected 5)")
            continue

        # Extract data from cells
        description_cell = cells[0]
        health_text = cells[1].get_text(strip=True)
        attack_text = cells[2].get_text(strip=True)
        counter_text = cells[3].get_text(strip=True)
        other_cell = cells[4]

        # Parse stat ranges
        health_min, health_max = parse_stat_range(health_text)
        attack_min, attack_max = parse_stat_range(attack_text)
        counter_min, counter_max = parse_stat_range(counter_text)

        # Extract ability text
        ability_text = extract_ability_text(description_cell)

        # Extract other stats
        other_stats = extract_other_stats(other_cell)

        # Create CardInfo
        leader = CardInfo(
            card_name=f"{tribe_name} Leader #{i}",
            card_type=CardType.LEADER,
            url=url,
            card_description=f"Leader card for {tribe_name} tribe.",
            tribe_exclusivity=tribe,
            abilities_specific=ability_text if ability_text else None,
            other_stats=other_stats,
            health_min=health_min,
            health_max=health_max,
            attack_min=attack_min,
            attack_max=attack_max,
            counter_min=counter_min,
            counter_max=counter_max,
        )

        leaders.append(leader)

    return leaders


def parse_leaders_page(html: str, url: str = "") -> list[CardInfo]:
    """Parse the Leaders wiki page and extract all leader cards.

    Args:
        html: The raw HTML content of the Leaders page
        url: Wiki URL for the Leaders page

    Returns:
        List of CardInfo objects for all leaders
    """
    soup = BeautifulSoup(html, "html.parser")
    all_leaders = []

    for tribe_name, tribe_enum in TRIBE_SECTION_MAP.items():
        # Find the tribe section heading
        heading = soup.find("span", {"id": tribe_name})

        if not heading:
            logger.warning(f"Could not find section for {tribe_name}")
            continue

        # Find the table following this heading
        # The heading is inside h3, and the table follows
        h3 = heading.find_parent("h3")
        if not h3:
            logger.warning(f"Could not find h3 parent for {tribe_name}")
            continue

        # Find the next sibling table
        table = h3.find_next_sibling("table")
        if not table:
            logger.warning(f"Could not find table for {tribe_name}")
            continue

        # Parse this tribe's leaders
        tribe_leaders = parse_leader_table(table, tribe_enum, start_num=1, url=url)

        logger.info(f"Parsed {len(tribe_leaders)} leaders for {tribe_name}")
        all_leaders.extend(tribe_leaders)

    logger.info(f"Total leaders parsed: {len(all_leaders)}")
    return all_leaders
