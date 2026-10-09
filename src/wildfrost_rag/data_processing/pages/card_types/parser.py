"""Parses the wiki's card navbox into a card type -> [card names] schema."""

from bs4 import BeautifulSoup

from wildfrost_rag.core.logger import logger


def parse_card_type_html_schema(html: str) -> dict[str, list[str]]:
    """Parse the card type schema page HTML into a card_type -> [card_names] mapping.

    Args:
        html: Raw HTML of the schema page

    Returns:
        Dict mapping category name (e.g. "companions") to a list of card names
    """
    soup = BeautifulSoup(html, "html.parser")

    card_schema = soup.find("table", {"class": "wikitable", "id": "navbox"})

    card_data: dict[str, list[str]] = {}

    if card_schema is None:
        return card_data

    rows = card_schema.find_all("tr")[1:]

    for row in rows:
        card_list = []
        card_type = row.find("th")
        card_type_data = row.find("td")
        if card_type is None or card_type_data is None:
            continue
        card_names = card_type_data.find_all("a")

        category_name = card_type.get_text().strip().lower().replace(" ", "_")

        logger.debug(f"Schema category: {category_name}")

        for n in card_names:
            card_name = n.get_text().strip()
            card_list.append(card_name)

        card_data[category_name] = card_list

    return card_data
