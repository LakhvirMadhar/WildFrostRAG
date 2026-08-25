"""Bling parsing for WildFrostRAG.

Parses the Bling wiki page (enemy drop values) and shop pages
(Charm Merchant, The Woolly Snail) for item/charm pricing.
"""

from dataclasses import dataclass

from bs4 import BeautifulSoup

from wildfrost_rag.utils.logger import logger


@dataclass
class EnemyBlingDrop:
    """An enemy's base bling drop value."""

    card_name: str
    amount: int


@dataclass
class ShopListing:
    """An item or charm listing in a shop."""

    card_name: str
    base_price: int


# Boss/miniboss bling values from wiki text (not in the table).
# "All minibosses have a base value of 13, as do Bam and Boozle.
#  All other bosses have a base value of 27."
_MINIBOSS_BLING = 13
_BAM_BOOZLE_BLING = 13
_OTHER_BOSS_BLING = 27
_BAM_BOOZLE = {"Bam", "Boozle"}
# Bamboozle splits into Bam & Boozle — only they drop bling, not Bamboozle itself
_NO_DROP_BOSSES = {"Bamboozle"}


def parse_bling_page(
    html: str, boss_names: list[str], miniboss_names: list[str]
) -> list[EnemyBlingDrop]:
    """Parse the Bling wiki page to extract enemy drop values.

    Combines the table data (regular enemies) with known boss/miniboss
    values from the wiki text.

    Args:
        html: Raw HTML of the Bling page
        boss_names: List of boss card names from card_type_schema
        miniboss_names: List of miniboss card names from card_type_schema

    Returns:
        List of EnemyBlingDrop objects
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", class_="wikitable")

    if not tables:
        logger.warning("No wikitables found on Bling page")
        return []

    # Table 0 is the enemy drops table (Image | Card Name | Gold)
    table = tables[0]
    rows = table.find_all("tr")

    drops: list[EnemyBlingDrop] = []
    seen_names = set()

    for row in rows[1:]:  # skip header
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        card_name = cells[1].get_text(strip=True)
        gold_text = cells[2].get_text(strip=True)

        try:
            amount = int(gold_text)
        except ValueError:
            logger.warning(f"Could not parse bling amount for {card_name}: '{gold_text}'")
            continue

        drops.append(EnemyBlingDrop(card_name=card_name, amount=amount))
        seen_names.add(card_name)

    # Add minibosses (base value 13)
    for name in miniboss_names:
        if name not in seen_names:
            drops.append(EnemyBlingDrop(card_name=name, amount=_MINIBOSS_BLING))
            seen_names.add(name)

    # Add bosses (Bam/Boozle = 13, others = 27)
    for name in boss_names:
        if name in _NO_DROP_BOSSES:
            continue
        if name not in seen_names:
            amount = _BAM_BOOZLE_BLING if name in _BAM_BOOZLE else _OTHER_BOSS_BLING
            drops.append(EnemyBlingDrop(card_name=name, amount=amount))
            seen_names.add(name)

    logger.info(f"Parsed {len(drops)} enemy bling drops from Bling page")
    return drops


def parse_shop_page(html: str) -> list[ShopListing]:
    """Parse a shop page (Charm Merchant or Woolly Snail) to extract listings.

    Both shops use the same table format: Image | Card Name | Description | Price.
    Only table 0 is the shop inventory; later tables are navigation/language.

    Args:
        html: Raw HTML of the shop page

    Returns:
        List of ShopListing objects
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", class_="wikitable")

    if not tables:
        logger.warning("No wikitables found on shop page")
        return []

    table = tables[0]
    rows = table.find_all("tr")

    listings: list[ShopListing] = []

    for row in rows[1:]:  # skip header
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        card_name = cells[1].get_text(strip=True)
        price_text = cells[3].get_text(strip=True)

        try:
            base_price = int(price_text)
        except ValueError:
            logger.warning(f"Could not parse price for {card_name}: '{price_text}'")
            continue

        listings.append(ShopListing(card_name=card_name, base_price=base_price))

    logger.info(f"Parsed {len(listings)} shop listings")
    return listings


def parse_clunker_prices(html: str) -> list[ShopListing]:
    """Parse the Clunkers wiki page to extract clunker shop prices.

    Table format: Image | Card Name | Scrap | Attack | Counter | Other | Description | Tribe-exclusive? | Price
    Price is the last column (index 8).

    Args:
        html: Raw HTML of the Clunkers page

    Returns:
        List of ShopListing objects with clunker prices
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", class_="wikitable")

    if not tables:
        logger.warning("No wikitables found on Clunkers page")
        return []

    table = tables[0]
    rows = table.find_all("tr")

    listings: list[ShopListing] = []

    for row in rows[1:]:  # skip header
        cells = row.find_all("td")
        if len(cells) < 9:
            continue

        card_name = cells[1].get_text(strip=True)
        price_text = cells[8].get_text(strip=True)

        try:
            base_price = int(price_text)
        except ValueError:
            logger.warning(f"Could not parse clunker price for {card_name}: '{price_text}'")
            continue

        listings.append(ShopListing(card_name=card_name, base_price=base_price))

    logger.info(f"Parsed {len(listings)} clunker prices")
    return listings
