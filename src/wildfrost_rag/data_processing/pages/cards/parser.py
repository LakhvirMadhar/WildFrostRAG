"""Parses an individual card's wiki page into one or more CardInfo objects.

One page can hold several infoboxes: phases of one card (Infernoko,
Truffle), variants of one card (Naked Gnome as enemy and companion), or
unrelated cards. parse_card_page sorts out which.
"""

import re
from collections import defaultdict
from typing import Any
from urllib.parse import unquote

from bs4 import BeautifulSoup, Tag

from wildfrost_rag.core.logger import logger
from wildfrost_rag.data_processing.cards import CardInfo, CardType
from wildfrost_rag.data_processing.phase_config import PHASE_ORDER_OVERRIDES, VARIANT_CARDS


def get_base_name(card_name: str) -> str:
    """Extract base name by removing parenthetical suffixes.

    Used for detecting multi-phase cards where phases have names like:
    - 'Truffle', 'Truffle (medium)', 'Truffle (small)' → all return 'Truffle'
    - 'Infernoko' → returns 'Infernoko'
    - 'Naked Gnome' → returns 'Naked Gnome'
    """
    return re.sub(r"\s*\([^)]*\)\s*$", "", card_name).strip()


def _extract_card_name_from_infobox(infobox: Tag) -> str | None:
    """Extract card name from infobox table.

    The card name is in the first row's <th> element:
    <tr><th colspan="3">Card Name</th></tr>
    """
    first_row = infobox.find("tr")
    if not first_row:
        return None

    th = first_row.find("th")
    if not th:
        return None

    return th.get_text(strip=True)


def _extract_phase_from_infobox(infobox: Tag) -> int | None:
    """Extract phase number from image filename inside infobox.

    Wiki convention:
    - Infernoko_(FrenzyBoss2).png → Phase 2
    - Truffle_(SummonBoss3).png → Phase 3
    - No number in parentheses → Phase 1

    Handles URL-encoded parentheses: %28 = (, %29 = )

    Returns None if no image found or can't determine phase.
    """
    img = infobox.find("img")
    if not img:
        return None

    # Try src or alt attribute
    raw_ref = img.get("src") or img.get("alt") or ""
    img_ref = raw_ref if isinstance(raw_ref, str) else ""

    # URL decode %28 and %29 to ( and )
    img_ref_decoded = unquote(img_ref)

    # Look for number at end of parentheses before .png: (SomeName2).png → 2
    match = re.search(r"\([^)]*?(\d+)\)\.png", img_ref_decoded, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # No number in parentheses - this is Phase 1
    if ".png" in img_ref_decoded.lower():
        return 1

    return None


def _extract_stats_from_infobox(infobox: Tag) -> dict[str, int | str | None]:  # noqa: C901
    """Extract stats dictionary from an infobox table.

    Returns dict with keys like 'health', 'attack', 'counter', 'scrap', 'other_stats'.
    """
    rows = infobox.find_all("tr")
    stats: dict[str, int | str | None] = {}

    if len(rows) >= 4:
        # Stats are in rows 2 (headers) and 3 (values)
        stats_headers = [th.text.strip().lower() for th in rows[2].find_all("th")]
        stats_values = [td.text.strip() for td in rows[3].find_all("td")]

        for header, value in zip(stats_headers, stats_values, strict=False):
            if value.strip() == "":
                stats[header] = None
            elif value.isdigit():
                stats[header] = int(value)
            else:
                stats[header] = value

    # Extract 'Other Stats' and 'Card Description' sections
    for i, row in enumerate(rows):
        th = row.find("th")
        if not th:
            continue

        header_text = th.get_text(strip=True)

        if header_text == "Other Stats":
            if i + 1 < len(rows):
                td = rows[i + 1].find("td")
                if td:
                    other_stats_text = td.get_text(separator=" ", strip=True)
                    other_stats_text = re.sub(r"\s+", " ", other_stats_text).strip()
                    stats["other_stats"] = other_stats_text if other_stats_text else None

        elif header_text == "Card Description":
            if i + 1 < len(rows):
                td = rows[i + 1].find("td")
                if td:
                    # Check for flavor text: italic text in gray/grey span
                    gray_span = td.find(
                        "span",
                        style=lambda s: s and ("color:gray" in s or "color:grey" in s),
                    )
                    italic = td.find("i")

                    if gray_span and italic:
                        # This is flavor text (like "Does absolutely nothing...")
                        flavor_text = italic.get_text(separator=" ", strip=True)
                        flavor_text = re.sub(r"\s+", " ", flavor_text).strip()
                        stats["flavor_text"] = flavor_text if flavor_text else None
                    else:
                        # This is ability text
                        ability_text = td.get_text(separator=" ", strip=True)
                        ability_text = re.sub(r"\s+", " ", ability_text).strip()
                        stats["abilities_specific"] = ability_text if ability_text else None

    return stats


def _create_card(
    card_data: dict[str, Any],
    card_type: CardType,
    url: str,
    html: str,
    description: str | None,
    phase: int | None = None,
    total_phases: int | None = None,
    base_name: str | None = None,
) -> CardInfo:
    """Create a CardInfo from parsed infobox data."""
    return CardInfo(
        card_name=card_data["card_name"],
        card_type=card_type,
        url=url,
        card_html=html,
        card_description=description,
        health=card_data["stats"].get("health"),
        attack=card_data["stats"].get("attack"),
        scrap=card_data["stats"].get("scrap"),
        counter=card_data["stats"].get("counter"),
        other_stats=card_data["stats"].get("other_stats"),
        abilities_specific=card_data["stats"].get("abilities_specific"),
        flavor_text=card_data["stats"].get("flavor_text"),
        phase=phase,
        total_phases=total_phases,
        base_name=base_name,
    )


def _detect_variant_card_type(card_name: str, default_type: CardType) -> CardType:
    """Detect card type from name suffix for variant cards."""
    if card_name.endswith("(Enemy)"):
        return CardType.NON_BOSS_ENEMIES
    elif card_name.endswith("(Companion)"):
        return CardType.COMPANIONS
    return default_type


def _sort_phases(group: list[dict[str, Any]], base_name: str) -> list[dict[str, Any]]:
    """Sort cards in a group by phase order."""
    if base_name in PHASE_ORDER_OVERRIDES:
        order_list = PHASE_ORDER_OVERRIDES[base_name]
        return sorted(
            group,
            key=lambda x: order_list.index(x["card_name"]) if x["card_name"] in order_list else 999,
        )
    # Sort by phase number extracted from image (Phase 1 has no number)
    return sorted(group, key=lambda x: x["phase_from_image"] if x["phase_from_image"] else 1)


def _parse_infoboxes(soup: BeautifulSoup, url: str) -> list[dict[str, Any]]:
    """Parse all infoboxes from HTML into a list of card data dicts."""
    infoboxes = soup.find_all("table", {"id": "infobox"})
    parsed_cards: list[dict[str, Any]] = []

    for infobox in infoboxes:
        card_name = _extract_card_name_from_infobox(infobox)
        if not card_name:
            logger.warning(f"Could not extract card name from infobox in {url}")
            continue

        parsed_cards.append(
            {
                "card_name": card_name,
                "base_name": get_base_name(card_name),
                "stats": _extract_stats_from_infobox(infobox),
                "phase_from_image": _extract_phase_from_infobox(infobox),
            }
        )

    return parsed_cards


def parse_card_page(
    html: str,
    card_type: CardType,
    url: str,
) -> list[CardInfo]:
    """Parse HTML that may contain multiple infoboxes (multi-phase cards).

    Handles:
    - Single infobox: Returns list with one CardInfo (phase=None)
    - Multiple infoboxes, same base name, same type: PHASES (linked via phase numbers)
    - Multiple infoboxes, same base name, different type: VARIANTS (separate cards, no phase)
    - Multiple infoboxes, different base names: MULTIPLE CARDS (separate cards, no phase)

    Args:
        html: The HTML content of the page
        card_type: The CardType for all cards on this page
        url: The URL of the page

    Returns:
        List of CardInfo objects, one per infobox
    """
    soup = BeautifulSoup(html, "html.parser")

    if not soup.find("table", {"id": "infobox"}):
        logger.warning(f"No infoboxes found in HTML for {url}")
        return []

    # Extract description from meta tag (shared across all cards on page)
    description_tag = soup.find("meta", attrs={"name": "description"})
    raw_desc = description_tag.get("content", "") if description_tag else None
    description: str | None = raw_desc if isinstance(raw_desc, str) else None

    # Parse all infoboxes
    parsed_cards = _parse_infoboxes(soup, url)
    if not parsed_cards:
        return []

    # Single infobox - no phases
    if len(parsed_cards) == 1:
        return [_create_card(parsed_cards[0], card_type, url, html, description)]

    # Multiple infoboxes - group by base name
    base_name_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card_data in parsed_cards:
        base_name_groups[card_data["base_name"]].append(card_data)

    result: list[CardInfo] = []

    for base_name, group in base_name_groups.items():
        if len(group) == 1:
            # Single card with this base name - no phases
            result.append(_create_card(group[0], card_type, url, html, description))

        elif base_name in VARIANT_CARDS:
            # Variant cards (e.g., Naked Gnome enemy/companion) - no phase linking
            for card_data in group:
                variant_type = _detect_variant_card_type(card_data["card_name"], card_type)
                result.append(_create_card(card_data, variant_type, url, html, description))

        else:
            # Multiple cards with same base name - these are PHASES
            group_sorted = _sort_phases(group, base_name)
            total_phases = len(group_sorted)
            for phase_num, card_data in enumerate(group_sorted, start=1):
                result.append(
                    _create_card(
                        card_data,
                        card_type,
                        url,
                        html,
                        description,
                        phase=phase_num,
                        total_phases=total_phases,
                        base_name=base_name,
                    )
                )

    return result
