"""
Configuration for multi-phase card detection edge cases.

The auto-detect heuristic (base name matching) handles most cases:
- Same base name + same type → PHASES
- Same base name + different type → VARIANTS
- Different base names → MULTIPLE CARDS

This config handles cases that can't be auto-detected:
- Cross-page transforms (different pages, different names)
- Variant cards (same page, multiple infoboxes, but NOT phases)
- Any other edge cases where the heuristic fails
"""

# Cards that transform into other cards across different wiki pages.
# Key: source card name
# Value: list of target card names (in transformation order, or all if simultaneous)
CROSS_PAGE_TRANSFORMS: dict[str, list[str]] = {
    "Bamboozle": ["Bam", "Boozle"],  # Splits into both simultaneously
}

# Cards with multiple infoboxes that are VARIANTS, not phases.
# These have the same base name but represent different versions (e.g., enemy vs companion).
# Each infobox should be treated as a separate card with NO phase linking.
VARIANT_CARDS: set[str] = {
    "Naked Gnome",  # Has enemy and companion variants on same page
}

# Manual phase order overrides for cards where image-based detection gets it wrong.
# Key: base_name
# Value: list of card_names in correct phase order (phase 1 first, then phase 2, etc.)
PHASE_ORDER_OVERRIDES: dict[str, list[str]] = {
    # Frost Guardian's images don't follow the standard naming pattern
    "Frost Guardian": ["Frost Guardian (Frost Wizard)", "Frost Guardian"],
}

# Enemy cards that can be recruited as companions if kept alive.
# Key: enemy card name (with suffix)
# Value: companion card name (with suffix)
# Creates [:CAN_BE_RECRUITED_AS] relationship from enemy -> companion
RECRUITABLE_ENEMIES: dict[str, str] = {
    "Naked Gnome (Enemy)": "Naked Gnome (Companion)",
}
