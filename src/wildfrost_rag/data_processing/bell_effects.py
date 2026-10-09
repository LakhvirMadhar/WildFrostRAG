"""Curated facts about what each bell does in Wildfrost.

None of this is on the Bells wiki page in a parseable form - it's hand-
curated game knowledge, kept here (like phase_config.py and crowns.py)
rather than in repositories/bells.py, so the game data and the Cypher that
writes it into Neo4j change independently.

Card type values are CardType values (e.g. "non_boss_enemies").
"""

BELL_GRANTS_KEYWORD: dict[str, str] = {
    "Bell of Death": "Injured",
    "Noomlin Sun Bell": "Noomlin",
}
"""Bell -> keyword the bell grants to cards."""

BELL_AFFECTS_KEYWORD: dict[str, str] = {
    "Breakfast Sun Bell": "Consume",
}
"""Bell -> keyword whose cards the bell affects."""

BELL_MODIFIES_STAT: dict[str, str] = {
    "Battle Bell": "Attack",
    "Frenzy Bell": "Frenzy",
    "Heart Bell": "Health",
    "Blood Bell": "Health",
    "Sun Bell of Health": "Health",
    "Sun Bell of Strength": "Attack",
}
"""Bell -> stat the bell modifies."""

BELLS_AFFECTING_BLING: list[str] = ["Blingsack Bell", "Gold Blade Bell", "Blingsnail Bell"]
"""Bells that change the Bling economy."""

BELL_ADDS_CARD_TO_FIGHT: dict[str, str] = {
    "Gobbler Bell": "Gobbler",
}
"""Bell -> card the bell adds to fights."""

BELL_TARGETS_CARD_TYPE: dict[str, list[str]] = {
    "Bombskull Bell": ["clunkers"],
    "Dread Bell": ["non_boss_enemies", "enemy_clunkers"],
    "Fog Bell": ["non_boss_enemies"],
    "Goat Bell": ["non_boss_enemies"],
    "Frostbourne Bell": ["non_boss_enemies"],
    "Frosthand Bell": ["non_boss_enemies"],
    "Icebourne Bell": ["non_boss_enemies"],
    "Gloom Bell": ["companions", "items"],
    "Battle Bell": ["companions"],
    "Blood Bell": ["companions", "leaders"],
    "Sun Bell of Health": ["leaders"],
    "Sun Bell of Strength": ["items"],
    "Frenzy Bell": ["items"],
}
"""Bell -> card types the bell targets."""

BELL_AFFECTS_MAP_EVENTS: dict[str, list[str]] = {
    # Gloom Bell: card rewards from these events can carry cursed charms
    "Gloom Bell": [
        "Frozen Travellers",
        "Treasure Chest",
        "Gnome Traveller",
        "The Woolly Snail",
        "Charm Merchant",
    ],
}
"""Bell -> map events the bell affects."""

BELLS_APPLYING_ALL_CURSED_CHARMS: list[str] = ["Gloom Bell"]
"""Bells that can apply every cursed charm."""

BELL_INTRODUCES_CROWN: dict[str, str] = {
    "Tyrant Bell": "Cursed Crown",
}
"""Bell -> crown the bell lets appear in a run (distinct from applying a charm)."""
