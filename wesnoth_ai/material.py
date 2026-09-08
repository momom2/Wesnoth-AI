"""Material: cost-weighted HP fraction, the mover's units minus the
enemy units the mover can see (the user's metric, 2026-09-06).

Two consumers: the encoder records it on every RawEncoded (the value
head reads it when the model is built with `value_material`), and the
per-phase value-head study compares it with the head against human
outcomes. One function, so the two never drift.
"""
from typing import Iterable

# The value head receives material / MATERIAL_SCALE: a typical
# mid-game margin is tens of gold-equivalents, a decided game a few
# hundred, so the input sits within a few units of zero.
MATERIAL_SCALE = 100.0


def material_of_units(units: Iterable, side: int) -> float:
    """Sum over `units` of cost x current HP / max HP, signed + for
    `side` and - for its opponents (sides other than 1 and 2 are
    ignored, as are units without hit points)."""
    total = 0.0
    for u in units:
        if u.side not in (1, 2) or u.max_hp <= 0:
            continue
        v = float(u.cost) * float(u.current_hp) / float(u.max_hp)
        total += v if u.side == side else -v
    return total


def material_score(game_state, side: int) -> float:
    """Material from `side`'s point of view over the units it can see
    (wesnoth_ai.visibility: own units, enemies inside the vision
    disc, hidden units excluded; every unit when the game has fog
    off)."""
    from wesnoth_ai.visibility import units_visible_to
    return material_of_units(units_visible_to(game_state, side), side)
