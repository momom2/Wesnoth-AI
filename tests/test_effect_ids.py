#!/usr/bin/env python3
"""A scenario `[effect]` names an ability or weapon special by `id=`.

Three different weapon specials share the `[chance_to_hit]` tag
(`wesnoth_src/data/core/macros/weapon_specials.cfg`):

    #define WEAPON_SPECIAL_MAGICAL
        [chance_to_hit]
            id=magical
            value=70

and `marksman` / `deflect` are the same tag with another id. Every
ability is likewise `[hides] id=submerge`, `[hides] id=ambush`
(`wesnoth_src/data/core/macros/abilities.cfg`).

Until 2026-09-13 `_apply_effect_to_unit` read the child TAG, so a
granted `magical` was stored as `chance_to_hit` and a granted
`submerge` as `hides`. Nothing consumes those names -- combat asks
`"magical" in weapon.specials` (`wesnoth_ai/combat.py`) and the fog
gate asks `"submerge" in unit.abilities` (`wesnoth_ai/visibility.py`)
-- so both were created and silently inert. `apply_to=new_ability` was
not dispatched at all.

2p Silverhead Crossing, one of the 21 Ladder-pool maps, grants both to
its side-3 Tentacle in a `prestart` `[object]` (351 corpus games play
this map), so these ran wrong in every game on it.

Dependencies: tools.scenario_events, wesnoth_ai.rules.scenario_pool, wesnoth_sim
Dependents:   pytest only
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.scenario_events import _effect_member_ids  # noqa: E402
from tools.replay_extract import parse_wml  # noqa: E402


def _node(text: str):
    return parse_wml(text)


def test_a_member_is_named_by_its_id_not_its_tag():
    """The three `[chance_to_hit]` specials must not collide."""
    root = _node(
        "[specials]\n"
        "[chance_to_hit]\nid=magical\nvalue=70\n[/chance_to_hit]\n"
        "[chance_to_hit]\nid=marksman\nvalue=60\n[/chance_to_hit]\n"
        "[damage]\nid=backstab\nmultiply=2\n[/damage]\n"
        "[/specials]\n")
    assert _effect_member_ids(root.first("specials")) == {
        "magical", "marksman", "backstab"}


def test_a_block_without_an_id_falls_back_to_its_tag():
    """A hand-written scenario may write `[berserk]` with no id; the
    tag IS the identity there."""
    root = _node("[specials]\n[berserk]\nvalue=30\n[/berserk]\n[/specials]\n")
    assert _effect_member_ids(root.first("specials")) == {"berserk"}
    assert _effect_member_ids(None) == set()


def test_silverhead_grants_a_working_submerge_and_magical():
    """End to end on the real scenario: the granted ability reaches the
    fog gate and the granted special reaches combat.

    Both assertions fail against the pre-2026-09-13 code, which stored
    `hides` and `chance_to_hit`.
    """
    from wesnoth_ai.rules.scenario_pool import (ScenarioSetup, build_scenario_gamestate,
                                     load_factions)
    from wesnoth_ai.rules.terrain_resolver import hides_cover
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.visibility import units_visible_to

    factions = load_factions()
    setup = ScenarioSetup(scenario_id="multiplayer_Silverhead_Crossing",
                          faction1="Rebels", faction2="Undead",
                          leader1=factions["Rebels"].leader_pool[0],
                          leader2=factions["Undead"].leader_pool[0])
    sim = WesnothSim(build_scenario_gamestate(setup),
                     scenario_id=setup.scenario_id, max_turns=4)

    tentacles = [u for u in sim.gs.map.units if "Tentacle" in u.name]
    assert tentacles, "Silverhead must field its side-3 Tentacle"
    t = tentacles[0]

    assert "submerge" in {str(a) for a in (t.abilities or ())}, \
        "the [object]'s apply_to=new_ability must grant submerge"

    code = sim.gs.global_info._terrain_codes.get((t.position.x, t.position.y))
    assert hides_cover(code, "submerge"), \
        f"the Tentacle stands on {code!r}, which must be deep water"
    for side in (1, 2):
        assert not any(u.id == t.id for u in units_visible_to(sim.gs, side)), \
            f"side {side} must not see a submerged Tentacle"

    evil_eye = [a for a in t.attacks if a.is_ranged]
    assert evil_eye, "the [object] adds a ranged 'evil eye'"
    assert "magical" in {str(s) for s in (evil_eye[0].weapon_specials or ())}, \
        "the granted [chance_to_hit] id=magical must reach combat as 'magical'"


def test_an_unmodelled_apply_to_is_reported(caplog):
    """Silence is how `new_ability` went missing for months."""
    import logging

    from tools.scenario_events import _APPLY_TO_GAPS_SEEN, _apply_effect_to_unit

    root = _node("[effect]\napply_to=attack_anim_nonsense\n[/effect]\n")
    eff = root.first("effect")
    _APPLY_TO_GAPS_SEEN.discard("attack_anim_nonsense")

    class _U:
        abilities: set = set()
        attacks: list = []
        statuses: set = set()

    with caplog.at_level(logging.WARNING, logger="scenario_events"):
        _apply_effect_to_unit(_U(), eff)
    assert any("attack_anim_nonsense" in r.getMessage() for r in caplog.records), \
        "an apply_to we do not model must warn, not vanish"


@pytest.mark.parametrize("apply_to,expected", [
    ("new_ability", {"regenerate", "submerge"}),
    ("remove_ability", {"regenerate"}),
])
def test_new_and_remove_ability_use_the_id(apply_to, expected):
    from tools.scenario_events import _apply_effect_to_unit

    root = _node(f"[effect]\napply_to={apply_to}\n[abilities]\n"
                 "[hides]\nid=submerge\n[/hides]\n[/abilities]\n[/effect]\n")

    class _U:
        def __init__(self):
            self.abilities = {"regenerate", "submerge"} if apply_to == "remove_ability" \
                else {"regenerate"}
            self.attacks = []
            self.statuses = set()

    u = _U()
    _apply_effect_to_unit(u, root.first("effect"))
    assert set(u.abilities) == expected
