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

Until 2026-09-13 the effect applier read the child TAG, so a granted
`magical` was stored as `chance_to_hit` and a granted `submerge` as
`hides`. Nothing consumes those names -- combat asks for "magical" among
a weapon's specials and the fog gate for "submerge" among a unit's
abilities -- so both were created and silently inert.
`apply_to=new_ability` was not dispatched at all. The Rust core applies
the effects (rust/wesnoth_core/src/effects.rs); these tests drive it.

2p Silverhead Crossing, one of the 21 Ladder-pool maps, grants both to
its side-3 Tentacle in a `prestart` `[object]` (351 corpus games play
this map), so these ran wrong in every game on it.

Dependencies: wesnoth_ai.game_core, wesnoth_ai.rules.scenario_pool, wesnoth_sim
Dependents:   pytest only
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.replay_extract import parse_wml  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402


def _node(text: str):
    return parse_wml(text)


def _applied(effect_wml: str, unit_type: str = "Spearman", **changes) -> dict:
    """A recruit of `unit_type` with `changes`, after the core applied the
    `[effect]` (`wesnoth_core.apply_effect_fields`)."""
    import wesnoth_core
    unit = gc.build_recruit_unit(unit_type, 1, 2, 2, 7, trait_seed_hex="00000001")
    fields = gc.unit_fields(unit)
    fields.update(changes)
    return wesnoth_core.apply_effect_fields(fields, gc.wml_tuple(_node(effect_wml).first("effect")))


def _melee_specials(fields: dict) -> set:
    return next(set(sp) for (_t, _n, _d, ranged, sp) in fields["attacks"] if not ranged)


def test_a_member_is_named_by_its_id_not_its_tag():
    """The three `[chance_to_hit]` specials must not collide."""
    out = _applied(
        "[effect]\napply_to=attack\nrange=melee\n[set_specials]\n"
        "[chance_to_hit]\nid=magical\nvalue=70\n[/chance_to_hit]\n"
        "[chance_to_hit]\nid=marksman\nvalue=60\n[/chance_to_hit]\n"
        "[damage]\nid=backstab\nmultiply=2\n[/damage]\n"
        "[/set_specials]\n[/effect]\n")
    assert {"magical", "marksman", "backstab"} <= _melee_specials(out)
    assert not {"chance_to_hit", "damage"} & _melee_specials(out)


def test_a_block_without_an_id_falls_back_to_its_tag():
    """A hand-written scenario may write `[berserk]` with no id; the
    tag IS the identity there."""
    out = _applied("[effect]\napply_to=attack\nrange=melee\n[set_specials]\n"
                   "[berserk]\nvalue=30\n[/berserk]\n[/set_specials]\n[/effect]\n")
    assert "berserk" in _melee_specials(out)


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


def test_an_unmodelled_apply_to_is_reported():
    """Silence is how `new_ability` went missing for months."""
    import wesnoth_core
    wesnoth_core.drain_warnings()
    _applied("[effect]\napply_to=attack_anim_nonsense\n[/effect]\n")
    assert any("attack_anim_nonsense" in w for w in wesnoth_core.drain_warnings()), \
        "an apply_to we do not model must warn, not vanish"


@pytest.mark.parametrize("apply_to,expected", [
    ("new_ability", {"regenerate", "submerge"}),
    ("remove_ability", {"regenerate"}),
])
def test_new_and_remove_ability_use_the_id(apply_to, expected):
    before = ["regenerate", "submerge"] if apply_to == "remove_ability" else ["regenerate"]
    out = _applied(f"[effect]\napply_to={apply_to}\n[abilities]\n"
                   "[hides]\nid=submerge\n[/hides]\n[/abilities]\n[/effect]\n", abilities=before)
    assert set(out["abilities"]) == expected
