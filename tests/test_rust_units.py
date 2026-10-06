"""The core's unit construction equals the Python builders.

`wesnoth_core` builds units itself (units.rs, core_units.rs, effects.rs):
`_build_unit` with and without the leader traits, `_build_recruit_unit`
with a seeded and a legacy (hashed) trait roll, `_build_plague_corpse`,
`_apply_effect_to_unit`, and `_maybe_advance_unit` with its choice
queue, pick-advance lists, uniform draw, AMLA, variation, feeding and
[object] effects. Every unit type of the database. The Python builders
are the oracle until the port's certification. Skipped without the
phase-19 wheel.
"""
from __future__ import annotations

import copy
import json
import random
from types import SimpleNamespace

import pytest

from wesnoth_ai import game_core as gc
from wesnoth_ai.paths import UNIT_STATS_PATH

pytestmark = pytest.mark.skipif(gc.game_core_class() is None, reason="the installed wesnoth_core wheel is older than game_core needs")


@pytest.fixture(scope="module")
def type_names():
    names = sorted(json.loads(UNIT_STATS_PATH.read_text(encoding="utf-8"))["units"])
    assert len(names) > 300
    return names + ["Not A Unit Type"]


def _norm(fields: dict) -> dict:
    """A unit field dict in comparable form: sets sorted, the defense
    table a dict, the core-only class ids dropped."""
    out = dict(fields)
    for k in ("abilities", "traits", "statuses"):
        out[k] = sorted(out[k])
    out["attacks"] = [(t, n, d, r, sorted(sp)) for (t, n, d, r, sp) in out["attacks"]]
    if out.get("defense_table") is not None:
        out["defense_table"] = dict(out["defense_table"])
    out["object_effects"] = [repr(e) for e in out.get("object_effects") or ()]
    out.pop("class_id", None)
    out.pop("class_slowed_id", None)
    return out


def _py(u) -> dict:
    return _norm(gc.unit_fields(u))


def test_build_unit_equals_the_python_builder(type_names):
    import wesnoth_core
    from tools.replay_dataset import _build_unit
    records = []
    for k, name in enumerate(type_names):
        base = {"uid": 3 + k, "type": name, "side": 1 + k % 2, "x": k % 7, "y": k % 5}
        records.append((dict(base, is_leader=True), True, 100))
        records.append((dict(base, is_leader=False), False, 70))
        if k % 17 == 0:
            records.append((dict(base, is_leader=True, max_hp=41, hp=30, cost=20), True, 30))
            records.append((dict(base, petrified=True), False, 100))
    for rec, leader_traits, exp_mod in records:
        py = _build_unit(dict(rec), apply_leader_traits=leader_traits, game_id="g7", exp_modifier=exp_mod)
        rs = wesnoth_core.build_unit_fields(rec, leader_traits, "g7", exp_mod)
        assert _norm(rs) == _py(py), rec


def test_recruits_and_corpses_equal_the_python_builders(type_names):
    """A recruit of every type under three seeds and the legacy hash, and
    the corpse each type's death raises. A seed of one recruit does not
    reproduce another's traits (the comparison is not vacuous)."""
    import wesnoth_core
    from tools.replay_dataset import _build_plague_corpse, _build_recruit_unit
    rng = random.Random(7)
    differs = 0
    for k, name in enumerate(type_names):
        seeds = [f"{rng.getrandbits(32):08x}" for _ in range(3)] + [""]
        for seed in seeds:
            py = _build_recruit_unit(name, 2, 4, 5, 40 + k, game_id="g1", trait_seed_hex=seed, exp_modifier=70)
            rs = wesnoth_core.build_recruit_fields(name, 2, 4, 5, 40 + k, "g1", seed, 70)
            assert _norm(rs) == _py(py), (name, seed)
        other = _build_recruit_unit(name, 2, 4, 5, 40 + k, game_id="g1", trait_seed_hex="0badf00d", exp_modifier=70)
        differs += _py(other)["traits"] != _norm(rs)["traits"]
        py = _build_plague_corpse(name, 1, 3, 3, 90 + k, "g2", 100)
        rs = wesnoth_core.build_corpse_fields(name, 1, 3, 3, 90 + k, "g2", 100)
        assert _norm(rs) == _py(py), name
    assert differs > 50


def _node(tag, attrs=(), children=()):
    from tools.replay_extract import WMLNode
    n = WMLNode(tag)
    n.attrs = dict(attrs)
    n.children = list(children)
    return n


def _effects():
    """[effect] nodes over every modelled apply_to, with percentages,
    negative and malformed values, quoting and member ids."""
    N = _node
    specials = N("set_specials", children=[N("firststrike", {"id": "firststrike"}), N("chance_to_hit", {"id": '"magical"'})])
    abilities = N("abilities", children=[N("hides", {"id": "submerge"}), N("regenerate")])
    return [
        N("effect", {"apply_to": "attack", "range": "ranged", "increase_damage": "50%"}, [specials]),
        N("effect", {"apply_to": "attack", "range": "melee", "increase_attacks": "-1", "increase_damage": "-100%"}),
        N("effect", {"apply_to": "attack", "increase_damage": '" 3"'}),
        N("effect", {"apply_to": "attack", "increase_attacks": "x"}),
        N("effect", {"apply_to": "new_attack", "range": "ranged", "type": "fire", "damage": "7", "number": "3"},
          [N("specials", children=[N("magical", {"id": "magical"})])]),
        N("effect", {"apply_to": "new_attack", "damage": "bad", "number": ""}),
        N("effect", {"apply_to": "remove_attacks"}),
        N("effect", {"apply_to": "hitpoints", "increase_total": "-100%"}),
        N("effect", {"apply_to": "hitpoints", "increase_total": "5", "set": "1"}),
        N("effect", {"apply_to": "hitpoints", "set": '"3"', "heal_full": "Yes"}),
        N("effect", {"apply_to": "movement", "set": "0"}),
        N("effect", {"apply_to": "movement", "increase": "-50%"}),
        N("effect", {"apply_to": "movement", "set": "q", "increase": "2"}),
        N("effect", {"apply_to": "status", "add": "petrified"}),
        N("effect", {"apply_to": "status", "add": '"slowed"', "remove": "resting"}),
        N("effect", {"apply_to": "new_ability"}, [abilities]),
        N("effect", {"apply_to": "remove_ability"}, [abilities]),
        N("effect", {"apply_to": "movement_costs"}),
        N("effect", {"apply_to": "image_mod"}),
    ]


def test_effects_equal_the_python_applier(type_names):
    import wesnoth_core
    from tools.replay_dataset import _build_recruit_unit
    from tools.scenario_events import _apply_effect_to_unit
    changed = 0
    for k, name in enumerate(type_names[::23]):
        base = _build_recruit_unit(name, 1 + k % 3, 2, 2, 10 + k, trait_seed_hex=f"{k:08x}")
        base.current_moves = max(0, base.current_moves - 1)
        for eff in _effects():
            py = copy.copy(base)
            _apply_effect_to_unit(py, eff)
            rs = wesnoth_core.apply_effect_fields(gc.unit_fields(base), gc.wml_tuple(eff))
            assert _norm(rs) == _py(py), (name, eff.attrs)
            changed += _py(py) != _py(base)
    assert changed > 100


def _carrier(unit, *, choices, uniform, salt, counter, pick, exp_mod):
    gi = SimpleNamespace(_experience_modifier=exp_mod, _advance_choices=list(choices), _last_advance_events=[],
                         _advance_uniform=uniform, _advance_salt=salt, _advance_counter=counter,
                         _pickadvance_game={(s, t): list(v) for (s, t, v) in pick})
    return SimpleNamespace(game_id="g", map=SimpleNamespace(units={unit}), global_info=gi)


def _advance_cases(type_names):
    """Units at or past their experience cap: every type that advances
    or AMLAs, with traits in roll order, a feeding count, a pick-advance
    list, a persistent [object] effect and several choice queues."""
    from tools.replay_dataset import _build_recruit_unit
    rng = random.Random(3)
    obj = _node("effect", {"apply_to": "attack", "range": "ranged"},
                [_node("set_specials", children=[_node("firststrike", {"id": "firststrike"})])])
    cases = []
    for k, name in enumerate(type_names):
        if k % 3:
            continue
        u = _build_recruit_unit(name, 1 + k % 2, 3, 4, 20 + k, trait_seed_hex=f"{rng.getrandbits(32):08x}",
                                exp_modifier=50)
        u.current_exp = u.max_exp * (1 + k % 3) + k % 4
        u.current_moves = k % 4
        if k % 5 == 0:
            u._feeding_count = 1 + k % 3
        if k % 7 == 0:
            u._object_effects = [obj]
        targets = list(json.loads(UNIT_STATS_PATH.read_text(encoding="utf-8"))["units"].get(name, {})
                       .get("advances_to", []))
        pick = []
        if len(targets) > 1 and k % 2 == 0:
            u._pickadvance = targets[1:]
            pick = [(u.side, targets[-1], targets[:1])]
        for choices, uniform in (([], False), ([1, 0, 1], False), ([-1], False), ([], True)):
            cases.append((u, choices, uniform, pick))
    return cases


@pytest.mark.slow          # 9-11 s on CI: see pytest.ini two-tier note
def test_advancement_equals_the_python_applier(type_names):
    """`_maybe_advance_unit` on a carrier and the core's advancement from
    the same unit and globals: the same unit, choice queue, events and
    draw counter."""
    from tools.replay_dataset import _maybe_advance_unit
    advanced = amla = 0
    for u, choices, uniform, pick in _advance_cases(type_names):
        carrier = _carrier(copy.deepcopy(u), choices=choices, uniform=uniform, salt="s1" if uniform else "",
                           counter=4, pick=pick, exp_mod=50)
        py = _maybe_advance_unit(carrier, next(iter(carrier.map.units)))
        gs = gc.CoreState.from_state(_one_unit_state(u))
        gs.core.set_advance_state(list(choices), [(s, t, list(v)) for (s, t, v) in pick], [])
        gs.core.set_global_int("advance_uniform", int(uniform))
        gs.core.set_global_int("advance_counter", 4)
        if uniform:
            gs.core.set_advance_salt("s1")
        gs.core.advance_unit_id(u.id)
        rs = gs.core.unit_export(u.id)
        assert _norm(rs) == _py(py), (u.name, choices, uniform)
        c_choices, _p, c_events = gs.core.advance_state_export()
        gi = carrier.global_info
        assert list(c_choices) == [c if isinstance(c, int) else -1 for c in gi._advance_choices], u.name
        assert [tuple(e) for e in c_events] == [tuple(e) for e in gi._last_advance_events], u.name
        assert gs.core.globals_export()["advance_counter"] == gi._advance_counter, u.name
        advanced += py.name != u.name
        amla += py.name == u.name and py.max_hp > u.max_hp
    assert advanced > 100 and amla > 10


def _one_unit_state(u):
    """A small scenario state holding only `u` (on one of its hexes)."""
    from wesnoth_ai.rules import scenario_pool as sp
    gs = sp.build_scenario_gamestate(sp.random_setup(random.Random(5)))
    gs.map.units = {copy.deepcopy(u)}
    gs.global_info._experience_modifier = 50
    return gs
