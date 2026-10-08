"""The core's fight outcomes equal `tools/combat_outcomes.py` exactly.

The core (rust/wesnoth_core/src/outcomes.rs) answers the defender's
weapon choice, an attack's outcome distribution with and without its
advancement branches, and a fight's statistics for a view bound to it.
On boards built from the scenario pool with pairs of adjacent enemies of
random unit types (random hit points, experience often one fight from
the cap, slow and poison, the time of day, pick-advance lists, a leader,
an illuminator or a backstabber beside them, a weapon given the
petrifies special), both answer every attack of every pair and must
agree to the last bit and in their order: the chosen weapon, each
candidate's strike table, each distribution.

The comparison runs twice, the second time with the reference's `sum()`
adding left to right as Python 3.11's does: the box image runs 3.11, and a
reference that followed the interpreter's `sum()` diverged from the core on
every 3.11 box (2026-10-01).

The Python applier's enumeration reports an advanced unit unpetrified;
an AMLA leaves a petrified unit petrified (AMLA_DEFAULT cures poison
and slow only) and the core reports it so, so fights with a petrifying
weapon are compared without advancement branches.
"""
from __future__ import annotations

import copy
import functools
import json
import operator
import random

import pytest

from wesnoth_ai import game_core as gc
from wesnoth_ai.classes import Position
from wesnoth_ai.paths import UNIT_STATS_PATH

pytestmark = pytest.mark.skipif(gc.game_core_class() is None,
                                reason="the installed wesnoth_core wheel is older than game_core needs")

_SPECIALS = {"berserk", "swarm", "drains", "poison", "slow", "plague", "firststrike", "backstab",
             "charge", "marksman", "magical", "deflect"}


@pytest.fixture(scope="module")
def db():
    return json.loads(UNIT_STATS_PATH.read_text(encoding="utf-8"))["units"]


def _pools(db) -> dict:
    def having(pred):
        return sorted(n for n, t in db.items() if pred(t))
    return {
        "all": sorted(db),
        "special": having(lambda t: any(_SPECIALS & set(a.get("specials", [])) for a in t.get("attacks", []))),
        "leader": having(lambda t: "leadership" in t.get("abilities", [])),
        "light": having(lambda t: "illuminates" in t.get("abilities", [])),
        "feeder": having(lambda t: "feeding" in t.get("abilities", [])),
        "branching": having(lambda t: len(t.get("advances_to", [])) > 1),
        "berserk": having(lambda t: any("berserk" in a.get("specials", []) for a in t.get("attacks", []))),
        "swarm": having(lambda t: any("swarm" in a.get("specials", []) for a in t.get("attacks", []))),
        "two_of_a_range": having(lambda t: len({a.get("range") for a in t.get("attacks", [])})
                                 < len(t.get("attacks", []))),
    }


def _board(seed: int, db, pools):
    """A pool scenario's board holding fighting pairs, and the pairs'
    positions (attacker, defender)."""
    from tools.abilities import hex_neighbors
    from tools.replay_dataset import _build_recruit_unit
    from wesnoth_ai.rules import scenario_pool as sp
    rng = random.Random(seed)
    gs = sp.build_scenario_gamestate(sp.random_setup(rng))
    gi = gs.global_info
    gi.turn_number = rng.randint(1, 18)
    modifier = rng.choice((70, 100))
    gi._experience_modifier = modifier
    gs.map.units = set()
    free = {(h.position.x, h.position.y) for h in gs.map.hexes}
    uid = [1]

    def place(name, side, pos):
        u = _build_recruit_unit(name, side, pos[0], pos[1], uid[0], game_id=gs.game_id,
                                trait_seed_hex=f"{rng.getrandbits(32):08x}", exp_modifier=modifier)
        uid[0] += 1
        u.current_hp = rng.randint(1, u.max_hp)
        u.current_exp = max(0, u.max_exp - rng.choice((1, 1, 2, 4, 8, 9, 17, u.max_exp)))
        for status, p in (("slowed", 0.2), ("poisoned", 0.2)):
            if rng.random() < p:
                u.statuses.add(status)
        targets = db[name].get("advances_to", [])
        if len(targets) > 1 and rng.random() < 0.5:
            u._pickadvance = rng.sample(targets, rng.randint(1, len(targets) - 1)) + ["Not A Unit Type"]
        if u.attacks and rng.random() < 0.05:
            u.attacks[0].weapon_specials = sorted(set(u.attacks[0].weapon_specials or ()) | {"petrifies"})
        gs.map.units.add(u)
        free.discard(pos)
        return u

    def free_neighbours(pos):
        return [p for p in hex_neighbors(*pos) if p in free]

    pairs = []
    for k in range(14):
        starts = sorted(p for p in free if free_neighbours(p))
        if not starts:
            break
        a_pos = rng.choice(starts)
        d_pos = rng.choice(free_neighbours(a_pos))
        a_pool = pools["special"] if rng.random() < 0.5 else pools["all"]
        if k < 2:
            a_pool = pools[("berserk", "swarm")[k]]
        d_pool = rng.choice((pools["branching"], pools["two_of_a_range"], pools["all"]))
        if rng.random() < 0.1:
            a_pool = pools["feeder"]
        place(rng.choice(a_pool), 1, a_pos)
        place(rng.choice(d_pool), 2, d_pos)
        pairs.append((a_pos, d_pos))
        for pool, side, near in (("leader", 1, a_pos), ("leader", 2, d_pos), ("light", 1, a_pos)):
            spots = free_neighbours(near)
            if spots and rng.random() < 0.25:
                place(rng.choice(pools[pool]), side, rng.choice(spots))
        behind = [p for p in free_neighbours(d_pos) if p not in hex_neighbors(*a_pos)]
        if behind and rng.random() < 0.3:
            place(rng.choice(pools["all"]), 1, rng.choice(behind))
    # Game-wide pick-advance lists: a unit advancing into one of these
    # types is offered part of its advancements.
    gi._pickadvance_game = {}
    for name in rng.sample(pools["branching"], 12):
        targets = db[name]["advances_to"]
        gi._pickadvance_game[(rng.choice((1, 2)), name)] = rng.sample(targets, len(targets) - 1)
    return gs, pairs


def _unit_at(gs, pos):
    return next(u for u in gs.map.units if (u.position.x, u.position.y) == pos)


def _items(d) -> list:
    return list(d.items())


def _petrifying(u) -> bool:
    return any("petrifies" in (a.weapon_specials or ()) for a in u.attacks)


def _plain_sum(values, start=0):
    """Python 3.11's `sum()` of floats: left to right, uncompensated."""
    return functools.reduce(operator.add, values, start)


@pytest.mark.parametrize("interpreter_sum", ["this interpreter's", "Python 3.11's"])
def test_fight_outcomes_equal_the_python(db, interpreter_sum, monkeypatch):
    from tools import combat_outcomes as co
    from tools.replay_dataset import build_attack_context
    if interpreter_sum == "Python 3.11's":
        monkeypatch.setattr(co, "sum", _plain_sum, raising=False)
    pools = _pools(db)
    seen = {"attacks": 0, "choices": 0, "advanced": 0, "amla": 0, "berserk": 0, "pick": 0, "exact": 0}
    for seed in range(12):
        gs, pairs = _board(seed, db, pools)
        cs = gc.CoreState.from_state(copy.deepcopy(gs))
        core = cs.core
        for a_pos, d_pos in pairs:
            att, dfd = _unit_at(gs, a_pos), _unit_at(gs, d_pos)
            for w in range(len(att.attacks)):
                seen["attacks"] += 1
                where = (seed, a_pos, att.name, d_pos, dfd.name, w)
                py_weapon, py_tables = co.counter_weapon_choice(gs, att, dfd, w)
                rs_weapon, rs_tables, _fallback = core.counter_weapon_choice(*a_pos, *d_pos, w)
                assert rs_weapon == py_weapon, where
                assert [(i, _items(t)) for i, t in rs_tables.items()] == \
                    [(i, _items(t)) for i, t in py_tables.items()], where
                seen["choices"] += len(py_tables) > 1
                a_st, d_st = co._stats_pair(build_attack_context(gs, att, dfd, w, py_weapon))
                rs_a, rs_d = core.fight_stats(*a_pos, *d_pos, w, py_weapon)
                for st, rs in ((a_st, rs_a), (d_st, rs_d)):
                    assert (st is None) == (rs is None), where
                    if st is not None:
                        assert {k: getattr(st, k) for k in rs} == rs, where
                seen["berserk"] += a_st.rounds > 1
                action = {"type": "attack", "start_hex": Position(*a_pos), "target_hex": Position(*d_pos),
                          "attack_index": w}
                choices = [None] if _petrifying(att) or _petrifying(dfd) else [None, "uniform"]
                for choice in choices:
                    py = co.enumerate_attack_outcomes(gs, action, advancement_choice=choice)
                    rs = core.attack_outcomes(*a_pos, *d_pos, w, choice == "uniform")
                    assert (py is None) == (rs is None), (where, choice)
                    if py is None:
                        continue
                    assert _items(rs[0]) == _items(py.probs), (where, choice)
                    assert (rs[1], rs[2]) == (py.attacker_id, py.defender_id), where
                    seen["exact"] += 1
                    for key in py.probs:
                        for name, hp, u in ((key[8], key[0], att), (key[9], key[1], dfd)):
                            if name and name != u.name:
                                seen["advanced"] += 1
                                seen["pick"] += bool(getattr(u, "_pickadvance", None))
                            elif name and hp > u.max_hp + 1:
                                seen["amla"] += 1
    assert seen["choices"] > 40 and seen["exact"] > 200, seen
    assert seen["advanced"] > 250 and seen["pick"] > 30 and seen["amla"] > 150 and seen["berserk"] > 5, seen


def test_a_bound_view_is_answered_by_its_core():
    """`combat_outcomes` routes a bound view to the core: a view whose
    core holds a different defender gets the core's answer."""
    from tools import combat_outcomes as co
    from tools.replay_dataset import _build_recruit_unit
    from wesnoth_ai.rules import scenario_pool as sp
    gs = sp.build_scenario_gamestate(sp.random_setup(random.Random(2)))
    gs.map.units = set()
    hexes = sorted((h.position.x, h.position.y) for h in gs.map.hexes)
    from tools.abilities import hex_neighbors
    a_pos = next(p for p in hexes if any(q in set(hexes) for q in hex_neighbors(*p)))
    d_pos = next(q for q in hex_neighbors(*a_pos) if q in set(hexes))
    gs.map.units.add(_build_recruit_unit("Elvish Fighter", 1, *a_pos, 1, game_id=gs.game_id))
    gs.map.units.add(_build_recruit_unit("Orcish Grunt", 2, *d_pos, 2, game_id=gs.game_id))
    other = copy.deepcopy(gs)
    other.map.units = {u for u in other.map.units if u.side == 1}
    other.map.units.add(_build_recruit_unit("Orcish Archer", 2, *d_pos, 2, game_id=gs.game_id))
    cs = gc.CoreState.from_state(other)
    gc.bind_view(gs, cs)
    try:
        att, dfd = _unit_at(gs, a_pos), _unit_at(gs, d_pos)
        # The Elvish Fighter's bow (weapon 1) meets the Archer's bow in the
        # core and nothing on the Grunt the view shows.
        assert co.choose_counter_weapon(gs, att, dfd, 1) == cs.core.counter_weapon_choice(*a_pos, *d_pos, 1)[0]
        assert co.choose_counter_weapon(gs, att, dfd, 1) >= 0
    finally:
        gc._VIEW_CORES.pop(id(gs), None)

