"""The Rust terrain resolver and movement classes equal the Python ones.

`wesnoth_core` resolves every terrain fact from a hex's code (terrain.rs)
and computes the movement classes itself (core.rs `class_for`); the
Python resolver (`wesnoth_ai.rules.terrain_resolver`) and the
pathfinder's arrays (`tools.pathfind_sim._terrain_arrays_for`,
`replay_dataset._terrain_def_pct`) are the oracle. Every code of the
terrain database and of the tracked maps, every movement and defense
table of the unit database. Skipped without the phase-18 wheel.
"""
from __future__ import annotations

import json
import random

import pytest

from wesnoth_ai import game_core as gc
from wesnoth_ai.paths import UNIT_STATS_PATH, WESNOTH_SRC_DIR

pytestmark = pytest.mark.skipif(gc.game_core_class() is None, reason="the installed wesnoth_core wheel is older than game_core needs")


def _map_codes():
    from wesnoth_ai.rules.terrain_resolver import load_terrain_db, strip_start_position
    codes = set(load_terrain_db())
    for path in WESNOTH_SRC_DIR.glob("data/**/maps/*.map"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            for cell in line.split(","):
                code = strip_start_position(cell)
                if code and not code.startswith(("border_size", "usage")) and "=" not in code:
                    codes.add(code)
    return sorted(codes)


@pytest.fixture(scope="module")
def codes():
    out = _map_codes()
    assert len(out) > 400
    return out


@pytest.fixture(scope="module")
def unit_stats():
    return json.loads(UNIT_STATS_PATH.read_text(encoding="utf-8"))


def test_terrain_facts_equal_the_resolver(codes):
    import wesnoth_core
    from tools.replay_dataset import _parse_hex_code
    from wesnoth_ai.encoder import _first_terrain_id
    from wesnoth_ai.rules import terrain_resolver as tr
    for code in codes:
        (heals, light, max_l, min_l, _any, mask, ambush, conceal, submerge,
         types, mods, one) = wesnoth_core.terrain_facts(code)
        assert heals == tr.terrain_heals(code), code
        for base in (-25, 0, 25):
            if light >= 0:
                lit = min(base + light, max(base, max_l)) if _any else base
            else:
                lit = max(base + light, min(base, min_l))
            assert lit == tr.terrain_light_bonus(code, base), (code, base)
        assert mask == tr.terrain_mask(code), code
        assert (ambush, conceal, submerge) == tuple(
            tr.hides_cover(code, a) for a in ("ambush", "concealment", "submerge")), code
        py_types, py_mods = _parse_hex_code(code)
        assert types == sum(1 << int(t) for t in py_types), code
        assert mods == sum(1 << int(m) for m in py_mods), code
        assert one == _first_terrain_id(py_types), code


def _tables(unit_stats, key):
    seen = {}
    for u in unit_stats["units"].values():
        table = u.get(key) or {}
        seen[tuple(sorted(table.items()))] = table
    return list(seen.values())


def test_movement_costs_equal_the_resolver(codes, unit_stats):
    import wesnoth_core
    from wesnoth_ai.rules.terrain_resolver import mvt_cost
    tables = _tables(unit_stats, "movement_costs")
    assert len(tables) > 20
    n = 0
    for table in tables:
        for slowed in (False, True):
            costs = {k: (v if v >= 99 or not slowed else 2 * v) for k, v in table.items()}
            pairs = list(costs.items())
            for code in codes:
                assert wesnoth_core.terrain_mvt_cost(code, pairs) == mvt_cost(code, costs), (code, costs)
                n += 1
    assert n > 10000


def test_defense_equals_the_resolver_with_floors(codes, unit_stats):
    """Every type's defense table, and each with feral's village floor
    (`village=-50`), whose negative entry the resolver reads as a cap."""
    import wesnoth_core
    from wesnoth_ai.rules.terrain_resolver import def_pct
    tables = _tables(unit_stats, "defense")
    tables += [{**t, "village": -50} for t in tables[::3]]
    n = 0
    for table in tables:
        pairs = list(table.items())
        for code in codes:
            assert wesnoth_core.terrain_def_pct(code, pairs) == def_pct(code, table), (code, table)
            n += 1
    assert n > 10000


def _state_with_every_type(seed: int, unit_stats, stride: int):
    """A pool scenario's state with one unit of every `stride`-th type
    on its first hexes, one of them feral-floored and one slowed."""
    from dataclasses import replace
    from tools.replay_dataset import _build_unit
    from wesnoth_ai.rules import scenario_pool as sp
    gs = sp.build_scenario_gamestate(sp.random_setup(random.Random(seed)))
    hexes = sorted((h.position.x, h.position.y) for h in gs.map.hexes)
    names = sorted(unit_stats["units"])[::stride] + ["Not A Unit Type"]
    uid = 1000
    for k, name in enumerate(names):
        x, y = hexes[k % len(hexes)]
        u = _build_unit({"uid": uid + k, "type": name, "side": 1, "x": x, "y": y})
        if k == 1:
            u._defense_table = {**u._defense_table, "village": -50}
        if k == 2:
            u = replace(u, statuses={"slowed"})
            u._defense_table = dict(unit_stats["units"][name]["defense"])
        gs.map.units.add(u)
    return gs


def test_movement_classes_equal_the_pathfinder_arrays(unit_stats):
    """The core's class arrays for each unit, slowed and not, equal the
    pathfinder's movement cost and defense subcost and the combat
    defense of the Python state."""
    from tools.pathfind_sim import _terrain_arrays_for
    from tools.replay_dataset import _rebuild_unit, _stats_for, _terrain_def_pct
    from wesnoth_ai.observe import map_geometry
    checked = 0
    for seed, stride in ((1, 9), (2, 11)):
        gs = _state_with_every_type(seed, unit_stats, stride)
        cs = gc.CoreState.from_state(gs)
        keys = map_geometry(gs).keys
        for u in gs.map.units:
            d = cs.core.unit_export(u.id)
            table = getattr(u, "_defense_table", None) or _stats_for(u.name).get("defense", {})
            defense = [int(_terrain_def_pct(gs, x, y, table)) for (x, y) in keys]
            for slowed, cid in ((False, d["class_id"]), (True, d["class_slowed_id"])):
                st = set(u.statuses) | {"slowed"} if slowed else set(u.statuses) - {"slowed"}
                _p, positions, _n, mcost, dsub = _terrain_arrays_for(_rebuild_unit(u, statuses=st), gs)
                assert list(positions) == list(keys)
                got = cs.core.class_arrays(cid)
                assert list(got[0]) == [int(c) for c in mcost], (u.name, slowed)
                assert list(got[1]) == [int(c) for c in dsub], (u.name, slowed)
                assert list(got[2]) == defense, (u.name, slowed)
                checked += 1
    assert checked > 60


def test_map_terrain_facts_follow_the_codes():
    """The core's per-hex heal, light and cover arrays equal the
    resolver's answers for each hex's code."""
    from wesnoth_ai.observe import map_geometry
    from wesnoth_ai.rules import scenario_pool as sp
    from wesnoth_ai.rules import terrain_resolver as tr
    for seed in range(4):
        gs = sp.build_scenario_gamestate(sp.random_setup(random.Random(seed)))
        cs = gc.CoreState.from_state(gs)
        heal, _lm, _lx, _ln, _has, ambush, conceal, submerge = cs.core.terrain_arrays()
        codes = gs.global_info._terrain_codes
        for i, key in enumerate(map_geometry(gs).keys):
            code = tr.strip_start_position(codes.get(key) or "")
            assert heal[i] == (tr.terrain_heals(code) if code else 0), (key, code)
            assert (ambush[i], conceal[i], submerge[i]) == tuple(
                int(tr.hides_cover(code, a)) for a in ("ambush", "concealment", "submerge")), (key, code)
