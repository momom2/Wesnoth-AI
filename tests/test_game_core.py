"""The Rust-owned game state round-trips the Python state.

`CoreState.from_state(gs).to_state()` must equal `gs` over every modeled
field (units field for field including statuses, traits, abilities and
attacks; sides; the turn scalars; the village owners, the uncovered and
rejected sets, the advancement queue, the last walk and strikes) and
share the hex set by identity. A fork must not share dynamic state
with its parent. The core's state key must agree with itself on equal
states and change with any modeled field. Skipped without the phase-13
wheel.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from wesnoth_ai import game_core as gc

pytestmark = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")


def _harvested():
    from tests.test_enumerate_vectorized import _states as harvest
    return harvest(n_attack=3, n_plain=4)


def _replay_states(n_games=2, per_game=6):
    from pathlib import Path
    from tools.replay_dataset import filter_competitive_2p, iter_replay_pairs
    out = []
    root = next((Path(d) for d in ("replays_dataset", "replays_dataset_imitation") if Path(d).exists()), None)
    if root is None:
        return out
    for gz in filter_competitive_2p(root)[:n_games]:
        k = 0
        for gs, _ai in iter_replay_pairs(gz):
            if k % 40 == 0:
                out.append(copy.deepcopy(gs))
            k += 1
            if len(out) >= per_game * n_games:
                break
    return out


def _decorate(gs):
    """Exercise the stash paths: rejections, uncovered hiders, a walk,
    strikes, advancement choices."""
    gi = gs.global_info
    hexes = sorted(gs.map.hexes, key=lambda h: (h.position.y, h.position.x))
    gi._recruit_rejected_hexes = {(hexes[3].position.x, hexes[3].position.y)}
    gi._move_rejected_hexes = {(hexes[5].position.x, hexes[5].position.y)}
    units = sorted(gs.map.units, key=lambda u: u.id)
    gi._uncovered_units = {units[0].id} if units else set()
    gi._last_move_walk = {"ordered": (1, 2), "landed": (1, 3), "stop_reason": "ambush"}
    gi._last_checkup_strikes = [{"chance": 60, "hits": True, "damage": 7}, {"dies": False}]
    gi._advance_choices = [1, 0]
    gi._pickadvance_game = {(1, "Spearman"): ["Swordsman"]}
    gi._last_advance_events = [(1, 0)]
    gi._rng_request_counter = 5
    if units:
        setattr(units[0], "_feeding_count", 2)
    return gs


def test_round_trip_equals_the_state():
    checked = 0
    for gs in _harvested() + _replay_states():
        gs = _decorate(gs)
        cs = gc.CoreState.from_state(gs)
        back = cs.to_state()
        diffs = gc.states_equal(gs, back)
        assert not diffs, "\n".join(diffs[:8])
        assert back.map.hexes is gs.map.hexes and back.map.mask is gs.map.mask
        assert back.global_info._terrain_codes is gs.global_info._terrain_codes
        assert cs.core.n_units() == len(gs.map.units)
        assert cs.core.n_classes() >= 2
        checked += 1
    assert checked >= 8


def test_fork_is_isolated_and_keys_follow_content():
    gs = _decorate(_harvested()[0])
    cs = gc.CoreState.from_state(gs)
    twin = gc.CoreState.from_state(copy.deepcopy(gs))
    assert cs.state_key() == twin.state_key()
    fork = cs.fork()
    assert fork.state_key() == cs.state_key()
    uid = sorted(cs.core.unit_ids())[0]
    before = cs.core.unit_export(uid)["current_hp"]
    fork.core.update_unit(uid, {"current_hp": max(1, before - 1)})
    assert cs.core.unit_export(uid)["current_hp"] == before
    assert fork.state_key() != cs.state_key()
    fork.core.update_unit(uid, {"current_hp": before})
    assert fork.state_key() == cs.state_key()
    # Unlike the Python fork (which aliases Unit objects and relies on
    # the replace-unit pattern, tests/test_fork_isolation.py), a core
    # fork materializes its own units: no leaf object is shared.
    parent_units = {u.id: u for u in cs.to_state().map.units}
    assert all(parent_units[u.id] is not u for u in fork.to_state().map.units)
    fork.core.remove_unit(uid)
    assert cs.core.n_units() == fork.core.n_units() + 1
    assert uid in cs.core.unit_ids() and uid not in fork.core.unit_ids()
    assert not gc.states_equal(gs, cs.to_state())


def _apply_both(gs, cmd):
    """The command on the Python state and on a core built from a
    deep copy (its own event latches); (python state, core wrapper, path)."""
    from tools.replay_dataset import _apply_command
    py = copy.deepcopy(gs)
    cs = gc.CoreState.from_state(copy.deepcopy(gs))
    _apply_command(py, list(cmd))
    path = cs.apply_command(list(cmd))
    return py, cs, path


def test_init_side_and_end_turn_equal_the_python_applier():
    rust = checked = 0
    for gs in _harvested() + _replay_states():
        for side in (1, 2):
            py, cs, path = _apply_both(gs, ["init_side", side])
            rust += path == "rust"
            diffs = gc.states_equal(py, cs.to_state(), stash=False)
            assert not diffs, (side, "\n".join(diffs[:6]))
            py2, cs2, path2 = _apply_both(py, ["end_turn"])
            assert path2 == "rust"
            diffs = gc.states_equal(py2, cs2.to_state(), stash=False)
            assert not diffs, (side, "end_turn", "\n".join(diffs[:6]))
            checked += 1
    assert checked >= 16 and rust >= 8


def test_lawful_bonus_equals_the_python_helper():
    from tools.replay_dataset import _lawful_bonus_at
    n = 0
    for gs in _harvested()[:3] + _replay_states(n_games=2, per_game=2):
        cs = gc.CoreState.from_state(gs)
        hexes = sorted(gs.map.hexes, key=lambda h: (h.position.y, h.position.x))
        for h in hexes[::37]:
            for turn in (1, 2, 5, 6, 9):
                assert cs.core.lawful_bonus(h.position.x, h.position.y, turn) == \
                    _lawful_bonus_at(gs, h.position.x, h.position.y, turn), (h.position, turn)
                n += 1
    assert n > 100


def test_move_and_attack_equal_the_python_applier():
    """Whole replays through both appliers, compared every fifth
    command and after every init_side and attack (tools/diff_core)."""
    from collections import Counter
    from pathlib import Path
    from tools.diff_core import diff_core
    from tools.replay_dataset import filter_competitive_2p
    root = next((Path(d) for d in ("replays_dataset", "replays_dataset_imitation") if Path(d).exists()), None)
    if root is None:
        pytest.skip("no replay corpus")
    counts = Counter()
    for gz in filter_competitive_2p(root)[:3]:
        assert diff_core(gz, every=5, counts=counts) == []
    assert counts[("move", "rust")] >= 100 and counts[("attack", "rust")] >= 20


def _states_for_encoding():
    return _harvested() + _replay_states(n_games=2, per_game=3)


def _vocab_of(states):
    """A unit-type vocab over the states' names but the last one (out
    of vocab, the overflow bucket) and the factions of their sides."""
    names = sorted({u.name for gs in states for u in gs.map.units})
    factions = sorted({s.faction for gs in states for s in gs.sides if s.faction})
    return {n: i for i, n in enumerate(names[:-1])}, {f: i for i, f in enumerate(factions)}


def _assert_observations_equal(py_obs, core_obs):
    """Observation records equal up to the unit order (the Python one
    follows the unit set's iteration order)."""
    from wesnoth_ai.observe import _ARRAY_FIELDS
    assert (py_obs.side, py_obs.fog_on, py_obs.leader_on_keep) == \
        (core_obs.side, core_obs.fog_on, core_obs.leader_on_keep)
    assert sorted(py_obs.unit_ids) == sorted(core_obs.unit_ids)
    perm = [py_obs.unit_ids.index(uid) for uid in core_obs.unit_ids]
    per_unit = ("unit_hex", "visible", "acting", "unit_can_move", "unit_can_attack", "landable")
    for k in _ARRAY_FIELDS:
        a, b = getattr(py_obs, k), getattr(core_obs, k)
        if a is None or b is None:
            assert a is None and b is None, k
            continue
        if k in per_unit:
            a = a[perm]
        assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), k


def test_observation_from_core_equals_observe():
    from wesnoth_ai.observe import observe
    n = 0
    for gs in _states_for_encoding():
        for fog in (True, False):
            gs.global_info._fog = fog
            cs = gc.CoreState.from_state(gs)
            for side in (1, 2):
                for reach in (False, True):
                    py = observe(gs, side, reach=reach)
                    assert py is not None
                    _assert_observations_equal(py, cs.observe(side, reach=reach))
                    n += 1
    assert n >= 32


def test_encode_raw_from_core_is_byte_identical():
    """Every RawEncoded field from the core equals the encoder's on the
    Python state: both bases, fog on and off, the enemy-village gate."""
    import dataclasses
    from wesnoth_ai.encoder import RawEncoded, encode_raw
    states = _states_for_encoding()
    type_to_id, faction_to_id = _vocab_of(states)
    n = 0
    for gs in states:
        for fog in (True, False):
            gs.global_info._fog = fog
            for side in (1, 2):
                gs.global_info.current_side = side
                cs = gc.CoreState.from_state(gs)
                for relevant, gate in ((False, False), (True, False), (True, True), (False, True)):
                    kw = dict(type_to_id=type_to_id, faction_to_id=faction_to_id,
                              relevant_set=relevant, fog_hides_enemy_villages=gate)
                    py = encode_raw(gs, **kw)
                    core = cs.encode_raw(**kw)
                    for f in dataclasses.fields(RawEncoded):
                        a, b = getattr(py, f.name), getattr(core, f.name)
                        label = (f.name, side, fog, relevant, gate)
                        if f.name == "observation":
                            _assert_observations_equal(a, b)
                        elif isinstance(a, np.ndarray):
                            assert isinstance(b, np.ndarray), label
                            assert a.dtype == b.dtype and a.shape == b.shape, (label, a.dtype, b.dtype, a.shape, b.shape)
                            assert a.tobytes() == b.tobytes(), label
                        else:
                            assert a == b, label
                    n += 1
    assert n >= 64


def _one_sim(seed: int, *, mini: bool, max_turns: int, use_core: bool):
    from tests.sim_test_helpers import scenario_setup
    from tools.scenario_pool import build_scenario_gamestate
    from tools.wesnoth_sim import WesnothSim
    setup = scenario_setup(seed, mini=mini)
    return WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id,
                      max_turns=max_turns, use_core=use_core)


def _twin_sims(seed: int, *, mini: bool, max_turns: int):
    """Two simulators from one starting state: the Python state of
    record and the core as the state of record."""
    from tests.sim_test_helpers import scenario_setup
    from tools.scenario_pool import build_scenario_gamestate
    from tools.wesnoth_sim import WesnothSim
    setup = scenario_setup(seed, mini=mini)
    gs = build_scenario_gamestate(setup)
    py = WesnothSim(copy.deepcopy(gs), scenario_id=setup.scenario_id, max_turns=max_turns, use_core=False)
    core = WesnothSim(copy.deepcopy(gs), scenario_id=setup.scenario_id, max_turns=max_turns, use_core=True)
    assert core.core is not None
    return py, core


def test_simulator_on_the_core_plays_the_python_game():
    """Deterministic drivers play twin simulators, the Python state of
    record against the core: the same commands with the same extras
    (the recorder's attack and advancement side channels included),
    the same end state, and a fork of the core sim leaves its parent
    untouched. One game fights (`Brawler`), one random-walks."""
    import wesnoth_ai.dummy_policy as dummy_policy
    from tests.sim_test_helpers import Brawler
    from wesnoth_ai.classes import state_key
    from wesnoth_ai.dummy_policy import DummyPolicy
    total = attacks = 0
    cap = dummy_policy._BOOTSTRAP_UNITS
    dummy_policy._BOOTSTRAP_UNITS = 8
    try:
        total, attacks = _twin_game(3, True, 10, Brawler(), state_key, total, attacks)
        total, attacks = _twin_game(5, False, 4, DummyPolicy(), state_key, total, attacks)
    finally:
        dummy_policy._BOOTSTRAP_UNITS = cap
    assert total >= 50 and attacks >= 5, (total, attacks)


def _twin_game(seed, mini, max_turns, pol, state_key, total, attacks):
    if True:
        py, core = _twin_sims(seed, mini=mini, max_turns=max_turns)
        forked = False
        while not py.done:
            action = pol.select_action(py.gs, game_label="det")
            assert not core.done
            if not forked and core.turn_number >= 2:
                before = state_key(core.gs)
                f = core.fork()
                f.step({"type": "end_turn"})
                assert state_key(core.gs) == before and state_key(f.gs) != before
                forked = True
            py.step(action)
            core.step(action)
            assert not gc.states_equal(py.gs, core.gs, stash=False), (seed, len(py.command_history))
        assert core.done and (py.winner, py.ended_by) == (core.winner, core.ended_by)
        assert state_key(py.gs) == state_key(core.gs)
        assert len(py.command_history) == len(core.command_history) > 10
        for a, b in zip(py.command_history, core.command_history):
            assert (a.kind, a.side, a.cmd, a.extras) == (b.kind, b.side, b.cmd, b.extras)
        assert py._rng_requests == core._rng_requests
        total += len(py.command_history)
        attacks += sum(1 for c in py.command_history if c.kind == "attack")
    return total, attacks
