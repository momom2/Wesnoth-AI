"""Rust phase-2 certification (docs/rust_port_plan.md): the batch
move/attack enumeration must produce EXACTLY the masks the Python
path builds — every tensor equal, including the oracle bias arrays
(which consume the rows). Drives real scenario states plus
dummy-game midstates so occupancy/ZoC/fog shapes vary.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

wesnoth_core = pytest.importorskip("wesnoth_core")

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from tools import pathfind_sim as pf  # noqa: E402


def _mid_states(n_games=2, per_game=4):
    """Deep-copied mid-game states from dummy-policy games (the
    bench_infer harvest pattern): real fights, captures, fog."""
    from tools.elo_ladder import _ScriptedAdapter
    from tools.eval_sim import _PolicyPair, _play_one_eval_game
    from tools.scenario_pool import (build_scenario_gamestate,
                                     random_setup)
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.dummy_policy import DummyPolicy
    import random as _r

    sink, taken = [], [0]

    class _Rec:
        def __init__(self, inner):
            self._i = inner
            self._seen = 0

        def select_action(self, gs, **kw):
            self._seen += 1
            if self._seen % 7 == 0 and taken[0] < per_game:
                sink.append(copy.deepcopy(gs))
                taken[0] += 1
            return self._i.select_action(gs, **kw)

        def __getattr__(self, name):
            return getattr(self._i, name)

    out = []
    for g in range(n_games):
        taken[0] = 0
        rng = _r.Random(500 + g)
        setup = random_setup(rng)
        gs = build_scenario_gamestate(setup)
        sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                         max_turns=14)
        _play_one_eval_game(
            sim,
            _PolicyPair(policy=_Rec(_ScriptedAdapter(DummyPolicy())),
                        label="a", side=1),
            _PolicyPair(policy=_Rec(_ScriptedAdapter(DummyPolicy())),
                        label="b", side=2),
            game_label=f"rustenum{g}")
        out.extend(sink)
        sink.clear()
    return out


def _masks_both(enc, state):
    from wesnoth_ai.action_sampler import _build_legality_masks
    encoded = enc.encode(state)
    saved = pf._RUST
    try:
        pf._RUST = None
        py = _build_legality_masks(encoded, state)
        pf._RUST = wesnoth_core
        rs = _build_legality_masks(encoded, state)
    finally:
        pf._RUST = saved
    return py, rs


_FIELDS = ("actor_valid", "target_valid", "target_valid_attack",
           "target_valid_move", "type_valid", "type_bias",
           "attack_bias")


def _assert_equal(py, rs, tag):
    for f in _FIELDS:
        a, b = getattr(py, f), getattr(rs, f)
        assert torch.equal(a, b), (
            f"{tag}: {f} differs "
            f"(sum py={a.sum().item()} rs={b.sum().item()})")


def test_rust_enumeration_matches_python_masks():
    from wesnoth_ai.action_sampler import _rust_enumerate_rows
    from wesnoth_ai.encoder import GameStateEncoder
    import wesnoth_ai.action_sampler as _as

    # Count actual Rust engagements: a green run where the fast
    # path never fired (e.g. the >=2-eligible-units gate ate every
    # state) certifies nothing.
    engaged = [0]
    _orig = _rust_enumerate_rows

    def _counting(*a, **k):
        out = _orig(*a, **k)
        if out is not None:
            engaged[0] += 1
        return out

    enc = GameStateEncoder()
    sim = fresh_scenario_sim()
    _as._rust_enumerate_rows = _counting
    try:
        py, rs = _masks_both(enc, sim.gs)
        _assert_equal(py, rs, "fresh scenario")
        for k, gs in enumerate(_mid_states()):
            py, rs = _masks_both(enc, gs)
            _assert_equal(py, rs, f"midstate {k}")
    finally:
        _as._rust_enumerate_rows = _orig
    assert engaged[0] >= 2, (
        f"rust path engaged on only {engaged[0]} states -- the "
        f"differential proved nothing; widen the state sample")
