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

# The kernel's current contract (its arguments and its token bound) is
# the one the mask builder takes, from `ENUMERATE_KERNEL_PHASE` on.
_PHASE = getattr(wesnoth_core, "__phase__", 0)
_served = pytest.mark.skipif(
    _PHASE < pf.ENUMERATE_KERNEL_PHASE,
    reason=f"wheel is phase {_PHASE}; enumerate_moves has its current contract "
           f"from phase {pf.ENUMERATE_KERNEL_PHASE}")


def _mid_states(n_games=2, per_game=4):
    """Deep-copied mid-game states from dummy-policy games (the
    bench_infer harvest pattern): real fights, captures, fog."""
    from tools.elo_ladder import _ScriptedAdapter
    from tools.eval_players import _PolicyPair, _play_one_eval_game
    from wesnoth_ai.rules.scenario_pool import (build_scenario_gamestate,
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


@_served
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


# ---------------------------------------------------------------------
# rows_from_landable bounds-checks the token index (lib.rs)
# ---------------------------------------------------------------------
#
# `move_rows[u * ht + tok]` with `tok >= ht` still lands inside the
# buffer for every unit but the last, so an overflowing token index
# silently sets a bit in the NEXT unit's legality row; only the last
# unit's overflow leaves the buffer and panics. The production callers
# build `tok_of_hex` and `ht` from the same hex-position list, so the
# invariant holds by construction -- this pins the kernel's own
# contract so a future basis change cannot break it quietly.


def _line_map_call(tok_of_hex, ht, un=2):
    """Three hexes in a line (0-1-2), unit 0 on hex 0 with the moves to
    reach both others, unit 1 parked on hex 2 and unable to move; no
    zone of control, enemy, ally or occupant anywhere."""
    import numpy as np
    nbrs = np.array([1, -1, -1, -1, -1, -1,
                     0, 2, -1, -1, -1, -1,
                     1, -1, -1, -1, -1, -1], dtype=np.int64)
    flat = np.zeros(3, dtype=np.uint8)
    return wesnoth_core.enumerate_moves(
        nbrs, np.asarray(tok_of_hex, dtype=np.int64),
        np.ones(3, dtype=np.int64), np.zeros(3, dtype=np.int64),
        np.array([0, 2][:un], dtype=np.int64), np.zeros(un, dtype=np.int64),
        np.full(un, 5, dtype=np.int64), np.zeros(un, dtype=np.uint8),
        np.array([1, 0][:un], dtype=np.uint8), np.zeros(un, dtype=np.uint8),
        flat, flat, flat, flat, np.zeros(0, dtype=np.int64), ht)


@_served
def test_token_index_past_the_row_width_is_rejected():
    """tok 1 with ht 1 used to write into unit 1's row and come back
    clean; it must raise instead."""
    with pytest.raises(ValueError):
        _line_map_call([0, 0, 1], ht=1)


@_served
def test_the_same_arrays_enumerate_normally_when_the_width_fits():
    """Positive control: identical inputs with ht 2 must succeed and
    produce the real rows, so the rejection above is about the bound
    and not about a malformed call the kernel would refuse anyway."""
    mv, at = _line_map_call([0, 0, 1], ht=2)
    # unit 0 reaches hexes 1 and 2 (tokens 0 and 1); unit 1 cannot move.
    assert mv.tolist() == [1, 1, 0, 0], mv.tolist()
    assert at.tolist() == [0, 0, 0, 0], at.tolist()
