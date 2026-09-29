"""The certification compares two observations of one position hex by hex,
whatever order each side's hex set iterates in. After an event changes the
terrain, the Python applier's hex set iterates in another order than the
core's; compared index by index, every Aethermaw game read as a divergence
while every model input was identical (2026-09-29 audit)."""
from __future__ import annotations

import dataclasses
import random
from types import SimpleNamespace

import numpy as np
import pytest

from wesnoth_ai import game_core as gc

pytestmark = pytest.mark.skipif(gc.game_core_class() is None,
                                reason="the installed wesnoth_core wheel is older than game_core needs")

PER_HEX = ("seen", "zoc", "enemy", "ally", "occupied", "inert", "recruit_row", "network",
           "relevant", "tok_of_hex")


def _observation():
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.rules import scenario_pool as sp
    setup = sp.random_setup(random.Random(3))
    sim = WesnothSim(sp.build_scenario_gamestate(setup), scenario_id=setup.scenario_id, max_turns=4)
    side = sim.gs.global_info.current_side
    return gc.core_of(sim.gs).observe(side, reach=True)


def _reordered(obs, perm: np.ndarray):
    """The same observation with its hexes in another order: new index j
    holds old index perm[j]."""
    new_index = np.empty(len(perm), dtype=np.int64)
    new_index[perm] = np.arange(len(perm), dtype=np.int64)
    fields = {k: getattr(obs, k)[perm] for k in PER_HEX if getattr(obs, k) is not None}
    fields["landable"] = obs.landable[:, perm]
    fields["unit_hex"] = np.where(obs.unit_hex >= 0, new_index[np.maximum(obs.unit_hex, 0)], -1)
    geometry = SimpleNamespace(keys=[obs.geometry.keys[i] for i in perm])
    return dataclasses.replace(obs, geometry=geometry, **fields)


def test_observations_compare_by_hex_position_whatever_their_order():
    from wesnoth_ai.core_compare import observation_differences
    obs = _observation()
    assert obs.landable is not None and (obs.unit_hex >= 0).any()
    perm = np.random.default_rng(0).permutation(len(obs.geometry.keys))
    moved = _reordered(obs, perm)
    assert observation_differences(obs, moved) == []
    seen = moved.seen.copy()
    seen[0] ^= 1
    assert observation_differences(obs, dataclasses.replace(moved, seen=seen)) == ["observation seen"]
