"""The observation (wesnoth_ai/observe.py, from the Rust core) on
harvested states: a recruit hex rejected this turn leaves the recruit
row, and the record pickles, as it travels inside a RawEncoded."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai import observe as obs_mod  # noqa: E402

pytestmark = pytest.mark.skipif(gc.game_core_class() is None,
                                reason="wesnoth_core.GameCore not available")


def _states():
    from helpers.played_states import _states as harvest
    return harvest(n_attack=3, n_plain=4)


def _as_set(obs, arr):
    keys = obs.geometry.keys
    return set(map(keys.__getitem__, np.nonzero(arr)[0].tolist()))


def test_rejected_recruit_hexes_leave_the_row():
    for state in _states():
        for side in (1, 2):
            base = obs_mod.observe(state, side)
            row = _as_set(base, base.recruit_row)
            if not row:
                continue
            victim = sorted(row)[0]
            state.global_info._recruit_rejected_hexes = {victim}
            try:
                again = obs_mod.observe(state, side)
            finally:
                state.global_info._recruit_rejected_hexes = set()
            assert _as_set(again, again.recruit_row) == row - {victim}
            return
    pytest.skip("no harvested state with a recruitable leader")


def test_the_observation_pickles():
    import pickle
    state = _states()[0]
    obs = obs_mod.observe(state, 1, reach=True)
    back = pickle.loads(pickle.dumps(obs))
    assert back == obs and back.visible_ids() == obs.visible_ids()
