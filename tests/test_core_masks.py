"""The decision's packed legality masks the core builds beside the encoding
(rust/wesnoth_core/src/core_masks.rs) are the Python builder's
(`action_sampler._build_legality_masks` packed by `server_priors.pack_masks`),
array for array, on played positions in every token basis: attacks,
moves, recruits, a rejected recruit hex, units out of moves, statues."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from wesnoth_ai.game_core import game_core_class  # noqa: E402

needs_core = pytest.mark.skipif(game_core_class() is None, reason="wesnoth_core wheel absent or stale")

# (relevant set, relevant-set version, parity observation): the production
# basis, obs8's and the full board.
BASES = [(True, 2, True), (True, 1, False), (False, 1, False)]
FIELDS = ("actor_mask", "type_valid", "attack_valid", "move_valid", "union_valid", "n_attacks",
          "n_units", "n_recruits", "n_hexes")


def _encoders(relevant_set: bool, version: int, parity: bool):
    from tools.inference_seam import RemoteEncoder
    kw = dict(relevant_set=relevant_set, server_priors=True, fog_hides_enemy_villages=True,
              terrain_multi_hot=True, observation_parity=parity, relevant_set_version=version)
    core = RemoteEncoder({}, {}, **kw)
    python = RemoteEncoder({}, {}, **kw)
    # The Python builder's masks: the same encoding without the core's.
    python.raw_of = lambda gs, _r=python.raw_of: _without_core_masks(_r(gs))
    return core, python


def _without_core_masks(raw):
    raw.core_masks = None
    return raw


def _same(a, b, where: str) -> None:
    for name in FIELDS:
        x, y = getattr(a, name), getattr(b, name)
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray), (where, name)
            assert x.dtype == y.dtype and x.shape == y.shape, (where, name, x.dtype, y.dtype, x.shape, y.shape)
            assert np.array_equal(x, y), (where, name)
        else:
            assert x == y, (where, name, x, y)


@needs_core
@pytest.mark.parametrize("relevant_set,version,parity", BASES)
def test_the_core_packs_the_masks_the_python_builder_packs(relevant_set, version, parity):
    from helpers.core_positions import core_positions
    core_enc, python_enc = _encoders(relevant_set, version, parity)
    seen = {"attack": 0, "recruit": 0, "rejected": 0, "idle_unit": 0, "statue": 0}
    for k, gs in enumerate(core_positions()):
        got = core_enc.encode(gs)
        assert got._raw.core_masks is not None
        want = python_enc.encode(gs)
        assert want._raw.core_masks is None
        _same(got._masks, want._masks, f"position {k}")
        m = got._masks
        U, R = m.n_units, m.n_recruits
        seen["attack"] += int(m.type_valid[:U, 0].any())
        seen["recruit"] += int(m.actor_mask[U:U + R].any())
        seen["rejected"] += int(bool(gs.global_info._recruit_rejected_hexes))
        seen["idle_unit"] += int(((got._raw.unit_is_ours != 0) & (m.actor_mask[:U] == 0)).any())
        obs = got._raw.observation
        seen["statue"] += int((obs.inert[obs.tok_of_hex >= 0] != 0).any())
    assert all(seen.values()), seen
