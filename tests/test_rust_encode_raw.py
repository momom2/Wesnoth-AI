"""Rust phase-2b certification (docs/rust_port_plan.md): the wheel's
`encode_raw_streams` must produce EXACTLY the arrays the Python
builders produce -- every numpy field of RawEncoded equal in dtype,
shape and bytes -- over real states: scenario starts, dummy-game
midstates (captures, fights, fog), fog off, recruit rejections,
owner-map entries off the village terrain, a petrified (scenery) unit,
out-of-vocab names, and the relevant-set hex stream. A coverage test
keeps a thin harvest from certifying vacuously. Skips when the wheel
is not built, predates phase 2b, or is disabled (WESNOTH_RUST=0).
"""
from __future__ import annotations

import copy
import dataclasses
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

wesnoth_core = pytest.importorskip("wesnoth_core")
if not hasattr(wesnoth_core, "encode_raw_streams"):
    pytest.skip("wesnoth_core wheel predates phase 2b: rebuild it",
                allow_module_level=True)

from tools import pathfind_sim as pf  # noqa: E402
from tools.abilities import hex_neighbors  # noqa: E402
from wesnoth_ai import encoder  # noqa: E402
from wesnoth_ai.classes import Terrain, TerrainModifiers  # noqa: E402
from wesnoth_ai.encoder import RawEncoded, encode_raw  # noqa: E402

if encoder._rust_encode_kernel() is None:
    # Two causes, and the wheel's phase is the common one: the encoder
    # refuses a kernel older than `_ENCODE_KERNEL_PHASE`, whose feature
    # widths are wrong, and parity against it would mean nothing.
    pytest.skip(
        f"Rust encode kernel unavailable: WESNOTH_RUST=0, or the wheel is "
        f"phase {getattr(wesnoth_core, '__phase__', '?')} and the encoder "
        f"needs {encoder._ENCODE_KERNEL_PHASE}",
                allow_module_level=True)


def _dummy_game_states(seed: int, per_game: int, max_turns: int):
    """Deep-copied midgame states from one dummy-policy game. The
    dummy's recruit cap is lifted for the game so armies form (its
    default stops at 3 units)."""
    from tools.elo_ladder import _ScriptedAdapter
    from tools.eval_players import _PolicyPair, _play_one_eval_game
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    from tools.wesnoth_sim import WesnothSim
    import wesnoth_ai.dummy_policy as dummy_policy
    from wesnoth_ai.dummy_policy import DummyPolicy

    sink = []

    class _Rec:
        def __init__(self, inner):
            self._i = inner
            self._seen = 0

        def select_action(self, gs, **kw):
            self._seen += 1
            if (self._seen % 5 == 0 and len(sink) < per_game
                    and gs.global_info.turn_number >= 2):
                sink.append(copy.deepcopy(gs))
            return self._i.select_action(gs, **kw)

        def __getattr__(self, name):
            return getattr(self._i, name)

    cap = dummy_policy._BOOTSTRAP_UNITS
    dummy_policy._BOOTSTRAP_UNITS = 12
    try:
        setup = random_setup(random.Random(seed))
        sim = WesnothSim(build_scenario_gamestate(setup),
                         scenario_id=setup.scenario_id, max_turns=max_turns)
        _play_one_eval_game(
            sim,
            _PolicyPair(policy=_Rec(_ScriptedAdapter(DummyPolicy())),
                        label="a", side=1),
            _PolicyPair(policy=_Rec(_ScriptedAdapter(DummyPolicy())),
                        label="b", side=2),
            game_label=f"rustenc{seed}")
    finally:
        dummy_policy._BOOTSTRAP_UNITS = cap
    return sink


def _leader(gs, side):
    return next(u for u in gs.map.units if u.side == side and u.is_leader)


def _with_recruit_rejections(gs):
    """Two castle hexes beside the mover's leader bounced this turn,
    plus an off-board key the slot lookup must skip."""
    gs = copy.deepcopy(gs)
    lead = _leader(gs, gs.global_info.current_side)
    castle = {(h.position.x, h.position.y) for h in gs.map.hexes
              if TerrainModifiers.CASTLE in h.modifiers}
    near = [p for p in hex_neighbors(lead.position.x, lead.position.y)
            if p in castle][:2]
    gs.global_info._recruit_rejected_hexes = set(near) | {(999, 999)}
    return gs


def _with_owner_map_quirks(gs):
    """Owner entries on the enemy keep (no village terrain: the
    village bit is still shown when its owner is visible) and off the
    board (skipped); one village given to each side."""
    gs = copy.deepcopy(gs)
    side = gs.global_info.current_side
    enemy_keep = _leader(gs, 3 - side).position
    villages = sorted((h.position.x, h.position.y) for h in gs.map.hexes
                      if Terrain.VILLAGE in h.terrain_types)
    owners = dict(getattr(gs.global_info, "_village_owner", None) or {})
    owners[(enemy_keep.x, enemy_keep.y)] = 3 - side
    owners[(998, 998)] = side
    if len(villages) >= 2:
        owners[villages[0]] = side
        owners[villages[-1]] = 3 - side
    gs.global_info._village_owner = owners
    return gs


def _with_fog_off(gs):
    gs = copy.deepcopy(gs)
    gs.global_info._fog = False
    return gs


def _with_village_modifiers(gs):
    """scenario_pool hexes carry the village TERRAIN without the
    village MODIFIER, and the static village bit keys on the modifier
    (encoder._build_static_hex_arrays); add it so unowned visible
    villages take the modifier-only branch."""
    gs = copy.deepcopy(gs)
    for h in gs.map.hexes:
        if Terrain.VILLAGE in h.terrain_types:
            h.modifiers = set(h.modifiers) | {TerrainModifiers.VILLAGE}
    return gs


def _with_attacked_leader(gs):
    """The dummy policy never attacks; flag the mover's leader as
    having attacked (moves spent) so that feature column is exercised."""
    gs = copy.deepcopy(gs)
    lead = _leader(gs, gs.global_info.current_side)
    lead.has_attacked = True
    lead.current_moves = 0
    return gs


def _with_petrified_enemy(gs):
    """Fog off so the statue's slot is exercised next to live enemies."""
    gs = _with_fog_off(gs)
    side = gs.global_info.current_side
    enemy = next((u for u in gs.map.units if u.side != side), None)
    if enemy is None:
        return None
    enemy.statuses = set(enemy.statuses or set()) | {"petrified"}
    return gs


@pytest.fixture(scope="module")
def states():
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    starts = [build_scenario_gamestate(random_setup(random.Random(s)))
              for s in (11, 12, 13)]
    mids = _dummy_game_states(500, per_game=24, max_turns=16)
    mids += _dummy_game_states(501, per_game=16, max_turns=12)
    out = starts + mids
    out += [_with_fog_off(gs) for gs in mids[::4]]
    out += [_with_recruit_rejections(gs) for gs in starts + mids[::6]]
    out += [_with_owner_map_quirks(gs) for gs in starts[:2] + mids[1::8]]
    out += [_with_owner_map_quirks(_with_fog_off(gs)) for gs in starts[:1] + mids[3::10]]
    out += [_with_attacked_leader(gs) for gs in mids[4::7]]
    out += [_with_village_modifiers(gs) for gs in starts[1:2] + mids[5::11]]
    out += [_with_village_modifiers(_with_fog_off(gs)) for gs in mids[6::13]]
    out += [s for s in (_with_petrified_enemy(gs) for gs in mids[2::9])
            if s is not None]
    return out


@pytest.fixture(scope="module")
def vocab(states):
    """Half the type names and one faction, so both the vocab hit and
    the overflow-bucket clamp run."""
    names = sorted({u.name for gs in states for u in gs.map.units}
                   | {r for gs in states for s in gs.sides for r in s.recruits})
    type_to_id = {n: i for i, n in enumerate(names[::2])}
    faction_to_id = {"": 0, states[0].sides[1].faction: 1}
    return type_to_id, faction_to_id


def _encode_both(gs, vocab, relevant_set, terrain_multi_hot=False):
    type_to_id, faction_to_id = vocab
    kw = dict(type_to_id=type_to_id, faction_to_id=faction_to_id,
              relevant_set=relevant_set, terrain_multi_hot=terrain_multi_hot)
    rust = encode_raw(gs, **kw)
    saved = pf._RUST
    pf._RUST = None
    try:
        assert encoder._rust_encode_kernel() is None
        py = encode_raw(gs, **kw)
    finally:
        pf._RUST = saved
    return py, rust


def _assert_identical(py: RawEncoded, rust: RawEncoded, label: str):
    for f in dataclasses.fields(RawEncoded):
        a, b = getattr(py, f.name), getattr(rust, f.name)
        if isinstance(a, np.ndarray):
            assert isinstance(b, np.ndarray), (label, f.name)
            assert a.dtype == b.dtype, (label, f.name, a.dtype, b.dtype)
            assert a.shape == b.shape, (label, f.name, a.shape, b.shape)
            assert np.array_equal(a, b), (label, f.name)
            assert a.tobytes() == b.tobytes(), (label, f.name)
        else:
            assert a == b, (label, f.name)


@pytest.mark.parametrize("relevant_set", [False, True])
@pytest.mark.parametrize("terrain_multi_hot", [False, True])
def test_arrays_byte_identical(states, vocab, relevant_set, terrain_multi_hot):
    for k, gs in enumerate(states):
        py, rust = _encode_both(gs, vocab, relevant_set, terrain_multi_hot)
        _assert_identical(py, rust, f"state {k} relevant_set={relevant_set} "
                                    f"terrain_multi_hot={terrain_multi_hot}")


def test_harvest_exercises_every_branch(states, vocab):
    """Engagement counters: the certification above is only as good
    as what the states contain."""
    raws = [_encode_both(gs, vocab, False)[1] for gs in states]
    units = [len(r.unit_ids) for r in raws]
    assert max(units) >= 6 and min(units) >= 1
    assert all(len(r.recruit_types) >= 1 for r in raws)
    assert any((r.unit_side_ids == 1).any() for r in raws)       # visible enemies
    assert any((r.unit_side_ids == 2).any() for r in raws)       # scenery
    assert any((r.unit_feats[:, 8] == 1.0).any() for r in raws)  # has_attacked
    assert len({int(a) for r in raws for a in r.unit_feats[:, 9:].argmax(1)}) >= 2
    overflow = encoder.MAX_UNIT_TYPES - 1
    assert any((r.unit_type_ids == overflow).any() for r in raws)
    assert any((r.unit_type_ids != overflow).any() for r in raws)
    assert any(r.hex_dynamic_flags[:, 0].sum() >= 2 for r in raws)  # rejections
    assert any(r.hex_dynamic_flags[:, 1].any() for r in raws)       # our village
    assert any(r.hex_dynamic_flags[:, 2].any() for r in raws)       # theirs, visible
    assert any(r.hex_modifier_flags[:, 0].sum() > r.hex_dynamic_flags[:, 1:].sum()
               for r in raws)                                       # neutral village seen
    assert {getattr(gs.global_info, "_fog", True) for gs in states} == {True, False}
    assert 1 in {r.their_faction_id for r in raws}
