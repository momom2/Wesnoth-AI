#!/usr/bin/env python3
"""Per-village ownership encoding + fog view rule (2026-07-11).

Dynamic hex flags 1-2 encode village ownership AS SEEN by the side to
move:
  - own villages: always flagged "ours" (you know what you own);
  - enemy-owned villages: flagged "theirs" only when the hex is in
    the side's vision disc (or fog is off); fogged enemy villages
    appear NEUTRAL (both flags 0);
  - neutral villages / non-village hexes: both flags 0.

Also pins the checkpoint-compat shim: legacy [d, 1]
dynamic_flag_proj weights pad with zero columns on load.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import numpy as np
import torch

from wesnoth_ai.classes import (GameState, GlobalInfo, Hex, Map, Position,
                     SideInfo, Terrain, TerrainModifiers, Unit)
from wesnoth_ai.encoder import NUM_HEX_DYNAMIC_FLAGS, encode_raw
from test_inference_snapshot import _gs, _u


def _village_gs(*, owner_map, current_side=1, fog=True):
    """_gs() plus three village hexes at (1,1), (5,5), (9,9).

    Side-1 units sit at (0,0)/(3,3): with default vision (~5-6),
    (1,1) and (5,5) are within side 1's discs; (9,9) is safely
    outside every side-1 disc.
    """
    gs = _gs()
    for pos in ((1, 1), (5, 5), (9, 9)):
        h = next(x for x in gs.map.hexes
                 if x.position.x == pos[0] and x.position.y == pos[1])
        gs.map.hexes.discard(h)
        gs.map.hexes.add(Hex(position=Position(*pos),
                             terrain_types={Terrain.VILLAGE},
                             modifiers={TerrainModifiers.VILLAGE}))
    gs.global_info.current_side = current_side
    setattr(gs.global_info, "_village_owner", dict(owner_map))
    if not fog:
        # Underscore attr: GlobalInfo.__deepcopy__ only carries
        # underscore attrs through MCTS state copies.
        setattr(gs.global_info, "_fog", False)
    return gs


def _flags_at(gs, pos):
    raw = encode_raw(gs, type_to_id={}, faction_to_id={})
    for i in range(len(raw.hex_positions)):
        p = raw.hex_positions[i]
        if (p.x, p.y) == pos:
            return raw.hex_dynamic_flags[i]
    raise AssertionError(f"hex {pos} not found")


def test_flag_count_is_three():
    assert NUM_HEX_DYNAMIC_FLAGS == 3


def test_own_village_always_visible_as_ours():
    gs = _village_gs(owner_map={(9, 9): 1})   # ours, out of vision
    f = _flags_at(gs, (9, 9))
    assert f[1] == 1.0 and f[2] == 0.0, \
        "own villages show as ours even in fog"


def test_enemy_village_in_vision_shows_theirs():
    gs = _village_gs(owner_map={(1, 1): 2})   # theirs, near our units
    f = _flags_at(gs, (1, 1))
    assert f[1] == 0.0 and f[2] == 1.0


def test_enemy_village_in_fog_appears_neutral():
    gs = _village_gs(owner_map={(9, 9): 2})   # theirs, out of vision
    f = _flags_at(gs, (9, 9))
    assert f[1] == 0.0 and f[2] == 0.0, \
        "fogged enemy village must appear neutral"


def test_enemy_village_fogless_shows_theirs():
    import copy
    gs = _village_gs(owner_map={(9, 9): 2}, fog=False)
    # Route through deepcopy: MCTS copies states before encoding,
    # and GlobalInfo.__deepcopy__ drops non-underscore attrs -- a
    # plain `fog` attr silently reverts to fog-on (adversarial
    # review 2026-07-11).
    gs = copy.deepcopy(gs)
    f = _flags_at(gs, (9, 9))
    assert f[2] == 1.0, "fog off -> true owner visible everywhere"


def test_neutral_village_and_plain_hex_carry_no_owner_flags():
    gs = _village_gs(owner_map={})
    f = _flags_at(gs, (5, 5))                 # unowned village
    assert f[1] == 0.0 and f[2] == 0.0
    f2 = _flags_at(gs, (2, 2))                # not a village
    assert f2[1] == 0.0 and f2[2] == 0.0


def test_perspective_flips_with_current_side():
    gs = _village_gs(owner_map={(1, 1): 2}, current_side=2)
    f = _flags_at(gs, (1, 1))
    assert f[1] == 1.0 and f[2] == 0.0, \
        "side 2 sees its own village as ours"


def test_owner_map_alone_marks_owned_village():
    """The sim no longer stamps TerrainModifiers.VILLAGE on capture
    (2026-07-29 fork-isolation fix); the encoder derives the
    owned-village bit from `_village_owner` directly. Pins that an
    owner-map-only village -- the sim-lineage representation,
    including scenario-pool pre-owned villages -- carries the static
    owned bit AND the ownership flags. (The legacy modifier is still
    honored for the live-Wesnoth converter path; see the disjunction
    in encoder.encode_raw.)"""
    gs = _village_gs(owner_map={(1, 1): 2})
    # Strip the modifier from (1,1): sim-lineage hexes never carry it.
    h = next(x for x in gs.map.hexes
             if (x.position.x, x.position.y) == (1, 1))
    gs.map.hexes.discard(h)
    gs.map.hexes.add(Hex(position=Position(1, 1),
                         terrain_types={Terrain.VILLAGE},
                         modifiers=set()))
    raw = encode_raw(gs, type_to_id={}, faction_to_id={})
    for i, p in enumerate(raw.hex_positions):
        if (p.x, p.y) == (1, 1):
            assert raw.hex_modifier_flags[i][0] == 1.0, \
                "owner-map-only village must carry the owned bit"
            assert raw.hex_dynamic_flags[i][1] == 0.0
            assert raw.hex_dynamic_flags[i][2] == 1.0  # enemy, in vision
            return
    raise AssertionError("hex (1,1) missing from the raw hex stream")


def test_fork_capture_does_not_mutate_parent_encoding():
    """MCTS forks alias Hex objects (Map.__deepcopy__ fast path). A
    village capture stepped on a FORK must not change the parent
    state's encoding. Pre-2026-07-29, _capture_village stamped
    TerrainModifiers.VILLAGE onto the shared hex, so a HYPOTHETICAL
    capture inside a search leaked into the real game's encoder input
    -- root priors on a fixed state stepped by ~1e-3 after a search,
    which is what made test_mcts_search_through_seam_matches_direct
    flaky (the direct search leaked before the seam search ran)."""
    import dataclasses
    from sim_test_helpers import fresh_scenario_sim
    from tools.abilities import hex_neighbors

    captured = False
    for seed in (21, 20, 22, 23, 24):
        sim = fresh_scenario_sim(seed=seed, max_turns=12, mini=True)
        gs = sim.gs
        villages = {(h.position.x, h.position.y) for h in gs.map.hexes
                    if Terrain.VILLAGE in h.terrain_types}
        occupied = {(u.position.x, u.position.y) for u in gs.map.units}
        side = gs.global_info.current_side
        raw0 = encode_raw(gs, type_to_id={}, faction_to_id={})
        mods0 = {(h.position.x, h.position.y): set(h.modifiers)
                 for h in gs.map.hexes}
        for u in sorted((u for u in gs.map.units
                         if u.side == side and u.current_moves > 0),
                        key=lambda u: u.id):
            for (nx, ny) in hex_neighbors(u.position.x, u.position.y):
                if (nx, ny) not in villages or (nx, ny) in occupied:
                    continue
                fork = sim.fork()
                try:
                    fork.step({"type": "move",
                               "start_hex": u.position,
                               "target_hex": Position(x=nx, y=ny)})
                except Exception:
                    continue
                if (nx, ny) in (getattr(fork.gs.global_info,
                                        "_village_owner", None) or {}):
                    captured = True
                    break
            if captured:
                break
        if captured:
            break
    assert captured, "premise: no reachable one-step village capture " \
                     "found across seeds 20-24 (test needs a new recipe)"

    raw1 = encode_raw(gs, type_to_id={}, faction_to_id={})
    for f in dataclasses.fields(raw0):
        a, b = getattr(raw0, f.name), getattr(raw1, f.name)
        if isinstance(a, np.ndarray):
            assert a.shape == b.shape and bool((a == b).all()), \
                f"fork capture leaked into parent encoding: {f.name}"
        else:
            assert a == b, \
                f"fork capture leaked into parent encoding: {f.name}"
    mods1 = {(h.position.x, h.position.y): set(h.modifiers)
             for h in gs.map.hexes}
    assert mods0 == mods1, "fork capture mutated shared Hex modifiers"


def test_pad_helper_covers_direct_encoder_loads():
    """Tools (eval_vs_builtin, supervised_train, collect_cliffness,
    eval_mcts_vs_reinforce) load encoder state WITHOUT going through
    TransformerPolicy.load_checkpoint; `pad_legacy_encoder_state` is
    their shim. Pins that (a) strict=False alone does NOT tolerate
    the legacy shapes (the crash the shim prevents), and (b) the
    padded state loads with zeros in the new slots."""
    from wesnoth_ai.encoder import NUM_SIDE_CODES, pad_legacy_encoder_state
    from wesnoth_ai.transformer_policy import TransformerPolicy
    p1 = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                           num_layers=1, num_heads=4, d_ff=64)
    legacy = {k: v.clone() for k, v in p1._encoder.state_dict().items()}
    legacy["dynamic_flag_proj.weight"] = \
        legacy["dynamic_flag_proj.weight"][:, :1].clone()
    legacy["side_embed.weight"] = legacy["side_embed.weight"][:2, :].clone()

    p2 = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                           num_layers=1, num_heads=4, d_ff=64)
    try:
        p2._encoder.load_state_dict(legacy, strict=False)
        crashed = False
    except RuntimeError:
        crashed = True
    assert crashed, "premise: strict=False must reject shape mismatches"

    padded = pad_legacy_encoder_state(legacy, p2._encoder)
    p2._encoder.load_state_dict(padded, strict=False)
    w = p2._encoder.dynamic_flag_proj.weight
    assert w.shape[1] == NUM_HEX_DYNAMIC_FLAGS
    assert torch.all(w[:, 1:] == 0.0)
    se = p2._encoder.side_embed.weight
    assert se.shape[0] == NUM_SIDE_CODES
    assert torch.all(se[2] == 0.0)
    # Input dict untouched (helper returns a new mapping).
    assert legacy["dynamic_flag_proj.weight"].shape[1] == 1


def test_legacy_optimizer_state_repairs_after_pad(tmp_path):
    """The pad shim fixes the WEIGHTS, but a resumed Adam state still
    carries old-shaped moments; the first optimizer.step() then
    crashes on the broadcast (production incident 2026-07-11:
    'output with shape [256, 1] doesn't match the broadcast shape
    [256, 3]'). Pins repair_optimizer_state_shapes end-to-end
    through load_checkpoint."""
    from wesnoth_ai.transformer_policy import TransformerPolicy
    p1 = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                           num_layers=1, num_heads=4, d_ff=64)
    opt1 = p1._trainer.optimizer
    for g in opt1.param_groups:              # populate Adam state
        for p in g["params"]:
            p.grad = torch.zeros_like(p)
    opt1.step()
    opt1.zero_grad()
    ck = tmp_path / "legacy.pt"
    p1.save_checkpoint(ck)

    raw = torch.load(ck, map_location="cpu", weights_only=False)
    es = raw["encoder_state"]
    es["dynamic_flag_proj.weight"] = \
        es["dynamic_flag_proj.weight"][:, :1].clone()
    es["side_embed.weight"] = es["side_embed.weight"][:2, :].clone()
    # Slice the matching Adam moments back to the legacy shapes
    # (identified by shape: dynamic_flag_proj is [32, 3], side_embed
    # [3, 32] at this test's d_model=32).
    n_sliced = 0
    for entry in raw["optimizer_state"]["state"].values():
        for k, t in list(entry.items()):
            if isinstance(t, torch.Tensor) and tuple(t.shape) == (32, 3):
                entry[k] = t[:, :1].clone(); n_sliced += 1
            elif isinstance(t, torch.Tensor) and tuple(t.shape) == (3, 32):
                entry[k] = t[:2, :].clone(); n_sliced += 1
    assert n_sliced >= 4, "premise: moments for both padded params"
    torch.save(raw, ck)

    p2 = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                           num_layers=1, num_heads=4, d_ff=64)
    p2.load_checkpoint(ck)
    opt2 = p2._trainer.optimizer
    for g in opt2.param_groups:
        for p in g["params"]:
            p.grad = torch.zeros_like(p)
    opt2.step()                              # crashed before the fix
    for p, s in opt2.state.items():
        for k in ("exp_avg", "exp_avg_sq"):
            if k in s and isinstance(s[k], torch.Tensor):
                assert s[k].shape == p.shape


def test_legacy_checkpoint_pads_dynamic_flag_proj(tmp_path):
    from wesnoth_ai.transformer_policy import TransformerPolicy
    p1 = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                           num_layers=1, num_heads=4, d_ff=64)
    ck = tmp_path / "legacy.pt"
    p1.save_checkpoint(ck)
    # Simulate a pre-ownership checkpoint: slice the proj to [d, 1].
    raw = torch.load(ck, map_location="cpu", weights_only=False)
    w = raw["encoder_state"]["dynamic_flag_proj.weight"]
    raw["encoder_state"]["dynamic_flag_proj.weight"] = \
        w[:, :1].clone()
    torch.save(raw, ck)

    p2 = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                           num_layers=1, num_heads=4, d_ff=64)
    p2.load_checkpoint(ck)
    w2 = p2._encoder.dynamic_flag_proj.weight
    assert w2.shape[1] == NUM_HEX_DYNAMIC_FLAGS
    assert torch.allclose(w2[:, :1], w[:, :1])
    assert torch.all(w2[:, 1:] == 0.0), \
        "new ownership columns start at zero (old behavior preserved)"
