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
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import numpy as np
import torch

from classes import (GameState, GlobalInfo, Hex, Map, Position,
                     SideInfo, Terrain, TerrainModifiers, Unit)
from encoder import NUM_HEX_DYNAMIC_FLAGS, encode_raw
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


def test_pad_helper_covers_direct_encoder_loads():
    """Tools (eval_vs_builtin, supervised_train, collect_cliffness,
    eval_mcts_vs_reinforce) load encoder state WITHOUT going through
    TransformerPolicy.load_checkpoint; `pad_legacy_encoder_state` is
    their shim. Pins that (a) strict=False alone does NOT tolerate
    the legacy shapes (the crash the shim prevents), and (b) the
    padded state loads with zeros in the new slots."""
    from encoder import NUM_SIDE_CODES, pad_legacy_encoder_state
    from transformer_policy import TransformerPolicy
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


def test_legacy_checkpoint_pads_dynamic_flag_proj(tmp_path):
    from transformer_policy import TransformerPolicy
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
