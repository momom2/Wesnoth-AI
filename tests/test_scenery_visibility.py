#!/usr/bin/env python3
"""Scenery & statues: always visible, neutral-coded, unattackable
(2026-07-11).

Wesnoth's fog hides UNITS, not board furniture: petrified statues and
scenery-side objects (impassable vortices, ToD fires) render under
fog. Before this fix they were fog-filtered like enemies, and when
visible they encoded as ordinary foes -- every mainline map with
scenery (Caves of the Basilisk, Sullas Ruins, Thousand Stings
Garrison...) leaked or misled.

Pins (semantics refined 2026-07-14: ARMED non-petrified side>=3
units are hostile COMBATANTS -- fog-gated, attackable; only
petrified or attackless units are scenery):
  1. units_visible_to: attackless non-player-side units and
     petrified units bypass the fog disc; real enemies (including
     armed side-3 tentacles) still don't.
  2. Encoder: neutral units carry side code 2 (NUM_SIDE_CODES=3),
     is_ours=0.
  3. Legality: a visible scenery unit is never an attack target
     (occupancy inert) and never an actor.
  4. Checkpoint compat: legacy [2, d] side_embed pads with a zero
     row.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import torch

from test_inference_snapshot import _gs, _u
from wesnoth_ai.visibility import units_visible_to


def _with_extra(gs, unit):
    gs.map.units.add(unit)
    return gs


def _scenery(uid, side, x, y):
    """A genuine scenery unit: ATTACKLESS (vortex / ToD fire).
    The _u() template carries a spear -- strip it, else the unit
    is a combatant under the 2026-07-14 classification."""
    u = _u(uid, side, x, y)
    u.attacks.clear()
    return u


def test_scenery_side_visible_through_fog():
    gs = _gs()
    # side-3 ATTACKLESS vortex far from every side-1 unit
    vortex = _scenery("vortex", 3, 9, 9)
    _with_extra(gs, vortex)
    vis_ids = {u.id for u in units_visible_to(gs, 1)}
    assert "vortex" in vis_ids, "scenery must bypass the fog disc"


def test_petrified_statue_visible_through_fog_any_side():
    gs = _gs()
    statue = _u("statue", 2, 9, 0)
    statue.statuses.add("petrified")
    gs.map.units.add(statue)
    vis_ids = {u.id for u in units_visible_to(gs, 1)}
    assert "statue" in vis_ids, \
        "petrified units are board furniture: visible under fog"


def test_armed_side3_unit_is_a_fog_gated_combatant():
    """2026-07-14: an ARMED side-3 unit (Mini_Maps tentacle) is NOT
    scenery -- it hides in fog like any enemy and is attackable."""
    gs = _gs()
    tentacle = _u("tent", 3, 9, 0)            # armed, out of vision
    _with_extra(gs, tentacle)
    vis_ids = {u.id for u in units_visible_to(gs, 1)}
    assert "tent" not in vis_ids,         "armed side-3 units respect the fog disc"

    from wesnoth_ai.visibility import is_scenery_unit
    assert not is_scenery_unit(tentacle)
    petrified = _u("statue", 3, 9, 1)
    petrified.statuses.add("petrified")
    assert is_scenery_unit(petrified), "petrified stays scenery"
    assert is_scenery_unit(_scenery("fire", 3, 9, 2)),         "attackless side-3 stays scenery"


def test_armed_side3_unit_is_attackable():
    import torch
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    from wesnoth_ai.transformer_policy import TransformerPolicy
    pol = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                            num_layers=1, num_heads=4, d_ff=64)
    gs = _gs()
    _with_extra(gs, _u("tent", 3, 3, 4))      # armed, adjacent to u1
    enc = pol._encoder.encode(gs)
    with torch.no_grad():
        out = pol._model(enc)
    priors = enumerate_legal_actions_with_priors(enc, out, gs)
    attacked = any(
        p.action.get("type") == "attack"
        and (p.action["target_hex"].x, p.action["target_hex"].y) == (3, 4)
        for p in priors)
    assert attacked, "armed side-3 combatants must be attack targets"


def test_fog_off_reveals_enemies_and_survives_deepcopy():
    """`_fog = False` (underscore attr: GlobalInfo.__deepcopy__ only
    carries underscore attrs through MCTS state copies) disables the
    sight-disc gate for real enemies too."""
    import copy
    gs = _gs()
    _with_extra(gs, _u("sneak", 2, 9, 0))     # out of vision range
    setattr(gs.global_info, "_fog", False)
    gs2 = copy.deepcopy(gs)                   # MCTS-style state copy
    vis_ids = {u.id for u in units_visible_to(gs2, 1)}
    assert "sneak" in vis_ids, "fog off -> all non-hidden units visible"


def test_real_enemy_still_fogged():
    gs = _gs()
    hidden = _u("sneak", 2, 9, 0)          # far from side-1 units
    _with_extra(gs, hidden)
    vis_ids = {u.id for u in units_visible_to(gs, 1)}
    assert "sneak" not in vis_ids, \
        "ordinary enemies must still respect the fog disc"


def test_neutral_side_code_in_encoder():
    from wesnoth_ai.encoder import NUM_SIDE_CODES, encode_raw
    assert NUM_SIDE_CODES == 3
    gs = _gs()
    _with_extra(gs, _scenery("vortex", 3, 4, 4))   # near, visible
    raw = encode_raw(gs, type_to_id={}, faction_to_id={})
    idx = raw.unit_ids.index("vortex")
    assert raw.unit_side_ids[idx] == 2, "scenery -> neutral side code"
    assert raw.unit_is_ours[idx] == 0.0
    own_idx = raw.unit_ids.index("u1")
    foe_idx = raw.unit_ids.index("u2")
    assert raw.unit_side_ids[own_idx] == 0
    assert raw.unit_side_ids[foe_idx] == 1


def test_scenery_never_an_attack_target_nor_actor():
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    from wesnoth_ai.transformer_policy import TransformerPolicy
    pol = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                            num_layers=1, num_heads=4, d_ff=64)
    gs = _gs()
    # ATTACKLESS vortex adjacent to our unit u1 at (3,3)
    _with_extra(gs, _scenery("vortex", 3, 3, 4))
    enc = pol._encoder.encode(gs)
    with torch.no_grad():
        out = pol._model(enc)
    priors = enumerate_legal_actions_with_priors(enc, out, gs)
    # Action dicts carry Position objects under start_hex /
    # target_hex (adversarial review 2026-07-11: the first draft
    # asserted on nonexistent target_x/unit_id keys and was
    # vacuously green).
    assert priors, "sanity: legal actions must exist"
    saw_real_attack = False
    for p in priors:
        act = p.action
        tgt = act.get("target_hex")
        src = act.get("start_hex")
        if act.get("type") == "attack" and tgt is not None:
            assert (tgt.x, tgt.y) != (3, 4), \
                "scenery must not be attackable"
            if (tgt.x, tgt.y) == (4, 3):
                saw_real_attack = True
        if act.get("type") == "move" and tgt is not None:
            assert (tgt.x, tgt.y) != (3, 4), \
                "scenery occupies its hex: not a move target"
        if src is not None:
            assert (src.x, src.y) != (3, 4), \
                "scenery must never be an actor"
    # Positive control: the same loop CAN see attacks (real enemy
    # u2 at (4,3) is adjacent to u1) -- proves the assertions bite.
    assert saw_real_attack, \
        "expected an attack on the real enemy at (4,3)"


def test_legacy_side_embed_pads(tmp_path):
    from wesnoth_ai.encoder import NUM_SIDE_CODES
    from wesnoth_ai.transformer_policy import TransformerPolicy
    p1 = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                           num_layers=1, num_heads=4, d_ff=64)
    ck = tmp_path / "legacy.pt"
    p1.save_checkpoint(ck)
    raw = torch.load(ck, map_location="cpu", weights_only=False)
    w = raw["encoder_state"]["side_embed.weight"]
    raw["encoder_state"]["side_embed.weight"] = w[:2, :].clone()
    torch.save(raw, ck)
    p2 = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                           num_layers=1, num_heads=4, d_ff=64)
    p2.load_checkpoint(ck)
    w2 = p2._encoder.side_embed.weight
    assert w2.shape[0] == NUM_SIDE_CODES
    assert torch.allclose(w2[:2], w[:2])
    assert torch.all(w2[2] == 0.0)
