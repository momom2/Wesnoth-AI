"""Contract tests for visibility.relevant_hex_positions /
relevant_hexes_in_slot_order (T2-B relevant-set encoder core).

Pins the three adopted requirements at the source-of-truth layer:
  1. SUPERSET of every mask-offerable target hex (checked against
     _build_legality_masks on real pool states).
  2. DETERMINISTIC, STABLE ORDERING across repeated calls on equal
     states (the trainer replays stored target indices).
  3. Degenerate states (no units / no leader) don't crash and still
     return the static components.
"""
import copy
import random

import pytest
import torch

from tools.scenario_pool import (random_setup, build_scenario_gamestate,
                                 load_factions)
from wesnoth_ai.visibility import (hexes_in_slot_order,
                                   relevant_hex_positions,
                                   relevant_hexes_in_slot_order)


@pytest.fixture(scope="module")
def pool_state():
    load_factions()
    setup = random_setup(random.Random(77), forced_faction=None)
    return build_scenario_gamestate(setup)


def test_superset_of_mask_targets(pool_state):
    """Every hex the legality mask offers must be in the set."""
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.action_sampler import _build_legality_masks
    gs = pool_state
    side = gs.global_info.current_side
    enc = GameStateEncoder()
    with torch.no_grad():
        encoded = enc.encode(gs)
    masks = _build_legality_masks(encoded, gs)
    tv = masks.target_valid.numpy()
    offered = {(encoded.hex_positions[j].x, encoded.hex_positions[j].y)
               for j in range(tv.shape[1]) if tv[:, j].any()}
    rel = relevant_hex_positions(gs, side)
    missing = offered - rel
    assert not missing, f"mask offers hexes outside relevant set: {missing}"


def test_ordering_stable_and_row_major(pool_state):
    gs = pool_state
    a = relevant_hexes_in_slot_order(gs)
    b = relevant_hexes_in_slot_order(copy.deepcopy(gs))
    assert [(h.position.x, h.position.y) for h in a] \
        == [(h.position.x, h.position.y) for h in b]
    # Ordering is the row-major slot order, filtered (never re-sorted).
    full = [(h.position.x, h.position.y)
            for h in hexes_in_slot_order(gs)]
    keep = {(h.position.x, h.position.y) for h in a}
    assert [(x, y) for (x, y) in full if (x, y) in keep] \
        == [(h.position.x, h.position.y) for h in a]


def test_statics_always_included(pool_state):
    """Villages and castles/keeps are in the set regardless of reach."""
    from wesnoth_ai.classes import Terrain, TerrainModifiers
    gs = pool_state
    rel = relevant_hex_positions(gs, gs.global_info.current_side)
    for h in gs.map.hexes:
        p = (h.position.x, h.position.y)
        mods = h.modifiers or set()
        if (Terrain.VILLAGE in h.terrain_types
                or TerrainModifiers.CASTLE in mods
                or TerrainModifiers.KEEP in mods):
            assert p in rel, f"static hex {p} missing"


def test_degenerate_no_units(pool_state):
    """A unit-less board: no crash, statics still present, subset
    strictly smaller than the full board."""
    gs = copy.deepcopy(pool_state)
    gs.map.units = set()
    rel = relevant_hex_positions(gs, 1)
    assert rel, "statics should remain on an empty board"
    assert len(rel) < len(gs.map.hexes)
    assert len(relevant_hexes_in_slot_order(gs)) == len(
        {p for p in rel
         if p in {(h.position.x, h.position.y) for h in gs.map.hexes}})


# --------------------------------------------------------------------
# Encoder wiring (T2-B item 1). The flag changes the ACTION SPACE's index
# basis, so these properties are load-bearing, not cosmetic: the trainer
# re-encodes stored states and replays target_idx against them.
# --------------------------------------------------------------------

def _pool_state(seed=901):
    import random
    from tools.scenario_pool import (random_setup, build_scenario_gamestate,
                                     load_factions)
    from tools.wesnoth_sim import WesnothSim
    load_factions()
    setup = random_setup(random.Random(seed), forced_faction=None)
    sim = WesnothSim(build_scenario_gamestate(setup),
                     scenario_id=setup.scenario_id, max_turns=20)
    return sim.gs


def test_encoder_flag_defaults_off_and_is_opt_in():
    from wesnoth_ai.encoder import GameStateEncoder
    assert GameStateEncoder().relevant_set_hexes is False
    assert GameStateEncoder(relevant_set_hexes=True).relevant_set_hexes is True


def test_relevant_encoding_is_subset_in_filter_order():
    """Ordering must be the FILTER of the canonical (y,x) sort, never a
    re-sort: slot indices have to stay comparable and deterministic."""
    from wesnoth_ai.encoder import GameStateEncoder
    gs = _pool_state()
    ef = GameStateEncoder(d_model=32).encode(gs)
    er = GameStateEncoder(d_model=32, relevant_set_hexes=True).encode(gs)
    pf = [(p.x, p.y) for p in ef.hex_positions]
    pr = [(p.x, p.y) for p in er.hex_positions]
    assert len(pr) < len(pf), "relevant set should be smaller on a real map"
    assert set(pr) <= set(pf)
    assert pr == [q for q in pf if q in set(pr)]


def test_relevant_encoding_is_deterministic_across_reencode():
    """The trainer re-encodes STORED states and replays target_idx; any
    nondeterminism here corrupts every replayed transition."""
    import copy
    from wesnoth_ai.encoder import GameStateEncoder
    gs = _pool_state()
    a = GameStateEncoder(d_model=32, relevant_set_hexes=True).encode(gs)
    b = GameStateEncoder(d_model=32, relevant_set_hexes=True).encode(
        copy.deepcopy(gs))
    assert [(p.x, p.y) for p in a.hex_positions] == \
           [(p.x, p.y) for p in b.hex_positions]


def test_default_encoder_still_emits_the_full_board():
    """Regression guard: the flag must not perturb the default path."""
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.visibility import hexes_in_slot_order
    gs = _pool_state()
    e = GameStateEncoder(d_model=32).encode(gs)
    assert e.hex_tokens.size(1) == len(hexes_in_slot_order(gs))


def test_marker_propagates_through_encode():
    """hex_subset must reach EncodedState, or the assert below is dead."""
    from wesnoth_ai.encoder import GameStateEncoder
    gs = _pool_state()
    assert GameStateEncoder(d_model=32).encode(gs).hex_subset is False
    assert GameStateEncoder(d_model=32,
                            relevant_set_hexes=True).encode(gs).hex_subset is True


def test_superset_assert_FIRES_when_the_set_is_short():
    """The guard is only worth having if it actually trips. Shrink the
    relevant set behind the encoder's back and require the mask build to
    raise -- a silently shrunken action space is the failure mode this
    exists to prevent (an excluded hex is an unorderable hex)."""
    import pytest, torch
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    import wesnoth_ai.visibility as vis

    gs = _pool_state()
    enc = GameStateEncoder(d_model=32, relevant_set_hexes=True)
    model = WesnothModel(d_model=32, num_layers=2, num_heads=4, d_ff=64).eval()

    # sanity: intact set enumerates without tripping
    encoded = enc.encode(gs)
    with torch.no_grad():
        out = model(encoded)
    enumerate_legal_actions_with_priors(encoded, out, gs)

    # now drop hexes from the stream while the mask still offers them
    full = vis.relevant_hexes_in_slot_order(gs)
    orig = vis.relevant_hexes_in_slot_order
    try:
        vis.relevant_hexes_in_slot_order = lambda g: full[: max(1, len(full) // 3)]
        import importlib, wesnoth_ai.encoder as enc_mod
        importlib.reload(enc_mod)
        enc2 = enc_mod.GameStateEncoder(d_model=32, relevant_set_hexes=True)
        e2 = enc2.encode(gs)
        with torch.no_grad():
            o2 = model(e2)
        with pytest.raises(AssertionError, match="relevant-set gap"):
            enumerate_legal_actions_with_priors(e2, o2, gs)
    finally:
        vis.relevant_hexes_in_slot_order = orig
        import importlib, wesnoth_ai.encoder as enc_mod
        importlib.reload(enc_mod)


def test_policy_threads_flag_to_both_encoders():
    """Trainer and inference encoders must AGREE: a split would make the
    replayed target_idx index a different hex basis than the one the action
    was chosen in."""
    from wesnoth_ai.transformer_policy import TransformerPolicy
    on = TransformerPolicy(d_model=32, num_layers=2, num_heads=4, d_ff=64,
                           relevant_set_hexes=True)
    assert on._encoder.relevant_set_hexes and on._inference_encoder.relevant_set_hexes
    off = TransformerPolicy(d_model=32, num_layers=2, num_heads=4, d_ff=64)
    assert not off._encoder.relevant_set_hexes
    assert not off._inference_encoder.relevant_set_hexes


def test_worker_learner_index_basis_seam_is_wired():
    """AST/source guard on the seam: worker must parse the flag, honour it
    in its policy, and STAMP the payload; the learner must forward the flag
    and REJECT a mismatched payload. Third bug of this class (dead spool
    telemetry, _combine_stats swallow, dead acting-side advice), so the
    boundary gets a test that reads the boundary."""
    import ast, pathlib
    w = pathlib.Path("tools/selfplay_worker.py").read_text(encoding="utf-8")
    assert "--relevant-set-hexes" in w
    assert '"relevant_set"' in w, "worker must stamp the payload"
    tree = ast.parse(w)
    tp = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
          and getattr(n.func, "id", None) == "TransformerPolicy"]
    assert tp and any(k.arg == "relevant_set_hexes" for k in tp[0].keywords)
    l = pathlib.Path("tools/sim_self_play.py").read_text(encoding="utf-8")
    assert '"--relevant-set-hexes"]' in l, "learner must forward the flag"
    assert "REJECTING" in l and "relevant_set" in l, \
        "learner must reject a mismatched index basis loudly"


def test_holdout_probe_is_discarded_across_an_index_basis_change(tmp_path):
    """holdout CE is the ONE curve we rely on being comparable across
    restarts -- that's why the probe is persisted. Restoring a probe
    encoded under the other hex basis would keep the curve looking
    continuous while silently making it a different measurement."""
    from tools.mcts_policy import MCTSPolicy
    from wesnoth_ai.transformer_policy import TransformerPolicy
    from tools.mcts import MCTSConfig
    import pickle

    arch = dict(d_model=32, num_layers=2, num_heads=4, d_ff=64)
    p_off = MCTSPolicy(TransformerPolicy(**arch), MCTSConfig(), holdout_size=4)
    # hand-write a probe stamped with the OPPOSITE basis
    f = tmp_path / "probe.holdout"
    f.write_bytes(pickle.dumps({"experiences": [], "games": 1,
                                "target": 4, "relevant_set": True}))
    assert p_off.load_holdout(f) is False, "must refuse a foreign basis"

    # matching basis loads (empty list is still a successful restore)
    f2 = tmp_path / "probe2.holdout"
    f2.write_bytes(pickle.dumps({"experiences": [], "games": 1,
                                 "target": 4, "relevant_set": False}))
    assert p_off.load_holdout(f2) is True
