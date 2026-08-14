"""GBC head-A tests: trunk tap alignment, head mechanics, metric
helpers -- production code paths (tools/gbc_heads.py) on a real
model + encoder, no mirrored logic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402
from tools.gbc_heads import (  # noqa: E402
    GBCHeads, TrunkTap, auc, ece, goal_token,
)


@pytest.fixture(scope="module")
def policy():
    torch.manual_seed(0)
    return TransformerPolicy(device=torch.device("cpu"), d_model=32,
                             num_layers=1, num_heads=4, d_ff=64)


def test_trunk_tap_alignment(policy):
    """The tap's unit slice must align with EncodedState.unit_ids --
    the id-keyed contract that avoids the fa95da5 slot-index trap.
    Verified by matching against the model's own unit_ctx-derived
    actor ordering: actor slot i (a UNIT slot) and tap unit i are
    THE SAME token."""
    sim = fresh_scenario_sim(seed=3, max_turns=6, mini=True)
    gs = sim.gs
    model = policy._inference_model
    enc = policy._inference_encoder
    tap = TrunkTap(model)
    try:
        with torch.no_grad():
            encoded = enc.encode(gs)
            output = model(encoded)
        sl = tap.slices(encoded)
        U = encoded.unit_tokens.size(1)
        assert sl["unit"].shape[1] == U == len(encoded.unit_ids)
        assert sl["global"].shape[1] == 1
        # Actor slot ordering is [units, recruits, end_turn]; the
        # actor head consumed exactly the tap's unit tokens.
        assert output.actor_logits.shape[1] >= U
        # goal_token resolves ids and village hexes, and returns
        # None for unknown entities (fog-honest miss).
        uid = encoded.unit_ids[0]
        t = goal_token(encoded, sl, ("u", uid))
        assert t is not None and t.shape == (32,)
        assert torch.equal(t, sl["unit"][0, 0])
        assert goal_token(encoded, sl, ("u", "no_such_unit")) is None
        pos = encoded.hex_positions[3]
        t2 = goal_token(encoded, sl, ("v", pos.x, pos.y))
        assert t2 is not None and torch.equal(t2, sl["hex"][0, 3])
        assert goal_token(encoded, sl, ("v", -99, -99)) is None
    finally:
        tap.remove()


def test_head_a_learns_synthetic_signal(policy):
    """Head A must fit a learnable synthetic mapping from real trunk
    tokens (frozen) -- the smallest possible 'the gradient path
    works' proof. Labels: unit token belongs to side 1."""
    sim = fresh_scenario_sim(seed=3, max_turns=6, mini=True)
    model = policy._inference_model
    enc = policy._inference_encoder
    tap = TrunkTap(model)
    try:
        with torch.no_grad():
            encoded = enc.encode(sim.gs)
            model(encoded)
        sl = tap.slices(encoded)
    finally:
        tap.remove()
    z_units = sl["unit"][0].detach()
    z_glob = sl["global"][0, 0].detach()
    by_id = {u.id: u for u in sim.gs.map.units}
    sides = torch.tensor(
        [1.0 if by_id[uid].side == 1 else 0.0
         for uid in encoded.unit_ids])
    heads = GBCHeads(d_model=32)
    opt = torch.optim.Adam(heads.parameters(), lr=3e-3)
    pred_idx = torch.zeros(len(z_units), dtype=torch.long)
    losses = []
    for _ in range(200):
        logits = heads.batch_a(z_units, z_glob, pred_idx)[:, 0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, sides)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] * 0.5, \
        f"head A failed to fit a trivially separable signal: " \
        f"{losses[0]:.3f} -> {losses[-1]:.3f}"


def test_auc_and_ece():
    assert auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert abs(auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) - 0.5) < 1e-9
    import math
    assert math.isnan(auc([0.5], [1]))
    # Perfectly calibrated probabilities -> ECE ~ 0.
    probs = [0.05] * 20 + [0.95] * 20
    labels = [0] * 19 + [1] + [1] * 19 + [0]
    assert ece(probs, labels) < 0.05
    # Confidently wrong -> large ECE.
    assert ece([0.99] * 10, [0] * 10) > 0.9
