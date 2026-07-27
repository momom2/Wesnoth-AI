"""Model-side advice path (docs/detector_training_signal.md): a separate,
zero-init cross-attention block that lets the policy condition on detector
advice tokens with a LEARNABLE per-actor gate.

The load-bearing properties:
  - zero-init graft: with advice_out=0 the advice contributes NOTHING, so
    (a) advice tokens don't change the output at init, and (b) an existing
    (advice=False) checkpoint loads into an advice=True model and behaves
    identically -- no checkpoint is invalidated.
  - once trained (advice_out != 0) the advice tokens DO steer the policy,
    and gradients reach the gate + cross-attention.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai.encoder import GameStateEncoder                  # noqa: E402
from wesnoth_ai.model import WesnothModel                        # noqa: E402
from sim_test_helpers import fresh_scenario_sim                  # noqa: E402

_ARCH = dict(d_model=32, num_layers=2, num_heads=4, d_ff=64)


def _encoded():
    sim = fresh_scenario_sim(seed=3, max_turns=10,
                             scenario_id="multiplayer_The_Freelands")
    return GameStateEncoder(d_model=_ARCH["d_model"]), sim.gs


def test_advice_zero_init_contributes_nothing():
    enc, gs = _encoded()
    model = WesnothModel(advice=True, **_ARCH).eval()
    encoded = enc.encode(gs)
    with torch.no_grad():
        out_none = model(encoded)                     # advice_tokens = None
        encoded.advice_tokens = torch.randn(1, 3, _ARCH["d_model"])
        out_adv = model(encoded)                      # advice present, zero-init
    # advice_out=0 -> advice contributes nothing -> identical policy.
    assert torch.allclose(out_none.actor_logits, out_adv.actor_logits, atol=1e-6)
    assert torch.allclose(out_none.target_logits, out_adv.target_logits, atol=1e-6)


def test_advice_steers_policy_once_trained():
    enc, gs = _encoded()
    model = WesnothModel(advice=True, **_ARCH).eval()
    encoded = enc.encode(gs)
    with torch.no_grad():
        base = model(encoded).actor_logits.clone()
        # simulate a trained advice path: give advice_out non-zero weights.
        model.advice_out.weight.normal_(0.0, 0.5)
        encoded.advice_tokens = torch.randn(1, 3, _ARCH["d_model"])
        steered = model(encoded).actor_logits
    assert not torch.allclose(base, steered, atol=1e-5)


def test_advice_gradients_flow():
    enc, gs = _encoded()
    model = WesnothModel(advice=True, **_ARCH).train()
    encoded = enc.encode(gs)
    with torch.no_grad():
        model.advice_out.weight.normal_(0.0, 0.5)     # active path
    encoded.advice_tokens = torch.randn(1, 3, _ARCH["d_model"])
    model(encoded).actor_logits.sum().backward()
    assert model.advice_gate.weight.grad is not None
    assert model.advice_gate.weight.grad.abs().sum() > 0
    assert model.advice_out.weight.grad is not None


def test_advice_false_checkpoint_grafts_cleanly():
    enc, gs = _encoded()
    base = WesnothModel(**_ARCH).eval()               # advice=False
    grafted = WesnothModel(advice=True, **_ARCH).eval()
    res = grafted.load_state_dict(base.state_dict(), strict=False)
    # only the advice params are missing; nothing unexpected.
    assert res.unexpected_keys == []
    assert all(k.startswith("advice_") for k in res.missing_keys), res.missing_keys
    encoded = enc.encode(gs)
    with torch.no_grad():
        out_base = base(encoded)
        out_grafted = grafted(encoded)                # advice_tokens = None
    assert torch.allclose(out_base.actor_logits, out_grafted.actor_logits, atol=1e-6)
    assert torch.allclose(out_base.value, out_grafted.value, atol=1e-6)


def test_build_advice_tokens_shape_and_forward():
    enc, gs = _encoded()
    model = WesnothModel(advice=True, **_ARCH).eval()
    encoded = enc.encode(gs)
    U, H = encoded.unit_tokens.shape[1], encoded.hex_tokens.shape[1]
    assert U > 0 and H > 0
    motif_ids = torch.tensor([0, 1], dtype=torch.long)
    feats = torch.tensor([[1., 0.5, 0., 0.], [1., 0.2, 0.01, 1.]])
    muidx = torch.tensor([0, min(1, U - 1)], dtype=torch.long)
    dhidx = torch.tensor([0, min(1, H - 1)], dtype=torch.long)
    tok = model.build_advice_tokens(encoded, motif_ids, feats, muidx, dhidx)
    assert tok.shape == (1, 2, _ARCH["d_model"])
    with torch.no_grad():
        encoded.advice_tokens = tok
        out = model(encoded)
    assert out.actor_logits.shape[0] == 1
    # empty opportunities -> empty tokens
    empty = model.build_advice_tokens(
        encoded, torch.zeros(0, dtype=torch.long), torch.zeros(0, 4),
        torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
    assert empty.shape == (1, 0, _ARCH["d_model"])


def test_advice_out_bootstraps_from_zero_init():
    """At the zero-init graft advice_out=0 (advice contributes nothing), but
    its GRADIENT is non-zero (gate*attn_out) -> it bootstraps up under
    training. The gate's own grad is ~0 until advice_out!=0 (intended)."""
    enc, gs = _encoded()
    model = WesnothModel(advice=True, **_ARCH).train()
    encoded = enc.encode(gs)
    encoded.advice_tokens = torch.randn(1, 3, _ARCH["d_model"])
    model(encoded).actor_logits.sum().backward()
    assert model.advice_out.weight.grad is not None
    assert model.advice_out.weight.grad.abs().sum() > 0


def test_reforward_advice_gives_advice_gradients():
    """Mimic the trainer reforward: prospective advisor -> features -> build
    tokens (grad on) -> forward -> backward reaches the advice params, so the
    gate path LEARNS from the policy loss."""
    from tools.detector_advisor import (
        prospective_opportunities, opportunities_to_features)
    from test_detector_advisor import _backstab_side_turn
    st, *_ = _backstab_side_turn()
    gs = st.pre_state
    gs.global_info.current_side = 1
    enc = GameStateEncoder(d_model=_ARCH["d_model"])
    model = WesnothModel(advice=True, **_ARCH).train()
    encoded = enc.encode(gs)
    opps = prospective_opportunities(gs, side=1)
    assert len(opps) >= 1
    mids, feats, mu, dh = opportunities_to_features(encoded, opps)
    encoded.advice_tokens = model.build_advice_tokens(encoded, mids, feats, mu, dh)
    model(encoded).actor_logits.sum().backward()
    assert model.advice_out.weight.grad.abs().sum() > 0
    assert model.advice_motif_embed.weight.grad is not None
    assert model.advice_feat_proj.weight.grad is not None


def test_batched_advice_equals_per_sample():
    """forward_batch's batched advice path == per-sample forward: advice
    refines the same actors, the key mask excludes pads, and no-advice rows
    (all-masked keys) contribute nothing without NaN. Mixed advice lengths
    (2 / none / 3) with an ACTIVE advice_out."""
    enc, gs = _encoded()
    model = WesnothModel(advice=True, **_ARCH).eval()
    with torch.no_grad():
        model.advice_out.weight.normal_(0.0, 0.3)
        model.advice_out.bias.normal_(0.0, 0.3)
    d = _ARCH["d_model"]
    e0 = enc.encode(gs); e0.advice_tokens = torch.randn(1, 2, d)
    e1 = enc.encode(gs)                                   # no advice
    e2 = enc.encode(gs); e2.advice_tokens = torch.randn(1, 3, d)
    encs = [e0, e1, e2]
    with torch.no_grad():
        batched = model.forward_batch(encs)
        per = [model.forward(e) for e in encs]
    for b, p in zip(batched, per):
        assert torch.allclose(b.actor_logits, p.actor_logits, atol=1e-5), \
            (b.actor_logits - p.actor_logits).abs().max()
        assert torch.allclose(b.target_logits, p.target_logits, atol=1e-5)
        assert torch.allclose(b.value, p.value, atol=1e-5)
