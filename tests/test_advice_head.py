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
