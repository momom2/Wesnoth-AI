"""Vectorized legal-action enumeration == the per-actor reference
(wesnoth_ai/action_sampler.py, plan step 1.2: enumeration was 6.9 ms
of the 12.4 ms Python per decision in the 2026-09-04 baseline)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from helpers.played_states import _states  # noqa: E402


def _policy():
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(1)
    return TransformerPolicy(d_model=32, num_layers=1, num_heads=2,
                             d_ff=64, device=torch.device("cpu"))


def _key(la):
    return (la.actor_idx, la.type_idx, la.target_idx, la.weapon_idx, la.action["type"])


@pytest.fixture(scope="module")
def enumerations():
    from wesnoth_ai.action_sampler import (_enumerate_legal_actions_reference,
                                           enumerate_legal_actions_with_priors)
    policy = _policy()
    enc, model = policy._inference_encoder, policy._inference_model
    pairs = []
    with torch.no_grad():
        for gs in _states():
            e = enc.encode(gs)
            o = model(e)
            pairs.append((_enumerate_legal_actions_reference(e, o, gs),
                          enumerate_legal_actions_with_priors(e, o, gs)))
    return pairs


def test_same_actions_same_order(enumerations):
    for ref, vec in enumerations:
        assert [_key(la) for la in ref] == [_key(la) for la in vec]
        assert [la.action for la in ref] == [la.action for la in vec]


def test_same_priors(enumerations):
    for ref, vec in enumerations:
        r = np.array([la.prior for la in ref])
        v = np.array([la.prior for la in vec])
        assert np.allclose(r, v, rtol=1e-5, atol=1e-9)
        assert abs(v.sum() - 1.0) < 1e-4


def test_every_branch_exercised(enumerations):
    kinds = {la.action["type"] for _, vec in enumerations for la in vec}
    assert {"recruit", "move", "attack", "end_turn"} <= kinds


def test_reference_env_switch(monkeypatch):
    import importlib
    import wesnoth_ai.action_sampler as sampler
    monkeypatch.setenv("WESNOTH_ENUM_REFERENCE", "1")
    importlib.reload(sampler)
    try:
        assert sampler._ENUM_REFERENCE is True
    finally:
        monkeypatch.delenv("WESNOTH_ENUM_REFERENCE")
        importlib.reload(sampler)
        assert sampler._ENUM_REFERENCE is False
