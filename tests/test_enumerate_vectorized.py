"""Vectorized legal-action enumeration == the per-actor reference
(wesnoth_ai/action_sampler.py, plan step 1.2: enumeration was 6.9 ms
of the 12.4 ms Python per decision in the 2026-09-04 baseline)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def _policy():
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(1)
    return TransformerPolicy(d_model=32, num_layers=1, num_heads=2,
                             d_ff=64, device=torch.device("cpu"))


def _has_attack(gs):
    """Any own unit adjacent to a visible enemy it may still attack."""
    from tools.abilities import hex_neighbors
    side = gs.global_info.current_side
    enemies = {(u.position.x, u.position.y) for u in gs.map.units if u.side != side}
    for u in gs.map.units:
        if u.side == side and not u.has_attacked:
            if any(n in enemies for n in hex_neighbors(u.position.x, u.position.y)):
                return True
    return False


def _states(n_attack=4, n_plain=6):
    """Scenario starts (recruit branch) plus dummy-played midgame
    states, some of which have legal attacks."""
    import copy
    from tools.elo_ladder import _ScriptedAdapter
    from tools.eval_sim import _PolicyPair, _play_one_eval_game
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.dummy_policy import DummyPolicy
    starts = [build_scenario_gamestate(random_setup(random.Random(s))) for s in (1, 2)]
    attack, plain = [], []

    class _Rec:
        def __init__(self, inner):
            self._inner = inner
            self._n = 0

        def select_action(self, gs, **kw):
            self._n += 1
            if gs.global_info.turn_number >= 3 and self._n % 4 == 0:
                if _has_attack(gs) and len(attack) < n_attack:
                    attack.append(copy.deepcopy(gs))
                elif len(plain) < n_plain:
                    plain.append(copy.deepcopy(gs))
            return self._inner.select_action(gs, **kw)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    g = 0
    while (len(attack) < n_attack or len(plain) < n_plain) and g < 30:
        setup = random_setup(random.Random(100 + g))
        g += 1
        sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id,
                         max_turns=25)
        _play_one_eval_game(
            sim,
            _PolicyPair(policy=_Rec(_ScriptedAdapter(DummyPolicy())), label="a", side=1),
            _PolicyPair(policy=_Rec(_ScriptedAdapter(DummyPolicy())), label="b", side=2),
            game_label=f"enum{g}")
    assert len(attack) >= 1, "no attack-bearing state harvested"
    return starts + plain + attack


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
