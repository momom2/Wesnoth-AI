"""A tiny policy, a few played states, and the comparison of two
enumerations of legal actions with priors: the fixtures of the
server-priors parity tests."""
import numpy as np
import torch


def _policy():
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(3)
    return TransformerPolicy(d_model=32, num_layers=1, num_heads=2,
                             d_ff=64, device=torch.device("cpu"))


def _states():
    from helpers.played_states import _states as harvest
    return harvest(n_attack=2, n_plain=3)


def _same(ref, got):
    assert [la.action for la in ref] == [la.action for la in got]
    assert [(la.actor_idx, la.type_idx, la.target_idx, la.weapon_idx) for la in ref] == \
           [(la.actor_idx, la.type_idx, la.target_idx, la.weapon_idx) for la in got]
    assert np.allclose([la.prior for la in ref], [la.prior for la in got],
                       rtol=1e-5, atol=1e-9)
