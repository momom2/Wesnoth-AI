"""The policy-gradient loss and the readings of the same forward
(wesnoth_ai/pg_loss.py): the signed weight's sign survives, the class
masses and the KL are the server's enumerated law's, and the head scores are
the derivatives of log pi along the heads' own weights."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai import pg_loss  # noqa: E402
from wesnoth_ai.server_priors import KIND_ATTACK, KIND_END_TURN, KIND_MOVE, KIND_RECRUIT  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
CPU = torch.device("cpu")


def _positions():
    """Positions of two played games, attacks among their legal actions."""
    from helpers.pg_games import command_positions, played_games
    games = played_games()
    return games, [p for name in ("g0", "g2") for p in command_positions(games.records[games.names[name]],
                                                                          games.vocab)]


def _compacts(padded, packs):
    from wesnoth_ai.server_priors import batched_priors
    with torch.no_grad():
        return batched_priors(padded, list(packs))


def _choose(compact, i: int):
    """Row i's action: a legal one of a kind that cycles over the rows,
    the first kind present in the cycle's order."""
    order = [KIND_ATTACK, KIND_RECRUIT, KIND_MOVE, KIND_END_TURN]
    kinds = np.asarray(compact.kind)
    for k in order[i % 4:] + order[:i % 4]:
        hits = np.flatnonzero(kinds == k)
        if len(hits):
            j = int(hits[(7 * i) % len(hits)])
            return (int(compact.actor[j]), int(compact.kind[j]), int(compact.target[j]), int(compact.weapon[j]))
    raise AssertionError("a position with no legal action")


def _batch(padded, packs, actions):
    U = max(u for u, _, _ in padded.sizes)
    R = max(r for _, r, _ in padded.sizes)
    H = max(h for _, _, h in padded.sizes)
    return pg_loss.stage_actions(list(range(len(packs))), list(packs), actions, U + R + 1, H,
                                 padded.type_logits.shape[2], padded.weapon_logits.shape[2], CPU)


def _setup(network=None):
    from helpers.pg_games import initial_forward
    games, positions = _positions()
    encoder, model = network or (games.encoder, games.model)
    raws, packs = [r for r, _ in positions], [m for _, m in positions]
    with torch.no_grad():
        padded = initial_forward(model, encoder, raws)
    compacts = _compacts(padded, packs)
    actions = [_choose(c, i) for i, c in enumerate(compacts)]
    return games, raws, packs, padded, compacts, actions


@needs_core
def test_a_negative_weight_lowers_the_log_probability_of_its_action():
    """The gradient of the signed loss keeps the weight's sign: a step down
    the loss of a row weighted -1 lowers its action's log probability, of
    one weighted +1 raises it, and the gradients are opposite. The
    imitation loss's `prob > 0` filter would give the negative row none."""
    from helpers.pg_games import initial_forward
    from wesnoth_ai.own_loss import own_policy_loss
    games, raws, packs, _, _, actions = _setup()
    model, encoder = games.model, games.encoder
    params = [p for p in list(model.parameters()) + list(encoder.parameters()) if p.requires_grad]

    def grad_and_logp(w_first: float):
        padded = initial_forward(model, encoder, raws[:1])
        batch = _batch(padded, packs[:1], actions[:1])
        loss = pg_loss.pg_loss(padded, batch, torch.tensor([w_first]))
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        lp = float(pg_loss.action_log_probs(padded, batch)[0].detach())
        filtered = own_policy_loss(padded, dataclasses.replace(batch, prob=torch.tensor([[w_first]])))
        return grads, lp, float(filtered.sum().detach())

    plus, lp0, _ = grad_and_logp(1.0)
    minus, _, filtered = grad_and_logp(-1.0)
    flat_plus = torch.cat([g.flatten() for g in plus if g is not None])
    flat_minus = torch.cat([g.flatten() for g in minus if g is not None])
    assert flat_plus.norm() > 0
    assert torch.allclose(flat_minus, -flat_plus, atol=1e-7)
    assert filtered == 0.0, "the imitation loss drops a negative weight: the trap this loss avoids"
    for sign, grads in ((-1.0, minus), (1.0, plus)):
        saved = [p.detach().clone() for p in params]
        with torch.no_grad():
            for p, g in zip(params, grads):
                if g is not None:
                    p -= 1e-2 * g / flat_plus.norm()
        with torch.no_grad():
            padded = initial_forward(model, encoder, raws[:1])
            lp = float(pg_loss.action_log_probs(padded, _batch(padded, packs[:1], actions[:1]))[0])
        with torch.no_grad():
            for p, s in zip(params, saved):
                p.copy_(s)
        assert (lp - lp0) * sign > 0, (sign, lp0, lp)


@needs_core
def test_the_class_masses_are_the_sums_of_the_servers_priors():
    _, _, packs, padded, compacts, actions = _setup()
    masses = pg_loss.class_masses(padded, _batch(padded, packs, actions)).numpy()
    want = np.array([[float(np.asarray(c.prior)[np.asarray(c.kind) == k].sum()) / float(np.sum(c.prior))
                      for k in (KIND_END_TURN, KIND_ATTACK, KIND_RECRUIT)] for c in compacts])
    assert (want[:, 1] > 0).any() and (want[:, 2] > 0).any(), "the positions hold attacks and recruits"
    assert np.allclose(masses, want, atol=2e-6)


@needs_core
def test_the_legal_kl_is_the_kl_of_the_enumerated_laws():
    """KL(p || q) of two networks' joint laws by the chain rule equals the
    sum over every legal action of p log(p / q), the actions enumerated by
    the server's priors; KL(p || p) is 0."""
    from helpers.pg_games import initial_forward, tiny_network
    games, raws, packs, padded, compacts, actions = _setup()
    _, enc_q, model_q = tiny_network(seed=11)
    with torch.no_grad():
        other = initial_forward(model_q, enc_q, raws)
    compacts_q = _compacts(other, packs)
    batch = _batch(padded, packs, actions)
    kl = pg_loss.legal_kl(padded, other, batch.views).numpy()
    want = []
    for c, d in zip(compacts, compacts_q):
        keys_p = list(zip(c.actor.tolist(), c.kind.tolist(), c.target.tolist(), c.weapon.tolist()))
        keys_q = list(zip(d.actor.tolist(), d.kind.tolist(), d.target.tolist(), d.weapon.tolist()))
        assert keys_p == keys_q
        p = np.asarray(c.prior) / np.sum(c.prior)
        q = np.asarray(d.prior) / np.sum(d.prior)
        want.append(float((p * np.log(p / q)).sum()))
    assert min(want) >= 0 and max(want) > 1e-3
    assert np.allclose(kl, want, rtol=1e-4, atol=1e-6)
    assert np.allclose(pg_loss.legal_kl(padded, padded, batch.views).numpy(), 0.0, atol=1e-7)


class _Network(torch.nn.Module):
    """The model's embedded forward, its parameters named as the pass
    names them (\"model.<name>\")."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, streams, memory):
        return self.model.forward_embedded(streams, memory=memory)


@needs_core
def test_the_head_scores_are_the_derivatives_along_the_heads_weights():
    """Scaling a head's final linear map by (1 + tau) in the network
    itself: d log pi(a) / d tau at 0, by autograd, is the head's score."""
    from torch.func import functional_call
    games, raws, packs, padded, compacts, actions = _setup()
    model, encoder = games.model, games.encoder
    batch = _batch(padded, packs, actions)
    scores = pg_loss.head_scores(padded, batch).numpy()
    network = _Network(model)
    params = dict(network.named_parameters())
    staged = encoder.stage_raws(list(raws), device=CPU)
    for h, head in enumerate(pg_loss.HEADS):
        tau = torch.zeros((), requires_grad=True)
        scaled = {n: params[n] * (1 + tau) for n in pg_loss.HEAD_PARAMETERS[head]}
        out = functional_call(network, scaled, (encoder.embed_staged(staged),
                                                [model.initial_memory(64) for _ in raws]))
        lp = pg_loss.action_log_probs(out, batch)
        for i in range(len(raws)):
            (g,) = torch.autograd.grad(lp[i], tau, retain_graph=True)
            assert float(g) == pytest.approx(float(scores[i, h]), rel=1e-4, abs=1e-6), (head, i)
    kinds = np.array([a[1] for a in actions])
    assert np.all(scores[kinds == KIND_END_TURN, 1:] == 0) and np.all(scores[kinds != KIND_ATTACK, 3] == 0)
    assert np.abs(scores[kinds == KIND_ATTACK, 3]).max() > 0
