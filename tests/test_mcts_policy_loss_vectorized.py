"""Optimization #5: the vectorized MCTS factored policy loss must
match the original per-tuple loop within float tolerance.

The vectorized path groups the per-(actor/type/target/weapon) NLL
terms per cached log-prob vector and reduces each with one
index_select+sum, collapsing the backward graph from O(visit-count
tuples) to O(unique vectors). This reassociates the float32
summation, so it is NOT bit-identical -- this test pins that the
divergence stays at ULP scale (loss AND every parameter gradient
within 1e-5 relative), which is the gate the optimization ships
behind (trainer.TrainerConfig.vectorized_mcts_policy_loss).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import torch  # noqa: E402

from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402
from wesnoth_ai.trainer import _mcts_factored_policy_loss  # noqa: E402
from sim_test_helpers import fresh_scenario_sim  # noqa: E402


def _build_case():
    """A real mini-state encoded + forwarded through a tiny model,
    with visit_counts synthesized from the enumerated legal actions
    (valid actor/type/target/weapon indices, varied counts)."""
    pol = TransformerPolicy(device=torch.device("cpu"),
                            d_model=64, num_layers=2, num_heads=4, d_ff=128)
    enc, mdl = pol._encoder, pol._model
    enc.eval(); mdl.eval()                       # deterministic forward
    sim = fresh_scenario_sim(seed=11, max_turns=6, mini=True)
    gs = sim.gs
    # Pre-register vocab once so re-encodes are identical.
    enc.register_names(gs)

    encoded0 = enc.encode(gs)
    output0 = mdl(encoded0)
    priors = enumerate_legal_actions_with_priors(encoded0, output0, gs)
    # 5-tuple schema (actor, target, weapon, count, type); varied
    # positive counts to exercise the weighting.
    vc = [(p.actor_idx, p.target_idx, p.weapon_idx, (i % 7) + 1, p.type_idx)
          for i, p in enumerate(priors)]
    return enc, mdl, gs, vc


def _loss_and_grads(enc, mdl, gs, vc, *, vectorized):
    mdl.zero_grad(set_to_none=True)
    enc.zero_grad(set_to_none=True)
    encoded = enc.encode(gs)
    output = mdl(encoded)
    loss, total_v, _ = _mcts_factored_policy_loss(
        encoded, output, gs, vc, vectorized=vectorized)
    loss.backward()
    grads = {
        n: p.grad.detach().clone()
        for n, p in (list(mdl.named_parameters())
                     + list(enc.named_parameters()))
        if p.grad is not None
    }
    return float(loss.item()), float(total_v), grads


def test_vectorized_policy_loss_matches_loop_within_tolerance():
    enc, mdl, gs, vc = _build_case()
    assert len(vc) >= 3, "need a few legal actions to exercise the buckets"

    l_ref, tv_ref, g_ref = _loss_and_grads(enc, mdl, gs, vc, vectorized=False)
    l_vec, tv_vec, g_vec = _loss_and_grads(enc, mdl, gs, vc, vectorized=True)

    assert tv_ref == tv_vec, "total_visits must be identical"
    # Loss within ULP-scale relative tolerance.
    assert abs(l_vec - l_ref) <= 1e-5 * abs(l_ref) + 1e-7, (
        f"loss mismatch: loop={l_ref} vectorized={l_vec}")

    # Every parameter gradient that the loop produced must match.
    assert g_ref, "loop produced no gradients -- test is vacuous"
    assert set(g_ref) == set(g_vec), "different params got gradients"
    worst = 0.0
    for n, gr in g_ref.items():
        gv = g_vec[n]
        scale = gr.abs().max().item()
        md = (gv - gr).abs().max().item()
        worst = max(worst, md)
        assert md <= 1e-5 * scale + 1e-6, (
            f"grad mismatch on {n}: max|diff|={md:.3e} scale={scale:.3e}")
    # Sanity: the two paths are genuinely close, not both-zero.
    assert l_ref != 0.0


def test_both_modes_run_and_are_differentiable():
    """Guards the vectorized path's basic contract: positive
    total_visits, finite scalar loss, real gradients."""
    enc, mdl, gs, vc = _build_case()
    l, tv, g = _loss_and_grads(enc, mdl, gs, vc, vectorized=True)
    assert tv > 0
    assert l == l  # not NaN
    assert any(v.abs().sum().item() > 0 for v in g.values())
