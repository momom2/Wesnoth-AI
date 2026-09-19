"""The batched factored policy loss step_mcts runs
(trainer._batched_factored_policy_loss) against the per-state
reference (trainer._mcts_factored_policy_loss_reference, the original
per-tuple loop) on real mid-game positions: the policy loss, the
"entropy" field and the gradient must agree within float32
reassociation noise through step_mcts at batch 1 and at batch 4
(chunks with different hex counts), with every kind of term present
(actor, type, attack / move / recruit targets, weapons, end_turn), a
legacy 4-tuple experience, an empty experience, unequal game and
policy weights, two combat-oracle anneal steps, and out-of-range
stored indices. Also the stage-timing hook.

Positions: the first bench states of configs/bench_states.json,
rebuilt from the replay dataset (skipped when it is not on this
machine; WESNOTH_BENCH_DATASET points at a packed copy).
"""
from __future__ import annotations

import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.bench_pipeline import load_states  # noqa: E402
from tools.bench_train_step import configure_trainer_like_az_loop  # noqa: E402
from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors  # noqa: E402
from wesnoth_ai.trainer import (  # noqa: E402
    STEP_MCTS_STAGES, MCTSExperience, _mcts_factored_policy_loss_reference,
)
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402

MANIFEST = ROOT / "configs" / "bench_states.json"
DATASET = Path(os.environ.get("WESNOTH_BENCH_DATASET",
                              ROOT / "replays_dataset_imitation"))
N_STATES = 6
LOSS_REL_TOL = 1e-5
COSINE_MIN = 0.9999


def _bench_states(n: int):
    import json
    if not MANIFEST.exists():
        pytest.skip(f"{MANIFEST} missing")
    first = json.loads(MANIFEST.read_text(encoding="utf-8"))["states"][0]["file"]
    if not (DATASET / first).exists():
        pytest.skip(f"bench dataset not at {DATASET} (WESNOTH_BENCH_DATASET)")
    return [gs for gs, _scenario in load_states(MANIFEST, DATASET, n)]


@pytest.fixture(scope="module")
def case():
    """A small policy configured as az_loop configures its trainer,
    and experiences on real positions whose visit tables cover
    every term kind."""
    states = _bench_states(N_STATES)
    torch.manual_seed(0)
    policy = TransformerPolicy(device=torch.device("cpu"),
                               d_model=64, num_layers=2, num_heads=4, d_ff=128)
    configure_trainer_like_az_loop(policy._trainer)
    enc, mdl = policy._inference_encoder, policy._inference_model
    rng = random.Random(0)
    exps = []
    kinds = Counter()
    for i, gs in enumerate(states):
        enc.register_names(gs)
        with torch.no_grad():
            e0 = enc.encode(gs)
            legal = enumerate_legal_actions_with_priors(e0, mdl(e0), gs)
        assert legal
        if i % 2 == 0:
            # Every legal action, varied counts: the Gumbel-root regime.
            chosen = [(la, float((j % 7) + 1)) for j, la in enumerate(legal)]
        else:
            # 32 draws: the loop's search budget.
            counts = Counter(rng.choices(range(len(legal)), k=32))
            chosen = [(legal[j], float(c)) for j, c in sorted(counts.items())]
        for la, _c in chosen:
            kinds[la.action.get("type")] += 1
        visits = [(la.actor_idx, la.target_idx, la.weapon_idx, c, la.type_idx)
                  for la, c in chosen]
        exps.append(MCTSExperience(
            game_state=gs, visit_counts=visits, z=rng.choice((-1.0, 1.0)),
            game_weight=1.0 / (1 + i % 3), policy_weight=0.5 if i == 1 else 1.0,
            game_id=f"g{i // 2}", decision_step=0 if i < 3 else 5000))
    # Legacy 4-tuples (no type: union target rows, no type term).
    exps[2].visit_counts = [v[:4] for v in exps[2].visit_counts]
    # No visits at all (step_control's held-out probes build these).
    exps.append(MCTSExperience(game_state=states[0], visit_counts=[], z=1.0,
                               policy_weight=0.0, game_id="empty"))
    for kind in ("attack", "move", "recruit", "end_turn"):
        assert kinds[kind] > 0, f"the bench positions carry no {kind} target: {kinds}"
    return policy, exps


def _flat_grad(trainer) -> torch.Tensor:
    params = list(trainer.model.parameters()) + list(trainer.encoder.parameters())
    return torch.cat([(p.grad.detach().flatten() if p.grad is not None
                       else torch.zeros(p.numel())) for p in params]).double()


def reference_policy_step(policy, exps):
    """Policy loss, "entropy" and gradient of the per-tuple reference,
    aggregated as step_mcts aggregated per-state losses (weighted mean
    over experiences by game weight x policy weight; each state's
    loss normalized by its total visits). Same padded forward as the
    batched path, so only the loss computation differs."""
    tr = policy._trainer
    enc, mdl = tr.encoder, tr.model
    enc.eval()
    mdl.eval()
    tr.optimizer.zero_grad(set_to_none=True)
    total_gw = sum(e.game_weight for e in exps)
    # The reference encodes what the trainer encodes: the encoder's own
    # switches (basis, fog gate, terrain view), never a bare encode_raw.
    raws = [enc.raw_of(e.game_state) for e in exps]
    encoded = enc.encode_from_raw_batch(raws)
    outputs = mdl.forward_padded(encoded).samples()
    total = torch.zeros(())
    visits = 0.0
    nlp = 0.0
    for e, en, out in zip(exps, encoded, outputs):
        loss, tv, mean_nlp = _mcts_factored_policy_loss_reference(
            en, out, e.game_state, e.visit_counts, vectorized=False,
            decision_step=e.decision_step)
        total = total + loss * e.game_weight * e.policy_weight / total_gw
        visits += tv
        nlp += mean_nlp * tv
    total.backward()
    grad = _flat_grad(tr)
    tr.optimizer.zero_grad(set_to_none=True)
    return float(total.item()), nlp / visits, grad


def batched_policy_step(policy, exps, batch_size: int):
    """step_mcts with the optimizer stubbed, no clipping and the value
    coefficient at 0: its policy loss, "entropy" and gradient."""
    tr = policy._trainer
    real_step = tr.optimizer.step
    cfg = tr.config
    saved = (cfg.grad_clip, cfg.value_coef, cfg.train_batch_size)
    tr.optimizer.step = lambda *a, **k: None
    cfg.grad_clip, cfg.value_coef, cfg.train_batch_size = 1e9, 0.0, batch_size
    try:
        stats = tr.step_mcts(exps)
    finally:
        tr.optimizer.step = real_step
        cfg.grad_clip, cfg.value_coef, cfg.train_batch_size = saved
    grad = _flat_grad(tr)
    tr.optimizer.zero_grad(set_to_none=True)
    return float(stats.policy_loss), float(stats.entropy), grad


def _assert_parity(ref, cand, label: str):
    loss_r, ent_r, g_r = ref
    loss_c, ent_c, g_c = cand
    loss_rel = abs(loss_c - loss_r) / abs(loss_r)
    ent_rel = abs(ent_c - ent_r) / abs(ent_r)
    cosine = float(g_c @ g_r) / (float(g_c.norm()) * float(g_r.norm()))
    rel_l2 = float((g_c - g_r).norm()) / float(g_r.norm())
    print(f"{label}: loss ref {loss_r:.6f} batched {loss_c:.6f} rel {loss_rel:.2e} | "
          f"entropy rel {ent_rel:.2e} | grad cosine {cosine:.7f} rel L2 {rel_l2:.2e}")
    assert loss_r != 0.0 and float(g_r.norm()) > 0.0
    assert loss_rel < LOSS_REL_TOL, (label, loss_r, loss_c)
    assert ent_rel < LOSS_REL_TOL, (label, ent_r, ent_c)
    assert cosine > COSINE_MIN, (label, cosine)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_batched_loss_matches_reference_through_step_mcts(case, batch_size):
    policy, exps = case
    ref = reference_policy_step(policy, exps)
    cand = batched_policy_step(policy, exps, batch_size)
    _assert_parity(ref, cand, f"B={batch_size}")


def test_out_of_range_indices_skip_loudly_and_match_reference(case, caplog):
    """Stored-index bounds guard: tuples whose indices exceed the
    re-encoded basis are skipped with a log.error by both paths, and
    the surviving terms still agree."""
    policy, exps = case
    bad = list(exps)
    good = bad[0].visit_counts
    a0, t0 = good[0][0], good[0][4]
    bad[0] = MCTSExperience(
        game_state=bad[0].game_state, z=bad[0].z, game_weight=bad[0].game_weight,
        game_id=bad[0].game_id,
        visit_counts=good + [(10_000, None, None, 3.0, None),
                             (a0, 10_000, None, 3.0, t0)])
    with caplog.at_level(logging.ERROR, logger="trainer"):
        ref = reference_policy_step(policy, bad)
        caplog.clear()
        cand = batched_policy_step(policy, bad, 4)
    assert sum("index out of range" in r.message for r in caplog.records) >= 2
    _assert_parity(ref, cand, "oob")


def test_stage_timings_hook(case):
    policy, exps = case
    tr = policy._trainer
    tr.optimizer.step = lambda *a, **k: None    # weights stay put for the other tests
    try:
        # The attribute channel (MCTSPolicy.train_step callers).
        tr.stage_timings = {}
        tr.step_mcts(exps)
        first = dict(tr.stage_timings)
        assert set(first) == set(STEP_MCTS_STAGES)
        assert all(v > 0.0 for v in first.values()), first
        # A second call adds to the same dict.
        tr.step_mcts(exps)
        assert all(tr.stage_timings[s] > first[s] for s in STEP_MCTS_STAGES)
        # The keyword wins over the attribute, which stays untouched.
        second = dict(tr.stage_timings)
        own = {}
        tr.step_mcts(exps, timings=own)
        assert set(own) == set(STEP_MCTS_STAGES)
        assert tr.stage_timings == second
        tr.stage_timings = None
        tr.step_mcts(exps)      # no sink: no timing, no error
    finally:
        del tr.optimizer.step
        tr.optimizer.zero_grad(set_to_none=True)
