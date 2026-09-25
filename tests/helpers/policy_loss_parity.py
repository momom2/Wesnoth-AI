"""The batched factored policy loss against its per-state reference,
through step_mcts: the two steps, the parity check, and the bench
positions the real-data pins use (configs/bench_states.json, rebuilt from
the replay dataset; skipped when it is not on this machine;
WESNOTH_BENCH_DATASET points at a packed copy)."""
import os
from pathlib import Path

import pytest
import torch

from tools.bench_pipeline import load_states
from wesnoth_ai.paths import CONFIGS_DIR, IMITATION_DATASET_DIR
from wesnoth_ai.trainer import _mcts_factored_policy_loss_reference

MANIFEST = CONFIGS_DIR / "bench_states.json"
DATASET = Path(os.environ.get("WESNOTH_BENCH_DATASET", IMITATION_DATASET_DIR))
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
