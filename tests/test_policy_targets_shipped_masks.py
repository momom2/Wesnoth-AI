"""MCTSExperience.masks: an experience that carries the legality masks
the actor packed (server_priors.PackedMasks) trains through step_mcts
without any host mask build, and produces the same policy loss,
"entropy" and gradient as an experience that makes the trainer
rebuild them; the rebuilt path stays pinned to the per-state
reference; masks packed on another action-space basis are refused.

Positions: the first bench states of configs/bench_states.json
(skipped when the dataset is not on this machine).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from test_batched_policy_loss import (  # noqa: E402
    _assert_parity, _bench_states, batched_policy_step, reference_policy_step,
)
from tools.bench_train_step import (  # noqa: E402
    configure_trainer_like_az_loop, experiences_from_states,
)
from tools.inference_seam import build_light_encoded  # noqa: E402
from wesnoth_ai import trainer as trainer_module  # noqa: E402
from wesnoth_ai.encoder import encode_raw  # noqa: E402
from wesnoth_ai.server_priors import pack_masks  # noqa: E402
from wesnoth_ai.trainer import MCTSExperience  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402

N_STATES = 4


def _packed(policy, game_state):
    """The masks as the actor packs them: from the RawEncoded of the
    same vocabulary, on a light EncodedState."""
    enc = policy._trainer.encoder
    raw = encode_raw(game_state, type_to_id=enc.unit_type_to_id,
                     faction_to_id=enc.faction_to_id)
    return pack_masks(build_light_encoded(raw, torch.device("cpu")), game_state)


@pytest.fixture(scope="module")
def case():
    states = _bench_states(N_STATES)
    torch.manual_seed(0)
    policy = TransformerPolicy(device=torch.device("cpu"),
                               d_model=64, num_layers=2, num_heads=4, d_ff=128)
    configure_trainer_like_az_loop(policy._trainer)
    # Without masks on purpose: the test ships them itself below and
    # compares against the host rebuild.
    exps = experiences_from_states(policy, states, sims=32, rng=random.Random(0), masks=False)
    exps[1].visit_counts = [v[:4] for v in exps[1].visit_counts]     # legacy 4-tuples
    exps[2].policy_weight = 0.5
    exps.append(MCTSExperience(game_state=states[0], visit_counts=[], z=1.0,
                               policy_weight=0.0, game_id="empty"))
    return policy, exps


def _with_masks(policy, exps):
    out = []
    for e in exps:
        kw = {f: getattr(e, f) for f in e.__dataclass_fields__}
        kw["masks"] = _packed(policy, e.game_state)
        out.append(MCTSExperience(**kw))
    return out


def test_shipped_masks_skip_the_host_build_and_match(case, monkeypatch):
    policy, exps = case
    assert all(e.masks is None for e in exps)
    ref = reference_policy_step(policy, exps)
    rebuilt = batched_policy_step(policy, exps, 4)
    _assert_parity(ref, rebuilt, "rebuilt")

    shipped_exps = _with_masks(policy, exps)

    def no_build(*a, **k):
        raise AssertionError("step_mcts rebuilt masks an experience already carried")
    monkeypatch.setattr(trainer_module, "_host_packed_masks", no_build)
    shipped = batched_policy_step(policy, shipped_exps, 4)
    loss_r, ent_r, g_r = rebuilt
    loss_s, ent_s, g_s = shipped
    assert loss_s == loss_r and ent_s == ent_r
    assert torch.equal(g_s, g_r)


def test_masks_on_another_basis_are_refused(case):
    policy, exps = case
    shipped_exps = _with_masks(policy, exps)
    p = shipped_exps[0].masks
    p.n_hexes += 1
    with pytest.raises(ValueError, match="shipped masks"):
        batched_policy_step(policy, shipped_exps, 4)
