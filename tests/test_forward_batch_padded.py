"""forward_batch (2026-09-04 rewrite: launch count independent of B,
per-sample outputs as views of padded head tensors) matches the
single-sample forward per sample, on CPU and after to_cpu()."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

FIELDS = ("actor_logits", "type_logits", "target_logits", "weapon_logits",
          "value", "value_logits", "cliffness")


def _policy():
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(2)
    return TransformerPolicy(d_model=32, num_layers=1, num_heads=2,
                             d_ff=64, device=torch.device("cpu"))


def _encoded(policy, n=3):
    from tools.bench_states import harvest_states
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    states = [build_scenario_gamestate(random_setup(random.Random(7)))]
    states += harvest_states(n - 1, seed=11)
    enc = policy._inference_encoder
    return [enc.encode(gs) for gs in states]


def _assert_same(single, batched):
    assert batched.num_units == single.num_units
    assert batched.num_recruits == single.num_recruits
    assert torch.equal(batched.actor_kind.cpu(), single.actor_kind.cpu())
    for f in FIELDS:
        a, b = getattr(single, f), getattr(batched, f)
        assert a.shape == b.shape, (f, a.shape, b.shape)
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), f


def test_batched_outputs_match_single_forward():
    policy = _policy()
    model = policy._inference_model
    encs = _encoded(policy)
    sizes = {(e.unit_tokens.size(1), e.hex_tokens.size(1)) for e in encs}
    assert len(sizes) > 1, "batch must mix sizes to exercise padding"
    with torch.no_grad():
        singles = [model(e) for e in encs]
        batched = model.forward_batch(encs)
    for s, b in zip(singles, batched):
        _assert_same(s, b)


def test_padded_to_cpu_samples_match():
    policy = _policy()
    model = policy._inference_model
    encs = _encoded(policy)
    with torch.no_grad():
        singles = [model(e) for e in encs]
        padded = model.forward_padded(encs).to_cpu()
    for s, b in zip(singles, padded.samples()):
        _assert_same(s, b)
    assert padded.actor_logits.shape[0] == len(encs)


def test_batched_priors_match_single(policy=None):
    """The enumeration reads the batched outputs exactly as it reads
    single ones: same legal actions and priors."""
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    from tools.bench_states import harvest_states
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    policy = _policy()
    model, enc = policy._inference_model, policy._inference_encoder
    states = [build_scenario_gamestate(random_setup(random.Random(7)))]
    states += harvest_states(2, seed=11)
    encs = [enc.encode(gs) for gs in states]
    with torch.no_grad():
        singles = [model(e) for e in encs]
        batched = model.forward_batch(encs)
        for gs, e, s, b in zip(states, encs, singles, batched):
            la = enumerate_legal_actions_with_priors(e, s, gs)
            lb = enumerate_legal_actions_with_priors(e, b, gs)
            assert [x.action for x in la] == [x.action for x in lb]
            assert all(abs(x.prior - y.prior) < 1e-5 for x, y in zip(la, lb))
