"""Server-side priors (wesnoth_ai/server_priors.py, plan 1.3): the
compact legal actions computed on the server from actor-shipped masks
equal the enumeration from raw outputs, through every layer of the
seam."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def _policy():
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(3)
    return TransformerPolicy(d_model=32, num_layers=1, num_heads=2,
                             d_ff=64, device=torch.device("cpu"))


def _states():
    from tests.test_enumerate_vectorized import _states as harvest
    return harvest(n_attack=2, n_plain=3)


def _same(ref, got):
    assert [la.action for la in ref] == [la.action for la in got]
    assert [(la.actor_idx, la.type_idx, la.target_idx, la.weapon_idx) for la in ref] == \
           [(la.actor_idx, la.type_idx, la.target_idx, la.weapon_idx) for la in got]
    assert np.allclose([la.prior for la in ref], [la.prior for la in got],
                       rtol=1e-5, atol=1e-9)


def test_batched_priors_match_enumeration():
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    from wesnoth_ai.server_priors import batched_priors, pack_masks, unpack_compact
    policy = _policy()
    enc, model = policy._inference_encoder, policy._inference_model
    states = _states()
    encs = [enc.encode(gs) for gs in states]
    packs = [pack_masks(e, gs) for e, gs in zip(encs, states)]
    with torch.no_grad():
        padded = model.forward_padded(encs)
        compact = batched_priors(padded, packs)
        for e, gs, out, c in zip(encs, states, model.forward_batch(encs), compact):
            _same(enumerate_legal_actions_with_priors(e, out, gs), unpack_compact(c, e))
    kinds = {k for c in compact for k in c.kind.tolist()}
    assert {0, 1, 2, 3} <= kinds, "attack, move, recruit and end_turn all exercised"


def test_seam_with_server_priors_matches_direct_path():
    """RemoteEncoder(server_priors) + RemoteModel over an in-process
    InferenceServer: the enumeration on the seam's outputs equals the
    direct path, and the outputs carry value/cliffness."""
    from tools.inference_seam import InferenceServer, RemoteEncoder, RemoteModel
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    policy = _policy()
    enc, model = policy._inference_encoder, policy._inference_model
    states = _states()
    with torch.no_grad():
        direct = [(enc.encode(gs), gs) for gs in states]
        ref = [enumerate_legal_actions_with_priors(e, model(e), gs) for e, gs in direct]
        renc = RemoteEncoder(enc.unit_type_to_id, enc.faction_to_id,
                         fog_hides_enemy_villages=enc.fog_hides_enemy_villages, server_priors=True)
        rmodel = RemoteModel(InferenceServer(model, enc))
        lencs = [renc.encode(gs) for gs in states]
        outs = rmodel.forward_batch(lencs)
        for (e, gs), le, out, r in zip(direct, lencs, outs, ref):
            assert out.legal_compact is not None
            _same(r, enumerate_legal_actions_with_priors(le, out, gs))
            assert torch.allclose(out.value, model(e).value, atol=1e-5)
        # single-leaf call takes the same path
        one = rmodel(lencs[0])
        _same(ref[0], enumerate_legal_actions_with_priors(lencs[0], one, states[0]))


def test_wire_round_trip_keeps_compact():
    from tools.inference_seam import InferenceServer, RemoteEncoder, output_from_wire, output_to_wire
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    policy = _policy()
    enc, model = policy._inference_encoder, policy._inference_model
    gs = _states()[0]
    renc = RemoteEncoder(enc.unit_type_to_id, enc.faction_to_id,
                         fog_hides_enemy_villages=enc.fog_hides_enemy_villages, server_priors=True)
    le = renc.encode(gs)
    out = InferenceServer(model, enc).infer_batch([(le._raw, le._masks)])[0]
    back = output_from_wire(output_to_wire(out))
    _same(enumerate_legal_actions_with_priors(le, out, gs),
          enumerate_legal_actions_with_priors(le, back, gs))


def test_mixed_batch_refused():
    import pytest
    from tools.inference_seam import InferenceServer, RemoteEncoder
    policy = _policy()
    enc, model = policy._inference_encoder, policy._inference_model
    gs = _states()[0]
    renc = RemoteEncoder(enc.unit_type_to_id, enc.faction_to_id,
                         fog_hides_enemy_villages=enc.fog_hides_enemy_villages, server_priors=True)
    le = renc.encode(gs)
    with pytest.raises(ValueError, match="mixed"):
        InferenceServer(model, enc).infer_batch([le._raw, (le._raw, le._masks)])


def test_actor_pool_play_command_carries_flag():
    repo = Path(__file__).parent.parent
    pool_src = (repo / "tools/actor_pool.py").read_text(encoding="utf-8")
    actor_src = (repo / "tools/actor_worker.py").read_text(encoding="utf-8")
    assert "bool(self.server_priors)" in pool_src and "server_priors=_sp" in actor_src
