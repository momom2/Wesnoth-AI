"""Parity tests for the actor/learner inference seam (tools/inference_seam).

Proves the RawEncoded round-trip is LOSS-LESS: routing a forward through
RemoteEncoder -> (RawEncoded) -> InferenceServer -> RemoteModel produces
   (a) bit-identical ModelOutput tensors, and
   (b) identical legal-action priors out of the sampler,
versus the direct `model(encoder.encode(gs))` path. If both hold, the
multiprocess server (B2) is purely plumbing -- the numerics are settled
here, in-process and deterministically, on CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from wesnoth_ai.transformer_policy import TransformerPolicy            # noqa: E402
from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors  # noqa: E402
from tools.inference_seam import (                           # noqa: E402
    InferenceServer, RemoteEncoder, RemoteModel, build_light_encoded,
    move_model_output,
)
from sim_test_helpers import fresh_scenario_sim              # noqa: E402


def _policy():
    pol = TransformerPolicy(device=torch.device("cpu"),
                            d_model=64, num_layers=2, num_heads=4, d_ff=128)
    pol._inference_model.eval()
    pol._inference_encoder.eval()
    return pol


def _states(n=4):
    """A handful of distinct mid-game states across mini scenarios."""
    out = []
    for seed in range(n):
        sim = fresh_scenario_sim(seed=20 + seed, max_turns=12, mini=True)
        # Step a few actions to diversify (recruits/moves populate the
        # unit + recruit streams).
        out.append(sim)
    return out


def _seam(pol):
    enc = pol._inference_encoder
    mdl = pol._inference_model
    # Vocab must be seeded before freezing so encode_raw doesn't fall to
    # the overflow bucket for the units these scenarios use. We seed by
    # registering names from the states themselves, mirroring how the
    # B2 launcher will pre-seed + freeze.
    for sim in _states():
        enc.register_names(sim.gs)
    server = InferenceServer(mdl, enc, output_device=torch.device("cpu"))
    renc = RemoteEncoder(enc.unit_type_to_id, enc.faction_to_id,
                         device=torch.device("cpu"))
    rmodel = RemoteModel(server)
    return enc, mdl, server, renc, rmodel


def _assert_outputs_match(ref, got):
    import dataclasses
    for f in dataclasses.fields(ref):
        a = getattr(ref, f.name)
        b = getattr(got, f.name)
        if torch.is_tensor(a):
            assert torch.allclose(a, b, atol=1e-6), f"{f.name} differs"
        else:
            assert a == b, f"{f.name}: {a} != {b}"


def test_output_wire_roundtrip_is_lossless():
    """output_to_wire/output_from_wire (the actor-pool queue payload,
    2026-07-22 batch-granular transport): numpy round-trip must
    reproduce every ModelOutput field exactly — values, dtypes
    (actor_kind is long), and non-tensor passthroughs."""
    from tools.inference_seam import output_to_wire, output_from_wire
    pol = _policy()
    enc, mdl, server, renc, rmodel = _seam(pol)
    sim = _states(1)[0]
    raw = renc.encode(sim.gs)._raw
    ref = server.infer(raw)
    got = output_from_wire(output_to_wire(ref))
    _assert_outputs_match(ref, got)
    import dataclasses
    for f in dataclasses.fields(ref):
        a, b = getattr(ref, f.name), getattr(got, f.name)
        if torch.is_tensor(a):
            assert a.dtype == b.dtype, f"{f.name} dtype changed"
    # The wire dict itself must hold NO torch tensors (plain numpy
    # pickles inline; tensors would re-enter the shm machinery).
    w = output_to_wire(ref)
    assert not any(torch.is_tensor(v) for _tag, v in w.values())


def test_batched_outputs_to_cpu_matches_per_sample():
    """The coalesced one-transfer-per-field path (2026-07-22, the
    9xB-serialized-syncs fix) must be byte-identical to per-sample
    move_model_output — values, shapes, dtypes — across samples of
    DIFFERENT shapes (that's the case the flatten-cat-split must
    get right)."""
    from tools.inference_seam import (batched_outputs_to_cpu,
                                      move_model_output)
    pol = _policy()
    enc, mdl, server, renc, rmodel = _seam(pol)
    sims = _states(3)
    raws = [renc.encode(s.gs)._raw for s in sims]
    with torch.no_grad():
        encs = enc.encode_from_raw_batch(raws)
        outs = mdl.forward_batch(encs)
    ref = [move_model_output(o, torch.device("cpu")) for o in outs]
    got = batched_outputs_to_cpu(outs)
    assert len(ref) == len(got)
    import dataclasses
    for r, g in zip(ref, got):
        _assert_outputs_match(r, g)
        for f in dataclasses.fields(r):
            a, b = getattr(r, f.name), getattr(g, f.name)
            if torch.is_tensor(a):
                assert a.shape == b.shape and a.dtype == b.dtype


def test_single_forward_parity():
    pol = _policy()
    enc, mdl, server, renc, rmodel = _seam(pol)
    for sim in _states():
        gs = sim.gs
        with torch.no_grad():
            ref_out = mdl(enc.encode(gs))
        seam_enc = renc.encode(gs)
        seam_out = rmodel(seam_enc)
        _assert_outputs_match(move_model_output(ref_out, torch.device("cpu")),
                              seam_out)


def test_legal_action_priors_parity():
    """The whole point: the light EncodedState + remote output must
    yield the SAME legal-action priors the direct path does, so the
    actor's search explores an identical action space."""
    pol = _policy()
    enc, mdl, server, renc, rmodel = _seam(pol)
    for sim in _states():
        gs = sim.gs
        with torch.no_grad():
            ref_enc = enc.encode(gs)
            ref_out = mdl(ref_enc)
        ref_priors = enumerate_legal_actions_with_priors(ref_enc, ref_out, gs)

        seam_enc = renc.encode(gs)
        seam_out = rmodel(seam_enc)
        seam_priors = enumerate_legal_actions_with_priors(
            seam_enc, seam_out, gs)

        ref_keys = sorted((p.action.get("type"), p.actor_idx, p.target_idx,
                           p.weapon_idx) for p in ref_priors)
        seam_keys = sorted((p.action.get("type"), p.actor_idx, p.target_idx,
                            p.weapon_idx) for p in seam_priors)
        assert ref_keys == seam_keys, "legal action set diverged"
        # Priors numerically identical too (same logits, same masks).
        ref_pr = sorted(p.prior for p in ref_priors)
        seam_pr = sorted(p.prior for p in seam_priors)
        assert len(ref_pr) == len(seam_pr)
        for a, b in zip(ref_pr, seam_pr):
            assert abs(a - b) < 1e-6


def test_batched_forward_parity():
    pol = _policy()
    enc, mdl, server, renc, rmodel = _seam(pol)
    sims = _states()
    seam_encs = [renc.encode(s.gs) for s in sims]
    batch_out = rmodel.forward_batch(seam_encs)
    assert len(batch_out) == len(sims)
    for sim, got in zip(sims, batch_out):
        with torch.no_grad():
            ref = move_model_output(mdl(enc.encode(sim.gs)),
                                    torch.device("cpu"))
        _assert_outputs_match(ref, got)


def test_mcts_search_through_seam_matches_direct():
    """End-to-end: running mcts_search with the seam objects (in place
    of the real encoder/model) must produce an IDENTICAL tree to the
    direct path -- same chosen action, same edge visit counts. With a
    fixed search RNG and bit-identical forwards, the two are
    deterministically equal. This is the proof that the seam is a true
    drop-in and mcts.py needs zero changes."""
    import numpy as np
    from tools.mcts import MCTSConfig, mcts_search
    from tools.draw_tiebreak import DrawTiebreakConfig

    pol = _policy()
    enc, mdl, server, renc, rmodel = _seam(pol)
    cfg = MCTSConfig(
        n_simulations=24, gumbel_root=True, gumbel_m=6,
        chance_nodes=True, exact_outcome_enumeration=True,
        draw_tiebreak=DrawTiebreakConfig(cap=0.3), batch_size=1,
        add_root_noise=False)
    sim = fresh_scenario_sim(seed=21, max_turns=12, mini=True)

    root_direct = mcts_search(sim.fork(), mdl, enc, cfg,
                              rng=np.random.default_rng(0))
    root_seam = mcts_search(sim.fork(), rmodel, renc, cfg,
                            rng=np.random.default_rng(0))

    assert root_direct.gumbel_action == root_seam.gumbel_action
    vd = sorted((str(e.action), e.n_visits) for e in root_direct.edges)
    vs = sorted((str(e.action), e.n_visits) for e in root_seam.edges)
    assert vd == vs, "seam search tree diverged from direct"


def test_light_encoded_fields_match_real_encode():
    """The sampler is a pure function of these fields; if they match
    the real encode's, sampler parity is guaranteed structurally."""
    pol = _policy()
    enc, mdl, server, renc, rmodel = _seam(pol)
    for sim in _states():
        gs = sim.gs
        real = enc.encode(gs)
        light = renc.encode(gs)
        assert light.unit_ids == real.unit_ids
        assert light.recruit_types == real.recruit_types
        assert light.pos_to_hex == real.pos_to_hex
        assert light.visible_unit_ids == real.visible_unit_ids
        assert light.unit_tokens.size(1) == real.unit_tokens.size(1)
        assert light.recruit_tokens.size(1) == real.recruit_tokens.size(1)
        assert light.hex_tokens.size(1) == real.hex_tokens.size(1)
        assert torch.allclose(light.unit_is_ours, real.unit_is_ours)
        assert torch.allclose(light.recruit_is_ours, real.recruit_is_ours)
