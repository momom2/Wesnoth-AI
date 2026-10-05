"""The parity-memory recipe's network (docs/parity_memory_design_20260929.md,
"The memory", "The belief head", "Model interface").

With the new flags off the network is obs8's: the state_dict keys, the
outputs through every path and the seeded construction equal what the code
before the recipe produced (tests/data/legacy_model_reference.json). With
them on: every forward path agrees; sightings and memory reach the trunk
and are never actors or targets; a loss at one decision reaches the write
of the decision before; a fresh write keeps most of the memory; slots past
k have no influence; the state stays float32 under bfloat16 autocast; the
faction posterior is the lookup for a one-hot vector; checkpoints carry the
structure; consumers that cannot keep the state refuse a memory model."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from helpers.model_streams import OUTPUT_FIELDS, hand_raw, legacy_reference

REFERENCE = Path(__file__).parent / "data" / "legacy_model_reference.json"
D = 32
SLOTS = 16
ARCH = dict(d_model=D, num_layers=2, num_heads=2, d_ff=64)
FIELDS = ("actor_logits", "type_logits", "target_logits", "weapon_logits", "value_logits",
          "belief_logits")
CPU = torch.device("cpu")


def _parity_pair(memory_slots: int = SLOTS, seed: int = 0, parity: bool = True):
    """The recipe's encoder and model at the tiny width; `parity` False
    gives obs8's observation, with or without a memory."""
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    torch.manual_seed(seed)
    encoder = GameStateEncoder(d_model=D, terrain_multi_hot=True, observation_parity=parity,
                               relevant_set_version=2 if parity else 1).eval()
    model = WesnothModel(**ARCH, observation_parity=parity, memory_slots=memory_slots).eval()
    return encoder, model


def _records(parity: bool = True):
    """Three decisions of different sizes: sightings, none, several."""
    return [hand_raw(11, U=3, R=2, H=9, parity=parity, S=2),
            hand_raw(12, U=1, R=0, H=5, parity=parity, S=0),
            hand_raw(13, U=4, R=1, H=12, parity=parity, S=3)]


def _states(ks, seed=7, scale=0.5):
    g = torch.Generator().manual_seed(seed)
    return [scale * torch.randn(k, D, generator=g) for k in ks]


def _grad(p: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(p) if p.grad is None else p.grad.clone()


def test_with_the_flags_off_the_network_is_obs8s():
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    got = legacy_reference()
    assert got["model_keys"] == ref["model_keys"]
    assert got["encoder_keys"] == ref["encoder_keys"]
    for path in ("single", "padded", "packed"):
        for i, (g, w) in enumerate(zip(got[path], ref[path])):
            for f in OUTPUT_FIELDS:
                a, b = np.asarray(g[f]), np.asarray(w[f])
                assert a.shape == b.shape, (path, i, f)
                assert np.allclose(a, b, rtol=1e-5, atol=1e-6), (path, i, f, np.abs(a - b).max())
    # The construction draws the generator as before: one seed, the same
    # parameters. Seeded draws differ between torch versions and machines,
    # so the values compare where the reference was recorded.
    assert got["construction"].keys() == ref["construction"].keys()
    if got["platform"] != ref.get("platform"):
        return
    for name, (total, squares) in ref["construction"].items():
        assert got["construction"][name] == pytest.approx([total, squares], rel=1e-9, abs=1e-9), name


def test_outputs_have_one_logit_per_hex_and_one_state_per_sample():
    encoder, model = _parity_pair()
    raws, ks = _records(), [8, 0, 16]
    with torch.no_grad():
        out = model.forward_embedded(encoder.encode_from_raw_embedded(raws), memory=_states(ks))
    A_max, H_max = 4 + 2 + 1, 12
    assert out.actor_logits.shape == (3, A_max)
    assert out.target_logits.shape == (3, A_max, H_max)
    assert out.belief_logits.shape == (3, H_max)
    assert [tuple(m.shape) for m in out.memory] == [(k, D) for k in ks]
    assert all(m.dtype == torch.float32 for m in out.memory)
    for raw, k, sample in zip(raws, ks, out.samples()):
        U, R, H = len(raw.unit_ids), len(raw.recruit_types), len(raw.hex_positions)
        # Sightings and memory slots are neither actors nor targets.
        assert sample.actor_logits.shape == (1, U + R + 1)
        assert sample.target_logits.shape == (1, U + R + 1, H)
        assert sample.belief_logits.shape == (1, H)
        assert sample.memory.shape == (k, D)


@pytest.mark.parametrize("parity, slots", [(True, SLOTS), (True, 0), (False, SLOTS)],
                         ids=["parity-memory", "parity-only", "memory-only"])
def test_every_forward_path_agrees(parity, slots):
    """Single state, padded and packed trunks, through EncodedStates and
    through stream-ordered embeddings, at mixed slot counts; with either
    stream alone too."""
    encoder, model = _parity_pair(slots, parity=parity)
    raws = _records(parity)
    states = _states([8, 0, 16]) if slots else None
    with torch.no_grad():
        singles = [model(encoder.encode_from_raw(r), memory=None if states is None else states[b])
                   for b, r in enumerate(raws)]
        encoded = encoder.encode_from_raw_batch(raws)
        streams = encoder.encode_from_raw_embedded(raws)
        batched = {
            "padded": model.forward_padded(encoded, packed=False, memory=states),
            "packed": model.forward_padded(encoded, packed=True, memory=states),
            "embedded padded": model.forward_embedded(streams, packed=False, memory=states),
            "embedded packed": model.forward_embedded(streams, packed=True, memory=states),
        }
    for name, out in batched.items():
        for b, (sample, single) in enumerate(zip(out.samples(), singles)):
            for f in FIELDS + ("memory",):
                x, y = getattr(sample, f), getattr(single, f)
                if y is None:
                    assert x is None, (name, b, f)
                    continue
                assert x.shape == y.shape, (name, b, f)
                assert torch.allclose(x, y, atol=1e-5, rtol=1e-4), (name, b, f)


def test_belief_logit_j_is_hex_token_js():
    """Reordering a record's hex stream reorders its belief logits the
    same way and leaves the actors' logits alone: the logits follow
    `hex_positions`, the alignment the belief loss relies on."""
    encoder, model = _parity_pair()
    raw = _records()[2]
    order = np.random.default_rng(5).permutation(len(raw.hex_positions))
    shuffled = hand_raw(13, U=4, R=1, H=12, parity=True, S=3)
    for name in ("hex_xs", "hex_ys", "hex_terrain_ids", "hex_modifier_flags", "hex_dynamic_flags"):
        setattr(shuffled, name, getattr(raw, name)[order])
    shuffled.hex_positions = [raw.hex_positions[j] for j in order]
    state = _states([8])[0]
    with torch.no_grad():
        base = model(encoder.encode_from_raw(raw), memory=state)
        moved = model(encoder.encode_from_raw(shuffled), memory=state)
    assert not torch.allclose(base.belief_logits, base.belief_logits[:, order])
    assert torch.allclose(moved.belief_logits, base.belief_logits[:, order], atol=1e-5)
    assert torch.allclose(moved.actor_logits, base.actor_logits, atol=1e-5)


def test_sightings_and_memory_reach_the_trunk():
    encoder, model = _parity_pair()
    raw = _records()[0]
    state = _states([8])

    def actor_logits(r, memory):
        with torch.no_grad():
            return model(encoder.encode_from_raw(r), memory=memory[0]).actor_logits
    base = actor_logits(raw, state)
    assert not torch.allclose(base, actor_logits(raw, _states([8], seed=8)))
    moved = hand_raw(11, U=3, R=2, H=9, parity=True, S=2)
    moved.sight_feats = moved.sight_feats + 0.5
    assert not torch.allclose(base, actor_logits(moved, state))


def test_a_loss_at_one_decision_reaches_the_write_of_the_decision_before():
    """Step 2's heads send gradient into every slot step 1 wrote, and
    through the write into its parameters, only along the carried memory:
    none when it is detached. The write's pad rows (sample 0 has 8 slots
    beside a 16-slot partner) are no one's state and get none."""
    encoder, model = _parity_pair()
    first, second = _records()[:2], _records()[1:]
    write = (model.slot_memory.candidate.weight, model.slot_memory.gate.weight)

    def run(detach: bool):
        model.zero_grad()
        start = [model.initial_memory(8), model.initial_memory(16)]
        out1 = model.forward_embedded(encoder.encode_from_raw_embedded(first), memory=start)
        out1.memory_padded.retain_grad()
        carried = [m.detach() if detach else m for m in out1.memory]
        out2 = model.forward_embedded(encoder.encode_from_raw_embedded(second), memory=carried)
        (out2.actor_logits.sum() + out2.value_logits.logsumexp(-1).sum()).backward()
        return out1.memory_padded.grad, [_grad(p) for p in write]
    written, params = run(detach=False)
    per_slot = written.abs().sum(-1)                        # [2, 16]
    assert bool((per_slot[0, :8] > 0).all()) and bool((per_slot[1] > 0).all())
    assert float(per_slot[0, 8:].sum()) == 0.0
    assert all(g.abs().sum() > 0 for g in params)
    written, params = run(detach=True)
    assert written is None and all(g.abs().sum() == 0 for g in params)


def test_a_fresh_write_keeps_most_of_the_memory():
    """b_z = -2: the new state is about 88% of the old one plus a small write."""
    encoder, model = _parity_pair()
    states = _states([SLOTS] * 3)
    with torch.no_grad():
        out = model.forward_embedded(encoder.encode_from_raw_embedded(_records()), memory=states)
    old, new = torch.cat(states), torch.cat(out.memory)
    kept = float((new * old).sum() / (old * old).sum())
    assert 0.8 < kept < 0.95, kept


def test_slots_past_k_have_no_influence():
    encoder, model = _parity_pair()
    raws, k = _records()[:1], 5

    def run():
        return model.forward_embedded(encoder.encode_from_raw_embedded(raws),
                                      memory=[model.initial_memory(k)])
    with torch.no_grad():
        before = run()
        model.slot_memory.initial[k:] += 1.0
        model.slot_memory.slot_embed.weight[k:] += 1.0
        after = run()
        # A batch partner with every slot active pads this sample's memory to 16.
        pair = model.forward_embedded(encoder.encode_from_raw_embedded(raws + _records()[1:2]),
                                      memory=[model.initial_memory(k), _states([SLOTS])[0]])
    for f in FIELDS:
        assert torch.equal(getattr(before, f), getattr(after, f)), f
        x, y = getattr(pair.samples()[0], f), getattr(before.samples()[0], f)
        assert torch.allclose(x, y, atol=1e-5, rtol=1e-4), f
    assert torch.equal(before.memory[0], after.memory[0])
    assert torch.allclose(pair.memory[0], before.memory[0], atol=1e-5, rtol=1e-4)
    model.zero_grad()
    out = run()
    (out.actor_logits.sum() + out.memory[0].sum()).backward()
    for p in (model.slot_memory.initial, model.slot_memory.slot_embed.weight):
        assert p.grad[:k].abs().sum() > 0
        assert p.grad[k:].abs().sum() == 0


def test_the_memory_stays_float32_under_bfloat16_autocast():
    encoder, model = _parity_pair()
    states = _states([8, 0, 16])
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model.forward_embedded(encoder.encode_from_raw_embedded(_records()), memory=states)
    assert out.actor_logits.dtype == torch.bfloat16          # the heads did run in bfloat16
    new = torch.cat(out.memory)
    assert new.dtype == torch.float32
    assert (new - new.bfloat16().float()).abs().max() > 0     # finer than bfloat16 holds
    # The write is float32 arithmetic under any autocast.
    h = torch.randn(3, SLOTS, D).bfloat16()
    batch = model.slot_memory.batch(states, CPU)
    with torch.no_grad():
        with torch.autocast("cpu", dtype=torch.bfloat16):
            inside = model.slot_memory.write(h, batch)
        outside = model.slot_memory.write(h.float(), batch)
    assert inside.dtype == torch.float32 and torch.equal(inside, outside)


def test_the_faction_posterior_is_the_lookup_when_one_hot():
    from wesnoth_ai.encoder import MAX_FACTIONS
    encoder, _ = _parity_pair()

    def global_token(probs):
        raw = hand_raw(21, U=2, R=1, H=4, parity=True, S=1, faction_probs=probs)
        with torch.no_grad():
            token = encoder.encode_from_raw(raw).global_token[0, 0]
            base = (encoder.global_proj(torch.from_numpy(raw.global_feats).unsqueeze(0))[0]
                    + encoder.our_faction_embed.weight[raw.our_faction_id])
        return token, base
    rows = encoder.their_faction_embed.weight.detach()
    for faction in (0, 3, 6):
        probs = np.zeros(MAX_FACTIONS, dtype=np.float32)
        probs[faction] = 1.0
        token, base = global_token(probs)
        assert torch.equal(token, base + rows[faction])
    probs = np.zeros(MAX_FACTIONS, dtype=np.float32)
    probs[2], probs[5] = 0.25, 0.75
    token, base = global_token(probs)
    assert torch.allclose(token, base + 0.25 * rows[2] + 0.75 * rows[5], atol=1e-6)


def test_each_side_refuses_the_other_observation():
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    parity_encoder, parity_model = _parity_pair()
    legacy_encoder = GameStateEncoder(d_model=D, terrain_multi_hot=True)
    with pytest.raises(ValueError, match="observation_parity"):
        legacy_encoder.encode_from_raw(_records()[0])
    with pytest.raises(ValueError, match="observation_parity"):
        parity_encoder.encode_from_raw_embedded([hand_raw(1, U=2, R=1, H=4)])
    with pytest.raises(ValueError, match="sighting stream"):
        parity_model(legacy_encoder.encode_from_raw(hand_raw(1, U=2, R=1, H=4)),
                     memory=parity_model.initial_memory(4))
    with pytest.raises(ValueError, match="memory_slots=16"):
        parity_model(parity_encoder.encode_from_raw(_records()[0]))
    with pytest.raises(ValueError, match="without observation_parity"):
        WesnothModel(**ARCH).forward_embedded(parity_encoder.encode_from_raw_embedded(_records()))


def test_a_checkpoint_carries_the_structure(tmp_path):
    from tools.eval_players import _load_policy, peek_checkpoint_arch
    from tools.inference_seam import build_inference_pair, inference_blueprint
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(0)
    policy = TransformerPolicy(**ARCH, observation_parity=True, memory_slots=8,
                               relevant_set_version=2)
    path = tmp_path / "parity.pt"
    policy.save_checkpoint(path)
    peek = peek_checkpoint_arch(path)
    assert (peek["observation_parity"], peek["memory_slots"], peek["relevant_set_version"]) == (True, 8, 2)
    loaded = _load_policy(path, CPU, "parity")
    want = policy._inference_model.state_dict()
    got = loaded._inference_model.state_dict()
    assert got.keys() == want.keys() and all(torch.equal(got[k], want[k]) for k in want)
    assert loaded._inference_encoder.relevant_set_version == 2
    with pytest.raises(RuntimeError, match="memory_slots|observation_parity"):
        TransformerPolicy(**ARCH).load_checkpoint(path)
    model, encoder = build_inference_pair(
        inference_blueprint(loaded._inference_model, loaded._inference_encoder), CPU)
    assert {k: v.shape for k, v in model.state_dict().items()} == {k: v.shape for k, v in got.items()}
    assert encoder.state_dict().keys() == loaded._inference_encoder.state_dict().keys()
    assert (encoder.observation_parity, encoder.relevant_set_version) == (True, 2)


def test_the_imitation_trainers_checkpoint_carries_the_structure(tmp_path):
    from tools.eval_players import _load_policy, peek_checkpoint_arch
    from tools.supervised_train import _save_checkpoint
    encoder, model = _parity_pair(memory_slots=8)
    opt = torch.optim.AdamW(list(model.parameters()) + list(encoder.parameters()))
    path = tmp_path / "imitation.pt"
    _save_checkpoint(path, model, encoder, opt, step=1, pairs=1, arch=ARCH)
    peek = peek_checkpoint_arch(path)
    assert (peek["observation_parity"], peek["memory_slots"], peek["relevant_set_version"]) == (True, 8, 2)
    loaded = _load_policy(path, CPU, "imitation")._inference_model.state_dict()
    assert loaded.keys() == model.state_dict().keys()
    assert all(torch.equal(v, loaded[k]) for k, v in model.state_dict().items())


def test_consumers_without_the_memory_state_refuse_a_memory_model():
    """A consumer that carries each side's memory takes a memory model only
    with its slots named (MCTS, the pool) or each side's memory (the turn
    search); one that does not carry it refuses it."""
    from tools.actor_pool import ActorPool
    from tools.mcts import mcts_search
    from tools.mcts_policy import MCTSPolicy
    from tools.turn_search import plan_turn
    from wesnoth_ai.graphed_serve import GraphedServe
    from wesnoth_ai.transformer_policy import TransformerPolicy
    policy = TransformerPolicy(**ARCH, memory_slots=4)
    model, encoder = policy._inference_model, policy._inference_encoder
    with pytest.raises(ValueError, match="memory of 4 slots"):
        MCTSPolicy(policy)
    assert MCTSPolicy(policy, memory_slots=4).memory_slots == 4
    with pytest.raises(ValueError, match="memory \\(4 slots\\)"):
        mcts_search(None, model, encoder)
    with pytest.raises(ValueError, match="memory \\(4 slots\\)"):
        plan_turn(policy, None, 1, 0, None, None, None, "", False)
    with pytest.raises(ValueError, match="memory of 4 slots"):
        ActorPool(SimpleNamespace(_inference_model=model), 1, None)
    with pytest.raises(ValueError, match="memory_slots=4"):
        GraphedServe(model, encoder, CPU, graphs=False)
    parity_encoder, parity_model = _parity_pair(memory_slots=0)
    with pytest.raises(ValueError, match="obs8's streams only"):
        GraphedServe(parity_model, parity_encoder, CPU, graphs=False)


def test_the_pool_plays_a_memory_model_only_where_the_memory_travels():
    """Self-play carries each leaf's memory on the priors protocol's
    request; the plan tournament and value grounding carry none."""
    from tools.actor_pool import ActorPool
    from tools.value_grounding import GroundingConfig
    _, model = _parity_pair(memory_slots=4)
    policy = SimpleNamespace(_inference_model=model)
    assert ActorPool(policy, 1, None, memory_slots=4)._memory_slots == 4
    with pytest.raises(ValueError, match="server-side priors"):
        ActorPool(policy, 1, None, memory_slots=4, server_priors=False)
    with pytest.raises(ValueError, match="plan tournament"):
        ActorPool(policy, 1, None, memory_slots=4, pt_cfg=object())
    with pytest.raises(ValueError, match="value grounding"):
        ActorPool(policy, 1, None, memory_slots=4, ground_cfg=GroundingConfig(enabled=True))


@pytest.mark.parametrize("memory_slots,parity,graphed", [(0, False, True), (4, True, False),
                                                         (0, True, False), (4, False, False)])
def test_the_graphed_pool_server_serves_eager_whatever_it_cannot_embed(memory_slots, parity, graphed):
    """The graphed path embeds obs8's streams only: a model with the parity
    observation or a memory falls back to the eager server. CUDA, bf16 and
    the packed trunk are only read here, so a CPU model stands in."""
    from tools.actor_pool import ActorPool
    _, model = _parity_pair(memory_slots=memory_slots, parity=parity)
    model.infer_packed_trunk = True
    pool = ActorPool(SimpleNamespace(_inference_model=model), 1, None, graphed_serve=True,
                     infer_bf16=True, device=torch.device("cuda"),
                     memory_slots=memory_slots if memory_slots else None)
    assert pool._graphed_serve_applies() is graphed
