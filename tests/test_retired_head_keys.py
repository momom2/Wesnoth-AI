"""Checkpoints saved with the auxiliary margin head load into the network
that has no such head: the loader leaves out exactly the head's two keys
(wesnoth_ai/checkpoint_structure.py, RETIRED_MODEL_KEYS), the network
computes every remaining output to the bit as the weights that were saved,
and a strict load still refuses any other key the network lacks or misses.

The old checkpoint is a fresh tiny network's, with the head's tensors and
flag written into it as the code that had the head saved them; one network
is shaped like the reference player (the parity observation and a learned
memory), one like the self-play checkpoints that carried the head (the
moves-left head). Hand-built records, no simulator."""
from __future__ import annotations

import logging

import pytest
import torch

from helpers.model_streams import hand_raw

CPU = torch.device("cpu")
ARCH = dict(d_model=16, num_layers=1, num_heads=2, d_ff=32)
OLD_HEAD = {"aux_score_head.weight": (1, 16), "aux_score_head.bias": (1,)}
FIELDS = ("actor_logits", "actor_kind", "type_logits", "target_logits", "weapon_logits",
          "value", "value_logits", "cliffness", "moves_left", "belief_logits", "memory")
SHAPES = {
    "reference": dict(observation_parity=True, memory_slots=4, relevant_set_version=2),
    "self-play": dict(moves_left=True),
}


def _source(shape: str):
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(5)
    return TransformerPolicy(device=CPU, **ARCH, **SHAPES[shape])


def _save_with_old_head(policy, path):
    """`policy`'s checkpoint as the code with the head wrote it: the head's
    weights in the model state and its flag at the top level."""
    policy.save_checkpoint(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    g = torch.Generator().manual_seed(9)
    for key, shape in OLD_HEAD.items():
        ckpt["model_state"][key] = torch.randn(shape, generator=g)
    ckpt["aux_score"] = True
    torch.save(ckpt, path)


def _outputs(policy):
    """Every output the inference network computes on four records, one
    game-side's decisions in a row (the memory carried from each to the
    next), through the single-state and the padded paths."""
    encoder, model = policy._inference_encoder, policy._inference_model
    parity = bool(encoder.observation_parity)
    raws = [hand_raw(s, U=u, R=r, H=h, parity=parity, S=3 if parity else 0)
            for s, u, r, h in ((1, 3, 2, 9), (2, 1, 0, 5), (3, 4, 1, 12), (4, 6, 3, 20))]
    slots = int(model.memory_slots)
    out = []
    with torch.no_grad():
        memory = model.initial_memory(slots) if slots else None
        inputs = []
        for raw in raws:
            inputs.append(memory)
            o = model(encoder.encode_from_raw(raw), **({"memory": memory} if slots else {}))
            out.append(o)
            memory = o.memory if slots else None
        padded = model.forward_padded(encoder.encode_from_raw_batch(raws), packed=False,
                                      **({"memory": inputs} if slots else {}))
    return out + padded.samples()


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_an_old_head_checkpoint_loads_without_it_and_plays_the_same(shape, tmp_path, caplog):
    from tools.eval_players import _load_policy
    source = _source(shape)
    path = tmp_path / "old_head.pt"
    _save_with_old_head(source, path)

    with caplog.at_level(logging.WARNING):
        loaded = _load_policy(path, CPU, label="old head")
    assert not [r for r in caplog.records if "partial" in r.getMessage()]
    for model in (loaded._model, loaded._inference_model):
        assert not any(k in model.state_dict() for k in OLD_HEAD)
    want, got = source._model.state_dict(), loaded._inference_model.state_dict()
    assert got.keys() == want.keys() and all(torch.equal(got[k], want[k]) for k in want)

    compared = 0
    for a, b in zip(_outputs(source), _outputs(loaded), strict=True):
        for f in FIELDS:
            x, y = getattr(a, f), getattr(b, f)
            assert (x is None) == (y is None), f
            if x is not None:
                assert torch.equal(x, y), f
                compared += 1
    assert compared >= 8 * 8


def test_a_strict_load_refuses_any_other_key(tmp_path):
    source = _source("reference")
    path = tmp_path / "old_head.pt"
    _save_with_old_head(source, path)
    _source("reference").load_checkpoint(path, strict=True)

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    ckpt["model_state"]["aux_score_head.scale"] = torch.ones(1)
    torch.save(ckpt, tmp_path / "extra.pt")
    with pytest.raises(RuntimeError, match="aux_score_head.scale"):
        _source("reference").load_checkpoint(tmp_path / "extra.pt", strict=True)

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    del ckpt["model_state"]["actor_head.weight"]
    torch.save(ckpt, tmp_path / "missing.pt")
    with pytest.raises(RuntimeError, match="actor_head.weight"):
        _source("reference").load_checkpoint(tmp_path / "missing.pt", strict=True)
