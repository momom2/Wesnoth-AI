"""Tests for the policy-head human-anchor rehearsal (F1, 2026-08-10).

Behavioral: anchor_policy_step drives the REAL trainer + model on a
real encoded state and must reduce the imitation CE on the rehearsed
pair -- the same "does the gradient reach the policy heads" question
the feature exists to answer. Cache-format validation is pinned so a
value-anchor pickle can't be silently fed to the policy path.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from tools.policy_anchor import (  # noqa: E402
    CACHE_VERSION, anchor_policy_step, load_policy_anchor,
    sample_pairs_game_normalized,
)
from tools.replay_dataset import ActionIndices  # noqa: E402
from wesnoth_ai.encoder import encode_raw  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402


def _tiny_policy() -> TransformerPolicy:
    return TransformerPolicy(device=torch.device("cpu"), d_model=32,
                             num_layers=1, num_heads=2, d_ff=64)


def _one_pair(policy: TransformerPolicy):
    """A real encoded state + an end_turn ActionIndices for it."""
    sim = fresh_scenario_sim(seed=11, max_turns=12, mini=True)
    enc = policy._encoder
    enc.encode(sim.gs)                      # grow the vocab
    raw = encode_raw(sim.gs, type_to_id=enc.unit_type_to_id,
                     faction_to_id=enc.faction_to_id)
    with torch.no_grad():
        out = policy._model.forward_batch(
            enc.encode_from_raw_batch([raw]))[0]
    n_actors = out.actor_logits.shape[-1]
    # end_turn is always the LAST actor slot (unit + recruit + end_turn).
    return raw, ActionIndices("end_turn", actor_idx=n_actors - 1)


def test_anchor_policy_step_reduces_imitation_ce():
    policy = _tiny_policy()
    pair = _one_pair(policy)
    trainer = policy._trainer
    first = anchor_policy_step(trainer, [pair])
    assert first["grad_norm"] > 0.0         # gradient actually flowed
    for _ in range(8):
        last = anchor_policy_step(trainer, [pair])
    assert last["policy_ce"] < first["policy_ce"], (
        f"rehearsal did not reduce CE: {first} -> {last}")
    assert last["actor_ce"] == last["actor_ce"]     # not NaN


def test_load_policy_anchor_rejects_value_cache(tmp_path):
    """The value anchor pickles a bare list -- feeding it to the
    policy loader must fail loudly, not train on garbage."""
    bad = tmp_path / "value_anchor.pkl"
    with bad.open("wb") as f:
        pickle.dump([("raw", 1.0, 0.5)], f)
    with pytest.raises(ValueError, match="policy-anchor"):
        load_policy_anchor(bad)


def test_load_policy_anchor_roundtrip(tmp_path):
    p = tmp_path / "policy_anchor.pkl"
    with p.open("wb") as f:
        pickle.dump({"version": CACHE_VERSION, "meta": {},
                     "games": [[("r", "ai")], [("r2", "ai2")]]}, f)
    assert load_policy_anchor(p) == [[("r", "ai")], [("r2", "ai2")]]


def test_load_policy_anchor_rejects_v1_flat_cache(tmp_path):
    """A pre-2026-08-16 (pair-uniform) cache must fail loudly with
    the rebuild command, never silently rehearse under the wrong
    normalization."""
    p = tmp_path / "old.pkl"
    with p.open("wb") as f:
        pickle.dump({"version": 1, "meta": {},
                     "pairs": [("r", "ai")]}, f)
    with pytest.raises(ValueError, match="Rebuild"):
        load_policy_anchor(p)


def test_game_normalized_draw_weights_games_equally():
    """The v2 property itself: a 300-pair game and a 3-pair game must
    be REPRESENTED equally in the draw, i.e. per-game inclusion is
    uniform, not proportional to length."""
    import random
    big = [("big", i) for i in range(300)]
    small = [("small", i) for i in range(3)]
    rng = random.Random(7)
    # k == n_games: exactly one pair from EACH game, every time.
    for _ in range(50):
        picks = sample_pairs_game_normalized([big, small], 2, rng)
        assert {p[0] for p in picks} == {"big", "small"}
    # k=1 over many draws: each game chosen ~half the time.
    counts = {"big": 0, "small": 0}
    for _ in range(800):
        counts[sample_pairs_game_normalized([big, small], 1, rng)
               [0][0]] += 1
    assert 300 < counts["small"] < 500, counts
    # k > n_games cycles: 5 picks over 2 games -> both represented.
    picks = sample_pairs_game_normalized([big, small], 5, rng)
    assert len(picks) == 5
    assert {p[0] for p in picks} == {"big", "small"}
    # Empty pool degrades to empty, never raises.
    assert sample_pairs_game_normalized([], 4, rng) == []
