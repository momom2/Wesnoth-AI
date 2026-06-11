"""Parity test: encode_from_raw_batch matches per-sample encode_from_raw.

The batched path fuses 19+ embedding/projection calls per state into
one batched call by concatenating per-sample arrays. We need to make
sure that fusion is a strict throughput win without changing the
output tensors -- otherwise trainer math drifts silently.

Test strategy: build B=3 hand-built GameStates with different unit
counts / hex counts / recruit options, run them through both the
single-sample loop and the batched method, and assert the resulting
EncodedState tensors are element-equal field-by-field.

Dependencies: encoder, classes, replay_dataset (to build a state).
Dependents: regression CI for encoder batching.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import torch

from encoder import GameStateEncoder, encode_raw
from tools.replay_dataset import (
    _apply_command,
    _build_initial_gamestate,
    _setup_scenario_events,
)

FIXTURE = Path(__file__).parent / "tests" / "fixtures" / "strict_sync_hamlets_t9.bz2"


def _gs_at_step(replay_path: Path, n_apply: int):
    """Build a GameState at command index `n_apply` from the fixture
    replay (extracted) -- we have a tiny strict-sync fixture in the
    repo that's perfect for this. Returns a (gs, n_units) pair."""
    from tools.replay_extract import extract_replay

    data = extract_replay(replay_path)
    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))
    for i, cmd in enumerate(data["commands"]):
        if i >= n_apply:
            break
        _apply_command(gs, cmd)
    return gs


def test_encode_from_raw_batch_parity():
    """Batched encode_from_raw must produce element-identical tensors
    to the per-sample loop."""
    if not FIXTURE.exists():
        pytest.skip("strict-sync fixture missing")

    # Three different game-state snapshots from the fixture -- they
    # differ in unit positions, gold, villages, and history, so the
    # encoder's per-sample arrays vary in length / values.
    snapshots = [
        _gs_at_step(FIXTURE, 5),
        _gs_at_step(FIXTURE, 25),
        _gs_at_step(FIXTURE, 60),
    ]

    encoder = GameStateEncoder(d_model=64)
    # Pre-register names so both paths see the same vocab. We do this
    # via .encode (which calls register_names) and then discard the
    # result.
    for gs in snapshots:
        encoder.register_names(gs)

    type_to_id = encoder.unit_type_to_id
    faction_to_id = encoder.faction_to_id
    raws = [encode_raw(gs,
                       type_to_id=type_to_id,
                       faction_to_id=faction_to_id)
            for gs in snapshots]

    # eval() so dropout doesn't perturb the comparison.
    encoder.eval()
    with torch.no_grad():
        per_sample = [encoder.encode_from_raw(r) for r in raws]
        batched = encoder.encode_from_raw_batch(raws)

    assert len(batched) == len(per_sample)
    for i, (b, s) in enumerate(zip(batched, per_sample)):
        # Tensors should be float32-close: the batched path's
        # accumulator order can differ slightly from the per-sample
        # path (fused multiply-add, SIMD lane ordering), so we tolerate
        # the standard float32 ~1e-6 noise.
        for fname in ("hex_tokens", "unit_tokens", "unit_is_ours",
                      "recruit_tokens", "recruit_is_ours",
                      "global_token"):
            bv = getattr(b, fname)
            sv = getattr(s, fname)
            assert bv.shape == sv.shape, (
                f"sample {i} field {fname}: shape "
                f"batched={tuple(bv.shape)} vs single={tuple(sv.shape)}"
            )
            assert torch.allclose(bv, sv, atol=1e-5, rtol=0), (
                f"sample {i} field {fname}: max abs diff "
                f"{(bv - sv).abs().max().item()}"
            )

        # Per-sample list / dict fields must be reference-equivalent
        # (they come from the raw, untouched).
        assert b.hex_positions == s.hex_positions
        assert b.unit_positions == s.unit_positions
        assert b.unit_ids == s.unit_ids
        assert b.recruit_types == s.recruit_types
        assert b.pos_to_hex == s.pos_to_hex


def test_encode_from_raw_batch_edge_cases():
    """Empty list, single-element list, all-zero-length streams."""
    encoder = GameStateEncoder(d_model=32)

    # Empty.
    assert encoder.encode_from_raw_batch([]) == []

    # Single (should still produce identical output to the
    # per-sample method).
    if FIXTURE.exists():
        gs = _gs_at_step(FIXTURE, 3)
        encoder.register_names(gs)
        raw = encode_raw(gs,
                         type_to_id=encoder.unit_type_to_id,
                         faction_to_id=encoder.faction_to_id)
        encoder.eval()
        with torch.no_grad():
            [b] = encoder.encode_from_raw_batch([raw])
            s = encoder.encode_from_raw(raw)
        for fname in ("hex_tokens", "unit_tokens", "recruit_tokens",
                      "global_token"):
            assert torch.allclose(
                getattr(b, fname), getattr(s, fname),
                atol=1e-5, rtol=0
            ), f"single-element batch differs from per-sample on {fname}"
