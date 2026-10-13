"""The own rows' staging (wesnoth_ai/own_loss.py `stage_own_batch`): the
masks, the rows and the targets share one buffer and one copy, and each
field holds what its definition says, for rows of differing actors, hexes
and target lengths, with and without the biases."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from wesnoth_ai import own_loss
from wesnoth_ai.sequence_records import OwnTarget
from wesnoth_ai.server_priors import (KIND_ATTACK, KIND_END_TURN, KIND_MOVE, KIND_RECRUIT, PackedMasks,
                                      mask_layout, write_masks)

CPU = torch.device("cpu")
T, W = 2, 4


def _masks(rng, U: int, R: int, H: int, biased: bool) -> PackedMasks:
    A = U + R + 1

    def bits():
        return np.packbits(rng.random((A, H)) > 0.5, axis=1)
    return PackedMasks(actor_mask=(rng.random(A) > 0.3).astype(np.uint8),
                       type_valid=(rng.random((A, T)) > 0.4).astype(np.uint8), attack_valid=bits(),
                       move_valid=bits(), union_valid=bits(), n_attacks=rng.integers(0, W + 1, A).astype(np.int8),
                       end_turn_bias=float(rng.normal()), n_units=U, n_recruits=R, n_hexes=H,
                       type_bias=rng.normal(size=(A, T)).astype(np.float32) if biased else None,
                       attack_bias=rng.normal(size=(A, H)).astype(np.float32) if biased else None)


def _target(rng, n: int, A: int, H: int) -> OwnTarget:
    kind = ((np.arange(n) + rng.integers(0, 4)) % 4).astype(np.int8)    # every kind in a target of 4 or more
    attack = kind == KIND_ATTACK
    return OwnTarget(actor=rng.integers(0, A, n).astype(np.int32), kind=kind,
                     target=np.where(kind == KIND_END_TURN, -1, rng.integers(0, H, n)).astype(np.int32),
                     weapon=np.where(attack, rng.integers(0, W, n), -1).astype(np.int8),
                     prior=rng.random(n) + 0.01, end_prior=0.2)


@pytest.mark.parametrize("biased", [False, True], ids=["no bias", "biases"])
def test_the_staged_own_batch_holds_each_field_as_defined(biased):
    rng = np.random.default_rng(11)
    shapes = [(3, 1, 9), (1, 0, 4), (2, 2, 13)]
    packs = [_masks(rng, U, R, H, biased and b != 1) for b, (U, R, H) in enumerate(shapes)]
    targets = [_target(rng, n, U + R + 1, H) for n, (U, R, H) in zip((5, 1, 3), shapes)]
    A_max = max(U for U, _, _ in shapes) + max(R for _, R, _ in shapes) + 1
    H_max = max(H for _, _, H in shapes)
    rows = [0, 2, 3]
    batch = own_loss.stage_own_batch(rows, packs, targets, A_max, H_max, T, W, CPU)

    assert batch.rows.dtype == torch.int64 and batch.rows.tolist() == rows
    K = max(len(t.prior) for t in targets)
    want = {name: np.zeros((3, K), np.int64) for name in ("actor", "kind", "target", "weapon")}
    want["kind"][:] = KIND_END_TURN
    prob = np.zeros((3, K), np.float32)
    for r, t in enumerate(targets):
        m = len(t.prior)
        want["actor"][r, :m], want["kind"][r, :m] = t.actor, t.kind
        want["target"][r, :m] = np.where(t.target < 0, 0, t.target)
        want["weapon"][r, :m] = np.where(t.weapon < 0, 0, t.weapon)
        prob[r, :m] = t.prior / t.prior.sum()
    for name, a in want.items():
        got = getattr(batch, name)
        assert got.dtype == torch.int64 and np.array_equal(got.numpy(), a), name
    assert batch.prob.dtype == torch.float32 and np.array_equal(batch.prob.numpy(), prob)
    assert set(targets[0].kind.tolist()) == {KIND_ATTACK, KIND_MOVE, KIND_RECRUIT, KIND_END_TURN}

    layout = mask_layout(3, A_max, H_max, T, W, type_bias=biased, attack_bias=biased)
    reference = np.zeros(layout.nbytes, np.uint8)
    views = layout.numpy_views(reference)
    write_masks(views, packs)
    assert set(batch.views) == set(views)
    for name, v in views.items():
        got = batch.views[name]
        assert got.shape == v.shape and np.array_equal(got.numpy(), v), name
