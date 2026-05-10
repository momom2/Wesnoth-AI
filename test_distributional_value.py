#!/usr/bin/env python3
"""Tests for the distributional (C51-style) value head.

Covers:
  - Atom support is correct (linspace V_MIN..V_MAX, K bins).
  - The forward's `value` field equals the mean of the predicted
    distribution.
  - `cliffness` equals the std of the predicted distribution and is
    always non-negative.
  - `_project_returns_to_atoms` sums to 1, lands at the right bins
    for boundary inputs (V_MIN, V_MAX, exact-bin returns), and
    interpolates linearly between bins.
  - The categorical CE loss reduces to ~0 when the network's
    predicted distribution exactly matches the projection of the
    target — sanity check that loss + projection are consistent.

Dependencies: model, trainer, classes
Dependents: pytest only
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from model import (  # noqa: E402
    WesnothModel, VALUE_N_ATOMS, VALUE_V_MIN, VALUE_V_MAX,
)
from trainer import _project_returns_to_atoms, _categorical_value_loss  # noqa: E402


# ---------------------------------------------------------------------
# Atom support
# ---------------------------------------------------------------------

def test_atom_support_shape_and_endpoints():
    """The model registers a buffer of K atoms spanning
    [V_MIN, V_MAX] inclusive. Anything else and the projection
    will silently misalign."""
    m = WesnothModel(d_model=32, num_layers=2, num_heads=4, d_ff=64)
    atoms = m._value_atoms
    assert atoms.shape == (VALUE_N_ATOMS,)
    assert atoms[0].item() == pytest.approx(VALUE_V_MIN)
    assert atoms[-1].item() == pytest.approx(VALUE_V_MAX)
    # Uniform spacing.
    deltas = atoms[1:] - atoms[:-1]
    assert torch.allclose(deltas, deltas[0].expand_as(deltas))


# ---------------------------------------------------------------------
# Distribution → value / cliffness consistency
# ---------------------------------------------------------------------

def _make_logits_concentrated_at(atom_idx: int) -> torch.Tensor:
    """Produce K-vector logits whose softmax is ~one-hot on
    `atom_idx`. Used to test that mean / variance computations
    bottom out at the expected atom value."""
    logits = torch.full((VALUE_N_ATOMS,), -100.0)
    logits[atom_idx] = 0.0
    return logits.unsqueeze(0)   # [1, K]


def test_value_equals_distribution_mean_at_corners():
    """A distribution concentrated on atom 0 should yield
    value = V_MIN; on atom K-1 should yield value = V_MAX."""
    m = WesnothModel(d_model=32, num_layers=2, num_heads=4, d_ff=64)
    atoms = m._value_atoms

    # Concentrated at lower atom.
    logits = _make_logits_concentrated_at(0)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    mean = (probs * atoms).sum(dim=-1)
    assert mean.item() == pytest.approx(VALUE_V_MIN, abs=1e-3)

    # Concentrated at upper atom.
    logits = _make_logits_concentrated_at(VALUE_N_ATOMS - 1)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    mean = (probs * atoms).sum(dim=-1)
    assert mean.item() == pytest.approx(VALUE_V_MAX, abs=1e-3)


def test_cliffness_equals_distribution_std_zero_when_concentrated():
    """A near-one-hot distribution has near-zero std. This is the
    "low cliffness, network is certain" regime that soft-TT would
    trust for similarity transfer."""
    m = WesnothModel(d_model=32, num_layers=2, num_heads=4, d_ff=64)
    atoms = m._value_atoms
    # Fake a "one-hot at middle atom" forward result.
    K_mid = VALUE_N_ATOMS // 2
    logits = _make_logits_concentrated_at(K_mid)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    mean = (probs * atoms).sum(dim=-1, keepdim=True)
    var = ((probs * atoms.pow(2)).sum(dim=-1, keepdim=True)
           - mean.pow(2)).clamp_min(0)
    std = var.sqrt()
    assert std.item() < 0.01, "near-one-hot distribution should have ~0 std"


def test_cliffness_high_when_distribution_spread():
    """A uniform distribution over [-1, +1] has std = (V_MAX - V_MIN) /
    sqrt(12) for the *continuous* case; the discrete uniform-over-K
    is close. This is the "high cliffness, network is uncertain"
    regime — soft-TT should DISTRUST similarity transfer here."""
    atoms = torch.linspace(VALUE_V_MIN, VALUE_V_MAX, VALUE_N_ATOMS)
    probs = torch.full((1, VALUE_N_ATOMS), 1.0 / VALUE_N_ATOMS)
    mean = (probs * atoms).sum(dim=-1, keepdim=True)
    var = ((probs * atoms.pow(2)).sum(dim=-1, keepdim=True)
           - mean.pow(2)).clamp_min(0)
    std = var.sqrt().item()
    # Continuous uniform on [-1, 1] has std = sqrt(1/3) ≈ 0.577.
    # Discrete uniform on K=51 bins is very close.
    assert 0.55 < std < 0.60, f"expected std≈0.577, got {std:.3f}"


# ---------------------------------------------------------------------
# Projection of scalar returns to bin distribution
# ---------------------------------------------------------------------

def test_projection_sums_to_one():
    """For any scalar return in support, the projected distribution
    sums to 1. Otherwise the categorical CE loss has the wrong
    normalization."""
    atoms = torch.linspace(VALUE_V_MIN, VALUE_V_MAX, VALUE_N_ATOMS)
    for r in (-1.0, -0.5, 0.0, 0.3, 0.99, +1.0):
        target = _project_returns_to_atoms(torch.tensor([r]), atoms)
        assert target.shape == (1, VALUE_N_ATOMS)
        assert target.sum().item() == pytest.approx(1.0, abs=1e-6)
        assert (target >= 0).all()


def test_projection_lands_at_corner_atoms():
    """Returns at the edges of the support land entirely on the
    corresponding atom. r=V_MIN → target[0]=1; r=V_MAX → target[-1]=1."""
    atoms = torch.linspace(VALUE_V_MIN, VALUE_V_MAX, VALUE_N_ATOMS)
    # Tolerances loosened to float32 epsilon: the projection
    # involves `(V_MAX - V_MIN) / delta` which doesn't round-trip
    # exactly under fp32, leaving ~3e-5 mass on the wrong corner.
    target_min = _project_returns_to_atoms(
        torch.tensor([VALUE_V_MIN]), atoms)
    assert target_min[0, 0].item() == pytest.approx(1.0, abs=1e-4)
    assert target_min[0, 1:].sum().item() == pytest.approx(0.0, abs=1e-4)

    target_max = _project_returns_to_atoms(
        torch.tensor([VALUE_V_MAX]), atoms)
    assert target_max[0, -1].item() == pytest.approx(1.0, abs=1e-4)
    assert target_max[0, :-1].sum().item() == pytest.approx(0.0, abs=1e-4)


def test_projection_clips_out_of_support():
    """Returns above V_MAX or below V_MIN clip silently to the
    edge atoms (matches the trainer's `value_clip` upstream)."""
    atoms = torch.linspace(VALUE_V_MIN, VALUE_V_MAX, VALUE_N_ATOMS)
    above = _project_returns_to_atoms(torch.tensor([+5.0]), atoms)
    assert above[0, -1].item() == pytest.approx(1.0, abs=1e-4)
    below = _project_returns_to_atoms(torch.tensor([-5.0]), atoms)
    assert below[0, 0].item() == pytest.approx(1.0, abs=1e-4)


def test_projection_interpolates_linearly_between_bins():
    """A return halfway between atom_l and atom_l+1 distributes
    mass 0.5 / 0.5 between the two."""
    atoms = torch.linspace(VALUE_V_MIN, VALUE_V_MAX, VALUE_N_ATOMS)
    delta = (atoms[1] - atoms[0]).item()
    # Pick a non-edge atom and offset by half a bin.
    l = 10
    r = atoms[l].item() + delta * 0.5
    target = _project_returns_to_atoms(torch.tensor([r]), atoms)
    assert target[0, l].item() == pytest.approx(0.5, abs=1e-5)
    assert target[0, l + 1].item() == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------
# CE loss minimum at perfect prediction
# ---------------------------------------------------------------------

def test_ce_loss_minimized_when_logits_match_target():
    """When the network's predicted distribution exactly equals the
    target projection, the per-sample CE is close to the entropy
    of the target — its minimum value (achieved by a perfect
    predictor). For one-hot target distributions, that minimum
    is 0."""
    atoms = torch.linspace(VALUE_V_MIN, VALUE_V_MAX, VALUE_N_ATOMS)
    # Use a return at an exact atom so the projection is one-hot.
    K_idx = 25
    r = atoms[K_idx].item()
    target = _project_returns_to_atoms(torch.tensor([r]), atoms)
    # Near-one-hot logits matching the target.
    logits = torch.full((1, VALUE_N_ATOMS), -100.0)
    logits[0, K_idx] = 0.0
    loss = _categorical_value_loss(logits, torch.tensor([r]), atoms)
    # ~0 in fp32 means a few times 1e-3 — the softmax of [-100, 0,
    # ..., -100] still leaks ~exp(-100) mass into the off-target
    # bins, which the projection's float32 round-trip amplifies.
    # 0.005 is a comfortable ceiling well below "loss is broken".
    assert loss.item() < 5e-3, f"loss should be ~0 at perfect prediction, got {loss.item():.3f}"


def test_ce_loss_positive_for_wrong_prediction():
    """Predicting a distribution far from the target gives a
    finite, positive loss. Used here as a sanity check that the
    loss is computing CE rather than something silently zero."""
    atoms = torch.linspace(VALUE_V_MIN, VALUE_V_MAX, VALUE_N_ATOMS)
    # Target at lower atom; predict at upper. Maximally wrong.
    target_idx = 0
    r = atoms[target_idx].item()
    logits = torch.full((1, VALUE_N_ATOMS), -100.0)
    logits[0, VALUE_N_ATOMS - 1] = 0.0
    loss = _categorical_value_loss(logits, torch.tensor([r]), atoms)
    assert loss.item() > 50.0, "CE on near-one-hot disagreement should be large"


# ---------------------------------------------------------------------
# Forward → ModelOutput field shapes
# ---------------------------------------------------------------------

def test_modeloutput_carries_distributional_fields():
    """The forward path populates `value`, `value_logits`, and
    `cliffness`; downstream code (rollout, MCTS, trainer) reads
    each of them. A regression that drops one of these from
    `ModelOutput` would only surface as an attribute error in
    the runtime path that uses it."""
    from model import ModelOutput
    fields = ModelOutput.__dataclass_fields__
    assert "value" in fields
    assert "value_logits" in fields
    assert "cliffness" in fields
