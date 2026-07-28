"""Gumbel q-transform: soft targets, offset invariance, reference match.

The distillation target is softmax(logits + sigma(completed_q)). Before
2026-07-28 sigma used the PAPER's c_scale=1.0 on RAW q in [-1,1] without
the paper's normalization, so Q differences were multiplied by ~50-80 and
the target collapsed to near-one-hot -- it taught argmax every iteration
instead of a policy improvement. We now match the reference implementation
(mctx: value_scale=0.1, maxvisit_init=50, rescale_values=True).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.mcts import MCTSConfig, _gumbel_sigma, _rescale_q   # noqa: E402


def _target(logits, qs, max_v, cfg):
    t = logits + _gumbel_sigma(np.asarray(qs, float), max_v, cfg)
    t = t - t.max()
    p = np.exp(t)
    return p / p.sum()


def test_reference_defaults():
    """Constants must match mctx's qtransform_completed_by_mix_value."""
    cfg = MCTSConfig()
    assert cfg.gumbel_c_scale == 0.1      # mctx value_scale
    assert cfg.gumbel_c_visit == 50.0     # mctx maxvisit_init
    assert cfg.gumbel_rescale_q is True   # mctx rescale_values


def test_rescale_maps_to_unit_interval():
    q = np.array([-0.8, -0.1, 0.3, 0.9])
    r = _rescale_q(q)
    assert r.min() == 0.0 and r.max() == 1.0
    # order preserved (monotone transform)
    assert list(np.argsort(r)) == list(np.argsort(q))
    # degenerate (all equal) must not divide by zero
    flat = _rescale_q(np.array([0.5, 0.5, 0.5]))
    assert np.all(np.isfinite(flat))


def _entropy(p):
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum())


def test_target_is_softer_than_the_old_transform():
    """The fix is RELATIVE: on the same state the reference transform must
    produce a strictly softer target (higher entropy, lower peak) than the
    pre-2026-07-28 setting, which collapsed to a one-hot label.

    Note we do NOT assert some absolute floor on the smallest mass: the
    min-max rescale always spans the full [0,1], so the logit range is a
    fixed c_scale*(c_visit+max_N) ~ 8.2 regardless of how close the Q
    values are. That IS the reference's behaviour -- the target is meant to
    sharpen toward better actions; it just must not become argmax.
    """
    new = MCTSConfig()
    old = MCTSConfig(gumbel_c_scale=1.0, gumbel_rescale_q=False)
    logits = np.log(np.array([0.25, 0.25, 0.25, 0.25]))
    qs = np.array([0.10, 0.05, -0.05, -0.10])          # 0.2 spread
    p_new = _target(logits, qs, 32.0, new)
    p_old = _target(logits, qs, 32.0, old)
    assert p_old.max() > 0.95, f"expected old collapse, got {p_old}"
    assert p_new.max() < 0.95, f"new target still ~one-hot: {p_new}"
    assert _entropy(p_new) > _entropy(p_old) + 0.1, (
        f"new target not softer: H_new={_entropy(p_new):.3f} "
        f"H_old={_entropy(p_old):.3f}")


def test_narrow_q_spread_does_not_annihilate_the_prior():
    """With MANY actions and a realistic prior, the target must still carry
    mass on non-argmax actions -- this is the property whose absence crushed
    recruit mass from ~0.16 to ~0.001 in the measured campaign."""
    cfg = MCTSConfig()
    n = 40
    priors = np.full(n, 1.0 / n)
    logits = np.log(priors)
    rng = np.random.default_rng(0)
    qs = rng.normal(0.0, 0.05, size=n)                  # tight, realistic
    p = _target(logits, qs, 32.0, cfg)
    assert p.max() < 0.9, f"collapsed with n={n}: max={p.max():.3f}"
    # the top-5 should not swallow everything
    assert np.sort(p)[-5:].sum() < 0.999


def test_offset_invariance():
    """Adding a constant to every Q must not move the target. This is what
    makes the target immune to a drifting value baseline (the measured
    side-to-move bias grew 0.06 -> 0.37 over 1M steps)."""
    cfg = MCTSConfig()
    logits = np.log(np.array([0.4, 0.3, 0.2, 0.1]))
    qs = np.array([0.2, 0.0, -0.1, -0.3])
    p0 = _target(logits, qs, 32.0, cfg)
    p1 = _target(logits, qs + 0.37, 32.0, cfg)
    assert np.allclose(p0, p1, atol=1e-9), f"offset changed target: {p0} vs {p1}"


def test_monotone_in_q():
    """Higher Q must still get more mass -- the transform stays a policy
    IMPROVEMENT operator, we only softened its temperature."""
    cfg = MCTSConfig()
    logits = np.log(np.array([0.25, 0.25, 0.25, 0.25]))
    p = _target(logits, np.array([0.3, 0.1, -0.1, -0.3]), 32.0, cfg)
    assert list(np.argsort(-p)) == [0, 1, 2, 3]
