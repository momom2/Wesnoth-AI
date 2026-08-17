"""TCS target link functions (user ruling 2026-08-17: evaluation
exposure must not buy probability under an uninformative grader).

The core pin is the DECOY INVARIANT, the miniature of the harness
that measured the leg-3 ratchet: under a pure-noise value head, a
force-included (always-evaluated) action gains expected target mass
under the exp link and must NOT under the linear link. Production
code paths throughout: `tcs_target_distribution` builds the targets,
`gumbel_top_k_alternatives` picks the evaluated sets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.mcts import MCTSConfig                        # noqa: E402
from tools.turn_search import (                          # noqa: E402
    TurnSearchConfig, gumbel_top_k_alternatives, tcs_target_distribution,
)

N_ACT = 20
ET = 0           # index of the force-included "end_turn" analog
DECOY = 1        # same prior, never force-included
NOISE_SD = 0.08  # ~2 C51 atoms, the verifier's harness value


def _priors() -> np.ndarray:
    p = np.full(N_ACT, (1.0 - 0.16) / (N_ACT - 2))
    p[ET] = 0.08
    p[DECOY] = 0.08
    return p


def _mean_drift(link: str, rng: np.random.Generator,
                trials: int = 3000) -> np.ndarray:
    """E[target - prior] per action under a PURE-NOISE grader, with
    the production evaluated-set sampler force-including ET."""
    p = _priors()
    cfg = MCTSConfig()
    acc = np.zeros(N_ACT)
    for _ in range(trials):
        incumbent = int(rng.integers(2, N_ACT))    # a normal action
        picks = gumbel_top_k_alternatives(p, incumbent, ET, 4, rng)
        evaluated = np.zeros(N_ACT, dtype=bool)
        evaluated[incumbent] = True
        for i in picks:
            evaluated[i] = True
        q = np.where(evaluated, rng.normal(0.0, NOISE_SD, N_ACT), 0.0)
        tgt = tcs_target_distribution(
            p, q, evaluated, v_root=0.0,
            max_visits=float(evaluated.sum()), mcts_config=cfg,
            link=link)
        acc += tgt
    return acc / trials - p / p.sum()


def test_decoy_invariant_linear_kills_the_exposure_ratchet():
    rng = np.random.default_rng(7)
    d_exp = _mean_drift("exp", rng)
    rng = np.random.default_rng(7)
    d_lin = _mean_drift("linear", rng)
    # exp: the always-evaluated action collects the convexity bonus.
    assert d_exp[ET] > 0.02, f"exp ET drift {d_exp[ET]:+.4f}"
    assert d_exp[ET] > 4 * abs(d_exp[DECOY])
    # linear: exposure buys nothing (residual renorm bias is small
    # and non-positive -- it may only push evaluated actions DOWN).
    assert abs(d_lin[ET]) < 0.005, f"linear ET drift {d_lin[ET]:+.4f}"
    assert d_lin[ET] < 0.001


def test_linear_still_moves_mass_toward_true_signal():
    """Exposure-invariance must not cost signal: a genuinely better
    evaluated action must gain mass over its prior."""
    p = _priors()
    evaluated = np.zeros(N_ACT, dtype=bool)
    evaluated[[ET, 2, 3, 4]] = True
    q = np.zeros(N_ACT)
    q[3] = 0.12                          # 3 atoms above its peers
    tgt = tcs_target_distribution(
        p, q, evaluated, v_root=0.0, max_visits=4.0,
        mcts_config=MCTSConfig(), link="linear")
    pn = p / p.sum()
    assert tgt[3] > pn[3] * 1.3
    assert tgt[ET] < pn[ET]              # the average action pays


def test_single_evaluation_is_neutral():
    """One evaluated action has no peers to compare against: the
    target must equal the (lam-discounted) prior exactly."""
    p = _priors()
    evaluated = np.zeros(N_ACT, dtype=bool)
    evaluated[5] = True
    q = np.zeros(N_ACT)
    q[5] = 0.9                           # huge -- and meaningless
    tgt = tcs_target_distribution(
        p, q, evaluated, v_root=0.0, max_visits=1.0,
        mcts_config=MCTSConfig(), link="linear")
    assert np.allclose(tgt, p / p.sum(), atol=1e-12)


def test_clipping_and_stats_out():
    p = np.array([0.4, 0.3, 0.3])
    evaluated = np.array([True, True, True])
    q = np.array([0.5, 0.0, -0.5])
    stats = {}
    tgt = tcs_target_distribution(
        p, q, evaluated, v_root=0.0, max_visits=3.0,
        mcts_config=MCTSConfig(), link="linear", beta=5.0,
        stats_out=stats)
    assert stats["link_clip_frac"] == 1 / 3       # the -0.5 action
    assert tgt[2] == 0.0
    assert tgt[0] > tgt[1] > 0.0
    assert abs(tgt.sum() - 1.0) < 1e-12


def test_exp_path_is_byte_identical_to_the_shared_transform():
    """link='exp' must remain the sigma transform verbatim -- the
    probe's 2026-08-14 baselines and the Gumbel-MCTS path depend on
    it not drifting."""
    from tools.mcts import _gumbel_sigma
    rng = np.random.default_rng(3)
    p = _priors()
    evaluated = np.zeros(N_ACT, dtype=bool)
    evaluated[[ET, 2, 7]] = True
    q = np.where(evaluated, rng.normal(0.0, 0.2, N_ACT), 0.0)
    cfg = MCTSConfig()
    got = tcs_target_distribution(p, q, evaluated, v_root=0.1,
                                  max_visits=3.0, mcts_config=cfg,
                                  link="exp")
    pn = np.maximum(p, 1e-12)
    pn = pn / pn.sum()
    pv = pn[evaluated]
    weighted = float((pv * q[evaluated]).sum() / pv.sum())
    v_mix = (0.1 + 3.0 * weighted) / 4.0
    completed = np.where(evaluated, q, v_mix)
    t = (float(getattr(cfg, "distill_prior_discount", 1.0))
         * np.log(pn) + _gumbel_sigma(completed, 3.0, cfg))
    t -= t.max()
    want = np.exp(t)
    want /= want.sum()
    assert np.allclose(got, want, atol=1e-15)


def test_config_default_is_linear():
    cfg = TurnSearchConfig()
    assert cfg.target_link == "linear"
    assert cfg.target_beta == 5.0
