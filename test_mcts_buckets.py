"""Tier-2 outcome bucketing integration (stage 1: copy-at-expansion).

Drives forced attack-edge simulations (the Gumbel inner loop) on a
real multi-outcome combat and checks the headline property: with
`outcome_buckets` ON, same-event-class outcomes share ONE network
forward (later members copy the representative's edges) -> strictly
fewer forwards than the per-outcome path, while the search still runs
cleanly and deterministically. Value-unbiasedness (members recurse
from their own real state) is checked loosely here; the precise
mass/Q invariants live in test_outcome_buckets for the pure core.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import torch  # noqa: E402

from transformer_policy import TransformerPolicy  # noqa: E402
from classes import state_key  # noqa: E402
from tools.abilities import hex_neighbors  # noqa: E402
from tools.draw_tiebreak import DrawTiebreakConfig  # noqa: E402
from tools.mcts import (  # noqa: E402
    MCTSConfig, MCTSNode, _expand, _run_one_sim,
)
from sim_test_helpers import fresh_scenario_sim  # noqa: E402


def _combat_sim():
    """A mini state with a friendly attacker teleported adjacent to an
    enemy, so the root has an attack edge with a multi-outcome combat."""
    sim = fresh_scenario_sim(seed=21, max_turns=10, mini=True)
    side = sim.gs.global_info.current_side
    att = next(u for u in sim.gs.map.units if u.side == side and u.attacks)
    dfd = next(u for u in sim.gs.map.units if u.side != side and u.attacks)
    occ = {(u.position.x, u.position.y) for u in sim.gs.map.units}
    spot = next(((nx, ny) for nx, ny in hex_neighbors(dfd.position.x,
                                                      dfd.position.y)
                 if (nx, ny) not in occ and 0 <= nx < sim.gs.map.size_x
                 and 0 <= ny < sim.gs.map.size_y), None)
    assert spot is not None
    att.position.x, att.position.y = spot
    # Raise HP so the combat has many "both survive" HP outcomes
    # (the bucket members the copy-at-expansion amortizes over).
    att.current_hp = att.max_hp = 60
    dfd.current_hp = dfd.max_hp = 60
    return sim


def _policy():
    pol = TransformerPolicy(device=torch.device("cpu"),
                            d_model=64, num_layers=2, num_heads=4, d_ff=128)
    pol._encoder.eval(); pol._model.eval()
    return pol


def _cfg(buckets: bool, sims: int = 48) -> MCTSConfig:
    return MCTSConfig(
        n_simulations=sims, gumbel_root=True, gumbel_m=8,
        chance_nodes=True, exact_outcome_enumeration=True,
        draw_tiebreak=DrawTiebreakConfig(cap=0.3), batch_size=1,
        add_root_noise=False, outcome_buckets=buckets)


def _forced_attack_run(pol, sim, buckets, seed, n_sims=48):
    """Build+expand a root, then run `n_sims` sims forced through the
    attack edge (mirrors the Gumbel inner loop), counting model
    forwards during the loop only. Returns (attack_edge, n_forwards,
    root)."""
    enc, mdl = pol._encoder, pol._model
    cfg = _cfg(buckets, n_sims)
    root = MCTSNode(sim.fork())
    _expand(root, mdl, enc, tiebreak=cfg.draw_tiebreak)
    atk = next((e for e in root.edges
                if isinstance(e.action, dict)
                and e.action.get("type") == "attack"), None)
    assert atk is not None, "no attack edge at the combat root"

    n_fwd = [0]
    handle = mdl.register_forward_pre_hook(
        lambda m, i: n_fwd.__setitem__(0, n_fwd[0] + 1))
    rng = np.random.default_rng(seed)
    tt = {state_key(root.sim.gs): root}
    try:
        for _ in range(n_sims):
            _run_one_sim(root, mdl, enc, cfg, tt, {"hits": 0, "misses": 0},
                         forced_first_edge=atk, sample_rng=rng)
    finally:
        handle.remove()
    return atk, n_fwd[0], root


def test_bucketing_reduces_forwards_and_runs_clean():
    pol = _policy()
    sim = _combat_sim()
    atk_off, fwd_off, root_off = _forced_attack_run(pol, sim, False, seed=0)
    atk_on,  fwd_on,  root_on = _forced_attack_run(pol, sim, True,  seed=0)

    # Both ran all sims through the attack edge.
    assert atk_off.n_visits == 48 and atk_on.n_visits == 48
    # The combat fanned out into several distinct outcomes (else the
    # test is vacuous -- nothing to bucket).
    assert len(atk_off.children) >= 3, (
        f"expected a multi-outcome combat; got {len(atk_off.children)}")
    # Headline: bucketing shares forwards across same-event-class
    # outcomes -> strictly fewer model forwards.
    assert fwd_on < fwd_off, (
        f"bucketing should cut forwards: on={fwd_on} off={fwd_off}")
    # A bucket representative WAS elected and reused (copy-at-expansion
    # actually fired).
    assert atk_on.bucket_rep, "no bucket representative recorded"


def test_bucketing_is_deterministic():
    pol = _policy()
    sim = _combat_sim()
    _, f1, r1 = _forced_attack_run(pol, sim, True, seed=7)
    _, f2, r2 = _forced_attack_run(pol, sim, True, seed=7)
    assert f1 == f2, "same seed must give the same forward count"
    # identical outcome-child visit structure
    v1 = sorted((str(k), c._total_visits) for k, c in r1.edges[0].children.items()) \
        if r1.edges else []
    # compare the attack edge's children visit multiset
    a1 = next(e for e in r1.edges if e.action.get("type") == "attack")
    a2 = next(e for e in r2.edges if e.action.get("type") == "attack")
    m1 = sorted(c._total_visits for c in a1.children.values())
    m2 = sorted(c._total_visits for c in a2.children.values())
    assert m1 == m2, "same seed must give identical child visit structure"


def test_root_value_estimate_in_band():
    """Loose unbiasedness sanity: the attack edge's Q under bucketing
    is in the same ballpark as the per-outcome path (both estimate the
    same underlying combat value; bucketing only approximates priors,
    and members still recurse from their own real states)."""
    pol = _policy()
    sim = _combat_sim()
    atk_off, _, _ = _forced_attack_run(pol, sim, False, seed=3, n_sims=64)
    atk_on, _, _ = _forced_attack_run(pol, sim, True, seed=3, n_sims=64)
    assert abs(atk_on.q_value - atk_off.q_value) < 0.25, (
        f"bucketed Q {atk_on.q_value:.3f} far from per-outcome "
        f"{atk_off.q_value:.3f}")
