"""Unit tests for the pure outcome-bucketing core (tools/outcome_buckets).

Pins the Tier-2 invariants in isolation (no torch / no MCTS):
event-class hard-split, mass conservation across splits, renorm
sampling, ground-stat retention on split, and the OGA value-
heterogeneity split trigger. The MCTS integration is tested
separately.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from tools.outcome_buckets import (   # noqa: E402
    Bucket, MemberStat, event_class, initial_buckets, propose_split, split,
)


def _k(a_hp, d_hp, a_sl=False, d_sl=False, a_po=False, d_po=False):
    return (a_hp, d_hp, a_sl, d_sl, a_po, d_po)


# ---- event class -----------------------------------------------------

def test_event_class_splits_on_death_and_status():
    assert event_class(_k(10, 8)) == (False, False, False, False, False, False)
    assert event_class(_k(0, 8)) == (True, False, False, False, False, False)
    assert event_class(_k(10, 0)) == (False, True, False, False, False, False)
    # poison/slow are part of the class
    assert event_class(_k(10, 8, d_po=True)) != event_class(_k(10, 8))


def test_initial_buckets_group_by_event_class_and_conserve_mass():
    probs = {
        _k(10, 8): 0.3, _k(10, 5): 0.2, _k(7, 8): 0.1,   # both alive
        _k(10, 0): 0.25,                                  # defender dead
        _k(0, 8): 0.15,                                   # attacker dead
    }
    buckets = initial_buckets(probs)
    assert len(buckets) == 3, "three distinct event-classes"
    # members partitioned, total mass conserved
    total = sum(b.prob_mass for b in buckets)
    assert abs(total - 1.0) < 1e-9
    all_members = set()
    for b in buckets:
        all_members |= set(b.members)
    assert all_members == set(probs)
    # the both-alive class holds exactly the 3 alive outcomes
    alive = next(b for b in buckets if b.event == (False, False, False, False, False, False))
    assert set(alive.members) == {_k(10, 8), _k(10, 5), _k(7, 8)}


# ---- representative + sampling --------------------------------------

def test_representative_is_a_real_member_near_centroid():
    b = initial_buckets({_k(10, 9): 0.5, _k(10, 1): 0.5})[0]
    rep = b.representative()
    assert rep in b.members
    # centroid d_hp = 5; both members equidistant (9 vs 1 -> |4| each),
    # tie-break by key picks the smaller -> (10,1,...)
    assert rep == _k(10, 1)


def test_sample_member_matches_renormalized_probs():
    probs = {_k(10, 8): 0.6, _k(10, 4): 0.3, _k(10, 2): 0.1}
    b = initial_buckets(probs)[0]
    rng = np.random.default_rng(0)
    counts = {k: 0 for k in probs}
    N = 20000
    for _ in range(N):
        counts[b.sample_member(rng)] += 1
    for k, p in probs.items():
        assert abs(counts[k] / N - p) < 0.02, f"{k}: {counts[k]/N} vs {p}"


def test_sample_member_renormalizes_within_subbucket():
    # A sub-bucket holding two of three outcomes must sample by mass
    # renormalized to its own members only.
    probs = {_k(10, 8): 0.6, _k(10, 4): 0.3, _k(10, 2): 0.1}
    b = initial_buckets(probs)[0]
    lo, hi = split(b, axis=1, threshold=4.0)   # d_hp<4 -> {2}; >=4 -> {8,4}
    assert set(lo.members) == {_k(10, 2)}
    assert set(hi.members) == {_k(10, 8), _k(10, 4)}
    rng = np.random.default_rng(1)
    counts = {k: 0 for k in hi.members}
    N = 20000
    for _ in range(N):
        counts[hi.sample_member(rng)] += 1
    # within hi, renorm: 0.6/0.9 and 0.3/0.9
    assert abs(counts[_k(10, 8)] / N - 0.6 / 0.9) < 0.02


# ---- split: mass + stat conservation --------------------------------

def test_split_conserves_mass_and_retains_member_stats():
    probs = {_k(10, 8): 0.4, _k(10, 5): 0.35, _k(10, 2): 0.25}
    b = initial_buckets(probs)[0]
    # record some visits/values before splitting
    b.record(_k(10, 8), 0.9)
    b.record(_k(10, 8), 0.7)
    b.record(_k(10, 2), -0.5)
    pre_mass, pre_visits, pre_vsum = b.prob_mass, b.visits, b.value_sum
    lo, hi = split(b, axis=1, threshold=5.0)   # <5 -> {2}; >=5 -> {8,5}
    # mass conserved
    assert abs((lo.prob_mass + hi.prob_mass) - pre_mass) < 1e-12
    # member stats RETAINED (nothing inherited, nothing lost)
    assert lo.visits + hi.visits == pre_visits
    assert abs((lo.value_sum + hi.value_sum) - pre_vsum) < 1e-12
    # the (10,8) member's accumulated stats followed it into `hi`
    assert hi.members[_k(10, 8)].visits == 2
    assert abs(hi.members[_k(10, 8)].value_sum - 1.6) < 1e-12
    assert lo.members[_k(10, 2)].visits == 1


def test_split_preserves_event_class():
    b = initial_buckets({_k(10, 8): 0.5, _k(10, 2): 0.5})[0]
    lo, hi = split(b, axis=1, threshold=5.0)
    assert lo.event == b.event == hi.event


# ---- OGA split trigger ----------------------------------------------

def test_no_split_below_v_min():
    b = initial_buckets({_k(10, 8): 0.5, _k(10, 2): 0.5})[0]
    b.record(_k(10, 8), 1.0)
    assert propose_split(b, v_min=10, tau=0.1) is None


def test_no_split_when_homogeneous():
    # both halves same value -> no heterogeneity -> no split
    probs = {_k(10, h): 0.25 for h in (8, 6, 4, 2)}
    b = initial_buckets(probs)[0]
    for k in probs:
        for _ in range(3):
            b.record(k, 0.5)         # identical value everywhere
    assert propose_split(b, v_min=4, tau=0.1) is None


def test_split_detects_value_heterogeneity_across_hp():
    # low-defender-HP outcomes are much better (killable next turn);
    # the value gradient along the defender-HP axis must trigger a
    # split on axis=1.
    probs = {_k(10, h): 0.25 for h in (8, 6, 4, 2)}
    b = initial_buckets(probs)[0]
    for _ in range(5):
        b.record(_k(10, 8), -0.6)
        b.record(_k(10, 6), -0.5)
        b.record(_k(10, 4), 0.5)
        b.record(_k(10, 2), 0.6)
    out = propose_split(b, v_min=8, tau=0.3)
    assert out is not None, "should detect the value gradient"
    axis, thr, gap = out
    assert axis == 1, "heterogeneity is on the defender-HP axis"
    assert gap > 0.3
    # splitting there separates the good (low-HP) from bad (high-HP)
    lo, hi = split(b, axis, thr)
    assert lo.mean_value > hi.mean_value   # lo = low defender HP = better


def test_repeated_splits_converge_to_singletons():
    # PARSS guarantee: under continued pressure, buckets refine to one
    # member each (= exact).
    probs = {_k(10, h): 1.0 / 6 for h in (10, 8, 6, 4, 2, 1)}
    work = initial_buckets(probs)
    # force splits by giving a monotonic value gradient and splitting
    # any splittable bucket until none remain.
    for b in work:
        for k in b.members:
            b.record(k, float(k[1]) / 10.0)   # value ~ defender HP
    frontier = list(work)
    singletons = []
    guard = 0
    while frontier and guard < 100:
        guard += 1
        b = frontier.pop()
        if len(b.members) == 1:
            singletons.append(b); continue
        out = propose_split(b, v_min=1, tau=0.0)   # tau=0 -> always split if heterogeneous
        if out is None:
            singletons.append(b); continue
        lo, hi = split(b, out[0], out[1])
        frontier.extend([lo, hi])
    assert all(len(b.members) == 1 for b in singletons)
    # mass still conserved over the fully-refined set
    assert abs(sum(b.prob_mass for b in singletons) - 1.0) < 1e-9
