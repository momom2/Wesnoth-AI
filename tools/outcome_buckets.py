"""Adaptive outcome bucketing for MCTS chance nodes (Tier 2).

Combat is stochastic: an attack fans out into up to (n+1)(m+1)
distinct outcomes (attacker-HP x defender-HP, x status flags). The
Tier-1 DP (tools/combat_outcomes) gives each outcome's exact
probability, but treating every distinct OutcomeKey as its own MCTS
child means one network forward per outcome and the visit budget
diluted across many near-identical futures.

This module is the PURE, torch-free core of the fix: group outcomes
into BUCKETS that share one representative forward, refine them only
where it matters, and keep statistics at the ground (per-member)
level so refinement is just a re-grouping. The MCTS integration
(representative forward, shared edges, per-visit member sampling,
backup) lives in tools/mcts.py; everything here is testable in
isolation.

Design (agreed 2026-06-15; see BACKLOG "Tier 2: adaptive outcome
bucketing"). Lit basis: PARSS (Hostetler et al.) for the
coarse->fine, split-in-half, asymptotic-convergence backbone;
OGA-UCT (Anand et al.) for the value-heterogeneity split trigger.

  * EVENT HARD-SPLIT (never merged): bucket first by the discrete
    part of OutcomeKey -- (a_dead, d_dead, a_slowed, d_slowed,
    a_poisoned, d_poisoned). Within an event-class only the
    continuous (a_hp, d_hp) varies, and legal actions are
    HP-independent, so a bucket's shared post-combat edges are valid
    for all members.

  * GROUND-STATS AGGREGATION (OGA model): statistics live per MEMBER
    (OutcomeKey -> visits, value_sum). A bucket's value is the
    aggregate of its members'. A split is a RE-GROUPING of retained
    member stats -> nothing inherited, nothing lost, warm AND
    unbiased by construction (this is why there's no "child visit
    inheritance" question).

  * SPLIT trigger (OGA) + mechanism (PARSS): once a bucket has
    visits >= V_min AND its members show value heterogeneity > tau
    along an HP axis (compare the visit-weighted mean member-value
    of the two halves either side of that axis's median), bisect at
    the median and recurse. Continued pressure -> one member per
    bucket = exact (PARSS convergence). Member stats are retained
    across the split.

Probability convention: each member stores its FULL-distribution
probability (from OutcomeDistribution.probs, which sums to ~1 across
all members of all buckets, minus the DP's dropped dust). So
`bucket.prob_mass` = P(land in this bucket); sampling a bucket by
prob_mass then a member by within-bucket renormalized prob is
exactly sampling a member by its full probability -> unbiased.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# (a_hp, d_hp, a_slowed, d_slowed, a_poisoned, d_poisoned) -- the same
# tuple tools/combat_outcomes produces. Imported lazily in callers;
# re-stated here so this module has no import-time dependency.
OutcomeKey = Tuple[int, int, bool, bool, bool, bool]

# Event-class = the discrete (non-HP) signature. `dead` is derived
# from hp <= 0; combat_outcomes._canonical already zeroes a dead
# unit's status flags, so the class is consistent.
EventClass = Tuple[bool, bool, bool, bool, bool, bool]


def event_class(key: OutcomeKey) -> EventClass:
    """Discrete signature of an outcome: (a_dead, d_dead, a_slowed,
    d_slowed, a_poisoned, d_poisoned). Outcomes in different classes
    are NEVER bucketed together (different units alive / statuses =
    structurally different game)."""
    a_hp, d_hp, a_sl, d_sl, a_po, d_po = key
    return (a_hp <= 0, d_hp <= 0, a_sl, d_sl, a_po, d_po)


@dataclass
class MemberStat:
    """Ground-level statistics for one concrete outcome (OutcomeKey).
    `prob` is the outcome's full-distribution probability (fixed);
    `visits`/`value_sum` accumulate as the search samples it."""
    prob:      float
    visits:    int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class Bucket:
    """A group of outcome members sharing one representative forward.
    Holds only the pure bookkeeping; the shared MCTS node/edges are
    attached by the caller (tools/mcts.py) and are NOT this module's
    concern."""
    event:   EventClass
    members: Dict[OutcomeKey, MemberStat]

    # ---- aggregate stats (OGA: bucket = aggregate of its members) ----
    @property
    def prob_mass(self) -> float:
        return sum(m.prob for m in self.members.values())

    @property
    def visits(self) -> int:
        return sum(m.visits for m in self.members.values())

    @property
    def value_sum(self) -> float:
        return sum(m.value_sum for m in self.members.values())

    @property
    def mean_value(self) -> float:
        v = self.visits
        return self.value_sum / v if v else 0.0

    # ---- per-visit operations ----
    def representative(self) -> OutcomeKey:
        """The member to run the shared forward on: the one closest to
        the probability-weighted centroid HP (snapped to a real member
        so the forward is on a real state). Deterministic."""
        keys = list(self.members)
        if len(keys) == 1:
            return keys[0]
        mass = self.prob_mass or 1.0
        cen_a = sum(k[0] * self.members[k].prob for k in keys) / mass
        cen_d = sum(k[1] * self.members[k].prob for k in keys) / mass
        # Manhattan distance to centroid; tie-break by key for
        # determinism.
        return min(keys, key=lambda k: (abs(k[0] - cen_a) + abs(k[1] - cen_d), k))

    def sample_member(self, rng) -> OutcomeKey:
        """Sample a member by its within-bucket renormalized
        probability. `rng` is a numpy Generator (uses .random()).
        Falls back to the representative if mass is degenerate."""
        keys = list(self.members)
        if len(keys) == 1:
            return keys[0]
        mass = self.prob_mass
        if mass <= 0:
            return self.representative()
        r = rng.random() * mass
        acc = 0.0
        for k in keys:
            acc += self.members[k].prob
            if r <= acc:
                return k
        return keys[-1]   # float-dust guard

    def record(self, key: OutcomeKey, value: float) -> None:
        """Attribute one backed-up value to a member (ground stat)."""
        m = self.members[key]
        m.visits += 1
        m.value_sum += value


def initial_buckets(probs: Dict[OutcomeKey, float]) -> List["Bucket"]:
    """Coarsest valid abstraction: ONE bucket per event-class (PARSS
    starts coarse; OGA refinement does the rest). `probs` is
    OutcomeDistribution.probs."""
    by_class: Dict[EventClass, Dict[OutcomeKey, MemberStat]] = {}
    for key, p in probs.items():
        by_class.setdefault(event_class(key), {})[key] = MemberStat(prob=p)
    return [Bucket(event=ec, members=mem) for ec, mem in by_class.items()]


def propose_split(bucket: "Bucket", v_min: int,
                  tau: float) -> Optional[Tuple[int, float, float]]:
    """Decide whether `bucket` should be refined, OGA-style. Returns
    (axis, threshold, gap) or None.

    axis: 0 = attacker HP, 1 = defender HP. threshold: split members
    with hp < threshold from hp >= threshold. gap: |mean value of the
    two halves|, the heterogeneity that justified the split.

    Criterion: visits >= v_min, and for some HP axis with >1 distinct
    value among VISITED members, splitting at that axis's
    visit-weighted median yields two non-empty visited halves whose
    mean member-values differ by > tau. Picks the axis with the
    largest gap (most-heterogeneous direction)."""
    if bucket.visits < v_min:
        return None
    visited = [(k, m) for k, m in bucket.members.items() if m.visits > 0]
    if len(visited) < 2:
        return None

    best: Optional[Tuple[int, float, float]] = None
    for axis in (0, 1):
        vals = sorted({k[axis] for k, _ in visited})
        if len(vals) < 2:
            continue   # no variation on this axis -> can't split here
        # Visit-weighted median HP on this axis (so the split balances
        # sampled mass, not raw member count).
        thr = _weighted_median_hp(visited, axis)
        lo = [(k, m) for k, m in visited if k[axis] < thr]
        hi = [(k, m) for k, m in visited if k[axis] >= thr]
        if not lo or not hi:
            continue
        gap = abs(_wmean(lo) - _wmean(hi))
        if gap > tau and (best is None or gap > best[2]):
            best = (axis, float(thr), gap)
    return best


def split(bucket: "Bucket", axis: int,
          threshold: float) -> Tuple["Bucket", "Bucket"]:
    """Partition `bucket` into (lo, hi) at `threshold` on `axis`,
    RETAINING each member's MemberStat (visits/value_sum). Mass and
    stats are conserved: lo.members + hi.members == bucket.members.
    Both halves keep the same event-class."""
    lo: Dict[OutcomeKey, MemberStat] = {}
    hi: Dict[OutcomeKey, MemberStat] = {}
    for k, m in bucket.members.items():
        (lo if k[axis] < threshold else hi)[k] = m
    return (Bucket(event=bucket.event, members=lo),
            Bucket(event=bucket.event, members=hi))


# ---------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------

def _wmean(items: List[Tuple[OutcomeKey, MemberStat]]) -> float:
    """Visit-weighted mean of member mean-values (so a member sampled
    more carries more weight in the heterogeneity test)."""
    tot_v = sum(m.visits for _, m in items)
    if tot_v <= 0:
        return 0.0
    return sum(m.value_sum for _, m in items) / tot_v


def _weighted_median_hp(items: List[Tuple[OutcomeKey, MemberStat]],
                        axis: int) -> float:
    """Smallest HP `h` on `axis` such that the visited mass with
    hp >= h is < half -- i.e. the cut that balances visited mass.
    Returns a value that puts at least one member on each side when
    the axis has >1 distinct value."""
    by_hp: Dict[int, int] = {}
    for k, m in items:
        by_hp[k[axis]] = by_hp.get(k[axis], 0) + m.visits
    hps = sorted(by_hp)
    total = sum(by_hp.values())
    # Walk HP ascending, cut after the point that first reaches >= half
    # the visited mass; threshold = next HP value (so the cut HP goes
    # to the `lo` side and there's a non-empty `hi`).
    acc = 0
    for i, h in enumerate(hps):
        acc += by_hp[h]
        if acc * 2 >= total and i + 1 < len(hps):
            return float(hps[i + 1])
    # Degenerate (all mass at top): cut just below the max.
    return float(hps[-1])
