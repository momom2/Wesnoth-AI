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

  * SPLIT trigger (OGA, significance-aware) + mechanism (PARSS):
    once a bucket has visits >= V_min AND the two halves either side
    of an HP axis's visit-weighted median have mean values differing
    by > z_sig standard errors (a value gap unlikely to be sampling
    noise -- no hand-tuned threshold), bisect at the median and
    recurse. Refines only where a value difference is statistically
    detectable, so truly value-homogeneous buckets stay merged (the
    OGA efficiency win); under sustained pressure a heterogeneous
    bucket refines toward one member each (PARSS convergence).
    Member stats (incl. value_sq_sum for the SE) are retained across
    the split.

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

# (a_hp, d_hp, a_slowed, d_slowed, a_poisoned, d_poisoned,
#  a_petrified, d_petrified, a_type, d_type) -- the same tuple
# tools/combat_outcomes produces. Imported lazily in callers; re-stated
# here so this module has no import-time dependency.
OutcomeKey = Tuple[int, int, bool, bool, bool, bool, bool, bool, str, str]

# Event-class = the discrete (non-HP) signature. `dead` is derived
# from hp <= 0; combat_outcomes._canonical already zeroes a dead
# unit's status flags / type, so the class is consistent.
EventClass = Tuple[bool, bool, bool, bool, bool, bool, bool, bool, str, str]


def event_class(key: OutcomeKey) -> EventClass:
    """Discrete signature of an outcome: (a_dead, d_dead, a_slowed,
    d_slowed, a_poisoned, d_poisoned, a_petrified, d_petrified,
    a_type, d_type). Outcomes in different classes are NEVER bucketed
    together (different units alive / statuses / TYPE = structurally
    different game). Petrified rides here (a stoned unit differs from a
    live one at the same HP); so does the type-name, so an ADVANCED unit
    buckets apart from its pre-advance self."""
    a_hp, d_hp, a_sl, d_sl, a_po, d_po, a_pe, d_pe, a_ty, d_ty = key
    return (a_hp <= 0, d_hp <= 0, a_sl, d_sl, a_po, d_po, a_pe, d_pe,
            a_ty, d_ty)


@dataclass
class MemberStat:
    """Ground-level statistics for one concrete outcome (OutcomeKey).
    `prob` is the outcome's full-distribution probability (fixed);
    `visits` / `value_sum` / `value_sq_sum` accumulate as the search
    samples it. `value_sq_sum` (sum of squared backed-up values) lets
    the split test estimate value variance -> standard error, so the
    split trigger is statistically significance-aware rather than
    keyed to a hand-tuned threshold."""
    prob:         float
    visits:       int = 0
    value_sum:    float = 0.0
    value_sq_sum: float = 0.0

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
        m.value_sq_sum += value * value


def initial_buckets(probs: Dict[OutcomeKey, float]) -> List["Bucket"]:
    """Coarsest valid abstraction: ONE bucket per event-class (PARSS
    starts coarse; OGA refinement does the rest). `probs` is
    OutcomeDistribution.probs."""
    by_class: Dict[EventClass, Dict[OutcomeKey, MemberStat]] = {}
    for key, p in probs.items():
        by_class.setdefault(event_class(key), {})[key] = MemberStat(prob=p)
    return [Bucket(event=ec, members=mem) for ec, mem in by_class.items()]


def propose_split(bucket: "Bucket", v_min: int, z_sig: float = 2.0,
                  min_half_visits: int = 2
                  ) -> Optional[Tuple[int, float, float]]:
    """Decide whether `bucket` should be refined, OGA-style with a
    SIGNIFICANCE-AWARE trigger. Returns (axis, threshold, gap) or None.

    axis: 0 = attacker HP, 1 = defender HP. threshold: split members
    with hp < threshold from hp >= threshold. gap: |mean_hi - mean_lo|.

    Criterion (both must hold):
      * visit pressure: bucket.visits >= v_min;
      * statistical significance: for some HP axis, splitting at that
        axis's visit-weighted median yields two halves each with
        >= min_half_visits visits whose mean backed-up values differ
        by MORE than `z_sig` standard errors of the difference --
        i.e. the bucket is masking a value gap unlikely to be
        sampling noise. Picks the axis with the largest STANDARDIZED
        gap (most significant direction).

    This replaces the earlier fixed-tau threshold: the test
    self-calibrates to the observed value spread, so there's no
    hand-tuned heterogeneity knob -- only `z_sig` (default 2 ~ 95%).

    CAVEAT (honest): MCTS value backups for a node are correlated,
    not i.i.d., so the SE computed from per-visit values
    UNDERESTIMATES the true uncertainty -> the test is somewhat
    split-happy. `v_min` + `min_half_visits` + a conservative `z_sig`
    provide margin; raise `z_sig` if splits proliferate. (Same
    spirit as confidence-based abstraction methods.)"""
    if bucket.visits < v_min:
        return None
    visited = [(k, m) for k, m in bucket.members.items() if m.visits > 0]
    if len(visited) < 2:
        return None

    best: Optional[Tuple[int, float, float]] = None   # (axis, thr, gap)
    best_z = z_sig
    for axis in (0, 1):
        if len({k[axis] for k, _ in visited}) < 2:
            continue   # no variation on this axis -> can't split here
        thr = _weighted_median_hp(visited, axis)
        lo = [(k, m) for k, m in visited if k[axis] < thr]
        hi = [(k, m) for k, m in visited if k[axis] >= thr]
        n_lo, mean_lo, se_lo = _half_stats(lo)
        n_hi, mean_hi, se_hi = _half_stats(hi)
        if n_lo < min_half_visits or n_hi < min_half_visits:
            continue
        gap = abs(mean_hi - mean_lo)
        se_diff = math.sqrt(se_lo * se_lo + se_hi * se_hi)
        # Standardized gap; se_diff==0 (perfectly-consistent halves)
        # means any nonzero gap is infinitely significant.
        std_gap = math.inf if se_diff == 0.0 else gap / se_diff
        if gap > 0.0 and std_gap > best_z:
            best, best_z = (axis, float(thr), gap), std_gap
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

def _half_stats(items: List[Tuple[OutcomeKey, MemberStat]]
                ) -> Tuple[int, float, float]:
    """(N, mean, standard_error) over the pooled per-visit values of a
    half. N = total visits; mean = visit-weighted mean value; SE =
    sqrt(sample_variance / N). SE is 0 for N<2 or perfectly-consistent
    values (caller treats SE=0 as 'any gap is significant')."""
    N = sum(m.visits for _, m in items)
    if N <= 0:
        return 0, 0.0, 0.0
    S = sum(m.value_sum for _, m in items)
    Q = sum(m.value_sq_sum for _, m in items)
    mean = S / N
    if N < 2:
        return N, mean, 0.0
    var = max(0.0, (Q - S * S / N) / (N - 1))   # sample variance, clamped
    return N, mean, math.sqrt(var / N)


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
