# Swap detector — locked design

**Criterion revised 2026-07-23 (user-approved): distributional
pathwise dominance**, replacing the original RNG-coupling formulation
(2026-07-22). The rest of the design — comparison vector, configurable
views, plugin registry, validation gates — is unchanged. First shipped
generator: `backstab_setup` (the others follow).

Offline detector for strictly-better reorderings of a side-turn's
actions, run over fully-recorded training replays. Observe-only in
v1: firing data calibrates future search-time ordering oracles; NO
reward/label channel (EV-mode and any training coupling deferred).

## Criterion: distributional pathwise dominance

No RNG coupling, no realized rolls. From the fixed pre-window state,
each ordering induces an OUTCOME DISTRIBUTION over end-states (combat
is a function of state + weapon stats, so the distribution is
determined). The swap is a strict improvement iff, per comparison
dimension, the candidate ordering's distribution weakly stochastically
dominates the baseline's (good direction; product order over dims) and
strictly on ≥1 dimension with positive probability.

Why distributional beats coupling here:

- **Kills the hindsight class structurally.** An `independent` reorder
  (same actions, params identical) has the SAME distribution → `=` on
  every dimension automatically. There is no realized difference to
  cherry-pick, so the coupling design's "NEVER flag realized
  differences" caveat becomes a theorem, not a guard.
- **Drops the strict-sync dependency.** No strike-for-strike stream
  alignment is required, so differing CTH / strike counts are fine and
  nothing is skipped for alignment reasons. v1 needs only the
  reconstructed pre-window state + the two action lists; the export's
  `[random_seed]` / `[checkup]` data is a gate-1 reconstruction check,
  not a criterion input.

**Bounds shortcut (the usual fast path).** While no unit's alive/dead
status varies across the outcome space, every dimension is monotone in
the hit-pattern, so its support is an interval `[lo, hi]` at the
all-miss / all-hit extremes. `lo(candidate) ≥ hi(baseline)`
(more-is-better) certifies almost-sure dominance on that dimension with
no enumeration; symmetric for less-is-better. Overlapping intervals are
inconclusive by bounds alone (v1 does not claim).

**Death = the one discontinuity.** When a unit's end-HP interval
straddles 0, its death restructures what follows (no retaliation, hex
freed, re-target changes), so bounds no longer suffice and the cases
split die/survive (and by timing). v1 renders a verdict only where a
sound per-dimension dominance survives the split; the GENERAL rule for
first-order dominance across death-branches with differing `P(death)`
is deferred (see "Deferred"). Backstab does not need it (see gate 2).

**Exactness gate on the outcome engine.** Distributions come from
`tools/combat_outcomes.enumerate_attack_outcomes`, which returns the
EXACT distribution or `None` — it never approximates. It returns `None`
for petrify, *possible advancement* (a level-up changes the unit
mid-window), and complexity blow-ups (berserk's refilled schedule past
`MAX_SCHEDULE=512` / `MAX_DP_STATES=4096`). Any attack in either
ordering that comes back `None` makes the comparison `inconclusive`;
the detector NEVER samples to fill the gap.

## Classification (mechanical, sim-decided)

- sequential: swapped order illegal in a forked sim -> unswappable.
- independent: swap legal AND all action params identical both ways
  -> distributions provably equal -> `=` on every dimension (no
  realized difference exists to cherry-pick).
- correlated: swap legal, params differ -> verify distributional
  dominance (above).

## Per-dimension comparison vector (the stored primitive)

Per comparison (per death-branch case when a unit's survival is
ambiguous), one symbol {<, =, >, incomparable} per dimension.
Dimension registry (FINE-GRAINED; views may coarsen):

- unit existence: own survivors superset-eq, enemy subset-eq
- HP: componentwise on the common survivor set (own >=, enemy <=)
- XP: per-own-unit >= (XP-safety falls out of the product order)
- MP remaining: per-un-acted-unit >= (instrumental, turn-scoped)
- villages / gold
- visibility: FILTRATION-indexed (observation set at each decision
  index, pointwise inclusion), NOT endpoint sets -- scout-first has
  equal endpoints but dominating intermediate information. Sound
  because fog-lifting is one-way for the observer.
- statuses, ONE DIMENSION PER STATUS: poisoned (own fewer-eq /
  enemy more-eq), slowed (ditto), etc.
- ability-interaction potentials, one per ability: units-under-
  leadership set, units-adjacent-to-healer set, illuminated set,
  curable-adjacency set. These are POSITIONAL POTENTIALS, not
  state deltas -- define each as end-of-pair set inclusion per
  side; keep fine so skeptical views can drop them.
- pure unit position (user 2026-07-22): discrete order -- '=' iff
  every surviving unit stands on the same hex in both orderings,
  else incomparable. Almost always gates the product order (by
  design: position value is the net's job, not the detector's) and
  sits at the bottom of most views, but the registry is exhaustive
  with it: a product-dominant flag with position '=' is the
  strongest possible claim, and position-aware views become
  possible later.

## Orders = configurable views over the vector

- product order over ALL dims = theorem-grade ("Tier-1"): sound
  against any consistent valuation.
- lexicographic views (partial-order lex: strict at first non-equal
  dim, incomparable if that dim incomparable) are labeled
  HYPOTHESES: shipped set = L1 existence>HP>XP>vis, L2
  existence>XP>HP>vis (user), each flag carries its certifying
  order; per-order fire-rate columns adjudicate trust.
- coarsening/merge semantics: a view may DROP or merge fine dims;
  a dropped dim reads as indifferent ('='). Finer registry never
  hurts: merges are views, the vector stays maximal-granularity.

## Plugin registry (candidate generators -> shared verifier)

v1: backstab_setup (pair), attacks_before_commit (normal form per
target: attacks as early as legality allows; kill-branches bank the
uncommitted surround moves' MP), village_first, scout_first,
strong_attacker_first (XP-reallocating; product-incomparable, lex
views only). Contract: verifier certifies dominance of the
CANDIDATE reordering; no optimality claims -- "some strict
improvement exists" is the signal.

## Validation gates + run plan

1. Replay-walk loader adapter over our own exports (reuse
   replay_dataset command walk; corpus-style loader on the embedded
   scenario snapshot). GATE: re-derive a game's final state, match
   its games.jsonl row, before trusting detector output.
2. Verifier unit-test theorem: thief backstab is a strict improvement
   whenever the setup is legal. Backstab doubles per-hit damage, so
   the defender-HP distribution stochastically drops, `P(kill)` weakly
   rises, and a kill removes retaliation so own-HP weakly rises:
   monotone stochastic dominance, no death-branch mixing needed. The
   test asserts the per-dimension dominance directly from the two
   `OutcomeDistribution`s, not from sampled rolls.
3. Run on the latest ~100 games (pull via
   `run_validation_batch.py --hf-pull`, repo `momom2/wesnoth-tier-a`).
   Report: per-motif and per-order fire rates, guaranteed-gain
   vectors, and the `inconclusive` census (what v1 skips: overlapping
   bounds, death-mixture, or `enumerate_attack_outcomes` -> None).

## Deferred (v1 marks `inconclusive`)

- **Death-branch mixture verdict.** First-order stochastic dominance
  across die/survive branches with differing `P(death)` between
  orderings, where interval bounds overlap. Needed by the non-monotone
  generators (`strong_attacker_first`, XP reallocation); NOT by
  backstab. Until designed, any comparison that reaches this case is
  `inconclusive`.
- **Non-exact combat** (`enumerate_attack_outcomes -> None`: petrify,
  possible advancement, berserk/complexity blow-up). Never sampled;
  always `inconclusive`.
- EV-mode and any reward/label/training coupling (as in the header).
