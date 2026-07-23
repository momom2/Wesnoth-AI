# Swap detector — locked design (2026-07-22, user-approved)

Offline detector for strictly-better reorderings of a side-turn's
actions, run over fully-recorded training replays. Observe-only in
v1: firing data calibrates future search-time ordering oracles; NO
reward/label channel (EV-mode and any training coupling deferred).

## Criterion: pathwise dominance under RNG coupling

Replay both orderings against the SAME per-strike RNG realizations;
branch only where realizations diverge structurally (= deaths:
early death truncates remaining strikes and downstream actions).
The swap is strictly better iff on EVERY branch its outcome weakly
dominates and on ≥1 positive-probability branch strictly. Pairs
whose orderings change strike counts / CTH so streams cannot align
strike-for-strike: classify `correlated-unscored`, count, skip.

## Classification (mechanical, sim-decided)

- sequential: swapped order illegal in a forked sim -> unswappable.
- independent: swap legal AND all action params identical both ways
  -> distributions exactly equal (only the RNG stream shuffles);
  NEVER flag realized-outcome differences (hindsight cherry-pick).
- correlated: swap legal, params differ -> verify dominance.

## Per-dimension comparison vector (the stored primitive)

Per coupled branch, one symbol {<, =, >, incomparable} per
dimension. Dimension registry (FINE-GRAINED; views may coarsen):

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
2. Verifier unit-test theorem: thief backstab -- strictly better
   iff the enemy survives one unbuffed hit (same CTH -> same rolls;
   damage doubles on hits; earlier death = fewer retaliations).
3. Run on the last ~100 games (HF bundles r001_i000007-12).
   Report: per-motif and per-order fire rates, guaranteed-gain
   vectors, correlated-unscored census (what v1 misses).
