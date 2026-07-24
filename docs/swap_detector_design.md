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

- **Kills the hindsight class structurally.** A genuinely `independent`
  reorder (identical params AND no resolved-fight interaction — see
  Classification) has the SAME distribution → `=` on every dimension
  automatically. There is no realized difference to cherry-pick, so the
  coupling design's "NEVER flag realized differences" caveat becomes a
  theorem, not a guard. (Same-target-killable pairs are NOT independent
  — kill-XP / MP differ by ordering — so they are classed `correlated`
  and dominance-verified, never force-`=`.)
- **Drops the strict-sync dependency.** No strike-for-strike stream
  alignment is required, so differing CTH / strike counts are fine and
  nothing is skipped for alignment reasons. v1 needs only the
  reconstructed pre-window state + the two action lists; the export's
  `[random_seed]` / `[checkup]` data is a gate-1 reconstruction check,
  not a criterion input.

**Bounds shortcut (the usual fast path).** A dimension's value is a
function of the hit-pattern; we certify `lo(candidate) ≥ hi(baseline)`
(more-is-better; symmetric otherwise) from per-dimension intervals. Two
levels:

- **Single stochastic action:** the interval is just `min`/`max` over
  that combat's exact DP support (`enumerate_attack_outcomes`).
  Ability-complete *for free* — the DP already models drain, slow,
  petrify, plague, feeding, berserk, … exactly — so NO per-ability
  reasoning is needed, and the endpoints are exact.
- **Multi-action window:** to avoid the joint we bound compositionally,
  which needs the WORST-/BEST-realizable hit-pattern per dimension —
  and the couplers flip which corner that is, so it is NOT the naive
  all-miss/all-hit. For own-HP, enemy hits lower it, but the unit's OWN
  hits RAISE it: via **drain** (self-heal), via **slow** (cuts the
  enemy's later damage), and via a **kill/petrify** (truncates
  retaliation). So the true min is at (own all-miss, enemy all-hit), and
  threshold events (death / petrify / kill-XP) split the branch (as
  "Death is a discontinuity" below). These live in a **per-ability bound
  table**, keyed `(dimension × ability) → (corner, threshold)`:

  > **Bound table (filled LAZILY).** Charted so far: base damage, drain,
  > slow, kill/petrify truncation. Any ability present in a
  > multi-action window with **no table entry for the dimension** →
  > **log a warning and proceed with the best bound we can** (fall back
  > to the exact joint where its support is tractable — always sound;
  > else a flagged best-effort corner). v1 is **observe-only** (no
  > reward/label channel), so a warning-flagged best-effort bound is
  > acceptable, and the warnings are the to-do list for growing the
  > table. Single-action support is always exact regardless.

Overlapping intervals are inconclusive by bounds alone (v1 does not
claim).

**Death and petrify are discontinuities.** When a unit's end-HP
interval straddles 0, its death restructures what follows (no
retaliation, hex freed, re-target changes); a surviving **petrify**
truncates the schedule the same way while the unit stays alive. So
bounds no longer suffice and the cases split die/survive (resp.
petrify/not, and by timing). v1 renders a verdict only where a sound
per-dimension dominance survives the split; the GENERAL rule for
first-order dominance across such branches with differing `P(event)`
is deferred (see "Deferred"). Backstab does not need it (see gate 2).

**Exactness gate on the outcome engine.** Distributions come from
`tools/combat_outcomes.enumerate_attack_outcomes`, which returns the
EXACT distribution or `None` — it never approximates. Current behaviour
(as of commits 8ee5b99 / 43bd6a7):

- **Petrify is modeled exactly** — a surviving petrifying hit stones the
  target and ends the fight (a second fight-ending discontinuity, see
  above); it no longer returns `None`.
- **Advancement is resolved into the distribution** when the detector
  passes `advancement_choice=` (`"uniform"` / a `{type: prob}` dict / a
  `callable` model head); the advanced type + full HP are keyed exactly.
  Without an `advancement_choice`, a fight that could level a unit
  returns `None` (the legacy bail).
- `None` remains for complexity blow-ups (berserk's refilled schedule
  past `MAX_SCHEDULE=512` / `MAX_DP_STATES=4096`).

Any attack that comes back `None` makes the comparison `inconclusive`;
the detector NEVER samples to fill the gap.

## Classification (mechanical, sim-decided)

- sequential: swapped order illegal in a forked sim -> unswappable.
- independent: swap legal AND the two actions do not INTERACT -- action
  params identical both ways AND no shared resolved-fight coupling.
  Then the distributions are provably equal -> `=` on every dimension.
  The interaction test keys on RESOLVED fight context, not the raw
  action dict: two attacks on the SAME defender that CAN die are NOT
  independent even with identical `(target, weapon)` -- whichever lands
  the lethal blow banks kill-XP (8x combat-XP) and keeps different
  surround-move MP, so the XP and MP dimensions differ by ordering.
  Such pairs fall to `correlated`.
- correlated: swap legal, but params differ OR the actions interact
  (shared killable target, or one's outcome feeds the other's
  XP / MP / position) -> verify distributional dominance (above).

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
- **Non-exact combat** (`enumerate_attack_outcomes -> None`:
  berserk/complexity blow-up, or an advancement fight when no
  `advancement_choice` is supplied). Never sampled; always
  `inconclusive`. (Petrify and choice-supplied advancement are now
  resolved exactly — see "Exactness gate".)
- EV-mode and any reward/label/training coupling (as in the header).
