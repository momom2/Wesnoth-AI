# XOD-P: provable combat overrides (design, 2026-09-24)

A teacher candidate for phase 2 (docs/plan_20260904.md 5): the
reference player, with its combat rewritten wherever a rewrite is
provably at least as good on every dimension of the resulting state.
It revises XOD (docs/selfplay_redesign_20260904.md 2), keeping its
source of information, the exact combat outcome enumerator, and
replacing its criterion.

## 1. Why

XOD as specified swaps the policy's attack for another when the expected
material swing of that one fight is better by a margin `m`. Its own
design names the weakness: it is "exact about the fight and blind about
everything after it". Combat in Wesnoth is fought over several turns,
so a trade that wins the fight can lose the position.

Within one side-turn, though, some rewrites cannot lose anything: the
same fights, in an order or with a weapon that leaves every unit at
least as well off. A rewrite of that kind needs no valuation of the
position and no view of later turns, because it never trades one thing
for another. The multi-turn judgement (which fights to take, when to
withdraw) stays with the policy. XOD-P overrides only those rewrites.

## 2. What exists

`tools/swap_detector.py` (July 2026; design in
docs/archive/swap_detector_design.md) is most of the engine:

- `reconstruct_side_turn_dist(pre_state, actions)` builds the exact
  joint distribution over end states of an ordered action list, driving
  the simulator's own applier through every hit/miss pattern
  (`enumerate_children_via_sim`), bit-faithful to
  `combat_outcomes.enumerate_attack_outcomes` by a parity test. It
  returns None past 512 particles or on an inconclusive fight.
- `compare_state_distributions` certifies a candidate order against a
  baseline by first-order stochastic dominance per dimension: per unit,
  existence, hp, poisoned, slowed and XP (the good direction mirrored
  for enemies); positions must have equal distributions; own gold.
- Generators: `backstab_setup` and `leadership_setup` (dominance
  certificates on the attack's own distribution), `attacks_before_commit`
  (reported as a heuristic: on the kill branch the mover ends elsewhere,
  so positions differ), `strong_attacker_first` (kill-XP reallocation,
  incomparable in the product order).

Four gaps separate it from an override:

1. It has only run offline, over exported games, observe-only. No fire
   rate was ever recorded.
2. `_verify_reorder` places the flanker or leader on its hex without
   checking that the move is legal before the attack; an override must
   apply the move.
3. The dimensions lack movement and attacks left per own unit, which is
   where "attack before committing the surround" pays, and village
   ownership.
4. It reads realized turns. The raw player decides one action at a
   time, so an override has to know the turn the policy intends.

## 3. The certificate

Two tiers, counted separately everywhere.

**Tier S, state dominance.** The candidate's end-state distribution
dominates the baseline's on every dimension, marginal by marginal:

| dimension | own units | enemy units |
|---|---|---|
| existence | more is better | less is better |
| hp (dead = -1) | more | less |
| poisoned, slowed | less | more |
| XP | more | less |
| moves left, attacks left | more | (not compared) |
| position | equal distributions | equal distributions |
| villages owned, gold | more | less |

Premise, written down: the game's value is monotone in each dimension
and additive across them, so marginal dominance implies a better
expected value. Joint (multivariate) dominance would cover every
monotone valuation and fire less; see section 8.

**Tier O, option dominance.** A unit that keeps its movement and can
still reach the hex the baseline moved it to dominates the unit that
made the move (`swap_detector.pos_mp_dominates`), because the
continuation can reproduce the baseline. This holds for the game's
value; the realized gain depends on the policy using the freed unit
well, which is why it is a separate tier.

## 4. Classes

Each class names the rewrite, the tier and what certifies it.

- **W, weapon.** At an attack, the same attacker, target and hex with
  another weapon whose fight distribution dominates on the attack's own
  dimensions (`swap_detector.ATTACK_DIMS` plus XP). Tier S. A single
  decision; no plan needed. Positions are identical by construction.
- **A, ability setup.** A planned move that puts a flanker on the
  backstab hex, a leadership unit next to a lower-level attacker, or an
  illuminator next to the fight, played before the attack instead of
  after. Tier S. The move is applied for real in the pre-attack state
  (legality), and the mover's end hex and movement left must equal the
  baseline's.
- **K, attack before a non-enabling move.** A killable attack followed
  in the plan by a move of another unit to a hex next to the same
  target, where the move does not change the attack's distribution.
  Played attack-first; on the kill branch the move is dropped and the
  unit returns to the policy. Tier O.
- **Q, order among adjacent attackers.** Attackers already next to the
  target, reordered, certified on the exact joint of the pair. Tier S.
  Expected to be rare: whichever unit lands a kill takes the kill XP,
  so most reorders with a kill in reach are incomparable on XP.

Out of scope, and recorded in section 8: which target, which hex to
attack from, who lands the kill, anything spanning turns, and XOD's
valued swaps.

## 5. The player (step 2)

A decode option of `tools/raw_player.RawPolicyPlayer`, on top of the
reference decode: procedure tag `raw:t0+eo-1.5+dom`.

1. **Plan.** At a decision whose argmax is an attack or a move, the
   player forks the state and plays its own argmax to the end of the
   turn, resolving each attack by its most probable outcome in which
   the target survives. The result is the longest turn the policy
   intends.
2. **Window.** The generators search the plan for rewrites involving
   the current argmax action (W, A, K, Q).
3. **Certificate.** Baseline and candidate windows are compared on
   their exact joint distributions (Tier S or O as the class says). No
   certificate, no override.
4. **Commit.** A certified window is executed in the candidate order.
   Actions whose target has died are skipped; then control returns to
   the policy, which re-plans from a state that dominates the one it
   would otherwise have reached. Nothing outside the window is touched.
5. **Instrument.** Every override is logged with its class, tier and
   certified gain; the override rate per decision is reported with each
   match.

There is no free parameter, unlike XOD's `m`, so the player is gated by
one match rather than a sweep: 800 decisive games against the
reference, PURE, sides alternated, ladder maps. A win is confirmed
against a second opponent before it is quoted (the panel's rule c,
docs/training_signal_panel_20260905.md 7), because a deterministic
reference can be exploited as well as outplayed. Distillation, the D of
XOD, waits for a win (plan rule 2).

## 6. Step 1: the census (pre-registered 2026-09-24, before any code or box)

**Question.** How often does the reference, left alone, play a turn
that one of the classes provably improves, and by how much? If rarely,
the override cannot move a match and step 2 is not built.

**Data.**

- Reference self-play: 200 games, `terrain` at `raw:t0+eo-1.5` on both
  sides, ladder maps, per-game combat salts, cap 60 turns. Every
  side-turn is recorded as its start state and its realized action
  list.
- Human baseline: 200 corpus games from the imitation manifest's
  training split (never the holdout), read through the same side-turn
  walk.

The census reads realized turns, as the swap detector does. That is the
override's opportunity set with one difference: a realized turn has
already branched on its outcomes, where the player plans on the
target-survives branch. The census therefore counts slightly fewer
K opportunities than the player would see.

**Estimands, per source.**

- The opportunity rate per class and tier: certified rewrites over all
  decisions (the player's denominator), and over attack decisions.
- The certified gain per opportunity: for W, A and Q the change in
  P(target killed), in the target's expected hp and in the attacker's
  expected hp; for K the expected movement banked, P(kill) times the
  movement kept.
- The inconclusive count (a fight the enumerator cannot resolve, a
  window past 512 particles), so a low rate is not silence.

**Rule.**

- **Stop** if the reference's Tier S rate is below 0.5% of decisions
  and its Tier S plus Tier O rate below 1%. The 0.5% bar is the one
  docs/selfplay_redesign_20260904.md 2.7 set for XOD: below it an
  override cannot move an 800-game match.
- **Build step 2** otherwise. The class list for the player is the
  classes above 0.1% of decisions.
- **Crash barrier, not a verdict:** an inconclusive share above 10% of
  attack decisions sends the census back with a larger particle cap
  before any reading.

**Predictions.**

| quantity | reference | humans |
|---|---|---|
| W, share of decisions | 0.2-1.0% | 0.1-0.5% |
| A, share of decisions | under 0.1% (thieves, leadership units and illuminators are few) | under 0.1% |
| K, share of decisions | 0.5-2% | 0.3-1.5% |
| Q, share of decisions | under 0.1% | under 0.1% |
| Tier S total | 0.3-1.2% | 0.2-0.7% |
| Tier S + O total | 0.8-3% | 0.5-2% |

So the stop rule fires with probability about 0.35, most likely on a
low W rate with K carrying the rest. The reference is expected above
the humans, because it imitates them one action at a time and cannot
reorder what it has not yet decided.

**Cost.** One 4090 box with about 32 cores, like 2026-09-23's ($0.56/h):
about 10 minutes of bring-up, 20 to 30 minutes for the 200 self-play
games, 10 to 20 for the census of both sources. About $0.50, ceiling
$1.20; the script cuts each stage at twice its estimate and the tool
writes its record as it goes.

## 7. Code plan

All in `tools/`, tests in `tests/`; no simulator change.

1. `tools/swap_detector.py`: the missing dimensions (moves and attacks
   left per own unit, village ownership); `_verify_reorder` applies the
   setup move through `_apply_command` and checks its legality instead
   of placing the unit; a side-turn source that reads an extracted
   record or a live simulator, besides exported bundles.
2. `tools/combat_dominance.py`, new: the four classes as functions over
   (state, plan or realized turn), returning certified rewrites with
   class, tier and gain; W and Q are new, A and K wrap the existing
   generators.
3. `tools/analysis/dominance_census.py`, new: plays the reference's
   self-play games (the eval worker's game loop, recording side-turns),
   reads the corpus games, runs the classes, writes the record.
4. `scripts/dominance_census_box.sh`: stage, games, census, upload.
5. Step 2 only: the plan-and-commit decode in `tools/raw_player.py`, its
   `run_elo_batch` flag and procedure tag.

Tests, each able to fail:

- the thief-backstab and leadership theorems through the new legality
  path, and a case where the setup move is illegal before the attack
  and no certificate is issued;
- a weapon case constructed so one weapon dominates (a ranged attack on
  a unit without a ranged weapon, equal damage) and one where the two
  are incomparable;
- a K case where the kill branch banks movement, and one where the
  "surround" move enables the attack (leadership), which must not fire
  as K;
- the census on a hand-built two-turn game with a known count.

## 8. Rejected, one line each

- rejected: joint (multivariate) dominance as the certificate, because
  it fires strictly less than the marginal form and the additive
  premise is the one every material-style valuation in the project
  already makes; revisit if step 2 wins and a counterexample appears.
- rejected: kill-XP allocation (`strong_attacker_first`) in the provable
  arm, because it moves XP from one own unit to another and no product
  order can rank that; a separate arm labelled as a heuristic, later.
- rejected: choosing the attack hex, because the attacker's end
  position differs and position is exactly what the policy judges and
  the certificate cannot.
- rejected: XOD's valued swaps (expected material swing with margin `m`)
  in the same match, because one factor at a time: they are a different
  criterion with a free parameter and get their own sweep if wanted.
- rejected: running the census on the laptop, because it generates
  self-play games (user ruling: none on the laptop).
