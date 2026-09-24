# XOD-P: combat overrides certified by distributional dominance (design, 2026-09-24)

A teacher candidate for phase 2 (docs/plan_20260904.md 5): the
reference player, with its combat rewritten wherever the rewrite's
outcome distribution is at least as good as the original's on every
dimension, or, in relaxed variants, almost so. It revises XOD
(docs/selfplay_redesign_20260904.md 2), keeping its source of
information, the exact combat outcome enumerator, and replacing its
criterion.

## 1. Why

XOD as specified swaps the policy's attack for another when the expected
material swing of that one fight is better by a margin `m`. Its own
design names the weakness: it is "exact about the fight and blind about
everything after it". Combat in Wesnoth is fought over several turns,
so a trade that wins the fight can lose the position.

Within one side-turn, though, some rewrites cannot lose anything: the
same fights, in an order or with a weapon whose outcome distribution
leaves every unit at least as well off. Combat is random, so the claim
is about distributions, never about one outcome. A rewrite of that kind
needs no valuation of the position and no view of later turns, because
it never trades one thing for another. The multi-turn judgement (which
fights to take, when to withdraw) stays with the policy.

Strict dominance is rare, so the design adds a ladder of relaxed
certificates (section 4): almost-dominant rewrites, which most
reasonable valuations prefer and which the games then judge.

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

## 3. The strict certificate (level R0)

Two tiers, counted separately everywhere.

**Tier D, distributional dominance.** The candidate's distribution over
end states first-order stochastically dominates the baseline's on every
dimension, marginal by marginal:

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
monotone valuation and fire less; see section 9.

**Tier O, option dominance.** A unit that keeps its movement and can
still reach the hex the baseline moved it to dominates the unit that
made the move (`swap_detector.pos_mp_dominates`), because the
continuation can reproduce the baseline. This holds for the game's
value; the realized gain depends on the policy using the freed unit
well, which is why it is a separate tier.

## 4. Relaxed certificates (levels R1 to R3)

None of these is a theorem. Each admits rewrites that most reasonable
valuations prefer, names the cases where it can be wrong, and is
checked twice: by playouts on a sample (section 7, the rider) and by its
own match (section 6). Levels are cumulative only when a match says so;
until then each is measured alone on top of R0.

**R1, experience as level-ups.** XP is compared only through level-up
events: per unit, the probability of levelling (or reaching an AMLA)
this turn, more for own units and less for enemy units. Residual XP is
ignored. It can be wrong when a unit ends just short of its threshold,
where the residual decides next turn's level-up; a variant keeps a
second flag, "within one kill of levelling", compared the same way.
R1 opens two classes: attacker order with a kill in reach (Q), where
kill XP otherwise makes every reorder incomparable, and giving the kill
to the unit it levels (F), which heals it fully and upgrades it.

**R2, almost first-order dominance.** For the many-valued dimensions
(hp), the candidate may fall below the baseline on a small part of the
range: the area where its distribution function is on the wrong side is
at most a fraction ε of the total area between the two functions
(Leshno and Levy 2002, "Preferred by 'all' and preferred by 'most'
decision makers: almost stochastic dominance", Management Science 48,
1074-1085). The binary dimensions (existence, so kills and deaths;
statuses; level-up) stay strict: no relaxation may lower the chance of
a kill or raise the chance of a loss. Two levels are measured, ε = 0.05
and ε = 0.15.

Worked example, a 6-4 attack against an 8-2 one at a 30% chance to hit
(70% defense), same range, so the defender's retaliation is the same:

| hits | 6-4 | 8-2 |
|---|---|---|
| 0 | 0.2401 | 0.49 |
| 1 | 0.4116 (6) | 0.42 (8) |
| 2 | 0.2646 (12) | 0.09 (16) |
| 3 | 0.0756 (18) | |
| 4 | 0.0081 (24) | |

The mean damage is 7.2 against 4.8. Against a target at 15 or 16 hp,
8-2 kills more often (0.090 against 0.084), so the strict kill
dimension rejects the swap there. At 17 to 18 hp, 6-4 kills with 0.084
and 8-2 cannot; at 19 to 24, 0.008 against nothing. At 20 hp the
target's hp distribution under 6-4 is almost dominant with ε = 0.113:
the violation is that 8-2 leaves the target at 12 hp or less more often
(0.51 against 0.35), which matters when a follow-up attacker needs
exactly that. So this swap passes at ε = 0.15 and fails at ε = 0.05.

**R3, position within two hexes.** A unit's end position may differ from
the baseline's by at most two hexes. Two forms are measured:

- *Literal*: any position within two hexes counts as equal.
- *Guarded*: the new hex must also give the unit at least the same
  defense, must not leave a village it held or would have taken, must
  not take it out of a healer's or leadership unit's adjacency, and
  must not let more enemy units reach a hex next to it on their next
  turn (`pathfind_sim.unit_reach`).

Known exceptions, which the guard covers only in part: a zone-of-control
wall with a gap opened by the move, a castle or keep hex, what the unit
sees under fog, a time area. R3 opens the attack-hex choice (H) and lets
K bank a mover whose alternative hex is near.

## 5. Classes

Each class names the rewrite and the lowest level that can admit it.

- **W, weapon** (R0; widened by R2). At an attack, the same attacker,
  target and hex with another weapon whose fight distribution dominates
  on the attack's own dimensions. A single decision; no plan needed.
- **A, ability setup** (R0). A planned move that puts a flanker on the
  backstab hex, a leadership unit next to a lower-level attacker, or an
  illuminator next to the fight, played before the attack instead of
  after. The move is applied for real in the pre-attack state
  (legality); the mover's end hex and movement left must equal the
  baseline's.
- **K, attack before a non-enabling move** (R0, tier O; widened by R3).
  A killable attack followed in the plan by a move of another unit to a
  hex next to the same target, where the move does not change the
  attack's distribution. Played attack-first; on the kill branch the
  move is dropped and the unit returns to the policy.
- **Q, order among adjacent attackers** (R0 when no kill is in reach;
  R1 otherwise). Attackers already next to the target, reordered,
  certified on the exact joint of the pair. The patterns it is expected
  to find: a slowing attack first (it halves the retaliation every later
  attacker faces), and the attacker most exposed to retaliation last,
  so an earlier kill spares it.
- **F, give the kill to the unit it levels** (R1). Among the planned
  attacks on a target, the order that makes a kill land on a unit it
  levels.
- **H, attack hex** (R3). The same attacker, target and weapon from
  another hex next to the target, with a fight distribution that
  dominates (better terrain defense for the attacker) and a position
  admitted by R3.

Out of scope, and recorded in section 9: which target, anything
spanning turns, and XOD's valued swaps.

## 6. The player (step 2)

A decode option of `tools/raw_player.RawPolicyPlayer`, on top of the
reference decode: procedure tag `raw:t0+eo-1.5+dom`, with the level as a
suffix (`+dom0`, `+dom1`, ...).

1. **Plan.** At a decision whose argmax is an attack or a move, the
   player forks the state and plays its own argmax to the end of the
   turn, resolving each attack by its most probable outcome in which
   the target survives. The result is the longest turn the policy
   intends.
2. **Window.** The generators search the plan for rewrites involving
   the current argmax action.
3. **Certificate.** Baseline and candidate windows are compared on
   their exact joint distributions at the arm's level. No certificate,
   no override.
4. **Commit.** A certified window is executed in the candidate order.
   Actions whose target has died are skipped; then control returns to
   the policy, which re-plans. Nothing outside the window is touched.
5. **Instrument.** Every override is logged with its class, level and
   certified gain; the override rate per decision is reported with each
   match.

Each level is one arm with no free parameter (ε is fixed per level), so
each is gated by one match: 800 decisive games against the reference,
PURE, sides alternated, ladder maps. The strict arm runs first; each
relaxation then runs alone on top of it. A win is confirmed against a
second opponent before it is quoted (the panel's rule c,
docs/training_signal_panel_20260905.md 7), because a deterministic
reference can be exploited as well as outplayed. Distillation, the D of
XOD, waits for a win (plan rule 2).

## 7. Step 1: the census (pre-registered 2026-09-24, before any code or box)

**Question.** How often does the reference, left alone, play a turn
that a class improves at each level, by how much, and how often are the
relaxed improvements wrong? If rarely, and often wrong, the override
cannot move a match and step 2 is not built.

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
target-survives branch. The census therefore counts slightly fewer K
and Q opportunities than the player would see.

**Estimands, per source.**

- The opportunity rate per class and level, over all decisions (the
  player's denominator) and over attack decisions. Each rewrite is
  counted at the strictest level that admits it, and its ε is recorded,
  so every level's rate can be read from one pass.
- The gain per opportunity: the change in P(target killed), in the
  target's expected hp and in the attacker's expected hp; for K the
  expected movement banked, P(kill) times the movement kept; for F the
  level-up probability gained.
- The inconclusive count (a fight the enumerator cannot resolve, a
  window past 512 particles), so a low rate is not silence.

**The rider: how often a relaxed rewrite is wrong.** From the
reference's side-turns, 25 rewrites per relaxed level (R1, R2 at 0.05,
R2 at 0.15, R3 guarded, R3 literal) that no stricter level admits, 125
in all, drawn at random. Each is graded by playouts from the side-turn's
start: the realized turn and the rewritten turn, 40 playouts each at
`raw:t0.5+eo-1.5`, the turn-gap tool's procedure
(docs/turn_gap_ref_prereg_20260921.md). Reported per level: the share of
rewrites graded worse than the original by 2 standard errors, and the
mean gap.

**Rule.**

- **Stop** if the reference's R0 rate (tiers D and O) is below 1% of
  decisions, its tier-D rate below 0.5%, and no relaxed level passes the
  next rule. The 0.5% bar is the one docs/selfplay_redesign_20260904.md
  2.7 set for XOD: below it an override cannot move an 800-game match.
- **A relaxed level earns its own match** if its rate beyond the
  stricter levels is at least 0.5% of decisions and at most 25% of its
  rider rewrites are graded worse. Otherwise it is dropped.
- **The class list** of any arm is the classes above 0.1% of decisions
  at that level.
- **Crash barrier, not a verdict:** an inconclusive share above 10% of
  attack decisions sends the census back with a larger particle cap
  before any reading.

**Predictions** (reference; the humans are expected at about half,
because the reference imitates them one action at a time and cannot
reorder what it has not yet decided):

| level, classes | rate beyond the stricter levels, share of decisions | rider rewrites graded worse |
|---|---|---|
| R0, tier D (W, A, Q without a kill in reach) | 0.3-1.2% | (not graded) |
| R0, tier O (K) | 0.5-2% | (not graded) |
| R1 (Q with a kill in reach, F) | 0.2-1% | under 10% |
| R2, ε = 0.05 (W) | 0.2-0.8% | under 10% |
| R2, ε = 0.15 (W) | 0.3-0.8% more | 10-20% |
| R3 guarded (H, K with a near mover) | 1-4% | 10-25% |
| R3 literal (as guarded) | 0.5-2% more | 20-40% |

So R0 alone passes with probability about 0.65, and R3 guarded is the
relaxation most likely to earn a match; R3 literal is expected to fail
the rider.

**Cost.** One 4090 box with about 32 cores, like 2026-09-23's ($0.56/h):
about 10 minutes of bring-up, 20 to 30 for the 200 self-play games, 10
to 20 for the census of both sources, about 50 for the rider (10,000
playouts at the 2026-09-23 rate of 3.5 a second). About $0.90, ceiling
$2; the script cuts each stage at twice its estimate and the tools
write their records as they go.

## 8. Code plan

All in `tools/`, tests in `tests/`; no simulator change.

1. `tools/swap_detector.py`: the missing dimensions (moves and attacks
   left per own unit, village ownership); `_verify_reorder` applies the
   setup move through `_apply_command` and checks its legality instead
   of placing the unit; a side-turn source that reads an extracted
   record or a live simulator, besides exported bundles.
2. `tools/combat_dominance.py`, new: the comparison at each level (the
   XP-to-level-up map, almost dominance with its ε, the two position
   rules) and the six classes as functions over (state, plan or
   realized turn), returning rewrites with class, level, ε and gain; W,
   Q, F and H are new, A and K wrap the existing generators.
3. `tools/analysis/dominance_census.py`, new: plays the reference's
   self-play games (the eval worker's game loop, recording side-turns),
   reads the corpus games, runs the classes, samples the rider's
   rewrites, writes the record.
4. `tools/turn_gap.py`: a mode that grades given pairs of turns (the
   realized and the rewritten action list) from one start state.
5. `scripts/dominance_census_box.sh`: stage, games, census, rider,
   upload.
6. Step 2 only: the plan-and-commit decode in `tools/raw_player.py`, its
   `run_elo_batch` flag and procedure tags.

Tests, each able to fail:

- the thief-backstab and leadership theorems through the new legality
  path, and a case where the setup move is illegal before the attack
  and no certificate is issued;
- the worked example of section 4: ε = 0.113 at 20 hp, admitted at 0.15
  and not at 0.05, and rejected at 16 hp by the kill dimension;
- R1: a reorder that shifts kill XP between units without a level-up is
  comparable under R1 and not under R0; one that moves a level-up away
  from a unit is rejected;
- R3 guarded: an attack hex with better defense is admitted, and the
  same hex is rejected when the move leaves a village the unit held;
- a K case where the kill branch banks movement, and one where the
  "surround" move enables the attack (leadership), which must not fire
  as K;
- the census on a hand-built two-turn game with a known count.

## 9. Rejected, one line each

- rejected: joint (multivariate) dominance as the certificate, because
  it fires strictly less than the marginal form and the additive
  premise is the one every material-style valuation in the project
  already makes; revisit if an arm wins and a counterexample appears.
- rejected: the probability-mass form of almost dominance (dominant
  except on outcomes of total probability p) as the relaxation, because
  it counts a violation by its probability alone, where Leshno and
  Levy's area form also weighs how far the distributions cross.
- rejected: relaxing the binary dimensions (kills, deaths, statuses,
  level-ups), because a relaxed rewrite that can lower a kill chance or
  raise a loss chance is no longer an improvement to the fight it keeps.
- rejected: choosing the target, because it changes which fight is
  fought, which is the multi-turn judgement the override leaves to the
  policy.
- rejected: XOD's valued swaps (expected material swing with margin `m`)
  in the same programme, because one factor at a time: they are a
  different criterion with a free parameter and get their own sweep if
  wanted.
- rejected: running the census on the laptop, because it generates
  self-play games (user ruling: none on the laptop).
