# XOD-P: combat overrides certified by distributional dominance (design, 2026-09-24)

**PARKED 2026-09-24 by user ruling, on branch `exp/xod-dominance`, not
merged.** The rewrites the checker certifies are worth a few Elo per
game at most, below what an 800-game match resolves. Section 10 holds
the measurement, the reopening condition and the catalogue of further
relaxations collected before parking.

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

Strict dominance is rare, so the design adds four relaxations
(section 4), each dropping or loosening one dimension: almost-dominant
rewrites, which most reasonable valuations prefer and which the games
then judge. Certifying a rewrite needs no network pass, so every
combination of relaxations is read from one census pass (section 7).

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

## 3. The strict certificate (R0, every axis off)

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
| seen by this side, per hex | more | (not compared) |
| visible to this side, per enemy unit | (not compared) | more |
| visible to the enemy, per own unit | less | (not compared) |

Premise, written down: the game's value is monotone in each dimension
and additive across them, so marginal dominance implies a better
expected value. Position, visibility and villages can block a rewrite
but never justify one: a candidate must be better on a unit's
existence, hp, statuses, level-up, experience or attacks left. Joint (multivariate) dominance would cover every
monotone valuation and fire less; see section 9.

**Visibility** is read at the end of the window with the simulator's
model (`wesnoth_ai.visibility`: `visible_hexes_for`, `units_visible_to`,
including hiders the window uncovers), which follows the engine since
0.4.6: a side sees the fog it has cleared during its turn, a mover
clearing along its path (docs/wesnoth_rules.md "Vision and fog"). With
positions equal, it moves only when a unit dies or a hider is
uncovered, so it rarely decides a strict comparison; it decides the
relaxed ones where positions change (R3, and K's mover on the kill
branch).

**Tier O, option dominance.** A unit that keeps its movement and can
still reach the hex the baseline moved it to dominates the unit that
made the move (`swap_detector.pos_mp_dominates`), because the
continuation can reproduce the baseline. This holds for the game's
value; the realized gain depends on the policy using the freed unit
well, which is why it is a separate tier.

## 4. Relaxed certificates: four axes

None of these is a theorem. Each admits rewrites that most reasonable
valuations prefer, names the cases where it can be wrong, and is
checked twice: by playouts on a sample (section 7, the rider) and by a
match (section 6). The axes combine freely: R1 on or off, R2 off or at
one of two values of ε, R3 off, guarded or literal, R4 on or off, 36
combinations in all, R0 being the one with every axis off.

A combination treats a dimension it drops as a valuation indifferent to
it would: the dimension neither blocks a rewrite nor counts as its gain.
So a looser combination can admit fewer rewrites than a tighter one,
when a rewrite's only gain lies on what it drops: a kill moved from one
unit to another raises one unit's experience and lowers the other's, and
under R1 it is no gain at all.

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
wall with a gap opened by the move, a castle or keep hex, a time area.
What the unit sees from its new hex is the visibility dimension's
business, so R3 alone still requires the new position to see at least
as much and show no more; R4 drops that. R3 opens the attack-hex choice
(H) and lets K bank a mover whose alternative hex is near.

**R4, visibility ignored.** The three visibility rows are dropped. It
can be wrong when the information decides the next moves: a scout whose
vision would have spotted an ambusher, a unit left in view of enemies
that would otherwise not know where it is. Alone it changes little,
since with positions equal visibility rarely differs; combined with R3
it is what lets most position changes through.

## 5. Classes

Each class names the rewrite and the relaxations it needs.

- **W, weapon** (R0; widened by R2). At an attack, the same attacker,
  target and hex with another weapon whose fight distribution dominates
  on the attack's own dimensions. A single decision; no plan needed.
- **A, ability setup** (R0). A planned move that puts a flanker on the
  backstab hex, a leadership unit next to a lower-level attacker, or an
  illuminator next to the fight, played before the attack instead of
  after. The move is applied for real in the pre-attack state
  (legality); the mover's end hex and movement left must equal the
  baseline's. Only for an attack that cannot kill: on the kill branch
  the baseline could still drop the move, which the rewrite commits.
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
  admitted by R3. Every candidate hex and the played one are next to
  the target, so R3's two-hex bound always holds for H and only the
  guard decides.

Every class also requires the rest of the turn to play as it did: no
later command lands on or recruits onto the hex a rewrite occupies, and
every attack between the rewrite's commands, or after them for H, next
to a hex a unit left or took, keeps its exact fight distribution with
the unit moved (backstab flanks, leadership, illumination).

Out of scope, and recorded in section 9: which target, anything
spanning turns, and XOD's valued swaps.

## 6. The player (step 2)

A decode option of `tools/raw_player.RawPolicyPlayer`, on top of the
reference decode: procedure tag `raw:t0+eo-1.5+dom`, with the
combination as a suffix naming the axes turned on (`+dom` for R0,
`+dom-r1-r3g-r4`, ...).

1. **Plan.** At a decision whose argmax is an attack or a move, the
   player forks the state and plays its own argmax to the end of the
   turn, resolving each attack by its most probable outcome in which
   the target survives. The result is the longest turn the policy
   intends.
2. **Window.** The generators search the plan for rewrites involving
   the current argmax action.
3. **Certificate.** Baseline and candidate windows are compared on
   their exact joint distributions under the arm's combination. No
   certificate, no override.
4. **Commit.** A certified window is executed in the candidate order.
   Actions whose target has died are skipped; then control returns to
   the policy, which re-plans. Nothing outside the window is touched.
5. **Instrument.** Every override is logged with its class, the
   relaxations it needed and its certified gain; the override rate per
   decision is reported with each match.

Each combination is one arm with no free parameter (ε is fixed per
level), so each is gated by one match: 800 decisive games against the
reference, PURE, sides alternated, ladder maps. The census chooses at
most three arms for the first round (section 7). A win is confirmed against a
second opponent before it is quoted (the panel's rule c,
docs/training_signal_panel_20260905.md 7), because a deterministic
reference can be exploited as well as outplayed. Distillation, the D of
XOD, waits for a win (plan rule 2).

## 7. Step 1: the census (pre-registered 2026-09-24, before any code or box)

**Question.** How often does the reference, left alone, play a turn
that a class improves under each combination of relaxations, by how
much, and how often are the relaxed improvements wrong? If rarely, and
often wrong, the override cannot move a match and step 2 is not built.

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

**One pass, every combination.** Every generator runs with every axis
at its loosest, and each candidate rewrite stores its full comparison
vector: the symbol of every dimension, the ε of each hp dimension, the
level-up flags, each unit's position distance and guard results, the
visibility symbols. A combination admits a rewrite when every dimension
it keeps passes and one it keeps that can justify the rewrite is better
(section 4), so all 36 are read from the stored vectors without
recomputing anything; each rewrite also records its *minimal sets*, the
smallest combinations that admit it. An opportunity is an attack with
at least one admitted rewrite of a class, however many it has; rates
count opportunities, and failed games stay out of every denominator.

**Estimands, per source and combination.**

- The opportunity rate per class, over all decisions (the player's
  denominator) and over attack decisions, and the rate beyond R0.
- The gain per opportunity: the change in P(target killed), in the
  target's expected hp and in the attacker's expected hp; for K the
  expected movement banked, P(kill) times the movement kept; for F the
  level-up probability gained.
- The inconclusive count (a fight the enumerator cannot resolve, a
  window past 512 particles), so a low rate is not silence.

**The rider: how often a relaxed rewrite is wrong.** From the
reference's side-turns, 150 rewrites that R0 does not admit, drawn at
random, stratified by minimal set so that each axis appears in at least
40 of them. Each is graded by playouts from the side-turn's start: the
realized turn and the rewritten turn, 40 playouts each at
`raw:t0.5+eo-1.5`, the turn-gap tool's procedure
(docs/turn_gap_ref_prereg_20260921.md). A graded rewrite carries its
vector, so the one sample estimates every combination's share of
rewrites graded worse than the original by 2 standard errors, over the
graded rewrites that combination admits.

**Rule.**

- **Stop** if the reference's R0 rate (tiers D and O) is below 1% of
  decisions, its tier-D rate below 0.5%, and no combination passes the
  next rule. The 0.5% bar is the one docs/selfplay_redesign_20260904.md
  2.7 set for XOD: below it an override cannot move an 800-game match.
- **A combination is eligible** if its rate beyond R0 is at least 0.5%
  of decisions, the rider graded at least 20 of its rewrites, and at
  most 25% of those were graded worse.
- **The first round** matches at most three arms: R0 if it clears its
  bars, then the eligible combinations with the highest rate beyond R0;
  between two combinations that differ by one axis, the one with fewer
  relaxations is taken unless the extra axis adds at least half again
  to the rate. Picking three of 36 on the rider's numbers overfits the
  rider, so the matches, not the rider, are the evidence.
- **The class list** of any arm is the classes above 0.1% of decisions
  under that combination.
- **Crash barrier, not a verdict:** an inconclusive share above 10% of
  attack decisions sends the census back with a larger particle cap
  before any reading.

**Predictions** (reference; the humans are expected at about half,
because the reference imitates them one action at a time and cannot
reorder what it has not yet decided). Rates are beyond R0, for each
axis alone and for the combinations expected to matter:

| combination | classes it adds | rate beyond R0, share of decisions | rider rewrites graded worse |
|---|---|---|---|
| R0, tier D (W, A, Q without a kill in reach) | | 0.3-1.2% (itself) | (not graded) |
| R0, tier O (K) | | 0.5-2% (itself) | (not graded) |
| R1 | Q with a kill in reach, F | 0.2-1% | under 10% |
| R2 at ε = 0.05 | W | 0.2-0.8% | under 10% |
| R2 at ε = 0.15 | W | 0.5-1.6% | 10-20% |
| R3 guarded | H where the new hex shows no less | 0.3-1.5% | 10-25% |
| R4 | little alone | under 0.3% | under 15% |
| R3 guarded + R4 | H, K with a near mover | 1-4% | 10-25% |
| R3 literal + R4 | as above, unguarded | 1.5-6% | 20-40% |
| R1 + R2 at 0.15 + R3 guarded + R4 | all of the above | 2-7% | 15-30% |

So R0 alone passes with probability about 0.65; R3 guarded with R4 is
the combination most likely to be matched, and R3 literal is expected to
fail the rider.

**Cost.** One 4090 box with about 32 cores, like 2026-09-23's ($0.56/h):
about 10 minutes of bring-up, 20 to 30 for the 200 self-play games, 10
to 20 for the census of both sources, about 60 for the rider (12,000
playouts at the 2026-09-23 rate of 3.5 a second). About $1, ceiling
$2.20; the script cuts each stage at twice its estimate and the tools
write their records as they go.

## 8. Code plan

All in `tools/`, tests in `tests/`; no simulator change.

1. `tools/swap_detector.py` stays as it is. What it lacked (attacks
   left per own unit, village ownership, the setup move applied through
   `_apply_command` with its legality checked, side-turns read from
   game records and corpus games) is in the new modules of item 2. K's
   widening by R3 (section 5) is not built: the census counts K under
   R0 and R4.
2. `tools/combat_dominance.py` and `tools/dominance_rewrites.py`, new:
   the full comparison vector (the dimensions of section 3 including
   visibility, the XP-to-level-up map, the ε of each hp dimension, each
   unit's position distance and guard results), the combination filter
   over it, and the six classes as functions over a realized side-turn,
   returning rewrites with class, vector, minimal sets and gain. A
   dropped dimension neither blocks nor justifies (section 4). Q and F
   compose
   the two fights from the exact fight tables (the first fight's, then
   the second's from each state the first leaves), checked against the
   simulator's own enumeration; walking the simulator through every
   strike pattern of both took over a minute on some pairs.
3. `tools/analysis/dominance_count.py`, new: reads corpus games and
   game records (tools/game_record.py; the reference's games come from
   an eval match, which records every game), runs the classes, counts
   every combination, keeps every fight distribution it computed per
   game, writes the record. The rider's sampling is still to add.
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
- visibility: a rewrite that leaves a unit where it sees less is
  rejected without R4 and admitted with it;
- the combination filter: a rewrite's minimal sets are exactly the
  smallest combinations under which the filter admits it;
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
- rejected: grading each combination by its own playout sample,
  because one sample of rewrites, each carrying its full vector,
  estimates every combination at once.
- rejected: choosing the target, because it changes which fight is
  fought, which is the multi-turn judgement the override leaves to the
  policy.
- rejected: XOD's valued swaps (expected material swing with margin `m`)
  in the same programme, because one factor at a time: they are a
  different criterion with a free parameter and get their own sweep if
  wanted.
- rejected: running the census on the laptop, because it generates
  self-play games (user ruling: none on the laptop).

## 10. Parked (2026-09-24): what was measured, and when to come back

### 10.1 Measured on human games

**300 games** (the first 300 of the training split; the driver of
10.4 reproduces the tool on the first 20 exactly): 8,024 side-turns,
85,776 decisions, 17,914 attacks, no failed game. Attacks with an
admitted rewrite: 84 at R0 (0.98 per 1,000 decisions); 487 under any
of the 36 combinations (5.68 per 1,000), the loosest alone giving the
same set; 680 (7.93 per 1,000) with the class Q2 of 10.4. The
pre-registered bars (section 7) are 1% of decisions for R0 and 0.5%
beyond R0 for a relaxed combination; humans sit at 0.1% and 0.57%.
These counts use status dimensions with a dead unit read as the extreme
value (10.3), which changes 4 admissions in 300 games.

**20 games**, the first read, with the gains per opportunity. The first 20 games of the imitation corpus's training split
(`tools/analysis/dominance_count.py --games 20`, 15 s on three cores):
611 side-turns, 7,310 decisions, 1,411 attacks. An opportunity is an
attack with at least one admitted rewrite of the class.

| combination | W | H | A | K | Q | F | any class | of decisions |
|---|---|---|---|---|---|---|---|---|
| R0 | 1 | 0 | 0 | 2 | 0 | 0 | 3 | 0.04% |
| r2-0.15 | 5 | 0 | 0 | 2 | 0 | 0 | 7 | 0.10% |
| r3g+r4 | 1 | 11 | 0 | 2 | 0 | 0 | 14 | 0.19% |
| r1+r2-0.15+r3l+r4 (loosest reported) | 5 | 17 | 0 | 2 | 3 | 1 | 28 | 0.38% |

Set aside on the way: 1,336 attack-hex candidates whose fight was not
better, 16 attack-hex candidates that a later command needed, 457
setup candidates on attacks that could kill, 2 fights the enumerator
could not resolve.

Gain per opportunity at the loosest combination (the best rewrite of
each attack):

| class | opportunities | mean gain | largest |
|---|---|---|---|
| H | 17 | attacker +1.35 hp; kill chance unchanged | +3.6 hp |
| W | 5 | kill chance +0.155, attacker +1.95 hp, target -1.7 hp | kill chance +0.296 |
| Q | 3 | no change in kill or level-up chance | |
| F | 1 | level-up chance +0.01 | |
| K | 2 | 3.3 movement points banked (tier O) | 4.2 |

Per game that is about 0.04 kills and 1.6 hp (1.4 opportunities per
game; the 300-game sample reads 1.6, or 2.3 with Q2). Near an even score one
point of win probability is about 7 Elo; if a kill were worth 10 to 20
points of win probability, the rewrites of a game are worth 3 to 6 Elo,
against a standard error of about 12 Elo on an 800-game match. The
value of a kill is an assumption, not a measurement; the rider of
section 7 measures a rewrite's worth directly by playouts.

### 10.2 When to come back

- The reference's own count. Every eval match records its games whole
  (`tools/game_record.py`, `tools/elo_eval_game.py`), so running
  `dominance_count.py --records` over the records of any match that runs
  anyway gives the reference's rates at no extra rental. Reopen if they
  are about ten times the human rates with gains of W's size (kill
  chance), not H's (a hp or two).
- A relaxation from 10.4 that multiplies the opportunities about tenfold
  with material gains.
- The input-side variant (10.5).

### 10.3 Known defects, not fixed

- A dead unit's outcome reads slowed, poisoned and petrified as false
  (`combat_outcomes._canonical` for `fight_dims`, `_after_fight` for
  `_joint_dims`). Enemy statuses count "more is better", so a candidate
  that kills more can read worse on a status the played attack also
  gives; the own-unit mirror reads a unit saved and left poisoned as
  worse. The right order: for an enemy, dead above alive with the status
  above alive without; for an own unit the reverse. Measured on 300
  games: the symbols change on 7,316 candidate vectors, admissions on 4
  (0.05 per 1,000 decisions).
- R2 lets an almost-dominant hp dimension pass but never counts it as the
  gain, so a rewrite whose only improvement is almost dominant is refused
  at every level (10 in 300 games); section 4 reads as if it should
  count.
- H's enemy-threat guard reads enemy reach on the played board, where the
  mover still stands on the played hex, blocking and zoning; it causes
  99 of the 123 guard failures.
- Q sees only two attacks in a row on one target: 1,607 of the 6,981
  same-target pairs in 300 games. 4,023 more have one move between them,
  usually the second attacker walking in (Q2, 10.4).
- A kill also shows as a status gain under the corrected reading:
  harmless for admission, a double count if gains are ever summed.

### 10.4 Further relaxations, collected before parking

From three independent agents: Wesnoth tactics, the dominance
literature, and measurement over corpus games. V marks a mechanic
checked in the 1.18.4 source or the local WML, J a judgement. None is
built. The agents' full reports are not kept; their essentials follow.

**From Wesnoth's mechanics.**
1. *Project enemy units to their next turn start* (V). Compare an enemy's
   hp and poison as they stand after its side's next init_side (the
   largest of village, regeneration and adjacent healer, plus 2 rest
   heal only for a `healthy` unit, since an attack clears resting,
   `attack.cpp:1374-1375`; poison 8, `game_config.cpp:43`), when no
   later command of the turn attacks it. Exact for the next moment hp
   has an effect; removes false gains such as 2 hp off a regenerating
   troll. Occasional to common (J).
2. *Own slow caused in the own turn is dropped* (V). Slow ends at the
   slowed unit's side's end of turn (`unit.cpp:1284`), and a unit that
   attacked cannot act again in the default era (`attack.cpp:1372`), so
   the slow's only effect was inside the fight, already in the hp
   distribution. Enemy slow stays: it lasts through the enemy's turn.
   Rare to occasional (J).
3. *Experience bands* (V for the mechanics). Residual experience compared
   through how far the unit is from levelling: levelled, one fight
   away, one kill away, two kills, far (kill experience 8 x level, 4 at
   level 0; fight experience = level, `game_config.cpp:40-41`; an
   advancing unit heals fully and is cured, `advancement.cpp:320-324`).
   Between R0 and R1; covers "no realistic hope of levelling". Common
   (J).
4. *Leader rules* (V: a side without a leader is defeated,
   `team.cpp:146,189`). (a) Never relax either leader's hp under R2-R4.
   (b) A rewrite that raises P(enemy leader dead) by at least a threshold
   and raises no own death is admitted whatever else it costs.
5. *Experience as progress weighted by the level-up's value*: the
   intelligent-unit case (V: intelligent is -20% max experience,
   `traits.cfg:186-200`; half the races draw it from four traits). Own
   experience pooled as the sum of weight x experience / max experience
   once kills, deaths, hp and statuses pass.
6. *Plague credit* (V: `attack.cpp:155-157`, `1287-1295`): P(an own unit
   is spawned) as a dimension, so an order that gives the kill to a
   plaguer counts. Rare.
7. *A kill that opens a path* to the enemy leader or a village for an own
   unit with moves left (ZoC, V; level-0 units emitting none, not
   verified). Occasional, often wrong (J).
8. Weaker: own units out of enemy reach projected to their next turn
   start; own hp in next-turn threat bands; cost-weighted merge of own
   units; on the last side-turn before the cap only the leader kill
   counts (a project convention, not a game rule).

**Measured on 300 human games** (per 1,000 decisions; "extra" is admitted
by the rule on top of some combination and by none of the 36 without
it). Scripts and outputs as run: `training/metrics/xod_20260924/`; the
per-candidate rows (8.8 MB, derived from corpus games) were not
committed and are regenerated by `census.py` in about 5 minutes on three
cores.

- Q2, a class added by the measurement: played "attack a1, move a2,
  attack a2", rewritten "move a2, attack a2, attack a1", the move
  replayed identically and a1's fight unchanged, with an option
  dimension for a2's move being committed on a1's kill branch. With it
  the admitted attacks go from 487 to 680.
- R0 refuses 7,909 candidates that are better somewhere. The largest
  families are trades, not near-dominance: the attacker safer against
  target damage or kill chance (17.6), own hp moved between the two
  attackers of Q (16.0), Q2's committed move (11.4), only own experience
  moved (9.4, of which R1 admits some), more target damage for more own
  damage (8.2). Among candidates blocked only by an incomparable hp
  dimension, 218 of 282 have epsilon above 0.5: mostly worse, not nearly
  better.

| rule | extra attacks (/1k) | gains of what it admits, mean |
|---|---|---|
| kill chance first: kill chance better, no own death chance worse, all else ignored | 453 (5.28) | kill +0.18, own hp -5.0, target hp -2.0 |
| same, gated: the kill opens something (another enemy next to the target, or the target on a village) | 272 (3.17) | kill +0.17, own hp -4.9 |
| same, gated: no follow-up attacker left | 174 (2.03) | kill +0.17, own hp -5.3 |
| same, gated: the target heals 8 or more next turn | 131 (1.53) | kill +0.18 |
| Q/Q2: drop the committed-move dimension, pool hp, attacks left and experience of the two attackers | 497 (5.79) | own hp +2.1 |
| Q/Q2: keep only the experience of the unit nearer its threshold, pool own hp | 445 (5.19) | own hp +0.5 |
| Q/Q2: pool the two attackers' hp | 303 (3.53) | own hp +1.6, level-up +0.025 |
| Q/Q2: exactly one attacker intelligent, keep only its experience, pool hp | 245 (2.86) | own hp +0.7 |
| Q/Q2: exactly one attacker intelligent, keep only its experience | 120 (1.40) | experience to it +1.4; nothing else |
| experience hopeless: drop a unit's experience when a kill now leaves it more than 16 short (8: 52, 0.61) | 76 (0.89) | experience moved only |
| an incomparable hp dimension decided by its mean | 74 (0.86) | kill +0.05, own hp +2.0 |
| R2 counted as a gain | 10 (0.12) | |
| enemy leader kill first | 6 to 11 (0.07 to 0.13) | kill +0.18 to +0.20, own hp -3.7 to -7.9 |
| target hp after its next heal | 1 to 6 | |
| reference only: XOD's valued swap (material swing above 0), which section 9 rejects | 2,433 (28.4) | median 0.68 gold |

The "experience hopeless" rule also tightens R1: of the 265 rewrites
only R1 admits, it refuses 79 (8) or 188 (16), which take experience
from a unit within reach of its threshold. On the user's examples: 711
attacks on the enemy leader have a candidate, 12 raise the leader-kill
chance and none of the 36 admits one (two go from 0.35 to 0.64 while
sparing our attacker, blocked only by an incomparable target hp); in
642 Q or Q2 rewrites experience moves toward the one intelligent
attacker at an equal kill chance, and 43 are admitted today.

**From the dominance literature.**
1. *Follow-up kill utilities* (target-based utility; Castagnoli and
   LiCalzi 1996, abstract read). A target's hp is valued only through
   the chance that each plausible follow-up this turn kills it: the
   rewrite must be at least as good for every follow-up plan (own units
   with an attack left that can reach the target), computed with
   `fight()` on the window's end states. Sound because no mainline
   special depends on the attacker's hp (grepped), so hp is worth its
   survival. On the section-4 example (6-4 against 8-2, 20 hp) it admits
   the swap for every follow-up tried (spearman 0.569 against 0.462,
   grunt 0.442 against 0.227, archer, fencer) where R2 admits it without
   knowing why; it should replace R2 for enemy hp. When the target
   surely survives the turn it falls back to first-order dominance.
2. *Terminal outcomes as extremes with a priority factor kappa*
   (derivation by the agent; lexicographic limit in Fishburn 1974,
   abstract only). The enemy leader's death counts as the best value of
   every dimension, the own leader's as the worst; admit when each
   dimension's worst-case shortfall on the non-terminal mass is at most
   (1 + kappa) x the gain in P(win); record kappa* per rewrite as epsilon
   is recorded. Example: a 33-hp fighter's sword 5-4 against a 10-hp
   leader wins 0.821 where the bow never does; R0 refuses it on own hp,
   kappa* = 0.164.
3. *Experience still needed, valued by a decreasing convex function*
   (increasing convex order; Tsetlin and Winkler 2018, abstract read):
   for every t, the pooled sum over own units of w x E[(t - r)+] must not
   fall, r the experience still needed. Admits the intelligent-unit kill
   transfer that R0 finds incomparable and R1 sees as no change; with a
   horizon cut it covers "no realistic hope".
4. *A band of valuations around a material scale* (multivariate almost
   dominance by optimal transport, Mueller and Wiesel 2026, arXiv
   2607.28215; Light 2026, arXiv 2607.29560; read through a summarizer):
   every valuation whose marginal values lie within [gamma, 1] of a
   nominal gold scale (the default AI's weights, `src/ai/default/
   attack.cpp`) must prefer the rewrite; gamma* recorded per rewrite.
   Spans R0 (gamma -> 0) to a valued swap (gamma = 1): a bridge to XOD's
   valued swaps, not a first arm.
5. *Certification over a set of valuations* (maximality; Troffaes 2007,
   abstract read): each rule above as linear constraints on the
   valuation, one small linear program per rewrite; it also settles
   which admitted rewrite to play (best worst-case gain).
6. Weaker: joint dominance within one unit as a guard; an S-shaped own-hp
   utility, superseded by rule 1's own-hp half.

Rejected by the literature pass: global second-order or convex dominance
on hp or damage (the game is risk-neutral in win probability and hp
matters by steps), higher orders, almost second-order variants,
dependence orders between units, acceptability indices, minimax regret,
lexicographic semiorders, and deriving kappa or the bounds from the
value head (config, not weights; the head failed as a pre-grader).

### 10.5 The input-side variant (not built)

Instead of overriding the policy, give it the fight: for each legal
attack option, the chance to kill, expected damage dealt and taken, the
chance to lose the attacker and level-up chances, so the policy weighs
them itself. It touches every attack (about 35 per side per game), not
one rewrite per game. Two facts found on the way: the weapon head reads
only the actor (`action_sampler.py`, "P(weapon | actor)"), so a unit
uses one weapon distribution whatever its target, which limits any
per-target fight input unless the weapon head sees the target; at
temperature 0 the damage is smaller, since the weapon and target heads
read the same actor context. It is a model change: a fresh arm and an
800-game match, on the user's word.

