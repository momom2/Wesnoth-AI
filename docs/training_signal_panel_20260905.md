# Training-signal panel: synthesis (2026-09-05)

21 proposals in five families (own-trunk value net and the lookahead
player it unlocks; rollout-graded expert iteration; preference
regression on paired turns; turn-level policy gradient; decode rules
and imitation data), three judges. Scores are on 1-5 for upside, cost,
fit to the rulings, and whether the kill can fire.

Rules applied in the ranking, from CLAUDE.md, docs/plan_20260904.md
section 3, BACKLOG.md rulings of 2026-09-05 and the judges' shared
objections:

- A strength claim is a PURE match against `raw:t0` (SE 0.018 at 800
  decisive games, 12 Elo); effects under about +25 Elo are invisible
  at that size, so a first test whose modal outcome is inside +-15
  Elo cannot fire either bar.
- Rule 2 blocks distilling from a teacher that has not won a match;
  the rollout-graded searcher cannot be matched ($50-300 per gate), so
  every proposal whose product is "distill confirmed pairs" needs the
  R2 ruling before its distillation step; only its harvest step is
  fundable now.
- Train sparingly: a 4-8 box-hour training path per attempt is a
  cost, not a neutral line item.
- The sigma_s readout on the recorded candidates (about 48 paired at
  the 160-playout truth) has SE about 0.05-0.06; a test whose
  prediction is a 0.03-0.08 move in sigma_s cannot pass or kill.
- The 369 holdout games are the source of every bench state and
  turn-gap position; positions drawn from them are eval-only. A
  proposal that trains on them destroys the project's only human
  holdout.
- The tempo direction ("the argmax ends its turn early") is the one
  lever the data already shows (2 of 3 confirmed gaps). Six proposals
  test it with training paths of $1.5-4.6; two test it for under $1
  as a decode rule or a config scalar with an exact null. The cheap
  form runs first; the trained forms are funded only where the cheap
  form shows the effect is state-specific.

## 1. Proposals worth testing, ranked

### 1. end_turn decided at the actor level (#14), with the end_turn logit offset as the attribution arm

Mechanism: the joint argmax compares end_turn's actor mass against
unit actions' four-way products, so end_turn wins whenever the "act"
mass is spread over several units; the rule plays end_turn only when
its actor mass is at least the largest unit or recruit actor marginal,
else the joint argmax among non-end actions. Nothing is trained; the
player is `raw:t0` plus one decode rule.

First test: 40-game screens of the rule and of the config end_turn
logit offset at -0.75 and -1.5 (the mini-category gate in
`wesnoth_ai/action_sampler.py:686-704` generalized, about 4 lines),
recording decisions per side-turn and capped games; then 1,300 games
vs `raw:t0` for 800 decisive for the rule, and the same for the best
offset only if the rule wins (attribution: rule-specific or "act
more"). Cost about $1.4 (bring-up $0.17, screens $0.06, 2 x $0.58).
Kill: decisions per side-turn under 9.7 in the screen; p <= 0.50 at
800 decisive.

Strongest objection: the "under-acts by 15%" evidence compares human
winners' turns with argmax turns on different position distributions
(army size confounds it), and forcing action when the act mass is
spread can walk units into retaliation late in a turn. Both are
answered by the match, not by the screen. A win ships as a procedure;
distilling it is a separate gated step.

### 2. Value-ranked whole-turn lookahead player (#18), step A = the pre-registered pre-grader measurement

Mechanism: at each boundary the player builds K = 8 candidates (base,
3 continue edits, 4 temperature-0.5 samples with recorded command
lists), plays one argmax reply on each fork, reads the current value
head at the mover's next boundary, and commits a candidate only if it
beats the base by a margin m; m = infinity is bit-identical to
`raw:t0`. Step A (docs/turn_proposer_design_20260905.md section 6,
already pre-registered) measures whether the head ranks candidates
within a position at all.

First test: step A on the box rented for test 1 ($0.03; the rider
re-realizing 4, 18, 59 included); step B, only on a pass, a
400-decisive-game match of `raw:t0+la8m` vs `raw:t0` ($1.1 with its
own bring-up, $0.95 on a shared box). Kill A: every grader at sigma_s
>= 0.30 or a confirmed alternative ranked below its base. Kill B:
p <= 0.50 at 400, or firing rate under 2%.

Strongest objection: the recorded prior puts the head at sigma_s
0.30-0.35, so the modal result is a $0.03 kill; and a step-B win
against a deterministic reference whose reply is computed exactly is
partly exploitation, so any win is confirmed at 400 games against
`2291k:t0` and `raw:t0.5` (#21 rule c, $0.3) before it is quoted.

### 3. Rating-weighted imitation (#15)

Mechanism: a Bradley-Terry rating over the corpus's player ids
(present in every replay's `[side]` blocks; the RCA AI as one player,
shrinkage by games played) and a fine-tune from the seed on the
winner-side pairs of top-quartile rated regulars, same recipe as the
relevant-set control arm (0.5 epoch, `--init-from seed.pt --max-pairs
1260000`). The imitated player moves from the corpus median toward its
strongest quartile; the labels already exist.

First test: index on the box ($0.03), training 4.6 h ($1.52), 1,300
games vs `raw:t0` for 800 decisive ($0.58), 600 vs the control arm for
400 decisive ($0.27), bring-up $0.17: about $2.6. Kill: filtered
subset under 250k pairs (switch to the soft-weight arm before
renting); p <= 0.50 vs `raw:t0`; top-rated holdout CE worse than the
control's by > 0.05 nat at equal pairs (barrier). Waits for the
control arm's 800-game number, which is in flight.

Strongest objection: winners already select the stronger side of
each game, so the top-quartile shift is smaller than a band shift in
chess; ratings over a corpus dominated by one-off players are noisy
and confounded with map and faction; 2-3 passes over 15-25% of the
pairs from a fit that plateaued at 1.2M can overfit, producing a null
that says nothing about rating selection.

### 4. Recruit type sampled from its marginal (#16)

Mechanism: when the joint argmax is a recruit, draw the type from the
recruit-actor marginals and play the argmax hex for that type;
everything else stays argmax. Restores composition variety where
sampling carries no move-quality tax.

First test: a 40-game screen recording `raw:t0`'s recruit-type entropy
per faction ($0.02, on the test-1 box); kill #1 if it already sits
within 0.3 bits of the human 2.55-2.78. Otherwise 1,300 games for 800
decisive ($0.58). Kill #2: p <= 0.50.

Strongest objection: the human entropy is across players and
matchups, not within a state, so the entropy-gap barrier is
confounded, and a flat marginal is model uncertainty rather than the
correct mix: sampling into it recruits counters the mode avoided.
Expected effect 0 to +20, mostly inside the resolution.

### 5. Blunder arm harvest on on-policy boundaries (#5, first test only)

Mechanism: continue edits (play the best non-end action at the base's
end_turn, then argmax; forced 1-3) on boundaries from `raw:t0.5`
self-play of the current weights, graded by a staged screen (10
playouts, escalate on best gap >= 0.4 or pooled z >= 1, replayed on
run1.json: passes 14 of 60 and keeps every confirmed position),
confirmed by replayed command lists with fresh salts, re-realized
under a second salt. The product is confirmed on-policy pairs and the
on-policy hit rate of the continue class.

First test: 200 on-policy positions, about 17,000 playouts, $1.5, 4
box-hours. Kill: fewer than 3 confirmed of 200. Run only after test 1
reads: if the decode rule wins, this measures what it leaves (the
offset check inside the test is then redundant and dropped); if the
rule loses, this tells whether the tempo gaps are position-specific.
The distillation step is blocked by rule 2 pending R2 and is not
funded here.

Strongest objection: 15-25 confirmed examples per $4 round against a
2.8M-decision prior cannot move an 800-game match, so as a training
signal it needs three rounds ($10) before one verdict; its worth now
is as the pair supply that makes the value net's sigma_s measurable
(item 6).

### 6. Own-trunk boundary value net from human outcomes (#1 and #7 merged; #2, #3, #4 behind it)

Mechanism: the seed's encoder and trunk copied into a value-only
checkpoint, trained on side-turn boundary states of the 16,755
imitation train games with the game outcome as label (16-per-game
RawEncoded cache built once, #1's cost accounting; #7's per-epoch
reconstruction is a box-day without it), never sharing a gradient
with the policy. It is the plan's phase-2 value function and the
grader every lookahead player needs.

First test: cache $0.33, training 4 epochs $0.45, readout $0.03,
about $1.2. Prerequisite that does not exist yet: a held-out
candidate set large enough to resolve sigma_s at 0.03-0.05 (about
150-200 paired candidates at >= 40 playouts against the 48 on file);
the continue-edit run on the 60 positions ($0.6, design doc section
6's next measurement) plus item 5's confirmed pairs supply it. Kill:
sigma_s >= 0.30 at the better read point, or not below the current
head by 2 SE. Not funded until the eval set exists.

Strongest objection (all three judges): outcome labels on real
trajectories train level discrimination, which the head already has
(AUC 0.82), not the within-position ranking of imagined siblings
(turn-scale AUC 0.53); the proposal's own prediction (0.25-0.30)
fails its own pass bar (0.20). #2 (auxiliary short-horizon targets)
and #3 (sibling-contrast fine-tune) are follow-on factors on this
net, each predicting a move below the readout's resolution; #4 (the
one-reply lookahead player on the new net) is #18 with a better
grader and needs no separate design.

### 7. Two-net league confirmation rule (#21, rule c only)

Mechanism: every win of a lookahead-type player against `raw:t0` is
confirmed at 400 games against `2291k:t0` (equal to the seed at
argmax, +9 +- 55) and against `raw:t0.5`, to separate "plays better"
from "computes the deterministic reference's exact reply". Cost
$0.3-0.6 per confirmed win, no run now. The playout-transfer run is
rejected (section 2).

## 2. Rejected, one line each

- #2 Auxiliary short-horizon targets: predicted -0.03 to -0.08 on
  sigma_s against a readout SE of 0.05-0.06 on 48 candidates; neither
  bar can fire; a later factor on item 6 once the eval set exists.
- #3 Sibling-contrast fine-tune: third link of a chain whose first
  two are predicted to miss; P = 4 labels (SD 0.45) cannot measure
  sigma_s to 0.2 on the self-play holdout; bakes in "T=1 samples are
  worse".
- #4 One-reply lookahead player on the new net: not testable until a
  grader at sigma_s <= 0.20 exists (cumulative P about 0.2-0.45); its
  reachable form today is #18.
- #6 Find arm (K = 8 with localization): position-specific labels,
  5-10 per $1.6, 0 to +15 Elo below any affordable gate; the
  localization verdict rests on 40-playout grades with SE 0.14.
- #8 Continue-edit preference pairs: trains the trunk on the 369
  holdout games (eval contamination) and on rollout-graded pairs
  before any teacher match (rule 2); about 350 effective labels fit
  only the global tilt that item 1 tests for $0.8.
- #9 Exact-combat edit pairs: same holdout contamination; 150
  effective labels learn a trust scalar the config margin already is;
  the enumerator as a gated player (the earlier panel's XOD rung 1,
  $0.55) tests the lever with an unbiased match. Stage A's override
  rate ($0.15) is the only new number and can ride on any box.
- #10 Whole-turn REBEL at P = 2: the verdict is a proxy (held-out r,
  ceiling 0.19), which rule 5 forbids; labels 96% noise; length bias
  rewards not ending the turn globally; uses nearly all holdout games
  as training positions; a pass funds a $40 round.
- #11 Forked-boundary RLOO: chi-squared 130-500 determines only
  global directions (tempo, aggression) that item 1 tests for under
  $1; 4-8 box-hours of training per attempt against "train sparingly";
  modal outcome inside the 800-game resolution.
- #12 Whole-game tempered REINFORCE: one outcome per ~300 decisions
  with a table baseline is the regime the project's first loop already
  ran without beating the prior; its stated product (P1 vs P2 within
  12 Elo) is unresolvable.
- #13 AWR on sequentially graded K = 4 turns: 70% of $3.4 grades a
  tail whose hit rate at T = 0.5 is unmeasured; ~5k up-weighted
  decisions (most of them neutral atoms of a good turn) against a
  2.8M-decision fit.
- #17 Uniform pair weighting: lowest information per dollar of its
  family by its own ranking; a tie with the control inside 2 SE
  decides nothing for $2.5; the quit-surrender exclusion is the more
  plausible lever on that axis and is not proposed here.
- #19 Policy-value consistency training: rule 2 blocks it until #18
  passes; the target is f(prior samples, own value head on imagined
  states), the family behind every failed leg, and a head at
  turn-scale AUC 0.53 weights the sampler's noise by its own biases.
- #20 Value-screened rejection sampling: the test grades only the
  head's top 2 of 30 edits with no random-edit control, so the
  enrichment it claims to measure is unidentified; its pairs feed a
  distillation the proposal itself rates under P 0.1. Resurrect with a
  random-edit arm only if #18 step A passes.
- #21 playout-transfer run: a within-position Spearman over 5
  candidates at 40 playouts (SE 0.14 per candidate against a 0.15-0.20
  spread) has a noise ceiling near 0.6-0.7, so the 0.5 and 0.8 reading
  thresholds are uncalibrated; only the confirmation-match rule
  survives (item 7).

## 3. Recommended order of the first three tests ($15 budget)

| order | test | cost | conditional on |
|---|---|---|---|
| 1 | end_turn decode rule + offset attribution arm (#14); same box: pre-grader step A (#18, $0.03), recruit-entropy screen (#16, $0.02), combat-edit override rate rho (#9 stage A, $0.15) | $1.6 | nothing |
| 2 | lookahead player match, #18 step B | $1.1 | step A pass (P about 0.25) |
| 3 | rating-weighted imitation (#15) | $2.6 | the relevant-set control arm's 800-game number |

Committed: $4.2 to $5.3. Held for the follow-ups the readings decide:
disjoint-seed confirmation of any winner ($0.58 each), #16's match if
its screen shows a gap ($0.58), the on-policy continue-edit harvest
(#5, $1.5) and the continue-edit run on the 60 positions ($0.6) as
the candidate set for the value net, then the value net itself
($1.2). Total inside $15 if at most two winners need confirmation.

Every result file carries the procedure tag, temperature and
precision; partial results are written as the run goes (ruling of
2026-09-05); a job past 1.5x its estimate is inspected and cut. If
the relevant-set retrain replaces the seed, tests 1 and 2 are
re-pinned on the winner (they are decode rules and transfer as
method); test 3's index and subset do not depend on the seed.

### Pre-registration draft: test 1, end_turn at the actor level

Question: does the reference under-act because of the joint argmax's
comparison of end_turn's actor mass against four-way products, and
does deciding end_turn at the actor level win games?

Estimand:
- Player A: `raw:t0+endm`, the seed at temperature 0 with end_turn
  played only when p_actor[end] >= max over unit and recruit slots of
  the actor marginal (sum of joint priors over that actor's legal
  actions); otherwise the joint argmax among non-end actions. Player
  B: `raw:t0`. Attribution arm: `raw:t0+eo<x>`, the seed with the
  end_turn logit offset x in {-0.75, -1.5}, everything else argmax.
- Match: PURE, sides alternated, ladder maps, persistent workers,
  `--jobs 10`, max 200 turns, seed base disjoint from every earlier
  match. Screen 40 games per arm; then games bought until 800 decisive
  for the rule (about 1,300 at the observed capped rate), and for the
  best offset only if the rule passes.
- Headline: the rule's decisive-game score p with SE 0.018; secondary
  read scoring capped games 0.5; decisions per side-turn and capped
  fraction per player; the offset arm's p as the attribution number.

Held fixed: the seed checkpoint, bf16 + compile on cuda, combat-oracle
alphas 0, the joint-prior argmax for every non-end decision, one
result directory per procedure pair (the outdir mismatch guard).

Bars:
- Kill 1 (screen): the rule's decisions per side-turn under 9.7
  against `raw:t0`'s 9.4 (the rule does not fire; dead at $0.02).
- Kill 2: p <= 0.50 at 800 decisive.
- Barrier: a capped rate against `raw:t0` above the reference's own
  17 of 40 by more than 2 SE reads as a stall tilt, reported next to
  W-L; the 0.5-scored read is quoted alongside.
- Pass: p >= 0.535 (428 of 800), then 800 decisive on a disjoint seed
  set before the number is quoted; the offset arm then runs to 800
  decisive. If the offset matches the rule within 1 SE, the lever is
  "act more" and the config scalar is the adopted form (rule 7).

PREDICTION (before the run): decisions per side-turn 9.4 -> 10.4
(range 9.8-11.2); rule p = 0.52 (range 0.46-0.58); P(pass) 0.3; the
offset at -0.75 within 0.02 of the rule; capped fraction against
`raw:t0` 0.35 (range 0.2-0.45, the reference's own 0.42 as the null).

Cost: bring-up $0.17; screens 3 x 40 games $0.06; 1,300 games $0.58;
attribution 1,300 games $0.58 (conditional); riders on the same box:
pre-grader step A $0.03, recruit-entropy screen $0.02, rho $0.15.
Total $1.0 unconditional, $1.6 with the attribution arm; about 5
box-hours.

### Pre-registration draft: test 2, the lookahead player (#18 step B)

Runs only if step A passes as pre-registered in
docs/turn_proposer_design_20260905.md section 6 (any grader at sigma_s
<= 0.2 with its 2-SE upper bound below 0.3 and all three confirmed
alternatives above their base). Step A's own resolution is stated
there: about 48 paired candidates, SE about 0.05.

Estimand:
- Player A: `raw:t0+la8m`: at each side-turn boundary, the base
  argmax turn via `record_spine`, 3 continue edits (forced 1, 2, 3
  non-end actions at the base's end_turn, argmax after) and 4
  temperature-0.5 samples with recorded command lists, deduplicated by
  post-turn `state_key`; each candidate gets one argmax reply by the
  opponent and one read of the grader that passed step A at the
  mover's next boundary; the best candidate is played only if its
  score exceeds the base's by m, with m set from the step-A rows so
  that the null control's firing rate on the 60 positions is under
  10%. m = infinity is bit-identical to `raw:t0` (asserted by a test on
  a fixed seed).
- Player B: `raw:t0`. PURE, sides alternated, ladder maps, persistent
  workers, seed base disjoint; 400 decisive games (buy 400-680).
- Headline: p with SE 0.025; firing rate (side-turns where the played
  turn differed from the base); K median both sides; capped fraction;
  class of the winning candidate (continue vs sample).

Held fixed: the seed, bf16 + compile, alphas 0, the grader and m
frozen before the match; the reply is the opponent's argmax under the
fork's salt.

Bars:
- Kill: p <= 0.50 at 400 decisive; or firing rate under 2%.
- Barrier: K median rising by more than 2 decisions against the base
  reads as the tempo effect of test 1 and is attributed there, not to
  the grader.
- Pass: p >= 0.535 at 400 (one-sided alpha 0.08), then 800 decisive on
  a disjoint seed set, then 400 decisive against `2291k:t0` and against
  `raw:t0.5` (item 7) before the number is quoted as strength.

PREDICTION (before the run): P(step A passes) 0.25; conditional on a
pass, p = 0.54 (range 0.47-0.60), firing rate 6% (range 2-15%);
unconditional P(a confirmed win at 800) about 0.1.

Cost: ~140 extra decisions per side-turn, 110-200 s of process time
per game, 400-680 games at 10 workers 1.7-2.8 box-hours, $0.55-0.95;
bring-up $0.17 if rented alone; about $1.1. Code before the box:
`tools/lookahead_player.py` (~150 lines), an argmax variant of
`project_value`, procedure tag and mismatch guard, the m = infinity
identity test.

### Pre-registration draft: test 3, rating-weighted imitation

Question: does the seed's argmax get stronger when it imitates the
corpus's strongest quartile instead of its median?

Estimand:
- Data: a player index over the 17,124 imitation replays (player ids
  from the `[side]` blocks, the RCA AI as one id), a Bradley-Terry fit
  over the corpus outcomes with shrinkage toward the mean by games
  played, `winner_rating` and `winner_games` written to the manifest.
  Arm 1: winner-side pairs of games whose winner has >= 30 corpus
  games and a rating in the top quartile of such players. Arm 2 (only
  if arm 1 is under 250k pairs, decided before renting): soft weight
  exp(k x rating_z) over all pairs, k = 1.
- Training: from the seed weights, the relevant-set control arm's
  recipe (`--init-from seed.pt --max-pairs 1260000`, per-game weight
  as configured, value loss as configured), same file order and seed.
- Matches: PURE vs `raw:t0`, 800 decisive (buy ~1,300); PURE vs the
  control arm, 400 decisive (buy ~600). Barrier: legality-masked
  holdout CE on holdout games won by top-rated players, at equal
  pairs, for the arm and the control.
- Headline: p vs `raw:t0` with SE 0.018; p vs the control with SE
  0.025 (separates "better data" from "another half epoch").

Held fixed: the seed, the recipe, the holdout split (the 369 holdout
games are excluded from the index's training subset and from the
rating fit's influence on labels; ratings are computed over all games
but no holdout pair is trained on), bf16 + compile at eval.

Bars:
- Kill 0 (before renting, plain python): arm 1 under 250k pairs ->
  arm 2.
- Kill 1: p <= 0.50 vs `raw:t0` at 800 decisive.
- Barrier: top-rated holdout masked CE worse than the control's by
  more than 0.05 nat at equal pairs (overfitting the subset).
- Pass: p >= 0.535 vs `raw:t0`, and above the control by at least 1
  SE; then 800 decisive on a disjoint seed set.

PREDICTION (before the run): arm 1 covers 18% of the pairs (range
12-28%); p vs `raw:t0` = 0.52 (range 0.46-0.57); p vs the control =
0.52 (range 0.47-0.57); P(pass) 0.3. The control arm itself is
predicted within +-30 Elo of `raw:t0`.

Cost: index on the box $0.03 (about 25 min single-core; not on the
laptop); training 1.26M pairs at 76 pairs/s, 4.6 h, $1.52; 1,300
games $0.58; 600 games $0.27; bring-up $0.17: about $2.6, 8
box-hours. Code: player index + Bradley-Terry (~150 lines,
torch-free, with tests), a manifest column, ~15 lines in the trainer's
weight map.

## 4. What today's engineering makes affordable, and what is not

Newly affordable (docs/box_specs.md, 2026-09-04/05 rows):

- An 800-game raw match is $0.36 through persistent workers (65 min)
  and about 45 min through shared inference once `raw:t0` is re-pinned
  through it. Every decode rule, config scalar and imitation arm can
  therefore be gated at 800 decisive games for under $1, and a
  1,600-3,200-game confirmation ($0.7-1.5) is affordable when an
  effect sits near +25 Elo. The measurement is no longer the
  constraint for procedure players; the players' strength is.
- Playouts at $7.5e-5 (11.4 process-seconds, cap 30 at temperature
  0.5): a 60-position turn-gap-style run is $0.6-0.9, a 200-position
  on-policy harvest $1.5, and a candidate set that resolves sigma_s at
  0.03-0.05 (about 200 candidates at 160 playouts, 32,000 playouts) is
  about $2.4. The value-net line is measurable for the price of one
  such run; it was not measurable at the pre-restart efficiency.
- A lookahead player at ~140 extra decisions per side-turn (38 ms of
  contended process time per decision) plays a 400-decisive match for
  about $1: the "cheap grader + edits + one reply" teacher of the
  design's R1 reading can be matched, which is what rule 2 requires.
- A 0.5-epoch imitation fine-tune is $1.5 (76 pairs/s); a value-only
  net over 536k boundary state-passes is $0.45 plus a $0.33 cache; a
  training path at 35 ms per experience (bf16 batch 16) prices a
  100k-experience update at about $0.35.
- `raw:t0.5` self-play at $3e-4 per decisive game: on-policy
  boundaries cost nothing next to their grading.

Still too expensive or unresolvable at $50:

- The rollout-graded searcher as a match player: $50-300 per 800
  games at any efficiency in reach; it stays an instrument (R1).
- A state-conditional turn policy from preference pairs: about 1e5
  effective pairs, $1,000 today, $100 after the phase-1 10x; the
  credit buys about 5,000 effective pairs, which determines a
  low-dimensional correction only (which class to shift, and when),
  and that correction is what the $0.8 decode rule and the $0.44
  config scalar already test.
- Turn-level policy gradient (RLOO, whole-game REINFORCE, AWR) at
  $3.4-4.6 per attempt with a 4-8 box-hour training path: its
  information (chi-squared 130-500) determines global directions and
  its modal outcome (+10 to +20 Elo) sits inside the 800-game
  resolution, so each attempt's verdict is "unresolved" and the
  written follow-up is a $10 escalation.
- Distilling confirmed pairs: 15-25 per $4 round (about $0.15-0.25
  each today, $0.05-0.10 with a pre-grader or the Rust worker path),
  three rounds before one gate can see them, and rule 2 blocks the
  step until R2 is ruled on; the budget buys about ten rounds in
  total.
- Any sigma_s comparison finer than 0.05 on the 48 recorded
  candidates; the fix is the $2.4 candidate set above, bought before
  the value-net tests, not a bigger model.
- A second-opponent playout truth (the two-net league's transfer run):
  its estimand has a noise ceiling below its own reading thresholds at
  any playout count the credit affords; only the confirmation matches
  are kept.

### Data lever: player ratings (test 3 tooling, 2026-09-05)

`tools/player_ratings.py` (tests: `tests/test_player_ratings.py`) builds
the player index and the Bradley-Terry fit of test 3. Player ids exist
only in the raw replays' `[side]` headers (the json.gz records carry
none); the raw corpus is on the laptop (`replays_raw/`, 218 MB for the
17,124 corpus games) and not on HF, so the index runs once on the
laptop (52 ms per header measured on 200 files: 15 core-minutes, about
3 minutes at `--workers 6`) or on a box after those files are shipped.
Every ai-controlled side is the one id `[ai]`; ratings use every game,
holdout included; the subset takes only non-holdout games; the AI is
rated but never in the top set. `fit --build-dataset` writes a
directory the trainer reads exactly like the full one (hardlinked game
files, manifest.jsonl, value_corpus_index.jsonl): the subset rows plus
the holdout games won by top-rated players flagged `holdout`, so the
trainer's periodic eval and `--eval-only` on the control checkpoint
give the barrier's two numbers on the same games. Commands:

    # laptop: index once, then kill 0 (seconds, plain python)
    python tools/player_ratings.py index --dataset replays_dataset_imitation --raw-root . --workers 6
    python tools/player_ratings.py fit --dataset replays_dataset_imitation --min-games 30 --quantile 0.75 --out ratings/arm1 --report
    # "KILL 0" in the summary -> arm 2; otherwise ship the index to the box:
    scp replays_dataset_imitation/players.jsonl BOX:/workspace/Wesnoth-AI/replays_dataset_imitation/

    # 4090 box, after scripts/eval_box_setup.sh staged seed.pt; the control
    # arm's checkpoint is CONTROL (scripts/relset_arms_box.sh leaves it at
    # /workspace/relset/control/arm.pt; stage it from HF otherwise)
    python tools/player_ratings.py fit --dataset replays_dataset_imitation --min-games 30 --quantile 0.75 --out /workspace/toprated/ratings --report --build-dataset replays_dataset_toprated
    ARCH="--d-model 384 --num-layers 8 --num-heads 12 --d-ff 1536"
    EVAL="--eval-every 50000 --eval-pairs 1200 --eval-pairs-per-game 8 --eval-sample-seed 0"
    python tools/supervised_train.py replays_dataset_toprated --checkpoint /workspace/toprated/arm.pt --init-from training/checkpoints/seed.pt --imitation-config configs/imitation.json $ARCH --epochs 30 --max-pairs 1260000 --seed 20260905 --bs 64 --lr 1e-4 --device cuda --workers 20 $EVAL --ckpt-every 2000 --log-every 100
    # barrier: the control arm's masked CE on the same top-rated holdout games
    python tools/supervised_train.py replays_dataset_toprated --eval-only --resume CONTROL --imitation-config configs/imitation.json $ARCH --device cuda $EVAL --eval-json /workspace/toprated/control_on_toprated_holdout.json
    # matches, PURE, raw:t0 both sides; tools/elo_collect.py OUTDIR --no-catalog --save-json OUTDIR.fit.json after each
    MATCH="--mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 --persistent-workers --no-infer-compile --device cuda --jobs 10"
    python tools/run_elo_batch.py --label-a toprated --spec-a /workspace/toprated/arm.pt --label-b seed_t0 --spec-b training/checkpoints/seed.pt --outdir /workspace/toprated/vs_seed --games 1300 --seed-base 40000 $MATCH --time-budget-min 120
    python tools/run_elo_batch.py --label-a toprated --spec-a /workspace/toprated/arm.pt --label-b control --spec-b CONTROL --outdir /workspace/toprated/vs_control --games 600 --seed-base 50000 $MATCH --time-budget-min 60

The training line is the control arm's recipe with the dataset swapped
and `--epochs 30`: the cosine schedule spans `--epochs` (T_max), so a
large value keeps the learning rate flat as in the control's half
epoch while `--max-pairs 1260000` stops the run, which is 2-3 passes
over a subset of the predicted size (18% of 2,566,963 winner-side
pairs is 462k; the 250k kill is 9.7%). `--eval-only` repeats the arch
flags because the trainer checks them against the checkpoint. The
holdout set of the built dataset is about 18% of the 369 holdout games
(66 at the prediction), so the stratified probe reads about 530 pairs
per eval instead of 1,200.
