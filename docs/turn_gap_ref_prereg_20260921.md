# Pre-registration: the turn-level value gap under the reference player (2026-09-21)

Status (2026-09-26): the RICH verdict below (7 of 60 confirmed, bar 6)
carries a caveat. The confirmation replayed each turn under the screen's
turn salt, so the in-turn dice of the best-of-four selection survived
into it; comparing every fight of the 7 confirmed turns with its exact
distribution, positions 15 and 57 carry luck in the alternative's favour
(+14.2 and +16.8 HP of the mover's), the other five none that explains
them. Re-realizing 15 and 57 under a second turn salt (the design's own
control, docs/turn_proposer_design_20260905.md 2.2; cents on a box)
decides whether RICH stands. Separately, 5 of the 16 replayed positions
of the pre-grader check came from maps with a third side, where the code
of the day read side 3's value after the turn (fixed in 0.7.7). A second
caveat (2026-09-26, docs/hidden_information_20260926.md): the candidates
were graded by playouts from the true post-turn state, the mover's hidden
enemies and the opponent's gold included; five of the seven confirmed
positions hold 2 to 4 enemy units hidden from the mover (the leader at 15
and 42), so a searcher limited to its own view may confirm fewer.
Re-grading the seven from sampled worlds needs a belief model first.

Phase 2's first measurement (docs/plan_20260904.md 5), taken again
against the current reference. Run 1 (docs/turn_gap_prereg_20260904.md,
2026-09-05) measured the seed at `raw:t0`; two of its three confirmed
gaps were the base ending its turn early
(docs/turn_proposer_design_20260905.md 2.1), and the reference has
since moved to `terrain` at `raw:t0+eo-1.5`, a decode that acts 1.42x
more per side-turn and is +263 +- 16 Elo over the seed's successor
(configs/reference_player.json). The question is the plan's: how often
does one of a few alternative whole turns beat the turn the reference
plays by enough that a rollout-graded turn search could see it. The
number prices every turn-level teacher design, and its kill sends the
value function to human data first.

Tool: `tools/turn_gap.py --reference` (the reference's checkpoint and
decode from the config, applied to the base turn, the sampled
alternatives and both sides of the playouts; provenance carries the
procedure tags); verdict: `tools/analysis/turn_gap_verdict.py`; box:
`scripts/turn_gap_ref_box.sh`; tests: tests/test_turn_gap.py,
tests/test_turn_gap_verdict.py.

## Estimand

As run 1 (its "Estimand" section holds), with these changes:

- Base turn: the whole sequence of atomic actions the reference
  player plays from the position, `raw:t0+eo-1.5` on the `terrain`
  checkpoint.
- Alternatives: K = 4 whole turns sampled at temperature 1 on the
  joint prior with the same end_turn offset (the offset is part of
  the player, so the proposer samples the reference's distribution,
  not the seed's).
- Playouts: both sides at `raw:t0.5+eo-1.5` (run 1 played `raw:t0.5`
  to avoid the deterministic stalls; the offset travels with the
  player), cap 30 turns after the boundary as run 1 ran.
- Grading: the sequential schedule of docs/turn_proposer_design_20260905.md
  2.5, which on run 1's recorded outcomes reproduced its 12 nominal
  hits with 7,790 playouts instead of 12,000. Screen: rounds of 10
  per surviving candidate, drop at gap + 2 SE < 0.25, stop as a
  nominal hit at gap >= 0.25 and gap - 2 SE >= 0, cap 40 per
  candidate. Confirmation, on every position the screen marks at gap
  >= 0.25: the screen's base and its best alternative replayed from
  their recorded actions under the screen's turn salt, 160 fresh
  playouts each in rounds of 20, confirmed at gap >= 0.25 and
  gap - 2 SE >= 0.10, rejected at gap + 2 SE < 0.25, else undecided
  at 160.
- Positions: the first 60 of `configs/bench_states.json` (holdout
  human games), the same 60 as run 1. Seed 21 (fresh salts; run 1
  used 1).

Headline: the CONFIRMED fraction, positions confirmed out of 60, with
its binomial standard error. Secondary, from the screen: the nominal
fraction against its permutation null, the split-half mean gap, the
base's and the alternatives' decisions per turn, and the capped
share of playouts.

## Rule

- KILL (the plan's): fewer than 3 of 60 positions confirmed (under
  5%). Rollout-graded turn search is then unaffordable as a teacher
  at this proposer, and phase 2 starts with the value function on
  human turn-boundary states (plan 5, fourth principle) rather than
  with a searcher.
- SPARSE: 3 to 5 confirmed, as run 1 read (3 to 4 of 60). The
  searcher stays an instrument that produces confirmed pairs to
  validate a cheap grader (docs/turn_proposer_design_20260905.md 5),
  not a teacher.
- RICH: 6 or more confirmed (10%). The pre-graded pipeline of the
  design (rows 1 to 7) is built and measured as the next factor.
- Validity, crash barriers only: more than half of the playouts at
  the cap sends the run back with a higher cap before any reading;
  a screen whose nominal fraction sits inside its permutation null
  carries no information on its own, and only the confirmation is
  read.

## Predictions

The base now acts until the end_turn head marks a turn worth
passing, so the early-end blunders that made two of run 1's three
confirmed gaps are gone from the base; what remains is the find
(a different plan), and temperature-1 sampling of a stronger player
proposes fewer of those. Screen: nominal hits 8 to 14 of 60 against a
permutation null near 0.18 (run 1: 12, null 0.179), split-half mean
gap +0.00 to +0.04 (run 1: +0.049 +- 0.041), base decisions per turn
12 to 14 (run 1: 9.4). Confirmed: 1 to 3 of 60, so the KILL fires or
the reading lands on its boundary; RICH is given 1 in 10. Capped
playouts 5 to 12% (run 1: 10%).

## Cost

Playouts: about 7,800 for the screen (the design's replay of run 1
under this schedule) and 1,500 to 2,500 for the confirmation (12
positions x 2 candidates x 60 to 100 at the stop rules), about
10,000 in all, at a mean playout length of 15 turns (run 1: 14.6):
about 150,000 game-turns.

Rate: the shared-inference turn-gap path measured 2026-09-05 (the
seed basis, per-process sims) ran 1,776 playouts in 2,350 s at 12
and at 24 workers, about 11 game-turns per second. The current stack
(the relevant-set basis at 2.7x fewer tokens per leaf, the Rust
kernels, the observation kernel) has not been timed on this path;
the eval path it shares moved 2 to 4x over the same period, so 22
to 44 game-turns per second is the expectation and 11 the floor.

| rate, game-turns per second | measurement wall | dollars at $0.56/h |
|---|---|---|
| 44 | 0.95 h | 0.55 |
| 22 | 1.9 h | 1.05 |
| 11 (the 2026-09-05 rate) | 3.8 h | 2.1 |

Plus bring-up (the code stage, the wheel build, the corpus and the
checkpoint): about 8 minutes, $0.10. Expected about $1.20, ceiling
$2.20. Box: a whole-CPU 4090 host of the 2026-09-18 class (Ryzen 9
5950X, $0.56/h); the path is worker-bound, so the CPU matters more
than the card. The script caps the screen at 3.5 h and the
confirmation at 2 h (`timeout`); the tool writes every completed
position to a `.partial.json` that `--summarize` reads, so a cut run
is still read, at fewer positions, and says so.

## Rulings

Written 2026-09-21 before any box is rented; the run waits for the
user's word on the cost above.

## Measured (2026-09-23, box 52267135: a 32-core slice of an EPYC 9684X host with an RTX 4090, $0.563/h)

**Verdict: RICH. 7 of 60 positions confirmed, 0.117 +- 0.041**
(positions 11, 15, 20, 24, 42, 47, 57; the bar is 6). Record:
`training/metrics/turn_gap_ref_20260921/` (screen, confirmation,
verdict, logs), run on `main` at 0.4.0 with the phase-11 wheel. About
1.9 h of rental, about $1.05, against the expected $1.20.

| quantity | predicted | measured |
|---|---|---|
| screen nominal hits | 8 to 14 of 60 | 16 (0.267 +- 0.057; permutation null 0.300) |
| split-half mean gap | +0.00 to +0.04 | +0.044 +- 0.048 |
| base decisions per turn | 12 to 14 | 10.2 (run 1: 9.4) |
| confirmed | 1 to 3 | 7 |
| capped playouts | 5 to 12% | 3.2% (screen), 1.8% (confirmation) |

The screen's nominal fraction sits inside its permutation null, so by
the rule only the confirmation is read. Of the 16 nominal hits it
replayed, 7 confirmed, 6 were rejected (4 with the alternative worse),
and 3 were undecided at 160 playouts. The screen played 7,810 playouts
in 2,216 s and the confirmation 3,160 in 2,259 s, at 24 workers.

Described, not pre-registered: in 6 of the 7 confirmed positions the
better turn takes more decisions than the reference's (14.4 against
10.3 on average) and in 5 it attacks more (37 attacks against 24 in
all). Run 1's confirmed gaps were mostly the base ending its turn
early; the -1.5 offset has not removed that pattern. The alternatives
are the best of four temperature-1 samples, so this is a description
of what won, not a measured cause.

Scope: all 60 positions are side-2 turn boundaries, as in run 1 (a
property of `configs/bench_states.json`), so the reading is about the
second player's turns.

By the rule, the next factor is the pre-graded pipeline of
docs/turn_proposer_design_20260905.md (rows 1 to 7).

## Pre-grader check on this run's candidates (pre-registered 2026-09-23, before the analysis)

The section-6 check of docs/turn_proposer_design_20260905.md: does a
forward-only grader rank candidate turns against the playout truth well
enough to pre-grade them? This run recorded both graders the tool reads
for every candidate, so the check is an analysis of the files above,
with nothing played and no box.

- Graders: `value_post`, the value head on the post-turn state (read
  from the opponent's observation, the side to move after the turn, and
  signed for the mover), and `hp_margin_post`, the mover's HP minus the
  opponent's. The design's other two (the value after one argmax reply,
  the expected material swing) were not recorded and are not measured
  here.
- Primary data: `confirm.json`, 16 positions, each the base and the
  screen's best alternative on up to 160 playouts. Secondary, as an
  attenuated check: `screen.json`, 60 positions and 299 candidates on 10
  to 40 playouts each.
- Estimand, per grader: the within-position residual SD of (playout mean
  - a x grader - b), one slope across all candidates, one intercept per
  position (`tools/analysis/turn_gap_pregrader.py`), with its standard
  error taken as SD / sqrt(2 df), df = candidates - positions - 1; and
  the ranking check: every alternative whose playout gap over its base
  is at least 0.25 in that file (8 of the 16 in the confirmation) must
  be ranked above its base.
- Rule (the design's): a grader passes at a residual SD <= 0.2 with its
  2-SE upper bound below 0.3 and every such alternative ranked above
  its base; it fails at >= 0.3 or with any such alternative ranked
  below; otherwise inconclusive. A pass builds the pre-graded pipeline
  (row 7) with that grader; two fails leave the pipeline without a
  pre-grader (rows 1 to 6). The tool's own verdict line (the 2026-09-05
  rule, value head only) is reported as well.
- Prior: the 2026-09-05 run on the seed's candidates
  (`training/metrics/turn_gap/pregrader1.json`, 12 positions, 160
  playouts; read today for the first time as a record): the value head
  at 0.156 within position with 2 of 8 such alternatives ranked below
  their base, a fail; the HP margin at 0.154 with 8 of 8 above, a pass
  under this rule.

Predictions: in `confirm.json`, the value head at 0.12 to 0.25 with 1
to 3 of 8 ranked below (a fail, probability 0.6; a pass 0.25); the HP
margin at 0.12 to 0.25 with 0 to 2 below (a pass, probability 0.5).
In `screen.json`, both at 0.20 to 0.35 on the noisier truth.

### Measured (2026-09-23): both graders fail

Records: `training/metrics/turn_gap_ref_20260921/pregrader_confirm.txt`
and `pregrader_screen.txt`.

| file | grader | within-position residual SD | winners ranked below their base | top pick agrees |
|---|---|---|---|---|
| confirm.json (primary) | value head | 0.323 +- 0.059 | 2 of 8 | 7 of 16 |
| confirm.json (primary) | HP margin | 0.322 +- 0.059 | 3 of 8 | 10 of 16 |
| screen.json | value head | 0.222 | 8 of 24 | 22 of 60 |
| screen.json | HP margin | 0.227 | 8 of 24 (2 ties) | 16 of 60 |

(+- 0.059 is SD / sqrt(2 x 15); 32 candidates, 16 positions.) Both
fail on the ranking, and both sit above 0.3 on the residual. On the
confirmation's candidates, the base and the screen's best alternative,
the value head barely separates them: correlation 0.07 with the playout
mean, slope 0.05. The HP margin's slope is near zero in HP units.

Against the predictions: the fails were predicted for the value head
(probability 0.6) and not for the HP margin (a pass at 0.5); both
primary residuals came out above the predicted 0.12 to 0.25; the screen's
0.22 sits inside its predicted 0.20 to 0.35. The prior run's value-head
residual of 0.156 used five candidates per position drawn without
selection; here each position holds a base and the alternative selected
as best, where the gaps are large, and the head does not see them.

By the rule: no forward-only pre-grader at this head; the pipeline
proceeds without one (rows 1 to 6 of the design), and a boundary value
net trained for this is the prerequisite for row 7, as the plan already
had it.
