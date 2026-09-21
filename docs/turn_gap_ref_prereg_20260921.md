# Pre-registration: the turn-level value gap under the reference player (2026-09-21)

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
