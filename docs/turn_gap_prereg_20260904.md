# Turn-level value gap: pre-registration (2026-09-04)

Question (docs/plan_20260904.md section 5, the phase-2 prerequisite):
at a turn boundary, how often does one of a few alternative whole
turns beat the turn the reference player actually plays, by enough
that a rollout-graded turn search could see it? This number prices
every turn-level teacher design. Tool: `tools/turn_gap.py`; tests:
`tests/test_turn_gap.py`.

## Estimand

- Position: a side-turn boundary as the mover sees it (the mover's
  init_side applied). Default set: the first 60 of the 200 holdout
  ladder positions in `configs/bench_states.json` (human games the
  seed never trained on, cut uniformly inside the game).
- Base turn: the whole sequence of atomic actions `raw:t0` (the seed
  at temperature 0, `tools/raw_player.py`) plays from the position
  until it ends its turn.
- Alternative turns: K = 4 whole turns played by the same weights at
  temperature 1 on the joint legal-action prior, one distinct
  sampling seed each. An alternative whose post-turn position equals
  the base's or an earlier alternative's (`state_key` of the state
  after the mover's end_turn, opponent's init_side included) is
  dropped and counted.
- Within one position every candidate turn is played with the same
  combat salt: the n-th random draw is the same across candidates
  (common random numbers). Each candidate turn is therefore one
  realization of its in-turn combat, and its post-turn position is
  fixed before the playouts.
- Playouts: each candidate post-turn position is played P = 40 times
  to the end of the game with `raw:t0` on both sides; playout r of
  candidate c of position i differs from the others only through the
  combat salt `turn_gap:<seed>:p<i>:c<c>:r<r>`. A playout ends
  undecided at the start of turn T0 + 41, T0 the boundary turn: the
  rest of turn T0 and 40 full turns are played.
- Outcome per playout from the mover's side: +1 win, -1 loss, 0 draw
  or undecided at the cap. Value of a candidate: mean over its P
  playouts.
- Gap of a position: best alternative value minus base value; 0 when
  no distinct alternative exists.
- Headline: the fraction of positions with gap >= 0.25, with its
  binomial standard error.

Held fixed: the seed `tier-b/a3/seed_imit_tierb_start.pt`; bf16 and
compiled inference on cuda, as the eval harness runs it (the
measured object is the seed under that precision); the joint-prior
argmax as the reference procedure; combat-oracle alphas at 0, so the
policy's decision counter does not alter priors between candidates.

## What the tool reports

Per position: the base and every alternative with decisions per
turn, every playout's outcome, capped flag, final turn and salt;
dropped alternatives with what they duplicated; the gap; whether the
base is best or tied. Summary: the headline fraction and its
standard error, the mean gap, a histogram of gaps at 0.25 width, the
fraction of positions where the base turn is best or tied, positions
without a distinct alternative, positions whose base turn ended the
game, playouts and capped playouts, decisions per turn, wall time
and dollars (`--dollars-per-hour`).

Two more numbers, because the headline is biased upward by
selection: the best of four noisy means beats an independent noisy
mean even when all five turns are equal. With decisive 50/50
playouts the standard error of one candidate value is 1/sqrt(40) =
0.16, the expected value of the best of four equal alternatives
exceeds the base by about 0.16, and the null probability of a gap
>= 0.25 is roughly 0.3. So:

- Permutation null: outcomes of a position reshuffled across its
  candidates (200 shuffles), the fraction of positions with gap >=
  0.25 that noise alone produces, computed from the run's own
  outcome distributions. The headline is read against it.
- Split-half gap: the best alternative is chosen on the even-numbered
  playouts and its gap to the base measured on the odd-numbered
  ones. Its mean over positions is an unbiased estimate of the
  chosen alternative's true advantage; reported with its standard
  error.

## Defaults

| knob | default | flag |
|---|---|---|
| positions | first 60 of `configs/bench_states.json` | `--n-states`, `--states-json`, `--dataset` |
| alternatives K | 4 | `--alternatives` |
| alternative temperature | 1.0 | `--temperature` |
| playouts P | 40 | `--playouts` |
| playout cap | 40 turns after the boundary | `--cap-turns` |
| gap threshold | 0.25 | `--gap-threshold` |
| seed | 1 | `--seed` |
| device, precision | cuda, bf16 + compile | `--device`, `--infer-bf16`, `--infer-compile` |
| parallelism | 10 positions at a time, one process and policy each | `--jobs` |

## Cost

Inputs (docs/box_specs.md, pipeline baseline of 2026-09-04): a
`raw:t0` vs `raw:t0` game through the eval worker on a 4090 box at
`--jobs 10` takes 34.6 s median for 48.5 turns median, 0.71 s per
game-turn per process with ten processes running. Box price taken
at $0.33/h.

Work: 60 positions x 5 candidate turns = 300 side-turns (about two
process-minutes, negligible); at most 60 x 5 x 40 = 12,000 playouts
(fewer when alternatives are dropped); policy load and compile once
per worker (about a minute each, in parallel).

Wall = 12,000 x L x 0.71 s / 10, L the mean playout length in turns:

| mean playout length L | box-hours | dollars at $0.33/h |
|---|---|---|
| 10 | 2.4 | 0.78 |
| 17 (the control matches' median game length) | 4.0 | 1.33 |
| 20 | 4.7 | 1.56 |
| 40 (every playout capped) | 9.5 | 3.12 |

The plan's "about $1.3" is the L = 17 row. L is unknown for
`raw:t0` self-play from mid-game positions: the benchmark's
argmax-vs-argmax games from turn 1 ran 48.5 turns median (outcomes
not recorded), so the capped row is the ceiling. Add about $0.17 for
box creation, image pull, bring-up and destruction (docs/box_specs.md,
2026-09-04 amendment). The tool logs one line per position with its
wall time and rewrites `<out>.partial.json` after every completed
position; `python tools/turn_gap.py --summarize <out>.partial.json`
reads the completed positions mid-run, so a run heading to the
ceiling can be stopped and read.

## Bars

- Kill criterion (docs/plan_20260904.md): under 5% of positions with
  gap >= 0.25 kills rollout-graded turn search; the value function
  must then come from human data first.
- Proposed reading rules, in addition (operator to confirm): the
  headline fraction is compared to the permutation null and counts
  as evidence for a gap only when it exceeds the null by at least two
  standard errors; the split-half mean gap must be positive by at
  least two standard errors. A headline above 5% that sits inside
  the null carries no information about the gap.
- Validity conditions, crash barriers only: if more than half of the
  playouts hit the cap, the outcome scale is compressed toward 0 and
  the run is rerun with a higher cap or a decisive rule before any
  reading; positions without a distinct alternative and base turns
  that end the game are reported and excluded from nothing.
- Limits of the estimand: alternatives come from temperature-1
  sampling of the seed, which sits about 400 Elo below its argmax
  (docs/raw_argmax_control_20260904.md), so the best of four is a
  floor on what a designed turn proposer could find; the in-turn
  combat of each candidate is one realization; positions are human
  holdout positions, not positions the seed reaches on its own.

## Operator rulings (2026-09-04, before the run)

- Reading rules adopted as written above: the headline counts as
  evidence only when it exceeds the permutation null by at least two
  standard errors, and the split-half mean gap must be positive by at
  least two standard errors.
- Kill criterion restated so it can fire under this noise: rollout-
  graded turn search is killed when the split-half mean gap is below
  0.02 with its two-standard-error upper bound below 0.05 AND the
  headline sits inside two standard errors of the null. The plan's
  "under 5% of positions" is unreadable at P = 40 (the null alone
  produces about 30%) and is superseded by this rule.
- Run settings: 60 positions, K = 4, P = 40, temperature 1, cap 30
  turns after the boundary (not 40: the capped ceiling at 40 is 9.5
  box-hours), `--jobs 14` on the 17.5-core box.

## Amendment before the run (2026-09-05 00:10, operator)

The temperature sweep that finished minutes earlier (docs/box_specs.md,
"Raw player temperature") showed that `raw:t0` against itself stalls:
17 of 40 games from turn 1 reached the 200-turn cap (median 125
turns), while `raw:t0.5` scored 22-18 against `raw:t0` with no game
past 133 turns (median 31). Playouts at temperature 0 with a 30-turn
cap would therefore mostly end undecided and compress the outcome
scale, which the validity condition above already forbids. Ruling:
the playouts run both sides at temperature 0.5
(`--playout-temperature 0.5`, one sampling seed per playout derived
from its salt, side 2 offset by one); the base turn, the alternatives
(temperature 1) and everything else stay as written. The estimand is
now the expected outcome under a decisive continuation policy of the
same strength as the reference within the sweep's resolution
(40 games, score 0.55 +- 0.08). Predictions unchanged.

## PREDICTION (operator, 2026-09-04, before the run)

- fraction of positions with gap >= 0.25: 0.35
- permutation null of that fraction: 0.30
- mean split-half gap: +0.03 (plausible range -0.02 to +0.08). The
  sampled turns are on average worse than the argmax turn (the
  sampler is about 400 Elo below the argmax over a game), but one of
  four is often no worse, and the selection on 20 playouts picks a
  small true edge in a minority of positions.
- fraction of playouts capped: 0.35 (mid-game boundaries, cap 30).
- expected cost: mean playout length about 22 turns -> 12,000 x 22 x
  0.71 s / 14 = 3.7 box-hours, about $1.25 at $0.33/h.

## Run

On a 4090 box brought up as in docs/box_specs.md (2026-09-04
amendment), with the seed at `training/checkpoints/seed.pt` and the
manifest's game files packed by
`python tools/bench_pipeline.py --pack-states /workspace/bench_dataset`
(as `scripts/bench_box.sh` does):

    python tools/turn_gap.py --checkpoint training/checkpoints/seed.pt \
        --device cuda --jobs 14 --cap-turns 30 --playout-temperature 0.5 --dollars-per-hour 0.33 \
        --states-json configs/bench_states.json --dataset /workspace/bench_dataset \
        --out training/metrics/turn_gap/run1.json

The JSON holds the config, provenance (procedures, precision, torch
version), every per-position record and the summary; the markdown
summary lands next to it. Results go in a section appended to this
document.

## Result (run 1, 2026-09-05, box 49875606)

Settings as amended: 60 positions, K = 4 at temperature 1, P = 40,
playouts at temperature 0.5, cap 30, `--jobs 14`. Records:
`training/metrics/turn_gap/run1.{json,md,log}`. Wall 2.71 h, $0.90.

| quantity | value | prediction |
|---|---|---|
| positions with gap >= 0.25 | 12/60 = 0.200 +- 0.052 | 0.35 |
| permutation null of that fraction | 0.179 | 0.30 |
| mean gap (selection-biased) | +0.113 +- 0.032 | - |
| mean split-half gap | +0.049 +- 0.041 | +0.03 (-0.02..+0.08) |
| base turn best or tied | 25/60 | - |
| playouts capped | 1,202 / 12,000 = 0.10 | 0.35 |
| distinct alternatives | 4.00 of 4 in every position | - |

Reading against the rules above: the headline exceeds its null by
0.02, far inside two standard errors (0.10), so it carries no
information; the split-half mean gap is positive by 1.2 standard
errors, short of the two required. The kill does not fire either:
the split-half gain is not below 0.02 and its two-standard-error
upper bound (0.13) is not below 0.05. Verdict: inconclusive between
"no gap" and "a small gap of about 0.05 per turn". The prediction
was inside the result on every line; the null and the capped
fraction came out lower than predicted because temperature-0.5
playouts are more decisive than the 50/50 assumption.

Gap histogram: 18 positions negative, 30 in [0, 0.25), 12 at or
above 0.25 (positions 2, 3, 4, 9, 14, 18, 20, 29, 42, 49, 57, 59;
nominal gaps 0.27 to 1.07).

What would resolve it: (a) four times the positions for the mean
(about $3.6, 11 box-hours at this efficiency); (b) cheaper and more
informative for phase 2, a confirmation run on the 12 nominal
big-gap positions with 160 fresh playouts per candidate (the same
sampled alternatives, playout salts offset past the first 40), which
tests directly whether any large gap is real. Pre-registered below.

## Confirmation run (pre-registered 2026-09-05, before the run)

Settings: `--positions 2,3,4,9,14,18,20,29,42,49,57,59 --playouts
160 --playout-offset 40`, everything else as run 1 (the alternatives
reproduce from their seeds; the in-turn combat salt is unchanged).
Estimand per position: the out-of-sample gap = mean over the 160 new
playouts of the alternative that run 1 selected, minus the base's
mean over the same new playouts. Cost: 12 x 5 x 160 = 9,600 playouts,
about 2.2 box-hours, $0.75 at `--jobs 12`.

Prediction: the confirmed mean gap over the 12 positions is about
+0.10 (regression from the nominal mean of +0.50); at most 3 of the
12 confirm at >= 0.25 (their standard error at P = 160 is about
0.06, so a confirmed 0.25 is four standard errors).

Reading: a confirmed position is one whose out-of-sample gap is at
least 0.25. Three or more confirmed positions, or a confirmed mean
above 0.15, means large turn-level gaps exist at this proposer
quality and rollout-graded turn search has something to find; zero
or one confirmed and a confirmed mean below 0.05 means the nominal
gaps were selection noise and the proposer (temperature-1 sampling
of the seed) must improve before turn search is priced again.

## Confirmation result (2026-09-05, box 49875606)

Settings as pre-registered (12 positions, K = 4 reproduced from their
seeds, 160 fresh playouts per candidate, salts offset by 40,
temperature 0.5, `--jobs 12`). Records:
`training/metrics/turn_gap/confirm1.{json,md,log}`. Wall 2.54 h, $0.84.
Capped playouts 11%.

Out-of-sample gap of the alternative run 1 selected, on the 160 new
playouts (the pre-registered estimand):

| position | run 1 nominal gap | out-of-sample gap | base value (new) |
|---|---|---|---|
| 2 | +0.40 | -0.04 | +0.55 |
| 3 | +0.27 | -0.18 | -0.54 |
| 4 | +1.07 | +0.96 | -0.28 |
| 9 | +0.38 | -0.11 | -0.48 |
| 14 | +0.43 | +0.03 | -0.66 |
| 18 | +0.35 | +0.28 | +0.24 |
| 20 | +0.60 | +0.21 | +0.06 |
| 29 | +0.55 | +0.22 | -0.71 |
| 42 | +0.45 | +0.12 | -0.23 |
| 49 | +0.75 | -0.42 | -0.39 |
| 57 | +0.32 | +0.11 | +0.21 |
| 59 | +0.42 | +0.38 | -0.74 |

Confirmed at >= 0.25: 3 of 12 (prediction: at most 3). Mean
out-of-sample gap +0.129 +- 0.098 (prediction: about +0.10). Eight of
twelve positive. The run's own best-of-four on the new playouts (a
second selection, but with a null of 0.015 at P = 160): 4 of 12 at
>= 0.25, mean +0.23 +- 0.08, split-half +0.18 +- 0.08.

Reading against the pre-registered rule: three confirmed positions
meet the evidence threshold exactly, and the confirmed mean (0.13)
sits between the two bars (0.05 and 0.15). Verdict: large turn-level
gaps exist at this proposer quality, but they are sparse: about 3 of
the 60 positions (5%, +- the count's own noise) carry an out-of-sample
gain of 0.3 to 1.0 in expected outcome, and the average gain of the
best of four sampled turns over all positions is about +0.05 per
turn. Position 4 (a -0.28 base turn against a +0.68 alternative) is
the kind of example a rollout-graded teacher would be built to find.

Price of the signal at this efficiency: finding these three cost
$0.90 of run 1 plus $0.84 of confirmation for 60 boundaries, i.e.
about $0.6 per confirmed large-gap example with temperature-1
sampling as the proposer and 40 + 160 playouts as the grader. That
number, not the fraction, is what a phase-2 teacher design has to
beat: a better proposer raises the hit rate, cheaper playouts (fewer
tokens per leaf, decisive continuation policies) lower the price.

## Audit of the confirmation (2026-09-05, docs/turn_proposer_design_20260905.md)

Re-reading the two JSON files: 12 of the 48 sampled alternatives in the
confirmation run were DIFFERENT turns from run 1's (decision counts
differ by 1 to 6): sampling from a seed is not reproducible across
runs under bf16 kernels, so the "same alternative, fresh playouts"
estimand held only where the resampled turn coincided. Positions 4, 18
and 59 (the three confirmed) are intact; at 20, 49 and 57 the
confirmation graded a turn nobody had selected; at 20 a matched
alternative scores +0.40 on 160 fresh playouts, a probable fourth
confirmed gap. The verdict above stands qualitatively (large gaps
exist and are sparse); the count is 3 to 4 of 60. Fix, in the tool
from this commit on: every candidate turn records its action list,
and confirmations replay the recorded actions instead of resampling.
The same audit finds two of the three confirmed gaps are base
blunders (all or most alternatives beat the base, the base turn 4-10
decisions shorter) and one a find, and it estimates that a sequential
screen would have found the same nominal hits with 35% fewer playouts.
