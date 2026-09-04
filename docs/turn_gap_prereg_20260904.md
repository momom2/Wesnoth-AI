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
