# Pre-registration: a value function that ranks candidate turns under obs8 (2026-09-25)

Written before any box runs it. Approved in principle by the user on
2026-09-24 and ordered on 2026-09-25 ("Proceed"). Revised the same day,
still before any run, after four independent reviews (data path,
statistics, box operations, literature) and a second statistics review
of the revision; the decision record is at the end.

## Why

Phase 2's teacher is a player that proposes a few whole turns, grades
them, and plays the best (docs/plan_20260904.md 5). Better turns exist:
under the reference, 7 of 60 positions have a sampled alternative turn
better by at least 0.25 in expected outcome
(docs/turn_gap_ref_prereg_20260921.md). Grading by full playouts costs
$0.006-0.013 per side-turn, so an 800-game gate of such a player costs
$50-300 (docs/turn_proposer_design_20260905.md); a teacher that can be
matched needs a grader that costs a forward pass or a few short
rollouts. The value head we have ranks candidate turns poorly: on the
2026-09-23 screen (60 positions, 10-40 playouts per candidate) its
read after the turn correlates 0.26 with the playout means within a
position, before any correction for playout noise.

## Question

Does a grader trained on outcome contrasts between candidate turns of
the same position under `obs8`'s own play, or `obs8`'s own value head
read a few turns into short rollouts, rank candidate turns well enough
to pre-grade them?

## Design

**Positions.** Turn starts of either side from the 800 recorded games
of `obs8` against `terrain` (HF
`tier-b/observation_retrain_20260924/games_obs_e1_vs_terrain.tar.gz`),
rebuilt from their records with the fingerprints checked
(tools/game_record.py `turn_starts`). From turn 2 on the games offer
41,328 turn starts (11 to 199 per game, median 47); up to 15 per game
are drawn with a seed of the game's own, 11,980 in all. Games are split
by a hash of their file name: 630 games (9,438 positions) are fit, 70
(1,046) choose each arm's configuration, and 100 (1,496) are held out
as the proxy games.

**Candidates.** At each position, as the turn-gap tool proposes them
(tools/turn_gap.py): `obs8`'s own turn at its decode
(`raw:t0+eo-1.5`), two turns sampled at temperature 1, and one
"continue" edit (the base turn without its end_turn, plus one more
argmax action). Candidates whose resulting position equals an earlier
one are dropped. Generation: tools/turn_value_data.py.

**Playouts.** One playout per candidate (two on the proxy games, so
their playout noise can be measured), both sides `obs8` at
`raw:t0.5+eo-1.5`, capped 30 turns after the position: +1 win, -1
loss, 0 draw or capped, from the mover's side. Every playout also
records, replayed from its commands (tools/playout_reads.py):
- horizon reads: `obs8`'s value from the side to move, signed to the
  mover, and the mover's HP margin, at each of the first 8 player turn
  starts after the candidate turn (read 0 is the position right after
  it; the mover's own turn starts k turns later are reads 2k - 1); a
  read after the game ended takes the outcome;
- fight luck: for each attack, the realized minus the exact expected
  change of the mover's HP margin and kill margin, from the exact
  outcome enumerator (tools/combat_outcomes.py); its conditional mean
  is zero, so it can correct a label without biasing it.

The candidate turn's own fights get the same luck reading, from its
recorded commands (computed with the features). One least-squares fit
of the outcome on both lucks, over the fit games' playouts, gives the
**adjusted outcome**: the outcome minus the fitted luck terms. It keeps
the expected value of the candidate turn over its own dice and sheds
noise; it is the truth every grader is judged against, because a
teacher replays the turn it picks with new dice and cannot use the ones
a grader saw.

**What a trained grader reads.** The position after the candidate
turn's last action and before its end_turn, encoded from the mover's
side by `obs8`'s encoder: what the mover observes when it is about to
end the turn. Each candidate records the commands, recruit rejections
and digest of that position; tools/turn_value.py rebuilds it, checks the
digest, and caches `obs8`'s global token (384 values). A candidate turn
that ends the game is left out of every measure: its outcome is known.

**Graders judged.**
- A, linear: standardized token to value, in closed form with a ridge.
- B, head: the shape of `obs8`'s value head, from its weights, trained
  with AdamW, the epoch chosen as below.
- D, rollout: no training; the mean over R = 8 playouts of the horizon
  read 2 turns after the candidate turn (read 3), at a cost of about
  R x 2 / 15 of a full playout per candidate.

A and B are fitted (tools/turn_value_fit.py) on the loss
level MSE + w x within-position MSE, with the label of a candidate the
mean over its playouts of lam x (horizon read 3) + (1 - lam) x
(the outcome, or the adjusted outcome). Each arm's configuration, among
lam in {0, 0.25, 0.5, 0.75}, the adjustment on or off, w in {0, 1, 10},
and the ridge (A, 1e-6 to 10) or the epoch (B, up to 40, epoch 0 being
`obs8`'s head), is the one whose grades correlate best, within
position, with the adjusted outcomes of the 70 choosing games. A
fine-tuned copy of the trunk is not in this registration.

**Reported, not judged:** `obs8`'s head on the pre-end_turn state (read
offline, and while playing), its read after the end_turn, the HP margin
after the turn, D at (R, read) = (4, 3), (8, 1) and (8, 7), each
grader's measure against the raw outcomes, each arm's selected
configuration, and the share of the outcome variance the lucks explain.

## Validation (the verdict's data)

All 200 positions of `configs/bench_states.json` (side-2 turn starts of
200 distinct human games). Candidates: `obs8`'s turn, three turns
sampled at temperature 1 and one continue edit; 28 playouts each under
the procedure above (seed 25), horizon reads and luck recorded, about
28,000 playouts. D's grade reads playouts 1 to 8 and its truth is
playouts 9 to 28; every other grader's truth is playouts 1 to 28. These are
positions of human games, where the training positions come from
`obs8`'s games: a grader that passes here ranks turns outside the
distribution it was fitted on.

## Measure

Within a position, center each candidate's grade and its mean adjusted
outcome on the position's means. The observed correlation r_obs pools
these deviations over the positions with two or more candidates.
Playout noise shrinks it: the reliability rho is the share of the means'
within-position variance that is not noise, with the noise of a
candidate's mean estimated as its adjusted outcomes' variance over its
playout count, times (1 - 1/C) for C candidates. The measure is r =
r_obs / sqrt(rho): an estimate of the grader's within-position
correlation with the candidates' expected values. Its standard error is
half the central 68% interval of 1,000 bootstrap resamples of the
positions (of games, on the proxy games). A synthetic check recovers a
known correlation of 0.5 and of 0.8 within 3 SE
(tests/test_turn_value.py).

## Rule

For each judged grader on the validation positions:
- **PASS:** r >= 0.7.
- **FAIL:** r <= 0.5.
- **INCONCLUSIVE:** in between.
- **UNDECIDED:** rho below 0.2 (the playouts do not separate the
  candidates) or an SE above 0.12.

0.7 is the design's bar: a grader error of 0.2 against a true
within-position spread of 0.15-0.2 is a correlation of about 0.7, the
level at which a pre-grader enriches gaps of 0.5 tenfold
(docs/turn_proposer_design_20260905.md). The second statistics review
simulated the rule as coded (200 positions x 5 candidates, 20 truth
playouts, raw outcomes, 50-60 runs per row, +- 0.06), at the 2026-09-23
spread of candidate values and at half its large gaps:

| true r | PASS / FAIL / INCONCLUSIVE / UNDECIDED |
|---|---|
| 0.50 | 0 / 0.40 / 0.60 / 0 (half the gaps: 0 / 0.40 / 0.50 / 0.10) |
| 0.55 | 0 / 0.18 / 0.82 / 0 (0.02 / 0.16 / 0.66 / 0.16) |
| 0.60 | 0.02 / 0.03 / 0.95 / 0 (0.06 / 0.08 / 0.80 / 0.06) |
| 0.70 | 0.57 / 0 / 0.43 / 0 (0.34 / 0 / 0.50 / 0.16) |
| 0.85 | 1.00 / 0 / 0 / 0 (0.90 / 0 / 0 / 0.10) |

The corrected r's spread was 0.045-0.057 at the 2026-09-23 spread and
0.07-0.08 at half, its bias 0 to +0.04. With 12 truth playouts, D was
UNDECIDED at 0.48-0.74 at half the gaps; its truth now has 20, as A's
and B's had in these rows, the adjusted outcomes raise the reliability
by the share of the variance the lucks explain, and the percentile SE
no longer follows the long tail of near-zero reliability resamples. No
grader's PASS below a true 0.6 appeared more than 6% of the time.

**Crash barrier, not a verdict:** on the proxy games, the observed
within-position correlation of A and of B with their own playouts must
be positive at 2 SE; an arm that misses it has learned nothing usable
at this size and its verdict is not read.

## Predictions

- `obs8`'s head before the end_turn: r 0.2 to 0.45 (its read after the
  turn reads 0.34 +- 0.10 corrected against the raw outcomes of the
  2026-09-23 screen).
- A: r 0.3 to 0.6, P(PASS) 0.1. B: r 0.35 to 0.65, P(PASS) 0.15.
- D: r 0.45 to 0.75, P(PASS) 0.3, P(UNDECIDED) 0.1: two turns of play
  turn most of a turn's consequences into material, which the head reads
  well late in a game, but part of what it reads is the turn's own dice,
  which the adjusted truth removes.
- Every grader reads higher against the raw outcomes than against the
  adjusted ones, by 0.05 to 0.15.
- Both arms pass the barrier; the selected configurations use the
  adjustment, lam 0.25 to 0.5 and a rank weight of 10.
- The lucks explain 0.1 to 0.3 of the outcome variance, the candidate
  turn's own a quarter to a half of that.

## Consequences

A PASS by A or B builds the pre-graded teacher with that head; a PASS
by D alone builds it with truncated rollouts; the teacher gets its own
pre-registration and an 800-game match against `obs8`. A head is fitted
on full-precision tokens and a teacher would read them through the
inference server's bf16 path, so before a teacher uses a head, its
grades on the validation positions are re-read through that path and
compared. All FAIL with
the barrier passed proposes a fine-tuned trunk on the better labels;
the barrier missed says the labels or the data do not carry the signal
at this size, and the turn-search route is re-priced before anything
else.

## Cost

One RTX 4090 host with 64 cores at about $0.48 per hour; the balance
is checked before renting ($27.52 on 2026-09-25).
- Validation: about 28,000 playouts at the 2026-09-23 rate of 3.5 per
  second, 2.2 h.
- Training data: 10,484 fit and choosing positions x up to 4 candidates
  x 1 playout, plus 1,496 proxy positions x up to 4 x 2, about 54,000
  playouts, 4.3 h. The readings add a replay of each playout, 0.4 s
  for a median playout on the laptop against several seconds of play:
  about 5-10%.
- Bring-up, tests, features and fits (24 configurations per arm): 0.6 h.

About 7.1 box-hours, $3.4. scripts/turn_value_box.sh runs the stages in
that order, uploads every record as it goes, resumes after a stop on the
same box or a new one, and stops the instance at the end. The two long
stages are cut at twice their estimates (12.8 h together) and every
other step has its own cap: all caps at once come to about 20
box-hours, $9.5.

## Measured (2026-09-26)

Box 52605483 (RTX 4090, 56 jobs) on the branch's code as staged
(`exp/turn-value` at ce7621b, version 0.6.0, `OBSERVATION_EPOCH` 8, the
reference's own), from about 14:40 on 2026-09-25 to 00:20 UTC: about 9.7
box-hours, $6.1 at $0.628 per hour. It stopped itself and was destroyed
after its records were pulled (training/metrics/turn_value_20260925/:
verdict.json and verdict.md are the record; the playouts, the caches and
the fitted arms stay on the model host, tier-b/turn_value_20260925/).

| grader | corrected r | SE | reliability | against raw outcomes | verdict |
|---|---|---|---|---|---|
| A, linear | 0.274 | 0.041 | 0.577 | 0.298 +- 0.042 | FAIL |
| B, head | 0.269 | 0.045 | 0.577 | 0.304 +- 0.046 | FAIL |
| D, rollout (read 3, playouts 1-8) | 0.422 | 0.051 | 0.511 | 0.449 +- 0.051 | FAIL |
| `obs8`'s head before the end_turn | 0.220 | 0.046 | 0.577 | 0.249 +- 0.046 | reported |
| `obs8`'s head after the turn | 0.226 | 0.096 | 0.577 | 0.272 +- 0.091 | reported |
| HP margin after the turn | 0.397 | 0.088 | 0.577 | 0.471 +- 0.077 | reported |
| rollout, 4 playouts, read 3 | 0.413 | 0.050 | 0.545 | 0.442 +- 0.050 | reported |
| rollout, 8 playouts, read 1 | 0.266 | 0.052 | 0.511 | 0.311 +- 0.054 | reported |
| rollout, 8 playouts, read 7 | 0.532 | 0.050 | 0.511 | 0.588 +- 0.050 | reported |

199 of the 200 validation positions carry the verdict: 5 candidate
turns that ended the game were left out, and one position kept fewer
than two candidates. On the proxy games both arms
pass the crash barrier (observed within-position correlation A 0.072 +-
0.016, B 0.099 +- 0.017), where `obs8`'s head does not (0.028 +- 0.016).
Selected on the stop split: A with lam 0.25, rank weight 10, ridge 10;
B with lam 0.75, rank weight 0, epoch 18; neither with the luck
adjustment. The lucks explain 0.072 of the outcome variance over 36,209
fit playouts.

**Verdict: every judged grader FAILS, with the barrier passed.** By the
rule's consequences this proposes a fine-tuned trunk on the better
labels; nothing here is a teacher.

Against the predictions: `obs8`'s head read 0.220, inside its 0.2-0.45;
A (0.274) and B (0.269) fell below their ranges (0.3-0.6, 0.35-0.65), D
(0.422) below its 0.45-0.75; every grader read higher against the raw
outcomes, by 0.02 to 0.07 (predicted 0.05-0.15); both arms passed the
barrier, as predicted, but the selected configurations did not use the
adjustment and B's rank weight was 0 (predicted: adjusted, lam 0.25-0.5,
rank weight 10); the lucks explained 0.07 of the variance (predicted
0.1-0.3).

Read, not judged (post hoc, for the next design): the HP margin after the
turn ranks candidates better than both fitted arms and than `obs8`'s
head; a rollout read deeper (read 7) ranks better than the pre-registered
read 3 (0.532 against 0.422 on the same 8 playouts), so the rollout's
value grows with its horizon at this spread. On the validation positions
the screen's rate of gaps of 0.25 or more equals its permutation null
(57 of 200 against 0.282, validation_summary.md): at 28 playouts the
large gaps between human-position candidates are noise, and the
reliability of 0.58 bounds what any grader can show here.

Caveat: 38 of the 200 validation positions come from maps with a
null-controller third side (Caves of the Basilisk 32, Sullas Ruins 3,
Silverhead Crossing 3). The staged code predates 0.7.7, so their playouts
gave that side an empty turn every round and every encoding read the
mover's own faction as the enemy's there, as `obs8` was trained to read
it. The empty turns change no unit on the statue maps (measured
2026-09-26); on Silverhead their only effect is the tentacle's rest heal.
The fit's positions come from `obs8`'s own games, built with two sides.

## Decision record

- Rejected: the pass bar of the 2026-09-23 pre-grader check (within-
  position residual SD <= 0.2 on the confirmation's pairs, every gap of
  0.25 or more ranked above its base), because at about 16 pairs a
  grader at the design's own bar fails about half the time (simulated
  400 times: PASS 0.14, FAIL 0.47) and a clearly weak one fails 0.69:
  a FAIL would not tell them apart.
- Rejected: a fixed rank weight of 1, because level plus ranking at
  weight 1 barely differs from plain squared error, and when the
  within-position signal lies in token directions the level does not
  use it recovers 0.20 of a recoverable 1.0 (simulated); the weight is
  chosen on the choosing games.
- Rejected: common random numbers across candidates' playouts, because
  after two turns diverge they share a median of 0 fights
  (docs/selfplay_redesign_20260904.md).
- Rejected: judging against the raw outcomes, because a grader that
  reads the realized material of a candidate turn earns correlation
  from that turn's own dice, which a teacher cannot use.
- Rejected: choosing configurations by within-position squared error,
  because it favours a well-scaled worse ranking (a correlation of 0.60
  at the right scale beats 0.65 at 1.5 times the spread), where the
  verdict is scale-free.
- Rejected: the bootstrap SD as the standard error, because resamples
  whose reliability falls near zero give it a long tail that overstated
  the SE of a good grader and left it UNDECIDED.
