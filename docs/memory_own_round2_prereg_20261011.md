# Pre-registration: training on its own trajectories, round 2 (2026-10-11)

Written by the project lead before any code or box for it. Round 1
(docs/memory_own_trajectories_prereg_20261011.md, confirmed in
docs/own_confirm_prereg_20261011.md) fine-tuned `parity3` on its own
recorded game-sides toward its memoryless reading, mixed with the human
corpus. The result, `own1`, is the reference: +88.6 +- 12.7 Elo over
`parity3`; within its own weights the memory costs about 22 Elo (-21.7 +-
12.3); its 0-slot reading beats `parity3`'s by +22.6 +- 12.3 and +37.4 +-
12.3 on two seed sets.

## Question

Does a second round on `own1`'s own trajectories improve the network again?
The reading that matters is the memoryless one (R2): if training on its own
games keeps improving the policy itself, the loop is a self-improvement
operator at a few dollars a round and is iterated; if not, it has done what
a memoryless teacher can do, and the next teacher must be better than the
policy.

## The method

Round 1's method and code (`exp/memory-own-trajectories`), with:

- **Own trajectories:** `own1`'s game-sides at 64 slots only (on-policy for
  the student), selected by its checkpoint's SHA-256: the 64-slot sides of
  the confirmation matches C2, C3 and C4 (HF `tier-b/own_confirm_20261011/`,
  about 3,200 game-sides), and a new match of `own1` at 64 slots against
  itself played on the box first (800 decisive games, seed base 113000,
  about 1,600 game-sides). Round 1's `parity3` trajectories are left out:
  the replay and the segmentation of a trajectory read the policy that
  played it, and a pooled round would need both policies.
- **Targets:** `own1`'s raw distribution at 0 slots, top 16 renormalised.
- **Init and schedule:** from `own1`, the same pass as round 1 (1 million
  own and 1 million human positions, the learning rate at 2.8e-5, linear to
  0 over the second half, own policy weight 1).
- **Barriers:** round 1's, with `own1` in place of `parity3`.

## Matches

PURE, both sides at `raw:t0+eo-1.5`, 800 decisive games each, against
`own1`:

| match | side A | side B | seed base |
|---|---|---|---|
| R1 | `own2`, 64 slots | `own1`, 64 slots (the reference) | 114000 |
| R2 | `own2`, 0 slots | `own1`, 0 slots | 115000 |
| R3 | `own2`, 64 slots | `own2`, 0 slots | 116000 |

## Predictions (the lead's)

R1 +15 Elo (-10 to +40). R2 +10 (-15 to +35). R3 -10 (-35 to +15).

## Readings

- **Compounding:** R2 at +25 or more (two standard errors). Training on its
  own games improves the policy itself again; round 3 is pre-registered
  next on `own2`'s trajectories, and the loop is priced as the program's
  engine.
- **Saturated:** R2 within two standard errors of 0. The memoryless target
  has given what it can; no further round of this form. R1 and R3 say what
  the round still did for the memory.
- **Regressed:** R2 at -25 or below: the round hurt the policy; the cause is
  found before anything else is trained.
- **R1 at +25 or more:** `own2` at 64 slots becomes the reference.

## Cost

Bring-up and tests about 20 minutes, the generation match about 15, the
corpus and pre-encoding 30, the candidates and targets about 50 (round 1's
targets ran at 390 positions a second), the pass about 2.6 hours (217
positions a second), three matches about 40 minutes: about 5.2 box-hours,
$2.2-3.3 at $0.42-0.63/h. BOX_MAX_H 8.

## Measured (2026-10-12)

Box 55454553 (RTX 4090, EPYC 7702, $0.463/h), stage
`tier-b/staging/stage_20261012_r2b.tar.gz` (commit bd5f33d of
`exp/memory-own-trajectories`), 22:11 to 03:30 UTC, about $2.5; a first box
(55449678) stopped at its test step on a test that read round 1's preset
in an environment holding round 2's variables, fixed with a test that fails
without the fix. Records in `training/metrics/memory_own_round2_20261011/`,
`own2` and the games on HF `tier-b/memory_own_round2_20261011/`.

**Data and barriers.** The generation match, `own1` against itself, read
-3.5 +- 12.3 Elo; its 1,600 game-sides and the confirmation's 64-slot sides
were the round's own trajectories; the replay reproduced 98.7% of the
decisions in turns 6 and later. Holdout imitation loss, `own1` against
`own2`: 2.812 against 2.805 at 0 slots, 2.709 against 2.705 at 64 slots.
Agreement with the target at the decode on held-out own game-sides: `own2`
0.910, `own1` 0.885. Every barrier passed.

**Matches**, 800 decisive games each:

| match | Elo of side A | prediction |
|---|---|---|
| R1, `own2` at 64 slots against `own1` at 64 | +16.5 +- 12.3 | +15 |
| R2, `own2` at 0 slots against `own1` at 0 | -6.1 +- 12.3 | +10 |
| R3, `own2` at 64 slots against `own2` at 0 | +13.0 +- 12.3 | -10 |

**Reading: Saturated.** R2 lies within two standard errors of 0: a second
round toward the memoryless reading did not improve the policy itself, so no
further round of this form is run. What the round did for the memory: within
`own2`'s weights the memory no longer costs strength (+13.0 +- 12.3 for 64
slots against 0, where it cost `parity3` 105.5 and `own1` 21.7). R1 is short
of +25, so `own2` is not the reference; `own1` stays. The next round needs a
teacher better than the policy's own reading.
