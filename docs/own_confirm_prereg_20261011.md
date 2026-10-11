# Pre-registration: confirming round 1 of the memory trained on its own trajectories (2026-10-11)

Written by the project lead before any box for it is rented. Round 1
(docs/memory_own_trajectories_prereg_20261011.md "Measured") fine-tuned
`parity3` on 5,472 of its own recorded game-sides toward its memoryless
reading, mixed with the human corpus. Against `parity3` at 0 slots the tuned
checkpoint read -4.3 +- 12.3 Elo at 64 slots (M1) and +37.4 +- 12.3 at 0
slots (M2). M2 was predicted at 0; a gain of three standard errors from a
first measurement is confirmed on fresh seeds before anything is built on
it, and the tuned checkpoint has not played the reference itself.

## Matches

`scripts/own_confirm_box.sh`, records in HF `tier-b/own_confirm_20261011`.
The tuned checkpoint is HF `tier-b/memory_own_trajectories_20261011/own.pt`
(SHA-256 70925e77...7f80, checked on the box). PURE, both sides at
`raw:t0+eo-1.5`, the Ladder maps with fog and uniform factions, 800 decisive
games each, every game recorded whole:

| match | side A | side B | seed base |
|---|---|---|---|
| C1 | tuned, 0 slots | `parity3`, 0 slots | 109000 |
| C2 | tuned, 64 slots | `parity3`, 64 slots (the reference) | 110000 |
| C3 | tuned, 64 slots | tuned, 0 slots | 111000 |
| C4 | tuned, 64 slots | tuned, 64 slots | 112000 |

C4 is not read for strength: its games, with C2's and C3's 64-slot sides,
are the tuned player's own trajectories for a round 2.

## Predictions (the lead's)

C1 +30 Elo (+5 to +55). C2 +95 (+65 to +125). C3 -40 (-70 to -10). C4
within one standard error of 0.

## Readings

- **C1 confirms the gain** when it lies above 0 by two standard errors (about
  +25 or more). Then training on the network's own trajectories improved
  it, and round 2 (its own new trajectories, its new memoryless reading as
  the target) is pre-registered next. C1 within two standard errors of 0:
  round 1's M2 was partly luck; the memory's repair (M1) stands on its own
  and round 2 is weighed on that alone.
- **C2** is the tuned checkpoint against the reference, the number a
  reference change would rest on: above +25 (two standard errors), the
  tuned checkpoint at 64 slots is proposed to the user as the reference.
- **C3** says what the memory still costs within the tuned weights; it is
  the quantity round 2 targets.

## Cost

Bring-up and tests about 15 minutes, four matches of 11 to 13 minutes
(round 1's: 699 and 654 s for 800 games, none capped), the final upload a
few minutes: about 1.2 box-hours, $0.50-0.76 at $0.42-0.63/h; BOX_MAX_H 2.5.

## Measured (2026-10-11)

Box 55438108 (RTX 4090, EPYC 7702, $0.427/h), stage
`tier-b/staging/stage_20261011_confirm.tar.gz`, 19:49 to 20:45 UTC, about
$0.45; records in `training/metrics/own_confirm_20261011/`, games on HF
`tier-b/own_confirm_20261011/`. 800 decisive games each, at most 7 capped:

| match | Elo of side A | prediction |
|---|---|---|
| C1, tuned at 0 slots against `parity3` at 0 | +22.6 +- 12.3 | +30 |
| C2, tuned at 64 against `parity3` at 64 (the reference) | **+88.6 +- 12.7** | +95 |
| C3, tuned at 64 against tuned at 0 | -21.7 +- 12.3 | -40 |
| C4, tuned at 64 against itself | -6.9 +- 12.3 | within one SE of 0 |

- **C1 does not confirm the gain at its bar** (1.8 standard errors against
  the 2 required). Round 1's M2 read +37.4 +- 12.3 on other seeds; the two
  together are about +30 +- 9, a pooling not pre-registered. The direction
  holds; its size is uncertain.
- **C2 clears +25:** the tuned checkpoint at 64 slots beats the reference
  by 88.6 Elo, in line with the chain through `parity3` at 0 slots (-4.3 +
  105.5). It is adopted as the reference `own1` (lead's decision under the
  user's delegation of 2026-10-09; configs/reference_player.json).
- **C3:** within the tuned weights the memory costs about 22 Elo (it cost
  `parity3` 105.5): round 2's target.
- C4 sits within one standard error of 0. Its games and the 64-slot sides
  of C2 and C3 are 3,200 game-sides of the tuned player's own play.
