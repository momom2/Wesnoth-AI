# Pre-registration: `parity3`'s baselines in play (2026-10-08)

Written before any box for it is rented: run Q2 of BACKLOG.md's queue.

## Why

- **The self-pin.** Every self-pin on record is of a player without a
  memory, the last `obs8`'s (side B +8 +- 12 Elo, 2026-10-02). Here both
  sides carry a memory through one shared inference server: the self-pin
  shows whether the harness treats them alike, and sets the noise every
  later match against `parity3` is read against.
- **The memory's share.** At 64 slots `parity2` lost to itself at 0 slots by
  37 +- 12 Elo, and at 16 by 30 +- 12, although the memory lowered the
  holdout loss (docs/parity_memory_prereg_20260929.md "Measured, pass 2");
  the investigation (tag `archive/exp-memory-in-play`) found no mismatch
  between training and play. `parity3` is that recipe trained two more
  epochs and annealed (holdout policy loss at 64 slots 2.688 against 2.852);
  whether its memory costs strength in play is not measured.
- **The size.** Self-play keeps the memory, its size open (user ruling
  2026-10-05; docs/selfplay_program_20261008.md starts from `parity3`).
  Active slots enter the trunk as tokens: 16 slots are 48 fewer than 64.

## Matches

`scripts/parity3_baselines_box.sh`, records in HF
`tier-b/parity3_baselines_20261008`. `parity3` (configs/reference_player.json,
SHA-256 checked) against itself: PURE, both sides at `raw:t0+eo-1.5`, sides
alternated, the Ladder maps with fog and both factions drawn uniformly, max
200 turns, 800 decisive games (at most 1,500 replacements), 20 persistent
workers, one shared inference server. Every game is recorded whole, and each
match's games go to HF as one tarball.

| match | side A | side B | seed base |
|---|---|---|---|
| 1. self-pin | `parity3a`, 64 slots | `parity3b`, 64 slots | 100000 |
| 2. the memory's share | `parity3`, 64 slots | `parity3_slots0`, 0 slots | 101000 |
| 3. the memory's size | `parity3`, 64 slots | `parity3_slots16`, 16 slots | 102000 |

Seed base S plays seeds S to S+799, and slot i's replacements S + i +
k x 1,000,000: two matches share a seed only when their bases lie within 800
of each other modulo 1,000,000. Every seed base named in a branch or tag of
the repository (scanned 2026-10-09) is 90000 or below, and no committed game
file carries a seed from 100000 to 102999.

## Bars

- **Crash barrier:** the match path's tests pass on the box's own wheel. A
  match is read only with 800 decisive games, else recorded as cut.
- **Resolved:** p >= 0.535 or p <= 0.465 over the decisive games, about +-25
  Elo (two standard errors at 800), the parity-memory matches' bar.

## Predictions (the lead's prior, before the run)

Self-pin within +-25 Elo of 0; 64 against 0 slots between -45 and +10 Elo;
64 against 16 slots within +-25 Elo.

## What each reading changes

- **Self-pin** within +-25: one standard error, about 12 Elo at 800 decisive
  games, is the noise of every later match against `parity3`. Outside: the
  two memory players are not treated alike, and no match against `parity3`
  is read until the cause is found.
- **64 against 0** at -25 or below: the memory costs `parity3` strength too,
  and the lead puts to the user whether the reference, which every gate of
  the program plays, plays at 0 slots. Otherwise it stays at 64 slots.
- **64 against 16** within +-25: 16 slots play as well as 64 at fewer tokens,
  and the lead proposes 16 for the program; resolved either way, the
  stronger size. The size is the user's ruling.

## Cost

An RTX 4090 with at least 24 usable cores (the matches of 2026-10-02 and
2026-10-04 ran 20 workers on 31 EPYC cores). Bring-up and tests about 10
minutes; each match 15 to 17 minutes (`parity3` against `parity2`, 944 games
in 929 s; one memory checkpoint against itself, 903 and 999 s); the final
upload a few minutes. About 1.1 box-hours, $0.46-0.69 at $0.42-0.63 an hour;
1.75 hours and $0.74-1.10 if every match runs 1.8x slower, as identical
repeats have. `BOX_MAX_H` 2.

## Measured (2026-10-09)

Box 54996204 (RTX 4090, EPYC 7713, $0.438/h), stage
`tier-b/staging/stage_20261009_q2.tar.gz`; its final upload did not reach
HF and a restart was refused at entry, so the third match was played again
on box 55010053 (Ryzen Threadripper PRO, $0.415/h) from the same stage, the
first two fits restored from HF. Records in
`training/metrics/parity3_baselines_20261008/`, games on HF
`tier-b/parity3_baselines_20261008/`. 800 decisive games each, PURE:

| match | games | Elo of side B against side A |
|---|---|---|
| 1. self-pin, 64 against 64 slots | 1,020 | -2.6 +- 12.3 |
| 2. 64 slots against 0 slots | 810 | **+105.5 +- 12.8 for 0 slots** |
| 3. 64 slots against 16 slots | 1,021 | 0.0 +- 12.3 |

- **The self-pin** is within one standard error: the harness treats two
  memory players alike, and about 12 Elo is the noise of a match against
  `parity3`.
- **The memory costs `parity3` about 105 Elo in play** (the prediction was
  -45 to +10; `parity2`'s cost was 37 +- 12). By the reading pre-registered
  above, whether the reference plays at 0 slots goes to the user.
- **16 slots play as 64 do**: the cost does not depend on the size.

Read from the same records afterwards (not pre-registered): in match 2 the
memory side made 9.6 decisions per side-turn (median; mean 9.8) against 11.0
(mean 11.1) for the side without it, over the 810 games, and 10 of the 810
games reached a cap where a memory against itself capped about one game in
five (220 of 1,020 in the self-pin). The memory player acts less, and
acting more is the largest lever on record (the end_turn offset -1.5, about
+229 Elo).

About $0.6 over three boxes (a third rental never started its
container and was destroyed).
