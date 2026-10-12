# Pre-registration: the reference's decode knobs, tuned against matches (2026-10-12)

Written by the project lead before any code or box for it; approved in
principle by the user the same day ("Do (1)").

## Why

The largest gain on record came from one decode scalar, not from training:
the end_turn logit offset -1.5, about +229 Elo for `relset`
(docs/endturn_rule_prereg_20260919.md), which said the network ranks the
actions within a kind well and calibrates between kinds badly. The
memory's tempo fault in `parity3` was the same symptom. Nothing else of the
kind has been tried: the offset is the only knob, tuned once on a memoryless
player, and `own1` plays under it unchanged. Tuning a few such knobs against
matches needs no value function, no teacher and no gradient, and each knob
is a readable, config-held part of the player (principle 3).

## The knobs

Added to the raw player's decode, each a logit offset on the compact
distribution before the joint argmax, every one recorded in the procedure
tag and the result:

- **e**, the end_turn offset (today -1.5), range -3.5 to 0;
- **a**, an offset on every attack, range -1.0 to +1.5;
- **r**, an offset on every recruit, range -1.0 to +1.0;
- **t**, an attack's time-of-day slope: the attack's offset gains t x the
  attacker's time-of-day damage modifier at the attack's hex (-0.25 to
  +0.25), range 0 to 4.

A move is the zero of the scale (offsets are defined up to a common shift).

## The search

Successive halving (Karnin, Koren and Somekh, ICML 2013; Jamieson and
Talwalkar, AISTATS 2016) over 24 knob vectors: today's decode (e -1.5, the
rest 0) and 23 drawn by a Latin hypercube over the ranges with a fixed seed.
Each vector plays `own1` at 64 slots under that decode against the
reference (`own1` at its own decode), PURE, the Ladder pool with uniform
factions, sides alternated:

1. 24 vectors x 100 decisive games; the 8 with the most wins go on;
2. 8 x 300 decisive games more (seeds disjoint from round 1); the 2 with
   the most wins over all their games go on;
3. 2 x 800 decisive games more; the better over all its games is the
   winner.

Then the winner against the reference, 800 decisive games on a seed base
no stage used (the confirmation, the only number read as a strength
claim). Seed bases, 1,000 apart so no two collide (a base plays S to S+799,
its replacements S + i + k x 1,000,000): stage 1's vector i at 200000 +
1000 i (i < 24), stage 2's at 230000 + 1000 j (j < 8), stage 3's at 240000
and 241000, the confirmation at 250000.

## Predictions (the lead's)

The confirmation +40 Elo (0 to +90). The winner's end_turn offset between
-3.0 and -1.5; its attack offset above 0.

## Readings

- **Gain:** the confirmation at +25 or more (two standard errors). The
  winner's knobs become the reference's decode (the checkpoint unchanged),
  and the knobs stay in config for the next checkpoint's search.
- **Flat:** within two standard errors of 0: today's decode is near the
  optimum of these knobs; recorded so they are not re-tuned unasked.
- Today's decode in stage 1 is the null arm: its 100 games against the
  reference must lie within two standard errors of 50%, or the stage's
  harness is suspect before anything is read.

## Cost

24 x 100 + 8 x 300 + 2 x 800 + 800 decisive games, about 7,200 games at
the measured 1.0-1.2 s a game on a 20-worker 4090 box: about 2.3 hours with
bring-up, $1.0-1.5. BOX_MAX_H 4.

## Measured (2026-10-12)

Box 55542194 (RTX 4090, EPYC 7742, $0.371/h), stage
`tier-b/staging/stage_20261012_knobs.tar.gz` (commit f4b7f6d of
`feature/decode-knobs`), 11:04 to 13:34 UTC, about $0.95; records in
`training/metrics/decode_knobs_20261012/` (`search.jsonl`, `plan.txt`,
`winner_decode.json`, `confirm.fit.json`), games on HF
`tier-b/decode_knobs_20261012/`. 34 matches, none failed, no game capped.

**Reading: Flat.** The search's winner, kv15 (e -3.0, a -0.59, r -0.01,
t 2.53), against `own1` at its own decode on the confirmation's seeds:
**+1.7 +- 12.3 Elo** (402-398 over 800 decisive games; predicted +40, 0 to
+90). The reference's decode stays at `raw:t0+eo-1.5`, and these four
knobs are not re-tuned unasked.

The search, stage by stage:

- **Stage 1:** the 24 vectors won 1,103 of 2,400 games (46.0%); the null
  arm, today's decode, 50-50, so the harness check passes. The 8 kept:
  kv10 60, kv09 56, kv07 53, kv15 53, kv19 53, kv00 50, kv17 50, kv03 48.
- **Stage 2:** over 300 games more, kv15 174, kv10 151, kv03 149, kv07
  146, kv17 143, kv09 140, kv00 137, kv19 133; the null arm 187 of 400
  over both stages (-22.6 +- 17.4 Elo, within two standard errors).
- **Stage 3:** over 800 games more, kv15 441, kv10 405; over all three
  stages kv15 668 of 1,200, kv10 616.

kv15's 800 stage-3 games alone read +35.7 +- 12.3; its confirmation, on
seeds no stage used, +1.7 +- 12.3, about two standard errors apart. The
two pooled read +18.7 +- 8.7, which is not a pre-registered estimand: it
bounds what these knobs could still be worth, below the +25 bar.

Predictions against the outcome: the winner's end_turn offset -3.0
(predicted -3.0 to -1.5, at its edge); its attack offset -0.59 (predicted
above 0: wrong). A least-squares fit of stage 1's win rates on the four
knobs, standardized: the attack offset -3.3 +- 1.3 points of win rate per
standard deviation of the knob (0.71), the other three within 1.3
standard errors of 0. The five vectors with an attack offset above +1.0
won 191 of 500 stage-1 games.

**What it says.** Around today's decode, offsets between the kinds of
actions find nothing further on `own1`: the end_turn offset's gain on
`relset` was a single correction of the imitation network's tempo, and the
network's other between-kind calibrations are near what such offsets can
reach. Adding attacks costs strength.
