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
