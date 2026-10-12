# Pre-registration: the look-ahead's material evaluator, fixed for gold and exposure (2026-10-12)

Written by the project lead before any code or box for it, on the user's
direction the same day: the exposure term is a fix to the evaluator, done
now.

## The two faults it fixes

The look-ahead player with the material evaluator played level with
`parity3` (-8.7 +- 12.3 Elo, docs/lookahead_material_gate_prereg_20261009.md
"Measured"), and its telemetry named the faults:

- **Recruiting is free.** The evaluator read the HP margin right after the
  action; a recruit adds its hit points and the gold it cost is invisible,
  so 5,817 of its 10,632 changed decisions became recruits.
- **Exposure is invisible.** A move or an attack that leaves a unit where
  the opponent kills it next turn reads as well as a safe one.

## The fixed evaluator

v(s) = tanh(M(s) / scale), from the deciding side's view, in the world it
expands in (the observed world: no unit it cannot see):

- **material in gold:** each unit's cost x its hit points over its maximum,
  own minus the opponent's, plus own gold minus the opponent's (the
  opponent's set as the observed world sets it); a recruit moves gold into
  a unit at its cost, so it reads about neutral when made;
- **minus the exposure:** the expected material the opponent's visible
  units can take on its next turn: every attack an opponent unit can make on
  an own unit it can reach (the core's reach, the exact outcome enumeration
  of `tools/combat_outcomes.py`), each attack's expected loss in the same
  gold units, a unit's deaths counted once, the attacks assigned greedily
  (each opponent unit to its best target, best first, a target's expected
  loss capped at its value), every opponent unit moving with its full
  movement. Every candidate is read on its own afterstate before any
  effect of the opponent's turn start: end_turn changes nothing on the
  board until that turn starts, so its candidate is read on the board as
  it stands, where the opponent's income, healing and refreshed units
  would otherwise count against end_turn and against no other candidate.

scale: the median absolute M over the first 50 timed decisions, fixed in the
configuration before the match.

## The gate

The look-ahead player (k 8, c 1, sigma 0.1, all kinds, the observed world,
as the material gate) on `own1` at 64 slots with the fixed evaluator,
against `own1` at the reference decode, PURE, 800 decisive games, seed base
160000. The same player with the old material evaluator is not replayed:
its gate is on record against `parity3`.

## Predictions (the lead's)

+5 Elo (-20 to +30); its changed decisions turning to recruits in under 15%
of cases (55% for the old evaluator).

## Readings

- **Teacher:** +25 or more: confirmed on fresh seeds, then distilled through
  the own-trajectory pipeline (pre-registered then).
- **Neutral or worse:** the one-step look-ahead with an exact evaluator
  does not beat the policy; recorded, and the evaluator stays for the
  tactical engine's design discussion.

## Cost

The exposure term's enumeration on top of the material look-ahead's 63 ms a
decision: an estimated 100-200 ms a decision, measured on the box first
(50 timed decisions); the match 25-45 minutes on a 20-worker 4090 box,
about $0.3-0.5.

## Decision record

- Revised 2026-10-12, before any run, from the implementation's timing: the
  end_turn candidate was read after the opponent's turn start, so its
  income and healing moved end_turn's M by -12 to +7 gold against a scale
  near 20 and flipped 4 of 6-7 end_turn choices to moves in two timing
  runs; every candidate is now read before that turn start.
