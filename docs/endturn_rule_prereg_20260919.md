# Pre-registration: end_turn decided at the actor level (2026-09-19)

Test 1 of the training-signal panel
(docs/training_signal_panel_20260905.md, "Pre-registration draft:
test 1"), written out against today's reference player and eval path,
before any box is rented. The code shipped 2026-09-19: the two decode
rules live in `tools/raw_player.py` (`end_turn_rule="actor"`,
`end_turn_offset`), reach a match through `run_elo_batch
--raw-end-turn-a/-b` and `--raw-end-turn-offset-a/-b`, and carry their
own procedure tags (`raw:t0+endm`, `raw:t0+eo<x>`), so their games
never share an outdir with plain `raw:t0`. Both rules are exact
client-side operations on the joint priors, on the compact arrays of a
shared inference server as on the enumerated list
(tests/test_end_turn_rule.py).

## Question

Does the reference under-act because the joint argmax compares
end_turn's actor mass against four-way products, and does deciding
end_turn at the actor level win games?

## Estimand

- Player A: `raw:t0+endm`, the reference player (`relset`) at
  temperature 0, end_turn played only when its actor mass is at least
  the largest actor marginal (the sum of joint priors over one unit's
  or recruit slot's legal actions); otherwise the joint argmax among
  the non-end actions. Player B: `raw:t0`, the reference itself.
- Attribution arm: `raw:t0+eo<x>`, the reference with the end_turn
  actor logit offset x in {-0.75, -1.5}, everything else the joint
  argmax.
- Match: PURE, sides alternated, Ladder maps, the recommended path
  (persistent workers, shared inference, 20 workers on a 4090), max
  200 turns, seed bases disjoint from every earlier match (41000 for
  the screens, 42000 for the rule's match, 43000 for the attribution
  arm). Screen 40 games per arm; then decisive results bought to 800
  for the rule (`--games 800 --max-extra-games 500`), and for the
  best offset only if the rule passes.
- Headline: the rule's decisive-game score p with its standard error
  (0.018 at 800); secondary: the same read with capped games scored
  0.5; decisions per side-turn (forwards over turns, per player, the
  raw player makes one forward per decision) and the capped fraction
  per arm; the offset arm's p as the attribution number.

Held fixed: the reference checkpoint `relset.pt`, bf16 packed serving
on cuda, combat-oracle alphas 0, the joint-prior argmax for every
non-end decision, one result directory per procedure pair.

## Bars

- Kill 1 (screen, $0.02): the rule's decisions per side-turn are not
  at least 3% above `raw:t0`'s in the same games (the draft's 9.7
  against 9.4 was the seed's; the reference's own rate is read off the
  screen's B side). The rule does not fire.
- Kill 2: p <= 0.50 at 800 decisive.
- Barrier: a capped rate against `raw:t0` above the reference's own
  self-match rate (17 of 40 on 2026-09-13; re-read on the screen's
  pair) by more than 2 SE reads as a stall tilt, reported next to
  W-L, with the 0.5-scored read alongside.
- Pass: p >= 0.535 (428 of 800), then 800 decisive on a disjoint seed
  set before the number is quoted; the offset arm then runs to 800
  decisive. If the offset matches the rule within 1 SE, the lever is
  "act more" and the config scalar is the adopted form.

## Predictions (the panel's, restated before the run)

Decisions per side-turn up 10% (range +4% to +19%); rule p = 0.52
(range 0.46-0.58); P(pass) 0.3; the offset at -0.75 within 0.02 of the
rule; the rule's capped fraction against `raw:t0` 0.35 (range
0.2-0.45; the reference's own self-match rate as the null).

## Cost

On a single-tenant 4090 host of the 2026-09-18 class (a 40-game
`raw:t0` match in 24-25 s there, 42-74 s on the shared hosts):
bring-up about 5 minutes; three screens about 2 minutes; the 800-
decisive match about 1,300 games, 15-30 minutes; the attribution arm
the same, conditional. About $0.30-0.60 at $0.40-0.60 per hour, one
box-hour at most. `scripts/endturn_rule_box.sh` runs it end to end
and writes the verdict under these bars.
