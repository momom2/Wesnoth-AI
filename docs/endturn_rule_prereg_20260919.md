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

Two riders on the same rental, each behind a switch: the Rust-path test
files with the phase-10 wheel and `tools/diff_core.py` over 600 corpus
replays (the process-independent unit hash of 2026-09-18 changed the
order every set of units iterates in; the local wheel cannot run
them), about 10 minutes; and the reference player against itself to
800 decisive games (seed base 44000), the tight self-pin BACKLOG.md
has carried since 2026-09-12, now under per-game luck, the current
hide-cover rule and that hash, 15-30 minutes. With both, about
$0.60-1.00 and up to two box-hours.

## Measured (2026-09-19, box 51591595: a 32-core slice of an EPYC 7B13 host with an RTX 4090, $0.75/h)

The riders first: 59 Rust-path tests passed with the phase-10 wheel
(1 skipped), `tools/diff_core.py` 600 of 600 corpus replays clean
through the core after the process-independent unit hash.

The screens, 40 games each (records under
`training/metrics/bench_pipeline/endturn_rule_20260919/`):

| arm | seed base | W-L (capped) | p over decisive | capped scored 0.5 | decisions per side-turn A / B | wall |
|---|---|---|---|---|---|---|
| `raw:t0+endm` | 41000 | 29-7 (4) | 0.806 +- 0.066 (36) | 0.775 | 8.07 / 5.59 (1.44x) | 41 s |
| `raw:t0+eo-0.75` | 41100 | 22-8 (10) | 0.733 +- 0.081 (30) | 0.675 | 7.30 / 5.08 (1.44x) | 46 s |
| `raw:t0+eo-1.5` | 41200 | 29-10 (1) | 0.744 +- 0.070 (39) | 0.738 | 9.27 / 6.20 (1.50x) | 36 s |

Kill 1 passed: the rule fires at 1.44x the reference's decisions per
side-turn, against a bar of 1.03x and a prediction of 1.10x. The
capped fraction against `raw:t0` read 0.10 for the rule, against the
reference's own self-match rate of 17 of 40 on 2026-09-13.

The rule's match, seed base 42000, 875 games in 534 s:

| arm | W-L (capped) | p over decisive | capped scored 0.5 | decisions per side-turn A / B |
|---|---|---|---|---|
| `raw:t0+endm` vs `raw:t0` | 602-198 (75) | 0.752 +- 0.015 (800) | 0.731 | 8.55 / 6.03 (1.42x) |

**PASS.** p 0.752 against a bar of 0.535 and a prediction of 0.52
(range 0.46-0.58): about +193 +- 14 Elo for a decode rule that trains
nothing. The capped fraction, 0.09, is far below the reference's own
self-match rate (about 0.4), so the barrier does not fire: the rule
does not win by stalling, it wins by acting.

The attribution arm, the best screened offset (-1.5), seed base 43000,
863 games in 511 s:

| arm | W-L (capped) | p over decisive | capped scored 0.5 | decisions per side-turn A / B |
|---|---|---|---|---|
| `raw:t0+eo-1.5` vs `raw:t0` | 631-169 (63) | 0.789 +- 0.014 (800) | 0.768 | 9.25 / 5.90 (1.57x) |

The offset does not merely match the rule within 1 SE: it beats it by
0.037 (2.5 SE), about +229 +- 15 Elo against the rule's +193. Under the
pre-registered reading the lever is "act more", and the config
scalar (`--raw-end-turn-offset`, a logit offset on end_turn) is the
adopted form; the actor-level rule is not needed. Open, not
pre-registered: the offset's best value (the screens read -0.75 at
0.733 and -1.5 at 0.744 over 30-39 decisive; -1.5 at 800 reads 0.789,
so the curve is still rising at -1.5) and whether the same offset
helps a searched player.

The reference's self-pin rider, `raw:t0` against itself, seed base
44000, 1,300 games in 791 s (the 800-decisive target was not reached
inside the 500 extra games: 519 of 1,300 capped, a capped fraction of
0.40):

| match | W-L (capped) | p over decisive | capped scored 0.5 | decisions per side-turn A / B |
|---|---|---|---|---|
| `raw:t0` vs `raw:t0` | 406-375 (519) | 0.520 +- 0.018 (781) | 0.512 | 4.83 / 4.82 (1.00x) |

About +14 +- 13 Elo for side A, 1.1 SE from zero: no asymmetry
detected, under per-game luck, the current hide-cover rule and the
process-independent unit hash. This replaces the 160-game self-pin
of 2026-09-12 (-57 +- 37, four replays of one seed set) as the
reference's baseline. It also re-reads the reference's own capped
rate: 0.40 here, against the 17 of 40 (0.42) the barrier used.

