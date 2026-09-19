# Pre-registration: do the day's two levers add? (2026-09-19)

Written after both results of the day and before this box is rented.
Two levers beat the reference player `relset` at `raw:t0` on
2026-09-19, each measured alone over 800 decisive games:

- the end_turn logit offset -1.5, a decode scalar: p 0.789 +- 0.014,
  about +229 Elo (docs/endturn_rule_prereg_20260919.md,
  docs/endturn_offset_sweep_prereg_20260919.md);
- the terrain-set arm, a checkpoint trained with `terrain_multi_hot`:
  p 0.562 +- 0.018, +44 +- 12 Elo (docs/terrain_multi_hot_prereg_20260919.md).

Whether they compose is the question the user's two rulings need, and
the candidate reference checkpoint needs its own self-pin.

## Estimand

Three PURE matches, sides alternated, Ladder maps, the recommended
eval path, max 200 turns, decisive results bought to 800 (`--games
800 --max-extra-games 500`), seed bases disjoint from every earlier
match:

| match | A | B | seed base |
|---|---|---|---|
| composed against composed-decode reference | terrain_e1 at `raw:t0+eo-1.5` | relset at `raw:t0+eo-1.5` | 47000 |
| composed against today's reference | terrain_e1 at `raw:t0+eo-1.5` | relset at `raw:t0` | 48000 |
| the arm's self-pin | terrain_e1 at `raw:t0` | terrain_e1 at `raw:t0` | 49000 |

Each side is served in its own terrain view by its own inference
server. Headline per match: p over decisive games with its standard
error, the Elo it implies, the capped fraction and decisions per
side-turn.

## Reading

- Match 1 is the checkpoint lever under the adopted decode. If p is
  within 1 SE of 0.562, the levers are independent (the terrain set
  helps moves and attacks, the offset helps turn ends); if it falls
  to 0.50 +- 1 SE, the offset already captures what the arm knew; if
  it rises, the arm profits more from acting.
- Match 2 is the total on the table: the best player we can field
  today against the reference every strength claim is made against.
  Reported as the number, with no bar.
- Match 3: an asymmetry beyond 2 SE of zero blocks the arm as a
  reference candidate until understood.

## Predictions

Match 1: p 0.55 (range 0.49-0.61); match 2: p 0.81 (0.76-0.86); match
3: p 0.50 +- 0.02, capped fraction about 0.4 (the argmax stall).

## Cost

Three 800-decisive matches at 9-15 minutes each on the EPYC 7B13 box
class, bring-up and wheel about 8 minutes: about 50 box-minutes,
$0.60. `scripts/composed_levers_box.sh` runs it end to end and leaves
ALL_DONE on HF on every exit.

## Measured (2026-09-19 night, instance 51625245: an EPYC 7C13 host with an RTX 4090, $0.60/h)

Records under `training/metrics/bench_pipeline/composed_levers_20260919/`
(fits, readouts and logs; the game records stayed on the box).

| match | seed base | games (capped) | W-L | p over decisive | Elo of A | decisions per side-turn A / B | wall |
|---|---|---|---|---|---|---|---|
| terrain_e1 `raw:t0+eo-1.5` vs relset `raw:t0+eo-1.5` | 47000 | 803 (3) | 430-370 | 0.537 +- 0.018 | +26 +- 12 | 9.37 / 8.84 | 489 s |
| terrain_e1 `raw:t0+eo-1.5` vs relset `raw:t0` | 48000 | 835 (35) | 656-144 | 0.820 +- 0.014 | +263 +- 16 | 10.03 / 5.67 | 536 s |
| terrain_e1 `raw:t0` against itself | 49000 | 1029 (229) | 402-398 | 0.502 +- 0.018 | +2 +- 12 | 5.94 / 5.93 | 812 s |

Match 1: the checkpoint lever survives the decode at +26 +- 12 Elo,
2.1 SE from zero and 1.4 SE below its lone +44: consistent with two
independent levers within noise, and with a partial overlap; the
800-game match cannot separate the two, and it does not need to for
the rulings (the arm is ahead either way). Match 2: the composed
player takes p 0.820 (predicted 0.81, range 0.76-0.86) against the
reference every strength claim is made against, about +263 Elo, with
4% of games capped against the reference's own 40%. Match 3: no
asymmetry (+2 +- 12 Elo for side A, 0.2 SE from zero); the arm caps
22% of its self-match games against the reference's 40% (predicted
about 0.4), so the terrain set also shortens the argmax stall. The
box ran 48 minutes, about $0.50.
