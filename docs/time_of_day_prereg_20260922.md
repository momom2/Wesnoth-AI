# Pre-registration: the time of day in the observation (2026-09-22)

Written before the box is rented, on the template of
docs/terrain_multi_hot_prereg_20260919.md.

## Question

Until 2026-09-22 the observation carried no time-of-day signal at all.
`GLOBAL_FEAT_DIM` was 6 -- turn number, side to move, our gold, our
income, our villages, theirs -- and neither the unit nor the hex
features carried the time of day or the lawful bonus, while combat
applied that bonus (`wesnoth_ai/combat.py`,
`rust/wesnoth_core/src/combat.rs` `combat_modifier`). A lawful unit's
damage swings by half between its best hour and its worst, and the
policy could not observe which it was in.

The turn number is not a stand-in. Two pool scenarios declare
`current_time=5` and start at second watch (Fallenstar Lake, Ruined
Passage), four minis roll `random_start_time=yes`, and two carry
`[time_area]` zones whose hexes run a cycle of their own. On those
maps the same turn number means different phases.

Does an imitation product that sees the time of day play stronger than
the reference, at the same recipe and pass?

## What is in the observation now

Two global features, not one:

| slot | value |
|---|---|
| 6 | this turn's lawful bonus / 25 |
| 7 | next turn's lawful bonus / 25 |

Both, because the bonus alone cannot tell dawn from dusk. Each carries
a zero bonus, and they are strategically opposite: at dawn the lawful
side is about to get stronger, at dusk the chaotic side is. One float
cannot express that; the pair reads 0/+1 against 0/-1.

Deliberately NOT in this arm, and queued as its own factor if this
passes: the per-hex bonus. Time areas make a hex's bonus differ from
the board's on two pool maps, and lit terrain overlays (campfire and
similar) clamp it on any map. `NUM_HEX_DYNAMIC_FLAGS` is built to grow
and `pad_legacy_encoder_state` already covers it, so that arm is a
separate one-factor change.

## Estimand

- Arm: the reference player's recipe from scratch with the new
  observation, everything else `scripts/terrain_arm_box.sh` to the
  letter: the relevant-set basis, `configs/imitation.json`, batch 64,
  lr 1e-4, cosine over 4 epochs stopped after one pass, the
  deduplicated corpus with the manifest split, fog gate on, terrain
  set on, pre-encoded records, run seed 20260909, arch 384/8/12/1536.
- Opponent: the reference player, `terrain` at `raw:t0+eo-1.5`
  (`configs/reference_player.json`), which observes six globals and
  loads through the zero pad, so it plays exactly the game it was
  trained on.
- Match: PURE, both sides at the reference decode, sides alternated,
  ladder maps, the recommended path (persistent workers, shared
  inference, 20 workers on a 4090), max 200 turns, each side served in
  its own encoding by its own inference server, seed base 62000
  (disjoint from every earlier match), decisive results bought to 800
  (`--games 800 --max-extra-games 1500`).
- Headline: the arm's decisive-game score p with its standard error
  (0.018 at 800) and the Elo it implies. Secondary: the capped
  fraction per side, the same read with capped games scored 0.5, the
  holdout probe at the end of the pass (CE, actor top-1, masked target
  CE, value AUC; proxies, never verdicts), and the per-phase value AUC.

Held fixed: the reference checkpoint, bf16 packed serving on cuda,
combat-oracle alphas 0, the joint-prior argmax, one result directory
per pair.

## Bars

- **Crash barrier, before the pre-encoding**: `tests/test_time_of_day_features.py`
  and the Rust encode parity (`tests/test_rust_encode_raw.py`) run on
  the box with the phase-11 wheel. The local suite cannot judge the
  Rust path: the laptop's wheel is phase 3, and the encoder now
  refuses a kernel below phase 11 and falls back to Python, so a local
  green run says nothing about the kernel. A parity failure stops the
  run before any training.
- **Kill**: p <= 0.50 against the reference at 800 decisive games. The
  feature is then not worth its two inputs at this recipe, and the
  per-hex arm does not follow.
- **Pass**: p >= 0.535, which is about +24 Elo, more than 1.9 standard
  errors. Then the per-hex arm is queued and this checkpoint is a
  candidate reference, pending its own self-pin.
- **Inconclusive**: between them. Recorded, and the per-hex arm
  decides whether the concept is worth a second look.
- **Barrier, not a verdict**: holdout CE more than 0.05 nat worse than
  the reference's on the same pairs suggests the recipe broke rather
  than the feature failing; investigate before reading the match.

## Prediction, before the run

p = 0.55 (range 0.50 to 0.62), so a pass is more likely than not. The
reasoning: the signal is cheap, it is genuinely absent today, and it
gates a mechanic worth 50% of a unit's damage. Against that, the
network may already infer the phase from the turn number on the 22 of
28 pool maps that start at dawn with a fixed cycle, which would leave
the feature buying little beyond the second-watch and random-start
maps. I hold a pass at about 0.6 probability.

Two specific expectations that will be checked whatever the headline
says: the arm should gain most on the maps where the turn number
misleads (Fallenstar Lake, Ruined Passage), and its decisions per
side-turn should not move much, since this is an observation change
rather than a decode change.

## Consequences for numbers

The observation changed, so `constants.OBSERVATION_EPOCH` is bumped
from 3 to 4: pre-encoded corpora and anchor caches from before this
refuse to load, and the corpus must be re-encoded on the box. Matches
run after this are **not** cross-build against earlier Elo, because
the reference loads through the zero pad and plays exactly as before;
what changed is that a new arm can see more.

## Cost

Pre-encoding the corpus and one imitation pass, as the terrain arm ran:
about 3 hours; the 800-decisive match about 18 minutes and $0.20; the
Rust wheel build and the parity tests about 10 minutes. Roughly $2.50
on a single-tenant 4090 host, one rental.
