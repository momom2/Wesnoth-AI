# Pre-registration: the hex's terrain as its full set (2026-09-19)

Written before the box is rented. The code shipped 2026-09-19: the
terrain database's aliases give every hex code its terrain SET
(`tools/terrain_resolver.terrain_members`, a forested hill is HILLS
and FOREST, a ford FLAT and SHALLOWWATER, an engine-defined full code
such as `Mm^Xm` its own list); the map parse, the scenario morphs and
the live-Wesnoth converter carry it on `Hex.terrain_mask`; behind the
checkpoint flag `terrain_multi_hot` the encoder's hex stream carries
the mask and the terrain term of the hex token is the sum of the
table's rows for the set bits (`GameStateEncoder.terrain_tokens`). The
flag is on for a fresh network, a checkpoint's own setting on load,
absent for every checkpoint before today, so the reference player's
observations are bit for bit what they were (tests/
test_terrain_multi_hot.py). It travels on the pre-encoder's
fingerprint, the checkpoint's struct flags, the inference blueprint,
the server hello and the actor pool's PLAY tuple, like the fog gate.

## Question

The one-class view labelled 1,356 of the Ladder pool's 1,572
forest-overlay hexes (86%) as something other than forest, 1,290 of
them as flat (`training/metrics/bench_pipeline/hide_cover_20260913/
census_terrain_mask_20260919.json`; the mask leaves 0 of 1,572 without
the forest bit). Nothing else in the observation carries a hex's
defense or movement class. Does an imitation product that sees the
set play stronger than the reference, at the same recipe and pass?

## Estimand

- Arm: the reference player's recipe from scratch with
  `--terrain-multi-hot`, everything else `scripts/seed2_relset_box.sh`
  to the letter: the relevant-set basis, configs/imitation.json,
  batch 64, lr 1e-4, cosine over 4 epochs stopped after one pass, the
  deduplicated corpus with the manifest split, fog gate on, pre-encoded
  records, run seed 20260909, arch 384/8/12/1536.
- Opponent: the reference player `relset`
  (`tier-b/seed2_relset_20260911/arm_epoch0.pt`), the same recipe at
  the same pass in the one-class view.
- Match: PURE `raw:t0`, sides alternated, Ladder maps, the recommended
  path (persistent workers, shared inference, 20 workers on a 4090),
  max 200 turns, each side served in its own encoding by its own
  inference server, seed base 61000 (disjoint from every earlier
  match), decisive results bought to 800 (`--games 800
  --max-extra-games 1500`).
- Headline: the arm's decisive-game score p with its standard error
  (0.018 at 800) and the Elo it implies; secondary: the capped
  fraction per side, the same read with capped games scored 0.5, the
  holdout probe at the end of the pass (CE, actor top-1, masked target
  CE, value AUC; proxies, not verdicts) and the per-phase value AUC.

Held fixed: the reference checkpoint, bf16 packed serving on cuda,
combat-oracle alphas 0, the joint-prior argmax, one result directory
per pair.

## Bars

- Crash barrier 1: `tests/test_terrain_multi_hot.py` and the Rust
  encode parity under the flag (`tests/test_rust_encode_raw.py`,
  parametrized today) run on the box before the pre-encoding; a
  failure does not stop the run but the verdict carries it and no
  number is quoted until it is understood.
- Crash barrier 2: the pass trains within 1% of the twin's 2,825,379
  pairs (the recipe's count; seed2's own run was 12% short for the
  out-of-memory drop found 2026-09-11). Outside it the recipe drifted
  and the match is a different comparison.
- Kill: p <= 0.50 at 800 decisive. The set is not what limits the
  imitation product at this recipe; the flag stays the fresh-network
  default (it costs nothing and is the engine's own view) and the arm
  is not a reference candidate.
- Pass: p >= 0.535 (428 of 800). The arm becomes the candidate
  reference player, pending the user's ruling and a self-pin; the
  number is quoted only after 800 decisive on this seed set, never
  chained onto an earlier match.

## Predictions

p = 0.53 (range 0.46-0.60), about +20 Elo; P(pass) 0.35. Forests are
7.6% of the pool's playable hexes and the units that live on them
(elvish 60-70% defense) are a third of the Ladder factions' rosters,
but a one-pass imitation product may read the hex from the units'
behaviour around it as much as from its label. The holdout CE moves
by less than 0.03 either way (it is not a strength proxy:
docs/model_cost_study_20260905.md 7).

## Cost

On a 4090 host: bring-up and wheel 5-10 minutes; pre-encoding at 30
workers 20-40 minutes; one pass of 2.83M pairs at the batched flow's
rate about 3 hours (the twin ran 138 pairs/s before the 1.86x batched
flow); the per-phase evaluation 5 minutes; the 800-decisive match
15-30 minutes; two 40-game self-timings 3 minutes. About 4-5 box-hours,
$2.5-3.5 at $0.50-0.75 per hour. `scripts/terrain_arm_box.sh` runs it
end to end; every exit, clean or not, leaves ALL_DONE on HF so the
laptop's watcher pulls the records and destroys the box.
