# Pre-registration: the serve batch cap, and plan 1.3's 3,000 leaves per second (2026-09-20)

Written before the box is rented, after the user asked whether the
3,000-leaves-per-second-per-4090 target of docs/plan_20260904.md 1.3
is really out of reach.

## What the records say

The best reading is 2,634-2,914 saturated leaf evaluations per second
on the quiet Core Ultra 9 285K host of 2026-09-18 (docs/box_specs.md
"The graphed server on a quiet host"), 3-14% under the target. That
run's own record (`graphed_default_20260918/pool_eager_a.json`) says
`max_batch = 16`: every serve batch held at most one actor's 16-leaf
request. The device span there was 95% of the server's infer time at
0.46-0.54 ms per leaf, and that was read as the GPU's roof. It is not
a FLOP roof: at about 320 tokens per leaf a leaf costs about 10 GFLOP,
so 0.5 ms per leaf is 20 TFLOPS, 12% of the card's dense bf16 peak
(docs/gpu_forward_design_20260904.md priced 56 GFLOP per leaf at 1,270
tokens; the relevant-set basis cut the tokens 4x). What fills 0.5 ms is
the fixed cost of about 400 small kernels per batch, each on a 5,000-
token problem. A batch four times larger runs the same kernels on
20,000 tokens for well under four times the time. The pool's queue
holds 8-11 requests at all times on that host (48 actors buying
in-flight leaves), so the batches can be filled. The one earlier
coalescing row (2026-09-04, `docs/box_specs.md` "Generation throughput
through the actor pool": 64-leaf cap, 30 leaves per batch on average)
read 1.14x on a CPU-bound host in the 1,270-token basis, where the
GPU was already busier per leaf.

So: no, the target is not known to be unattainable. It was measured
under a 16-leaf cap the az legs used, on the one host class where the
device was the bound. This box measures the cap itself.

## Estimand

`tools/bench_pool.py`, the committed bf16 packed configuration (server
priors, packed trunk, packed embed, eager server), 48 actors and games,
32 evaluations, leaf batch 16 per actor request, max 30 turns, the
reference player's twin `relset` as the checkpoint (the same tokens per
leaf as the reference; comparability with the 2026-09-18 rows), on a
single-tenant 4090 host. Arms, interleaved in two pairs:

| arm | --max-batch |
|---|---|
| b16_a, b16_b | 16 (the default: one request per batch) |
| b64_a, b64_b | 64 (up to four requests coalesced) |

Then one arm each of `--max-batch 96` and of 64 with `--graphed-serve`,
reported, not part of the rule; and one arm at 16 with the reference
checkpoint `terrain` (the first pool run under `terrain_multi_hot`),
a smoke of the actor pool's flag plumbing, reported.

Headline per arm: saturated leaves per second (the server's own
counter), iteration leaves per second, games per dollar, the mean
serve batch and the device ms per leaf from the serve stages.

## Rule

- QUIET: the two b16 arms' saturated rates agree within 5%; else the
  box is noisy and the pair is reported as noise.
- DEFAULT 64: b64 / b16 saturated >= 1.15 in BOTH pairs and games per
  dollar >= 0.97 in both; then `az_loop` and `bench_pool` take 64 as
  the default cap. The numerics are the same weights and math (the
  packed trunk pads nothing; bf16 batch composition moves the last
  bit, the same noise the graphed server was ruled not a cross-build
  for), so no match is needed.
- TARGET: plan 1.3's 3,000 is MET when any arm's saturated rate reads
  at least 3,000 on a 4090; the plan's table and BACKLOG record it
  either way.

## Predictions

b64 / b16 saturated 1.4x (range 1.15-1.9x): the device time per leaf
at four times the tokens per batch falls to 0.25-0.35 ms; iteration
leaves per second 1.2x (tail-bound); games per dollar 1.1x; mean serve
batch 45-60 of 64 at 48 actors. The 96 cap adds under 10% over 64
(the queue holds about 150 leaves). The target is met on the quiet
host class (2,700 x 1.4 = 3,800) and probably not on the 2026-09-14
class (1,750 x 1.4 = 2,450, where the host cost per batch is the
bound; the graphed arm there would be the one to read).

## Cost

Seven pool arms at about 7 minutes each plus bring-up: about 65
box-minutes, $0.50 at $0.40-0.50 per hour on the quiet host class.
`scripts/serve_batch_box.sh` runs it end to end and leaves ALL_DONE on
HF on every exit.

## Measured (2026-09-21, instance 51884357: a whole-CPU Ryzen 9 5950X 4090 host, 30.7-core quota, 32 GB, $0.56/h, 48 box-minutes, about $0.45)

Records under `training/metrics/bench_pipeline/serve_batch_20260920/`
(`verdict.txt`, one JSON and log per arm). 48 actors and games, 32
evaluations, leaf batch 16, bf16 packed, eager server unless noted.

| arm | cap | saturated leaves/s | iteration leaves/s | games per $ | leaves per batch | device ms per leaf | queue depth | wall s |
|---|---|---|---|---|---|---|---|---|
| b16_a | 16 | 2,398 | 1,454 | 918 | 17.1 | 0.76 | 21.7 | 407 |
| b64_a | 64 | 3,223 | 1,623 | 1,074 | 38.9 | 0.50 | 7.7 | 349 |
| b16_b | 16 | 2,402 | 1,644 | 1,096 | 17.2 | 0.74 | 22.7 | 342 |
| b64_b | 64 | 3,222 | 1,815 | 1,196 | 40.0 | 0.48 | 8.4 | 314 |
| b96_a | 96 | 3,237 | 1,686 | 1,127 | 38.1 | 0.51 | 5.0 | 333 |
| b64g_a | 64, graphed | 3,205 | 2,257 | 1,441 | 38.0 | 0.46 | 7.8 | 225 |
| ref16_a | 16, `terrain` checkpoint | 2,354 | 1,679 | 1,034 | 17.2 | 0.78 | 24.2 | 362 |

QUIET: the cap-16 arms repeat to 0.2%. Pair a: saturated 1.344x,
iteration 1.116x, games per dollar 1.169x; pair b: 1.341x, 1.104x,
1.091x. **DEFAULT 64: YES.** `az_loop` and `bench_pool` take 64 from
this commit. **Plan 1.3's 3,000 leaves per second per 4090: MET**, at
3,222-3,237 on a mid-class host (this host read 2,126-2,269 at cap
16 on 2026-09-18, the quiet Core Ultra 9 host 2,634-2,914).

What the columns say. The cap was binding at 16: the queue held 22
requests, the batches 17 leaves. At 64 the batches hold 39-40 leaves
(3.7 requests), the device time per leaf falls from 0.75 to 0.49 ms,
and the queue drains to 8. At 96 nothing moves (3,237, batches of
38): with the server this fast only 5 requests wait, so the queue,
not the cap, binds next; more actors would refill it (docs/box_specs.md
"Actors buy in-flight leaves"). The graphed server at 64 is not a
reading: three quarters of its batches (5,567 of 7,395) exceeded the
largest captured bucket, 12,288 tokens, which a 40-leaf batch of
305-token leaves just fills, and fell back to eager; its iteration
column is the tail swing. A fair graphed-64 arm needs bucket caps
past 16k tokens; the graphed default stays OFF. The reference
checkpoint `terrain` ran through the pool at cap 16 without error
(the first production run under `terrain_multi_hot`; 323 tokens per
leaf, 2,354 saturated, within the cap-16 arms' band), so the actor
pool's flag plumbing holds.

Against the predictions: 1.34x against 1.4x predicted (range
1.15-1.9x); batches of 39-40 against 45-60 predicted; the 96 cap
under 10% over 64 as predicted (0.4%); the target met, on a host class
the prediction had not counted on.
