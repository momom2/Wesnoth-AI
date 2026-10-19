# Pre-registration: fp8 products for self-play generation (2026-10-19)

Written on branch `feature/generation-speed-6` before any box for it, for the
lead to price and the user to approve. Generation round 5
(`training/metrics/generation_speed_r5_20261015/`) left the inference
server's device work as the bound: 12.1 device ms a batch of 47 leaves at
its fastest configuration, the trunk's matrix products about half of it
(4.7 of 9.3 profiled ms). This round measures one lever, the trunk's four
linear layers a layer on float8 e4m3 operands (`torch._scaled_mm` on the
4090; `wesnoth_ai/fp8_inference.py`), for speed and for play.

fp8 changes the arithmetic. It is for self-play generation only, behind
`--fp8-inference-a/-b`, off by default, never in a strength match; the games
record it per side (`infer_fp8_a`, `infer_fp8_b`) and no outdir mixes it.

## Question

1. Does the compiled fp8 loop serve self-play generation measurably faster
   than round 5's fastest configuration, read from the inference server's
   counters?
2. Does a player served with fp8 play as well as the same checkpoint served
   without it?

fp8 is adopted for generation only when both answers are yes.

## Design

Code: `feature/generation-speed-6`, `scripts/fp8_inference_box.sh`, readout
`tools/fp8_inference_readout.py`. One 4090 box; every match is `own1`
against itself at the reference decode (`raw:t0+eo-1.5`, 64 memory slots,
argmax), 120 games in flight with six a worker process, on the Ladder maps
with fog and both factions drawn uniformly.

**The gate.** The fp8 CUDA tests (`tests/test_fp8_inference_cuda.py`), which
no machine has run: `_scaled_mm` on the layouts it takes, eager and compiled
by inductor with no recompile over three batch shapes, the outputs near the
bf16 stream's, no statically scaled input over 448, on random weights at the
reference's shape and on `own1`'s weights, and the device milliseconds of a
56-leaf batch. Unless every one passes, none skipped, the entry finishes
before any arm (`FP8_GATE_FAILED`).

**The speed arms**, in this order, 600 games each on round 5's seeds
(130000-130599), each server writing its counters to a timeline every 5 s:

| arm | configuration |
|---|---|
| ref_1, ref_2, ref_3 | round 5's fastest: the compiled loop on the bf16 residual stream (`--compile-packed --bf16-residual`) |
| fp8c_1, fp8c_2 | ref with fp8 products (`--fp8-inference-a --fp8-inference-b`: one server for both sides) |
| bf16e | the bf16 stream without the compiled loop |
| fp8e | bf16e with fp8 products |
| profile_ref, profile_fp8c | ref and fp8c with 40 batches under torch.profiler after 400; not timed |

Order: ref_1, fp8c_1, bf16e, fp8e, ref_2, profile_ref, profile_fp8c, fp8c_2,
ref_3. The three ref runs, before, between and after, bracket the drift.
Each arm fails alone (a crash, a lost game, a failed request, a compiled
loop that fell back to eager, an fp8 arm whose server did not play fp8) and
the next one runs.

**The play check.** `own1` with fp8 on side A (`own1fp8`) against `own1`
without it on side B, both on ref's configuration (the compiled loop on the
bf16 stream), so fp8 is the only difference: 3,200 decisive games from seed
140000, capped games replaced (up to 1,500), every game recorded whole. The
two sides take two servers on the one GPU.

## Readings

**Speed** (`speed.json`), from the servers' counters, never a match's wall
(identical repeats have differed 1.6-1.8x in wall):
- **Primary:** the server's decisions a second over a steady window of its
  timeline: from 20 s after the first request, 60 s long, inside the 120
  games in flight (`tools/box_probe.steady_rate`). The whole-run figure
  (requests over the server's wall, round 5's reading) is reported beside it
  and replaces it for every arm when an arm lacks a window.
- The session is **valid** when each ref run lies within 5% of the ref runs'
  mean.
- fp8's gain is **resolved** when the fp8c runs' mean exceeds the ref runs'
  mean by more than the largest ref run's deviation from that mean and by at
  least 3%.
- Reported: device ms a leaf (fp8c against ref; the batch size moves with
  the rate, so a leaf is the unit of device work), device and infer ms a
  batch, mean batch, decisions a game (a rounding change plays other games),
  games an hour, fp8e against bf16e, the profiles' device time by kernel
  class, and the gate's device milliseconds of a 56-leaf batch.

**Play** (`play.json`): the fp8 side's Elo against the bf16 side, PURE, on
the decisive games (`tools/elo_collect.py`), with its standard error, about
6.1 Elo at 3,200 decisive games.
- **Kill:** the fp8 side reads below -2 standard errors (about -12 Elo). fp8
  measurably changes play; it stays off.
- **Pass:** otherwise. At an estimate near 0 this bounds the loss at about 10
  Elo (one-sided 95%); a true loss of 20 Elo is caught with probability
  about 0.9, one of 10 with about 0.35.
- **Not read:** fewer decisive games than 3,200, or games or servers that do
  not carry the sides' switches (side A fp8, side B not). The entry finishes
  as failed (`FP8_INFERENCE_INCOMPLETE`).
- Reported: each side's decisions a game (round 5's memory flaw showed as
  fewer actions a side-turn), the fp8 side's wins.

**The decision:** fp8 becomes generation's configuration when the session
is valid, the gain is resolved and the play check passes. Otherwise it stays
off, and the reason is recorded here.

A bound of about 7 Elo instead of 10 needs 6,400 decisive games: about 15
more minutes, $0.10-0.12. 3,200 is proposed because the speed gain predicted
below is worth less than the difference between the two bounds.

## Predictions (the branch's, before any box)

| reading | prediction | range |
|---|---|---|
| the gate passes | yes, probability 0.7 | the compiled path through inductor's `_scaled_mm` lowering with dynamic token counts never ran |
| fp8c against ref, decisions a second (window) | 1.07 | 1.02 to 1.12 |
| the gain resolved | probability 0.55 | |
| fp8c against ref, device ms a leaf | 0.80 | 0.72 to 0.90 |
| fp8e against bf16e, decisions a second | 1.00 | 0.95 to 1.05: the casts and the first layer's batch maxima run as passes of their own |
| decisions a game, fp8c | as ref | 440 to 480 (round 5: 442 to 478) |
| the gate's 56-leaf batch, compiled fp8 against the compiled bf16 stream | 0.82 | 0.70 to 0.95 |
| play: Elo of the fp8 side | -3 | -15 to +10; Pass with probability 0.85 |

Where they come from: round 5's profile of ref reads 4.7 of 9.3 device ms a
batch in matrix products, most of them the trunk's: a batch of 43 leaves at
about 400 tokens each (the branch's estimate) carries about 490 GFLOP of
trunk products, 104 TFLOP/s if the 4.7 ms were the trunk's alone, 63% of the
4090's bf16 rate. The 4090's fp8 rate with float32 accumulation is twice its
bf16 rate (NVIDIA's Ada architecture whitepaper: 330 against 165 dense
TFLOP/s), so the trunk's products can at most halve, about 2 ms; the casts
fuse into the kernels that write their inputs when the loop is compiled. The server was
busy on the device 63% of its wall at ref, the rest host work and waiting
for requests, so a 20% cut of device work a leaf moves the decision rate by
less. The trainer's fp8 trunk read an output RMS error of 2.1% against its
bf16 trunk (docs/trainer_speed_20261013.md "fp8 products"): enough to flip
argmax near ties, which is why play is measured, not inferred.

## Box

One RTX 4090 with at least 32 usable cores, 64 GB of memory and 60 GB of
disk (the script's `box-needs`), at about $0.42-0.46 an hour.

| step | minutes |
|---|---|
| bring-up, wheel, the tests of the generation path and of this branch | 20-25 |
| the earlier rounds' CUDA tests and the fp8 gate (inductor compiles the fp8 loop twice) | 5-10 |
| nine speed arms of 600 games (round 5: 121-166 s each, plus about a minute of compile for a compiled server) | 20-27 |
| the play check, 3,200 decisive games (round 5's 4.8 games a second at 120 in flight, slower with two servers) | 12-16 |
| uploads and the stop | 3-5 |

About 1.2 box-hours, $0.50-0.55. `BOX_MAX_H` 2.5 caps it at $1.05-1.15
plus the final round; the rental's funds check asks for 4 hours' price.

Launch (docs/box_runbook.md), after a green CI run on the commit staged:

    python tools/stage_code.py --out /tmp/stage_20261019_fp8.tar.gz \
        --script scripts/fp8_inference_box.sh --upload tier-b/staging/stage_20261019_fp8.tar.gz
    python scripts/rent_box.py search --min-hours 4 --min-cpu 32 --disk 60
    python scripts/rent_box.py create OFFER_ID --onstart fp8_inference_box.sh \
        --stage tier-b/staging/stage_20261019_fp8.tar.gz --hours 1.3 --disk 60

## Records

On HF under `tier-b/fp8_inference_20261019/`, pulled to
`training/metrics/fp8_inference_20261019/`:
- `speed.json`: every arm's figures and the reading, rewritten after every
  arm; `arm_<name>.json` (the match), `arm_<name>.inference_server_0.json`
  (the server's stats: counters, host stages, the compiled loop's state, the
  fp8 scales and bounds, the profile), `arm_<name>.inference_server_0.timeline.jsonl`,
  `arm_<name>.worker_stages.json`, `arms_failed.txt`;
- `fp8_cuda_tests.log` (the gate, run with `-s`: the errors, the scales, the
  56-leaf device milliseconds), `tests.log`, `cuda_tests.log`, `box.txt`;
- `play.json` (the reading), `play.fit.json`, `play.server_<0|1>.json` and
  their timelines, `timing_play.txt`, `games_play.tar.gz` (every game
  whole: the fp8 player's own trajectories);
- `status.txt`, `stages.txt`, `walls.txt` as on every box.

## Not measured here

- Round 5's own rounding changes (the compiled loop, the bf16 stream) have
  no play check: both sides of this one run them. The match driver applies
  them to every server, so a check of them against the eval configuration
  needs per-side switches first.
- In a policy-gradient run (docs/selfplay_pg_design_20261015.md) the
  behaviour policy would be the fp8 server's and the trained one the bf16
  network's: their log-probabilities differ by fp8's rounding. That design
  decides whether it corrects for the difference; this round measures its
  size in play only.
