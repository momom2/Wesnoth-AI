# Async replies in the serve loop (2026-10-29)

**The numbers here are a stand-in for the GPU, measured on a CPU-only
benchmark of the serve loop, on the desktop (Windows 10, Ryzen 5 7600),
while stage 1's self-play ran beside it** (12 worker processes and a server
on the RX 6600, below-normal priority, about 3 of the 12 threads). The
GPU A/B that settles the change is at the end, for the lead to run.

**Verdict on the stand-in: +4.1% decisions a second in stage 1's regime
(three interleaved pairs, +3.6 to +4.6%), above the 3% kill line; but
-1.8% and -13.1% in two other regimes (one pair each).** So the writer
thread is opt-in (`--inference-async-replies`), not the default.

## The problem

Stage 1's second self-play match (`selfplay_t1_local/match_s01.log`, its
server's stats line): 1,910,926 requests in 27,521 batches (mean 69.4),
4,091 s of wall, of which infer 3,659 s (the GPU busy 3,412.7 s), **reply
331.9 s**, idle 41 s, window 55.8 s. The reply time is
`tools/serve_loop.py`'s serve thread writing each batch's replies, one
`send_bytes` per request, before it reads the next batch: 0.174 ms a
request, 8.1% of the wall, with the GPU idle.

Fitted over the 273 thirty-second intervals of the server's timelines in
stage 1's first two matches (`games_s00`, `games_s01`), least squares on
batches and leaves:

| per interval | per batch | per leaf |
|---|---|---|
| GPU ms | 2.47 | 1.636 |
| infer ms | 6.83 | 1.838 |
| wall - infer - idle ms (reply and window) | 0.95 | 0.187 |

The forward's cost is nearly all per leaf, so on this GPU a smaller batch
costs little more per leaf (the local bench's 539 and 535 positions a
second at batch 24 and 48 agree, `local_gpu_20261028`).

## The change (`feature/async-replies`)

`InferenceService(async_replies=True)`: the serve thread counts the batch,
hands its replies to a writer thread (`_Replier`) in one queue item, and
goes on to collect and run the next batch. While the writer runs it alone
writes to and closes the connections (hellos, error replies and closes go
through it too); a failed write posts the connection back to the serve
thread, which drops it between batches. Off by default; on with
`--async-replies` (the server) or `--inference-async-replies`
(`run_elo_batch`). The server's hello, info and stats record
`async_replies`, and the stats line adds the writer's seconds (`send_s`)
and its failed writes. The bytes a client receives are the same on both
paths.

## Why a write cost 0.17 ms

Not the pipe's own cost. A reply that does not fit the named pipe's 8 kB
buffer (`multiprocessing.connection.BUFSIZE`; the replies here average
8.5 kB, 9 of the 20 sizes larger) completes only as its client reads it,
and a client reads only once its thread holds its process's GIL, which
the other games of that worker hold during their steps. The benchmark
shows it (`calibration.jsonl`, inline path, 10 s windows):

| GIL held per 2 ms step, ms | 0 | 0.3 | 0.4 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|
| ms a write | 0.023 | 0.194 | 0.226 | 0.280 | 0.621 | 1.276 |
| reply share of the wall | 1.2% | 9.0% | 10.5% | 12.6% | 23.8% | 39.5% |

The writer thread waits for those reads instead of the serve thread.

## The benchmark (`bench_replies.py`)

- One server process: `InferenceService` on a named pipe, its forward a
  sleep (it releases the GIL as a device wait does) of 6.8 ms + 1.84 ms a
  leaf, the infer fit above: 134 ms at stage 1's mean batch.
- Replies of the real wire (`byte_wire.encode_replies`), n legal actions
  drawn from the 20-quantile midpoints of 75,464 decisions of stage 1's
  second match (mean 437 actions); requests of 23,825 bytes (the mean of
  300 random-game positions encoded with own1's structure, memory kept on
  the server, `tools/bench_serve_host.py`).
- 12 client processes of 8 threads, each thread a game on its own
  connection (96 in flight): send, wait for the reply, a 2 ms step, again.
  The step holds the GIL for 0.3 ms (a busy loop) and sleeps the rest: the
  Rust core's calls release the GIL (`py.allow_threads` in its encode,
  observe, move and step calls). 0.3 ms puts the inline path's write at
  0.19 ms and its reply share at 9.0%, against stage 1's 0.174 ms and 8.1%.
- Every game connects and gets its hello before the start (Python takes
  about 2 s to start here; started on a timer, the last client process of
  the first smoke joined 23 s late). Then 10 s of warm-up and 40 s
  measured; the server reads its counters at the window's bounds.
- Runs interleaved, the order alternating by pair, at below-normal priority.

    python training/metrics/async_replies_20261029/bench_replies.py --out <dir> --reps 3

## Results: stage 1's regime (`main/`, `main.log`)

| pair | order | inline, decisions/s | async, decisions/s | async over inline |
|---|---|---|---|---|
| 0 | inline, async | 471.8 | 488.9 | +3.6% |
| 1 | async, inline | 467.9 | 486.9 | +4.1% |
| 2 | inline, async | 465.4 | 487.0 | +4.6% |
| mean | | 468.4 (sd 3.2) | 487.6 (sd 1.1) | **+4.1% (sd 0.5 points)** |

The server's stats over pair 0's measured 40 s:

    inline: 18872 requests in 204 batches (mean 92.5); idle 0.1s window 0.4s (reads 0.3s) infer 36.3s
            reply 3.34s (writer 0.0s, 0 failed) of 40.0s wall (stand-in gpu 36.1s)
    async:  19555 requests in 404 batches (mean 48.4); idle 0.3s window 0.8s (reads 0.3s) infer 39.0s
            reply 0.01s (writer 3.4s, 0 failed) of 40.0s wall (stand-in gpu 38.7s)

- The stand-in GPU's busy share rises from 89.6% to 96.5% (means of the
  three pairs), +7.7%; the decisions rise 4.1%.
- The difference is the batch: 93 leaves inline, 48 with the writer. A
  batch's games get their replies while the next batch already runs, so
  they cannot join it: the 96 games split into two cohorts that alternate.
  Each batch pays the fixed 6.8 ms, so the stand-in GPU's cost per leaf
  rises from 1.913 to 1.980 ms (+3.5%, means of the three pairs).
- The inline path reads stage 1's rate (468 against 467 decisions a
  second), with larger batches (93 against 69): the stand-in's games are
  away 2 ms a step, stage 1's longer. Where stage 1's batches are smaller,
  the cohort split costs less, so the GPU A/B may read more than 4.1%; it
  could also read less. The stand-in does not say.
- No write failed, no error in any game.

## Results: other regimes (one pair each, 30 s measured)

| regime | inline | async | async over inline | mean batch, inline / async |
|---|---|---|---|---|
| cheap writes: a step holding no GIL (0.023 ms a write, as a Unix socket's buffer would take a reply) (`cheap_writes/`) | 499.5 | 490.5 | -1.8% | 73.8 / 55.3 |
| a launch-bound forward (15 ms + 0.3 ms a leaf), cheap writes, 96 games (`launch_bound/`) | 1,764.7 | 1,533.1 | **-13.1%** | 76.4 / 52.6 |
| the same, 20 games, max batch 20 (`launch_bound_20/`) | 611.6 | 775.3 | +26.8% | 13.3 / 19.0 |

- What the writer changes most is how batches form, not the writes'
  cost. With cheap writes and a forward mostly paid per batch, it splits
  the games into cohorts and loses 13%.
- At 20 games it is the inline loop that splits: the games left out of a
  batch are waiting when it ends, so the collect window (1.5 ms) opens at
  once and closes before the just-answered games return from their 2.7 ms
  step. With the writer, nothing is waiting; the serve thread waits for
  the first request, and all 20 return inside the window.
- So the serve loop's batch shape depends on the window, the step and the
  forward's fixed cost together; the reply path is one input to it. The
  box regime (Linux sockets, a launch-bound 4090) was not measured: these
  two stand-ins of it disagree in sign.

## Decision

Opt-in. On the stand-in of stage 1 it clears the kill line (+4.1%
against 3%), so the code stays on `feature/async-replies` for the GPU A/B;
it is not the default, because it was worse in two regimes and the box
runs share the serve loop. If the A/B confirms a gain, the local scripts
(`scripts/selfplay_t1_local.py`, `scripts/match_local.py`) would pass
`--inference-async-replies`; the box scripts stay as they are until a box
run measures it.

Not verified: the GPU (below); Linux and a box; the share of stage 1's
real game step that holds the GIL, which is inferred from the write's
cost (0.3 ms fits 0.174 ms a write), not measured.

## The GPU A/B (for the lead, on an idle GPU)

Two 192-game self-play runs of `own2` at temperature 1, 96 in flight,
stage 1's flags and the seeds of the 2026-10-29 one-against-two-servers
A/B (872700), from a checkout of `feature/async-replies`, back to back.
The old loop:

    set HIP_VISIBLE_DEVICES=1
    %USERPROFILE%\venvs\wesnoth_rocm\Scripts\python.exe tools/run_elo_batch.py ^
      --label-a own2a --spec-a training/checkpoints/own2.pt --memory-a 64 ^
      --label-b own2b --spec-b training/checkpoints/own2.pt --memory-b 64 ^
      --mcts-sims 0 --raw-temperature-a 1 --raw-temperature-b 1 --log-decisions ^
      --outdir %USERPROFILE%\wesnoth_runs\async_replies_ab\inline ^
      --games 192 --max-extra-games 0 --seed-base 872700 ^
      --persistent-workers --shared-inference --no-infer-compile --device cuda ^
      --jobs 96 --worker-games 8 --min-free-mb 300 --infer-fp16 ^
      --inference-timeline-s 10 --time-budget-min 15 --no-inference-async-replies

The new loop: the same with `--outdir ...\async_replies_ab\async` and
`--inference-async-replies` in place of `--no-inference-async-replies`.

Read each run's server stats line in its log (decisions a second over the
wall, GPU busy, mean batch, reply and writer seconds) and the steady rate
from its timeline, the requests a second between the lines at 30 s and
at 70% of the run:

    python -c "import json,sys; L=[json.loads(l) for l in open(sys.argv[1])]; T=L[-1]['wall_s']; a=min(L,key=lambda r:abs(r['wall_s']-30)); b=min(L,key=lambda r:abs(r['wall_s']-0.7*T)); print((b['requests']-a['requests'])/(b['wall_s']-a['wall_s']))" <outdir>\.inference_server_0.timeline.jsonl

The one-against-two-servers A/B read +2.5% and called it the noise
level; if this one reads under about 3% either way, play the pair again
in the other order before reading it.
