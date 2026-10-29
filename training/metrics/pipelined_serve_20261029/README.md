# The pipelined serve loop (2026-10-29)

**On the GPU (the lead, 20:28-20:51, idle RX 6600): +9.2% decisions a
second at 96 games in flight and +11.1% at 144** (section "The GPU A/B:
results" at the end). That is above the stand-in's +7.1% and +6.4%. The
loop stays opt-in in `run_elo_batch` (boxes were not measured), and the
desktop's scripts turn it on.

**The numbers below, up to that section, come from a CPU stand-in for the
GPU, not from the GPU.**
The serve loop ran on the desktop (Windows 10, Ryzen 5 7600, 12 threads)
with a sleeping stand-in device. Stage 1's self-play was running beside it
on the RX 6600: 12 worker processes and a server at below-normal priority,
about 9% CPU a core. Every benchmark process ran at IDLE priority, one run
at a time. The GPU A/B that settles the change is at the end, for the lead
to run.

**Verdict on the stand-in: +7.1% decisions a second at 96 games in flight,
+6.4% at 144, +6.1% at 192 (two interleaved pairs each, sd 0.2 to 0.4
points).** That is above the 5% kill line at every depth. Pipelined at 144
or 192 in flight, the stand-in reads +9.2% and +10.0% over today's serial
loop at 96. The change is opt-in (`--inference-pipelined`) and not merged.

## The problem

The serve loop (`tools/serve_loop.py`, `InferenceService._serve`) does
everything in one thread, one step after another:
1. collect a batch;
2. stage and launch it (Rust host stages: parse, stage, memory, the forward
   launch, the priors);
3. wait for the device;
4. write the replies.

The device idles through steps 1, 2 and 4.

Take the idle-GPU A/B's steady state (`pipe_buffer_20261029`, arm
`biaspipe`, the server's timeline between 30 s and 70% of the run). A
batch of 80.5 leaves takes 151 ms of wall. Of that, infer is 139.6 ms
(the GPU's events read 118 ms), the wait for a first request 1.6 ms, and
the window, reads and reply writes 10 ms. Stage 1's match s01 gives the
host stages per batch: about 7.3 ms of launch work (parse 0.17, stage
1.18, memory 0.17, forward launch 4.42, priors 1.37) and 0.29 ms of
replies.

## The change (`feature/pipelined-serve`)

`tools/serve_pipelined.py` `PipelinedInferenceService`: the serve thread
launches batch N+1 (its host work done, its device work queued) before it
completes batch N (the device wait, the replies). It collects N+1 while the
device runs N. A waiter thread waits on each launched batch's device work
and wakes the serve thread's wait, so that:
- the collect waits for a first request only while N runs;
- N's end closes the collect's window, since the device would otherwise
  idle.

`tools/serve_stage.py` `launch_requests` splits `serve_requests` into
halves (the second in `tools/serve_launched.py`):
- the launch, which ends with a CUDA event and a `query()` that submits
  the queued work;
- `complete()`, which waits on that event and writes the replies.

Each batch keeps its pinned host buffers until its replies. For a given
batch, the replies are the bytes `serve_requests` gives.

Why it is safe (argued in the module docstrings; tested):
- **Games.** A game sends its next request only after its reply, so
  batch N+1 never holds a game of N.
- **Memory states.** The memory states the server keeps live on the device.
  The batches read and write them in queue order, on one stream. The books
  (`MemoryTable`) hold one pending plan, and N+1 is planned before N
  completes, so a batch commits its books at its launch. Committing after
  the device ran, as the serial path does, would lose N's commit.
  - A failure past the launch leaves the versions that batch wrote held,
    in rows the device may not have written. None is read before it is
    written again. After a device fault the batch's replies are errors,
    so no client is handed them. Otherwise each request is retried alone,
    and the retry reads version v again and writes v + 1 again into the
    same row before its reply hands it out. A request that fails alone
    too gets an error, and its client never reads v + 1.
  - A connection that closes while its request runs frees its rows. A new
    game-side that takes one of them writes it after N, in stream order.
- **Failures.** A launch or a completion that raises is answered as a
  failed batch: the health probe, then each request alone through the
  serial path. A request whose connection closed while its batch ran is
  not retried: its memory states were released at the close, and a retry
  would hold new ones that nothing releases.
- **Stop.** stop() answers the batch in flight before the connections
  close.

Switches:
- `eval_inference_server --pipelined`, `run_elo_batch
  --inference-pipelined`, off by default. The hello and the stats record
  `pipelined`.
- It is refused with `--graphed`, `--no-rust-stage` and `--profile-batches`.
- The stats gain `busy_wait_s`: time spent waiting for a first request
  while a batch runs. `idle_s` keeps the waits with nothing on the device.

The default path changes by refactors only:
- `_recover` and `_deliver` are factored out;
- `_wait` and `_collect` take the in-flight batch's event;
- the wake pipe gets a lock, since two threads write it now.

## The benchmark (`bench_pipelined.py`)

It starts from `async_replies_20261029/bench_replies.py`. One server
process runs the service on the server's own pipes (`pipe_listener`,
256 kB). 12 client processes of 8, 12 or 16 threads each play one game a
thread, on its own connection: send a 23,825-byte request, wait for the
reply, step 2 ms, send again. Of the step, 0.3 ms holds the GIL and the
rest sleeps. The replies are the real wire, sized from stage 1's legal
actions.

The stand-in batch has three parts:

| part | cost | from |
|---|---|---|
| launch (host, holds the GIL) | 6.55 ms + 0.014 ms a leaf | s01's parse, stage, memory, forward and priors seconds; the split per leaf from the idle runs' batches of 31 against s01's 69 |
| device (a queue on a clock; a wait on it sleeps, releasing the GIL) | 2.47 ms + 1.636 ms a leaf | the GPU fit of stage 1's timelines (`async_replies_20261029`, 273 intervals) |
| completion (host, after the wait) | 0.0042 ms a leaf | s01's `t_replies` |

A batch's device work starts when its launch ends and the batch before it
is done. Stage 1's own records show no overlap of the device with the
launch (s01: `t_finish` 125.2 ms a batch against 124.0 GPU ms), so neither
loop gets any. The serial loop runs the three parts back to back; the
pipelined loop launches the next batch first.

Calibration (`calibration/`, serial only, 12 s):

| game step | decisions/s | mean batch | stand-in device busy |
|---|---|---|---|
| 2 ms | 544.8 | 86.0 | 0.907 |
| 6 ms | 516.8 | 48.1 | 0.872 |
| 12 ms | 514.8 | 47.9 | 0.869 |
| the GPU (biaspipe, steady) | 531.8 | 80.5 | 0.78 by its events, 0.87 by its waits |

At 6 ms and more, the serial loop splits into two cohorts of games. Stage 1
does not show that split, so the main runs use a 2 ms step. The stand-in's
serial loop then reads 2.4% above the GPU's rate.

    python training/metrics/pipelined_serve_20261029/bench_pipelined.py --out <dir> --reps 2 \
        --threads 8 --device-ms 2.47 --device-per-leaf-ms 1.636

Each run has 10 s of warm-up and 30 s measured. The pairs are
interleaved, alternating which loop goes first.

## Results (`g096/`, `g144/`, `g192/`, logs beside them)

| games in flight | pair | serial, decisions/s | pipelined, decisions/s | pipelined over serial |
|---|---|---|---|---|
| 96 | 0 (serial first) | 544.5 | 584.3 | +7.3% |
| 96 | 1 (pipelined first) | 547.4 | 584.8 | +6.8% |
| 144 | 0 | 560.9 | 595.5 | +6.2% |
| 144 | 1 | 560.0 | 597.5 | +6.7% |
| 192 | 0 | 567.8 | 601.6 | +5.9% |
| 192 | 1 | 564.2 | 599.7 | +6.3% |

Means:
- 96: serial 546.0, pipelined 584.5, **+7.1%**;
- 144: serial 560.5, pipelined 596.5, **+6.4%**;
- 192: serial 566.0, pipelined 600.6, **+6.1%**.

No game errors.

Per batch in pair 0, from the server's counters over the measured window:

| | mean batch | device busy | launch ms | device wait ms | reply writes ms | window ms | first-request wait ms |
|---|---|---|---|---|---|---|---|
| serial, 96 | 86.4 | 0.906 | 7.8 | 144.4 | 1.9 | 2.2 | 1.4 |
| pipelined, 96 | 42.9 | 0.990 | 7.2 | 61.7 | 0.9 | 1.8 | 1.3 (busy) |
| serial, 192 | 167.0 | 0.937 | 8.9 | 276.2 | 4.2 | 2.9 | 2.8 |
| pipelined, 192 | 94.5 | 1.000 | 7.9 | 140.8 | 2.9 | 2.4 | 2.4 (busy) |

How to read this:
- **Pipelined, the stand-in device is busy all the time**, at every depth.
  The serve thread's own cycle (launch, wait, writes, window) adds up to
  the device's time per batch, so the loop is bound by the device.
- **The batches halve.** A batch's games are answered while the next
  batch runs, so they cannot join it, and the games split into two cohorts
  that alternate (as the writer thread's did, `async_replies_20261029`).
  Each batch pays the device's fixed 2.47 ms, so at 96 the device's time
  a leaf rises from 1.665 to 1.693 ms (+1.7%). That is most of the
  difference between the device's busy gain (+9.3%) and the rate's (+7.1%).
- **More games in flight help both loops a little.** Serial gains +2.7%
  from 96 to 144 and +1.0% from 144 to 192. Pipelined gains +2.1% and then
  +0.7%. 192 does not pay over 144 in the stand-in, so the GPU A/B stops
  at 144.

## What the stand-in does not settle

- **When the GPU starts on queued work.** On the GPU, infer exceeds the
  events' GPU time plus the host stages by about 14 ms a batch at steady
  state (139.6 against 118 + 7.6), and that gap is not explained. Two
  cases:
  - If it is device time the events miss, the stand-in has it: the device
    calibration includes it.
  - If it is the host's wake after the device finishes, the pipelined loop
    hides it too, and the GPU would gain more than the stand-in.

  A Windows driver may hold queued work back until a wait. The launch's
  `query()` and the waiter thread's `synchronize()` should submit it, but
  that is not verified on HIP.
- **Cross-thread waits on HIP.** The waiter thread calls
  `Event.synchronize()` while the serve thread launches the next batch on
  the same stream. If the HIP runtime held a lock across that wait, the
  launches would stall behind it. In the A/B, check that the pipelined
  arms' launch seconds a batch (`t_stage`, `t_forward`, `t_priors` in the
  server's stats line) stay at the serial arms' level. If they grow, make
  the waiter poll `query()` instead.
- **CPU.** A wait on a default event may spin a core (CUDA's default
  schedule does). The serial loop spins one during its wait; the pipelined
  loop may spin two, the waiter and the serve thread's completion. Stage 1
  left the CPU about 90% idle today. `torch.cuda.Event(blocking=True)` for
  the batch's `done` event (`serve_stage._launch_eager`) would make both
  waits yield the core instead, at some wake latency. It is not exposed as
  a switch: that is not a one-line change through the server's flags. Try
  it as a variant if the A/B shows the spinning.
- **VRAM.** Only the pinned host buffers of two batches live at once:
  the staging and output buffers, a few MB. A batch's device tensors are
  freed when its launch returns, and the caching allocator reuses them for
  the next batch in stream order. Pipelined batches are about half the
  serial ones (about 63 leaves against about 130 at 144 in flight), so the
  device's peak should not rise. Neither was tried on the RX 6600.
- **Boxes.** Linux, the 4090 and the graphed path were not measured, and
  `--pipelined` refuses `--graphed`.
- **The games.** The stand-in's games are lighter than real ones (a 2 ms
  step, a sleep for the Rust part). Real steps load the CPU, which the
  stand-in's do not.

## Tests

- `tests/test_eval_inference_server.py`:
  - the serial loop's tests run on both loops (counted before sent, the
    window and max batch, a closed connection, a failing request at the
    launch and at the completion, the device fault);
  - every client gets its own reply, and two batches on the device never
    share a connection;
  - stop() answers the batch in flight;
  - a batch whose finishing raises leaves the next batch answered, once;
  - a retry one at a time skips a client that closed while its batch ran;
  - the end-to-end match runs with `--inference-pipelined`; this variant is
    in the slow tier.
- `tests/test_serve_stage.py::test_launched_batches_overlap_and_keep_the_memory`:
  - each batch is launched before the previous one completes;
  - the replies are `serve_requests`'s bytes for the same batches;
  - the versions advance by one;
  - a refusal inside a launched batch is answered;
  - a connection closes while its batch runs.

  With the commit after the device ran, the second batch's plan discards
  the first's, and this test fails.
- `tests/test_generation_identity.py::test_a_game_with_the_memory_on_the_server_plays_as_with_it_on_the_wire`
  (slow): a third server, pipelined, plays the same game as the wire.

## The GPU A/B (for the lead, on an idle GPU)

`ab_gpu.sh` has four arms from one checkout of this branch:
- the serial and the pipelined loop;
- at 96 games in flight (12 workers of 8) and at 144 (12 workers of 12);
- 192 games each, `own2` at temperature 1, seeds 872700, stage 1's flags;
- two rounds, the second in reverse order.

Eight runs, about 22 minutes:

    bash training/metrics/pipelined_serve_20261029/ab_gpu.sh
    python training/metrics/pipe_buffer_20261029/ab_steady.py C:/Users/momom2/wesnoth_runs/pipelined_ab_20261029

Each arm is `run_elo_batch` with stage 1's flags, as in `ab_idle.sh`:
`--jobs 96 --worker-games 8` or `--jobs 144 --worker-games 12`, with
`--inference-pipelined` or `--no-inference-pipelined`.

Read the steady rate and `t_finish`, not a GPU-busy share from the
events: the events' device milliseconds are not comparable between the
arms.
- A pipelined batch's bracket opens only when the device reaches it,
  right after the batch before it.
- Both arms' brackets include the device's waits on the host's kernel
  launches.

On this branch the start event is recorded at a batch's first copy to the
device, not before its host staging, as on `main`. So the serial arm's
`gpu_ms` reads about the host staging time (part of `t_stage`, at most
about 1 ms a batch) below `main`'s.

Beyond the steady rate, read each run's server stats line in its log:
- the launch seconds a batch (above);
- `t_finish` a batch;
- `busy-wait`.

The one-against-two-servers A/B read +2.5% and called it the noise, and
two rounds of one arm have differed by 2.2%. If an arm pair reads under
about 3% either way, play it again before reading it.

## The GPU A/B: results (the lead, 2026-10-29 20:28-20:51)

`ab_gpu.sh` as above, at 89e995c6, on the idle RX 6600 right after stage 1
ended. Every run exited 0. Steady decisions a second by `ab_steady.py`;
the server's counters are from its stats line in each run's log (out
`C:\Users\momom2\wesnoth_runs\pipelined_ab_20261029`):

| Arm | Run 1 | Run 2 | Mean | Over serial | Wall a run (s) | `t_finish` / wall | Launch ms a batch | Mean batch (`ab_steady`) |
|---|---|---|---|---|---|---|---|---|
| serial, 96 | 517.3 | 523.1 | 520.2 | | 162, 168 | 0.74, 0.76 | 7.1, 5.9 | 77.0, 73.9 |
| pipelined, 96 | 569.2 | 567.0 | 568.1 | **+9.2%** | 150, 155 | 0.64, 0.63 | 5.5, 5.0 | 41.2, 41.7 |
| serial, 144 | 518.4 | 508.4 | 513.4 | | 161, 163 | 0.76, 0.77 | 7.2, 6.6 | 92.2, 91.0 |
| pipelined, 144 | 577.5 | 562.9 | 570.2 | **+11.1%** | 140, 146 | 0.66, 0.64 | 5.8, 5.7 | 58.7, 57.1 |

"Launch ms a batch" is `t_stage + t_forward + t_priors` over the batches.

What it settles from "What the stand-in does not settle":
- **The gain is larger than the stand-in's** (+9.2% against +7.1% at 96).
  So part of the 14 ms a batch the events did not explain is host-side,
  and the pipelined loop hides it. That is the README's second case.
- **Cross-thread waits do not stall the launches.** A pipelined batch's
  launch takes 5.0-5.8 ms against 5.9-7.2 for a serial one, though the
  serial batches are larger. The waiter thread's `synchronize()` holds no
  lock the launches need.
- **144 in flight does not pay over 96** for the pipelined loop (+0.4%,
  inside the 2-3% between identical runs). The desktop stays at 96.
- **Against stage 1's code**, the pipelined loop at 96 reads 568.1 against
  485.1 (`pipe_buffer_20261029`'s idle A/B, another session of the same
  day): about +17% for the bias fold, the pipe buffer and the pipelined
  loop together. That comparison crosses sessions, so it is looser than the
  pairs above.
- **VRAM** was not read; no run failed for memory.
- **The CPU: the pipelined server spins about one core.** One more run of
  each 96-games arm (20:51-20:57, out
  `C:\Users\momom2\wesnoth_runs\pipelined_cpu_20261029`) sampled the
  server process's CPU time every 5 s:
  - serial: 35.8 CPU seconds over 168 s, 0.21 of a core;
  - pipelined: 172.2 over 148 s, 1.17 cores.

  The serial loop waits with a stream synchronize (`complete()`, `done` is
  None there). The pipelined waiter waits on a default event
  (`serve_stage`'s `done = torch.cuda.Event()`). On HIP the event's
  synchronize most likely spins, while the stream's yields. The rate
  gained anyway, but the desktop loses a core to it. The next variant to
  try is a blocking event (`torch.cuda.Event(blocking=True)`).
- **A machine limitation, not this change:**
  `test_serve_stage.py::test_on_cuda_the_kept_memory_plays_as_the_wire[True]`
  fails on the RX 6600, here and at fc560ab9 (before the merge). Every
  CUDA graph capture fails on ROCm for Windows ("Cannot copy between CPU
  and CUDA tensors during CUDA graph capture unless the CPU tensor is
  pinned"), so the eager path serves, with the same bytes. The desktop
  never runs the graphed path. The other 18 GPU tests of the file pass.
