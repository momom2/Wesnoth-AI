# The local machine's GPU, measured (2026-10-28)

The user's desktop, offered for tests and training on 2026-10-28. Question:
can stage 1 of the training design (`scripts/selfplay_t1_box.sh`, priced
for a 4090 box) run here instead of on Vast?

**Answer, in the afternoon: no.** The GPU ran this model about two orders of
magnitude slower than a 4090 box. ROCm on Windows has no fused attention
kernel for its architecture, and a self-play pilot reset the display driver
within seconds.

**Answer, in the evening: yes, in about 11.5 hours.** A HIP attention
kernel, fp16 serving and a fused layer norm took `own2` from 42 to 535
positions a second. Self-play now serves 457 decisions a second, a sixth of
a 4090 box's pool (21,966 games an hour at 475.7 decisions, about 2,900 a
second), at no cost ("After the kernels" below).

## The machine

- **CPU and memory:** AMD Ryzen 5 7600 (6 cores, 12 threads), 63 GB RAM.
- **GPUs:**
  - AMD Radeon RX 6600: RDNA2, gfx1032, 8 GB, 224 GB/s. It also drives the
    display.
  - The CPU's iGPU, gfx1036.
- **No NVIDIA GPU**, so no CUDA.
- **GPU torch:** `torch 2.14.0+rocm10.1.0` (HIP 7.16) in a second venv,
  `%USERPROFILE%\venvs\wesnoth_rocm`, outside the repository. It comes from
  AMD's stable index:

  ```
  pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1032]==2.14.0+rocm10.1.0"
  ```

  The repository's `.venv` keeps `torch 2.14.1+cpu`.
  - TheRock's SUPPORTED_GPUS.md (read 2026-10-28) marks gfx1032 on Windows
    as build passing, sanity tested and release ready.
  - `torch-directml` was not used: its last release, 0.2.5.dev240914, stops
    at Python 3.12 and torch 2.4.
- **`HIP_VISIBLE_DEVICES=1`** makes `cuda:0` the RX 6600. Device 0 is the
  iGPU, which the gfx1032 wheel has no kernels for.
- **The Rust core:** built with the GNU toolchain
  (`stable-x86_64-pc-windows-gnu`, rustc 1.99.1). The machine has no Visual
  Studio, so the MSVC toolchain cannot link. maturin 1.15.0 built the wheel
  into `C:/Users/momom2/.cargo-target` without a Defender block, and
  `kernel_status` reads phase 38, current, in both venvs.

## Raw rates

4096² matmul on the RX 6600:

| precision | TFLOP/s |
|---|---|
| fp32 | 6.5 |
| fp16 | 12.5 |
| bf16 | 2.4 |

RDNA2 has no bf16 matrix units, so bf16, the 4090 boxes' default for
serving and training, is the slowest path here.

## `own2`'s forward and train step (`bench_local.py`)

The 200 positions of `configs/bench_states.json`, each with the 64-slot
initial memory, run through `forward_padded`:
- forward: eval mode, batch 64;
- train step: forward, backward and AdamW, batch 32, a surrogate loss over
  every head.

Positions a second, with `rates_cuda.json` holding all 200 positions (three
warm-up batches, three passes):

| | GPU fp32 | GPU fp16 | GPU bf16 |
|---|---|---|---|
| forward | 20.7 | 19.0 | 17.1 |
| train step | 3.0 | 3.3 | 0.6 |

GPU against CPU on the same work: the first 64 positions, one warm-up batch,
one pass (`rates_cuda_64.json`, `rates_cpu.json`):

| | GPU fp32 | GPU fp16 | GPU bf16 | CPU fp32, torch 2.14.1+cpu, 6 threads |
|---|---|---|---|---|
| forward | 15.3 | 12.9 | 12.0 | 10.2 |
| train step | 3.05 | 3.42 | 0.62 | 1.22 |

So the GPU runs 1.5 times the CPU's forward rate and 2.5 times its
train-step rate.
- **Batch size.** A single forward batch of 32 (the first 32 positions)
  took 0.58 s, about 55 positions a second, so batch 64 is past this card's
  best size. The match server's batches are capped at 32.
- **The ROCm build on CPU.** It is far slower than the CPU build: a first
  CPU run under the ROCm venv had not finished its fp32 forwards (768
  positions) after 19 minutes, under 0.7 a second, and was stopped.

**Why it is slow.** A profile of one forward batch of 32 took 0.58 s:
- 65% of the GPU time is `aten::_scaled_dot_product_attention_math`: the
  padded `[B, heads, L, L]` score matrix, the mask and a `_safe_softmax`
  (`where`, `isneginf`, `all`);
- the linear layers take 13%.

On this build, flash and memory-efficient SDPA both fail for gfx1032 in
fp16, fp32 and bf16 ("No available kernel"), so only the math path runs.
The packed trunk (`wesnoth_ai/packed_trunk.py`) needs the flash varlen
kernel, so it is off as well.

**Against the records.** A 4090 box's trainer, configuration B, runs 1,477
positions a second (BACKLOG, 2026-10-15). The pool generates about 2,400
decisions a second (`generation_speed_r5_20261015`). Those figures come from
other code paths: packed bf16 trunks with flash attention, compiled.

## Self-play pilots (`run_elo_batch`, stage 1's player)

Both pilots played `own2` against itself at `raw:t1`, 64 slots, with
`--log-decisions` and `--max-extra-games 0`, persistent workers and shared
inference. Their records are in `%USERPROFILE%\wesnoth_runs`, not
committed.

**GPU pilot** (seeds 870000-870031): 32 games in flight, fp32 serving (`--no-infer-bf16`).
- At 17:01:03, 16 s after the server started, its first forwards failed with
  `hipErrorLaunchFailure` ("unspecified launch failure").
- At the same second, the System log recorded event 4101: "Display driver
  amduw23g-205305-f16b1d83 stopped responding and has successfully
  recovered". That is a Windows TDR, the 2-second GPU timeout.
- The server died and all 32 games failed. The workers do not touch the GPU.
- No other reset was logged that day. The bench (one thread, batch 64,
  16:30 to 16:59), the profile and the attention-backend test before the
  pilot ran without one.
- Not investigated further: each attempt can blank the user's display, and
  can crash the user's GPU-accelerated programs.

**CPU pilot** (seeds 871000-871015): 16 games in flight, the server on CPU.
- No game finished within run_elo_batch's 20-minute per-game limit.
- The server used about 10 cores. It was stopped after 21 minutes.
- It ran under the ROCm venv, whose CPU kernels are the slow ones above. The
  CPU build's 10.2 forwards a second bound a CPU run at about 80 games an
  hour (465 decisions a game), all cores busy.

## Stage 1, re-priced for this machine

The GPU's forward rate bounds its generation rate from above. Over the first
32 bench positions, batch 32 took 0.58 s, about 55 positions a second. At
the argmax game length of about 465 decisions, that is at most about 425
games an hour; the temperature-1 game length is unmeasured.

- **Generation:** stage 1 plays 30,000 self-play games, plus 12,800 for D0
  and 3,200 for the memory report. At that bound it takes at least 108
  hours, and only if the TDR is solved. The 4090 box plan is about 3.1 hours
  and $1.8, at most $3.75.
- **Stage C:** its gradient pass at 3 positions a second, against the 500 to
  1,500 a second that stage C's plan assumes, is out of reach here.

## After the kernels (the evening of 2026-10-28)

The user asked for every drop of the card, custom kernels included. Three
changes made it a generation box (commits 085117c8 and 115be5d1, and
docs/selfplay_t1_local_prereg_20261028.md):

**1. A varlen attention kernel in HIP** (`wesnoth_ai/hip_kernels.py`).
- hiprtc, which ships in the ROCm SDK wheel, compiles it at run time, and
  ctypes launches it on torch's stream. No host compiler is needed; the
  machine has none.
- The packed trunk (`wesnoth_ai/packed_trunk.py`), which skips the padding
  but had only flash as a kernel, runs on it.

One layer's attention over 48 segments of 141-692 tokens, 12 heads of 32:

| version | time | rate |
|---|---|---|
| padded math SDPA, fp16 | 47 ms (32 segments) | |
| one query row a thread | 3.65 ms | 4.3 TFLOP/s |
| two rows a thread | 2.0 ms | 7.8 TFLOP/s |
| three rows a thread, 32 threads, 16-key tiles | 1.65 ms | 9.5 TFLOP/s |
| four rows a thread | | registers spill |

**2. fp16 serving** (`--infer-fp16`). This card runs bf16 matmuls at a
fifth of fp16's rate.

**3. A fused residual add and layer norm** in the model's ROCm fp16 loop.
It writes the float32 residual and its fp16 copy in one pass: 0.52 ms a
layer against 0.97 for the add, norm and cast it replaces.

`own2` on the 200 bench positions, positions a second, against the padded
fp32 trunk on the real slots:

| path | batch 24 | batch 48 | largest logit difference | actor argmax |
|---|---|---|---|---|
| padded fp32 | 42 (batch 32) | | (reference) | |
| packed fp16, attention v1 | 429 (batch 32) | 429 (batch 64) | 0.054 | 200/200 |
| packed fp16, attention v2 | 503 | 494 | | |
| packed fp16, v2 and the fused loop | 539 | 535 | 0.038 | 200/200 |

Where a 48-leaf batch's time goes now (`T` = 20,693 tokens, one layer):
- the four GEMMs, 6.95 ms at 9.5-12.1 TFLOP/s, near what rocBLAS reaches
  on this card;
- the attention, 1.65 ms;
- the two add-and-norms, 1.04 ms;
- the ReLU, 0.66 ms. hipBLASLt's fused bias-ReLU was slower: 3.28 ms
  against 3.05 for the GEMM then the ReLU.

**Self-play** (`run_elo_batch`, `own2` against itself at temperature 1, 64
slots, 192 games, 96 in flight, eight a worker):

| path | steady decisions/s | mean batch | games |
|---|---|---|---|
| attention v1, unfused | 361 | 37.6 | 192 in 200 s |
| attention v2, fused loop | 457 | 30.1 | 192 in 163 s |

- The server waits on the GPU for most of its time (`t_finish` 126 of
  about 140 s).
- A worker spends 1.4 ms of CPU a decision.
- The games average 378.6 decisions over 17.1 turns, with no capped game.
- No display-driver reset was logged during the pilots.
- On the 192 games, stage 1's walk, labels and readout run on Windows and
  pass all five gates (I-sample, legal mass, I-count, dice control, I-V).

**Stage 1, re-priced.** About 4,300 temperature-1 games an hour, about
11.5 hours for the design's 46,000 games
(docs/selfplay_t1_local_prereg_20261028.md).

## Reproduce

```
set HIP_VISIBLE_DEVICES=1
%USERPROFILE%\venvs\wesnoth_rocm\Scripts\python.exe training/metrics/local_gpu_20261028/bench_local.py ^
    --root . --dataset <repo>/replays_dataset_imitation --device cuda --out rates_cuda.json
```
