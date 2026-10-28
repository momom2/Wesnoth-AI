# The local machine's GPU, measured (2026-10-28)

The user's desktop, offered for tests and training on 2026-10-28. Question:
can stage 1 of the training design (`scripts/selfplay_t1_box.sh`, priced
for a 4090 box) run here instead of on Vast?

**Answer: no, not at the design's scale.** The GPU runs this model about two
orders of magnitude slower than a 4090 box. ROCm on Windows has no fused
attention kernel for its architecture, and a self-play pilot reset the
display driver within seconds. Stage 1 stays a Vast run.

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

## Reproduce

```
set HIP_VISIBLE_DEVICES=1
%USERPROFILE%\venvs\wesnoth_rocm\Scripts\python.exe training/metrics/local_gpu_20261028/bench_local.py ^
    --root . --dataset <repo>/replays_dataset_imitation --device cuda --out rates_cuda.json
```
