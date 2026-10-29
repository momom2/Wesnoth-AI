# The linear biases folded into the HIP kernels (2026-10-29)

**Question.** On ROCm, `F.linear` with a bias runs addmm, which copies the
bias into the output and then runs the GEMM with beta=1. The lead's probe
measured own2's four GEMMs at a 48-leaf batch's 20,693 tokens, on the busy
GPU: 24.0 ms a layer with their biases and 15.2 ms without. Does the fp16
serving loop (`WesnothModel._run_hip_fp16_trunk`) get faster if the GEMMs
run without biases and each bias is added by the kernel that already reads
that output?

**Answer: yes, about 7% on the whole forward, measured on a contended GPU.**
The change passes all three gates. The kill criterion was a forward less
than 3% faster; it is 7% (an earlier run read 12 to 14%). Every number
here was read while stage 1's
self-play run (96 games in flight) shared the RX 6600, so the timings are
inflated and noisy. The lead confirms on an idle GPU.

## The change

Where each bias goes (commit f490f25c):
- **The in-projection's bias:** the attention adds it as it loads q, k and
  v. This is a third variant of the HIP attention. The add is in fp16, so
  it is correctly rounded, as torch's fp16 add is.
- **The out-projection's and the second feed-forward linear's biases:** the
  add and norm adds them to the sublayer output in float32, before the
  residual add.
- **The first feed-forward linear's bias:** one bias-and-ReLU pass, in
  place on the fp16 output. This applies only where `layer.activation` is
  a ReLU (own2's is `F.relu`); any other activation keeps its linear's bias.

Where the attention's bias variant is not served, the loop runs as before.
That variant has its own self-test on first use, so if it fails, the
training path's kernels still serve.

The add and norm moved to `wesnoth_ai/hip_elementwise.py`, next to the bias
and ReLU, and uses `hip_kernels`' loader. `hip_kernels.py` went from 636 to
615 lines.

**Numerics.** The old loop rounded `acc + b` to fp16 once. The new loop
rounds `acc` to fp16, then adds the bias:
- in the attention and the ReLU, with one more fp16 rounding;
- in the add and norm, in float32, which is closer to fp32 than before.

## Gate 1: tests

- `tests/test_hip_kernels.py`: 26 passed on the RX 6600 (ROCm venv).
- The CPU venv over the model-path test files (packed trunk, trunk modes,
  fp8, compile, memory model, pg pass, inference server and seam, server
  priors): 101 passed, 26 skipped. The skips are the GPU tests.
- ruff is clean.

New tests:
- **The attention's bias variant** is bit for bit the plain kernel run on
  `q + b, k + b, v + b` rounded by torch. It also passes against fp32 SDPA.
- **The add and norm with a bias** matches `F.layer_norm(x + (y + b))` to
  1e-4 in float32.
- **The bias and ReLU** is bit for bit `F.relu` of the fp32 sum rounded to
  fp16. A NaN stays NaN, and the pass runs in place.
- **The fused loop with every bias random** passes against the padded fp32
  trunk and against the unfolded loop, with a ReLU and with a GELU. The
  GELU takes the fallback branch.

The kernels without a bias are unchanged:
- **Bit for bit.** A test pins the SHA-256 of their outputs at 041bd7ca on
  fixed inputs, on gfx1032. This covers the attention, its log-sum-exp
  variant (the training path's forward) and the add and norm.
- **Instruction for instruction.** `same_code.py` compiles both sources for
  gfx1032 with hiprtc and disassembles them; `same_code.json` holds the
  result. All four attention kernels and all five add-and-norm widths are
  identical. So the no-bias kernels' speed is unchanged by construction.
  The interleaved timings below agree within noise: 2.05 ms (old) against
  2.01 ms (new).

## Gate 2: accuracy on own2

The 200 bench positions, with the 64-slot initial memory. Each loop runs
in chunks of 40 and is compared with the padded fp32 trunk over the real
slots (`ab_contended.json`). Largest absolute logit difference:

| | actor | type | target | weapon | value | actor argmax | target argmax (per actor row) |
|---|---|---|---|---|---|---|---|
| old loop vs fp32 | 0.0092 | 0.0055 | 0.0382 | 0.0287 | 0.0299 | 200/200 | 5,249/5,259 |
| new loop vs fp32 | 0.0059 | 0.0079 | 0.0434 | 0.0223 | 0.0269 | 200/200 | 5,247/5,259 |
| new vs old | 0.0078 | 0.0054 | 0.0547 | 0.0234 | 0.0313 | 200/200 | 5,251/5,259 |

The gate was 0.06 at most and 200/200; both hold.
- The old loop reproduces the record's 0.038 on the target head.
- The target rows that disagree are about ten in either loop. The count
  moved by up to two rows between runs of the same old loop (5,247 to
  5,249).

## Gate 3: speed, interleaved in one process (contended)

`bias_fold_ab.py`, from `ab_contended.json`:
- **The old loop** is 041bd7ca's `_run_hip_fp16_trunk`, verbatim, on that
  commit's kernels: the script reads `hip_kernels.py` from git and builds
  it next to the new one.
- **The pairing:** each pair is one batch through the old loop and the new
  one, the order swapped every round. A batch is 48 or 24 bench positions.
- **The rounds:** 16 rounds over every batch, 64 pairs at B=48 and 128 at
  B=24. The ratio is the new time over the old.

| | B=48 (21,975 tokens in its first batch) | B=24 |
|---|---|---|
| forward, median new/old | **0.932** (quartiles 0.825, 1.064) | **0.926** (0.620, 1.363) |
| forward, median ms old to new | 216.6 to 205.1 | 112.8 to 97.9 |
| trunk alone, median new/old | 0.877 (0.751, 0.997) | 0.939 (0.473, 1.919) |

An earlier full run had 12 rounds, before commit 4a4b2146 rewrote the add
and norm's source to the same arithmetic in another instruction order. It
read a forward ratio of 0.883 at B=48 and 0.858 at B=24, and a trunk ratio
of 0.824 and 0.874.

### One layer's kernels

At layer 0's shapes on the first B=48 batch's 21,975 tokens: 60 rounds,
call by call, with CUDA events and the order reversed every other round.
The other process's work lands inside these events, so every median is
inflated.

| call | old (with bias) | new (without) |
|---|---|---|
| GEMM in-projection | 3.80 ms | 3.24 |
| GEMM out-projection | 1.26 | 0.75 |
| GEMM ff1 | 4.13 | 2.17 |
| GEMM ff2 | 2.32 | 2.18 |
| attention | 2.05 (old kernel) | 2.01 (no bias), 2.14 (bias variant) |
| add and norm | 0.598 (old kernel) | 0.597 (no bias), 0.627 (bias variant) |
| activation | 0.785 (`F.relu`) | 0.744 (bias and ReLU, in place) |
| **one layer, summed** | **15.54** | **12.47** (0.80) |

What the table shows:
- Without their biases, the four GEMMs take 3.2 ms less a layer.
- The bias costs the attention about 0.09 ms and each add and norm about
  0.03.
- The bias and ReLU is no slower than the ReLU it replaces.
- On the idle GPU on 2026-10-28, the GEMMs took 6.95 ms a layer with their
  biases, about 89.7 ms of a 48-leaf batch in all. If the same 28% saving
  held there, it would be about 1.9 ms a layer, or 15 ms a batch. That is
  an extrapolation, not a measurement.

## Two display-driver resets, and the fix in the script

The first two attempts at the forward A/B reset the display driver (System
log event 4101, at 14:06:10 and 14:18:09):
- **What failed:** the script's own process (`hipErrorLaunchFailure`).
- **The stage 1 run survived both:** none of its games failed. Around the
  first reset, its server served 2,274 to 2,602 requests in each 30-second
  window, against about 14,000 before. Around the second it served 4,922,
  against about 6,700 at the time. It recovered both times. Its rate had
  halved from 14:14:45, before the second attempt, while other sessions'
  jobs shared the machine.
- **What was running:** both times, the warm-up forwards. These queued
  twelve B=48 forwards with no host sync, and the packed forward issues
  none.
- **What ran cleanly:** every path that syncs per batch. That includes the
  accuracy pass, the tests, and the server itself.

With a sync after every forward (`fwd`), five runs completed without a
reset: one-round probes at B=24 and B=48, two full A/B runs, and a
kernels-only run.

The cause is not established. The likely one is a long stretch of queued
GPU work that Windows could not preempt for the other process within its
2-second timeout. Until it is established, a script that runs beside a
server should:
- sync each batch;
- run the padded fp32 reference in small chunks (`--ref-chunk 4`): at 40
  positions its score matrices hold GBs;
- cap the process's memory (`--memory-fraction 0.25`).

## Reproduce

```
set HIP_VISIBLE_DEVICES=1
%USERPROFILE%\venvs\wesnoth_rocm\Scripts\python.exe training/metrics/hip_bias_fold_20261029/bias_fold_ab.py ^
    --root . --checkpoint <own2.pt> --dataset <repo>/replays_dataset_imitation --out ab.json
%USERPROFILE%\venvs\wesnoth_rocm\Scripts\python.exe training/metrics/hip_bias_fold_20261029/same_code.py ^
    --root . --objdump %USERPROFILE%\venvs\wesnoth_rocm\Lib\site-packages\_rocm_sdk_core\lib\llvm\bin\llvm-objdump.exe
```

`--parts accuracy,forward,kernels` picks the parts, and `--rounds` and
`--kernel-rounds` set the repetitions. The full run took 2 min 44 s on the
contended GPU.

**Not measured:** any of this on an idle GPU, and self-play's decision
rate.
