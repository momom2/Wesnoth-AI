# Pre-registration: int8 linear layers on the RX 6600 (2026-10-29)

Branch `feature/int8-linear`, cut from `feature/local-gpu-stage1`. Written
and committed before any int8 accuracy number exists. Records go to
`training/metrics/int8_linear_20261029/`.

## Why

`own2` serves self-play on the local RX 6600 (gfx1032) at about 470
decisions a second in fp16 (`WesnothModel._run_hip_fp16_trunk`). The four
linear layers of each trunk layer take about 62% of the GPU time: 6.95 ms a
layer for a 48-leaf batch of 20,693 tokens, at 9.5-12.1 TFLOP/s from rocBLAS
fp16 (`training/metrics/local_gpu_20261028/README.md`). fp16's peak on this
card is about 17.9 TFLOP/s. RDNA2's `v_dot4_i32_i8` multiplies four int8
pairs into an int32 sum per lane per instruction, a peak of about 35.7 TOPS.
`torch._int_mm` fails here (hipBLASLt has no gfx1032 kernels), so int8 needs
a kernel of our own.

int8 changes the arithmetic, as fp8 did. fp8 failed its gate on the boxes:
per-tensor scales put own1's actor logits 0.331 of the head's scale from the
bf16 stream, against a bound of 0.25 (`docs/fp8_inference_prereg_20261019.md`,
"Measured"). The project accepts bf16 batched numerics for generation (the
boxes' servers and every box match ran under a bf16 autocast), so the
yardstick here is how far bf16 sits from fp32.

## The scheme

Per trunk layer, the four linears (in-projection, out-projection, both
feed-forward linears) compute

    out = (x_q @ w_q^T) * row_scale[:, None] * col_scale[None, :] + bias

in fp16, the product exact in int32.
- **Weights:** int8, one scale per output channel (absmax / 127, rounded to
  nearest), quantized once from the frozen weights.
- **Activations:** int8, one scale per token (the row's absmax / 127,
  rounded to nearest), read at every call.
- Everything else is the current fp16 path's arithmetic: the float32
  residual stream, the fused add-and-norm, the HIP attention, the heads
  under the fp16 autocast.

**Arm P** is that scheme. **Arm S** adds SmoothQuant's per-input-channel
migration (Xiao et al., "SmoothQuant", 2023), strength alpha = 0.5: for
each linear and input channel k, s_k = max|X_k|^0.5 / max_j|W_jk|^0.5. The
weight is quantized as W s, and the input x / s. The activation ranges
max|X_k| are read in float32 from 200 calibration positions disjoint from
the bench set: the first 200 entries of
`tools/bench_pipeline.build_state_manifest(dataset, 400, seed=2)` whose
game file is not one of `configs/bench_states.json`'s. S runs only if P
fails the gate.

**Deviation (recorded 2026-10-29, after P's reading and before any S
number).** The calibration draw above is infeasible. The corpus holds 369
holdout games of at least 4 turns. The draw at seed 2 exhausted the ladder
ones and found only 52 off the bench set: the bench's 200 positions take
most of them. S's 200 calibration positions are therefore drawn by the
same procedure (`build_state_manifest`'s rule: a uniform game on a ladder
map, a uniform cut turn in [2, n_turns - 1], seed 2) from the corpus's
non-holdout games. No game is shared with the bench set, which is all
holdout. The manifest is
`training/metrics/int8_linear_20261029/calibration_states.json`. Nothing
else about S changes.

## Speed (read first)

- **GEMM:** the kernel against `F.linear` in fp16 at M = 20,693 on the four
  shapes:
  - qkv [M,384]x[384,1152]
  - out [M,384]x[384,384]
  - ff1 [M,384]x[384,1536]
  - ff2 [M,1536]x[1536,384]

  Both run in one process, interleaved call by call (the generation run
  shares the GPU and skews both alike). The reading is the ratio of median
  times, per shape and for the sum of the four. The activation
  quantization's own time is reported beside it.
- **Stop rule:** if, after a fair attempt, the kernel's summed time over
  the four shapes is not below fp16's, the round stops there. The record
  then gives the achieved TOPS, the registers and LDS a block uses, and
  what limited it. No accuracy is measured and nothing is wired.

## Accuracy: the measurement

**Positions.** The 200 positions of `configs/bench_states.json`, rebuilt
from `replays_dataset_imitation` (read only).
- `own2` (`training/checkpoints/own2.pt`), eval mode.
- Each position with the 64-slot initial memory.
- Batches of 24 in the manifest's order; the last batch holds 8.

**The reference R** is the padded trunk in float32, with no autocast.

**The paths compared with R**, all in one process:
- **bf16:** the padded trunk under a bf16 autocast. This card's bf16 path
  has no packed kernel.
- **fp16:** the packed trunk under an fp16 autocast. This is the current
  serving path (`_run_hip_fp16_trunk`).
- **int8 (P, or S):** the fp16 path with the four linears of every layer
  in int8.

**Gated statistics**, over the 200 positions' real slots (`PaddedOutput.sample`):
- **E_h(X)**, for each head h in actor_logits, type_logits, target_logits,
  weapon_logits and value_logits: the largest |X_h - R_h| over all
  positions and entries, divided by the largest |R_h|. This is the fp8
  gate's statistic.
- **D(X):** the number of positions whose actor argmax differs from R's.

**Reported, not gated:**
- the RMS error over R's RMS, per head;
- the mean over positions of the total variation distance between X's and
  R's actor softmax;
- the same E statistic for belief_logits and for the new memory state;
- for arm P, each of the four linear classes alone in int8, a diagnostic
  of where the error comes from.

## The gate and the decisions

**An int8 arm passes** iff E_h(int8) <= E_h(bf16) on all five heads and
D(int8) <= D(bf16), with bf16 measured in the same run. The bounds do not
move after the measurement.

- **P passes:**
  1. Measure `own2`'s whole forward on the bench positions, int8 against
     the current fp16 path, interleaved, at batches 24 and 48
     (positions a second, and their ratio).
  2. Wire int8 behind an explicit opt-in that games record, as
     `--infer-fp16` is: the model attribute `infer_int8`, the server's
     `--infer-int8`, `run_elo_batch --infer-int8`, `infer_int8` in each
     game's record, and an outdir that never mixes it. Nothing changes by
     default.
  3. It is **not** adopted for generation here. Adoption needs a measured
     forward-rate gain and a play check: a match of the int8-served player
     against the fp16-served one, as fp8's design had it. That check is
     proposed separately, after stage 1 frees the GPU.
- **P fails and S passes:** the same, with S.
- **Both fail:** "rejected: int8 linears with per-token and per-channel
  scales, because <the statistic that failed>". The kernel, this
  pre-registration and the measured failure stay on the branch as the
  record, and nothing is wired. The per-class diagnostics say which inputs
  a later round would keep in fp16.

## Predictions (before any kernel ran)

| reading | prediction | range |
|---|---|---|
| int8 GEMM over fp16, summed over the four shapes | 1.5x | 1.1x to 2.0x |
| int8 GEMM, TOPS on ff1 | 18 | 12 to 24 |
| whole forward, int8 over fp16 (if P or S passes) | 1.2x | 1.05x to 1.35x |
| bf16's largest E_h, all heads | 0.02 | 0.005 to 0.08 |
| P passes the gate | probability 0.25 | |
| S passes the gate, given P fails | probability 0.35 | |

Where they come from:
- **Rounding.** bf16 rounds each operand to a relative 2^-8. A per-token
  int8 activation is off by up to 1/254 of its row's largest magnitude:
  about 0.004 to 0.01 of a typical element for a layer norm's output, and
  more for the ReLU's heavy-tailed output, which feeds ff2. A per-channel
  int8 weight is off by about the same.
- **The error budgets.** The int8 path's error is mostly the linears'. The
  bf16 path's also carries bf16 attention and bf16 heads, which the int8
  path keeps in fp16.
- **Speed.** At 8x8 outputs a thread, a 4-byte K step costs 64 dot4
  instructions for four 16-byte LDS reads, so the kernel is VALU-bound. At
  half the dot4 peak it would run about 1.5x rocBLAS's fp16 rate. The
  quantization passes take some of that back.

## Running beside stage 1

A 46,000-game generation run owns the GPU until about 19:30, and the card
drives the display: a launch that stalls it past 2 s resets the driver
(Windows TDR, System event 4101) and stops that run.
- **Kernels:** every new kernel runs first on tiny shapes, its grid and
  loops bounded.
- **GPU processes:** short ones only. Timing runs take at most about 2
  minutes each, and none is left running.
- **Resets:** the System log is checked for event 4101 after each GPU
  session, and any reset is reported.
- **Rates:** read low under the contention, so they are reported only as
  ratios to fp16 measured interleaved.

## Measured (2026-10-29)

Records in `training/metrics/int8_linear_20261029/` (README.md there):
`gemm_rates.json`, `accuracy_P.json` (P's reading), `accuracy_PS.json`
(S's, with P rerun in the same process), `calibration_states.json`.

**Speed passes.** The int8 GEMM's summed time over the four shapes is
below fp16's: fp16 over int8 is 2.02 (ratio of medians, 60 interleaved
rounds), against a stop rule of 1.
- qkv 2.46, out 2.13, ff1 (ReLU fused) 2.58, ff2 1.47.
- The ReLU output's quantization costs 0.61 ms a 48-leaf layer and the
  attention output's 0.17; `add_ln_q` costs what `add_ln` does.

**The gate fails, both arms.** E over bf16's E per head, and D:

| arm | actor | type | target | weapon | value | D (bf16's) |
|---|---|---|---|---|---|---|
| P | 3.26 | 3.59 | 4.30 | 3.64 | 1.82 | 7 (1) |
| P, rerun beside S | 3.04 | 4.61 | 5.22 | 4.04 | 1.82 | 7 (1) |
| S | 27.09 | 35.91 | 91.76 | 48.01 | 6.25 | 5 (1) |

**Verdict: rejected.**
- "Rejected: int8 linears with per-token and per-channel scales, because
  every head's largest logit error is 1.8 to 4.3 times bf16's, with 7
  actor argmax flips against bf16's 1."
- "Rejected: SmoothQuant at alpha 0.5 with its reference clamp, because
  3,819 channels with a near-zero calibration range got the floor's factor
  of 1e5, and S on ff2 alone carries the whole failure (E up to 1.05 of
  the target head's scale)." Those channels are presumably ReLU units
  that never fired in calibration; that was not checked channel by
  channel.

Nothing is wired, and the whole forward was not measured.

**Diagnostics.** The error sits on the layer norms' outputs.
- On the actor head, qkv alone reads 3.2 times bf16 and ff1 alone 4.7.
  Out alone reads 1.5 and ff2 alone 0.7.
- Smoothing helps the norms' outputs: ff1 alone goes from 4.7 to 1.25.
- No int8 configuration measured passes the whole gate, even with one
  linear class alone.

| reading | predicted | measured |
|---|---|---|
| int8 GEMM over fp16, the four shapes summed | 1.5x (1.1x to 2.0x) | 2.02x |
| int8 GEMM on ff1, TOPS | 18 (12 to 24) | 23.6 (median, under contention) |
| whole forward, int8 over fp16 | 1.2x | not measured (gate failed) |
| bf16's largest E over the heads | 0.02 (0.005 to 0.08) | 0.0207, and 0.0165 in the second process |
| P passes | probability 0.25 | no |
| S passes, given P fails | probability 0.35 | no |

**Run-to-run variation.** The two processes read different E for the
same paths: bf16's actor E is 0.0207 in one and 0.0165 in the other.
- The value head and D match in both processes.
- Within a process the forward is bit-reproducible, so the gate's
  comparisons stand.
- The cause is not pinned (README.md, "Run-to-run variation").

No display-driver reset (System event 4101) was logged on 2026-10-29.
