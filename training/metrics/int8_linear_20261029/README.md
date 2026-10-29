# int8 linear layers on the RX 6600 (2026-10-29)

Pre-registration: `docs/int8_linear_prereg_20261029.md` (committed before
any number). Code: `wesnoth_ai/hip_int8.py` (the kernels),
`wesnoth_ai/int8_trunk.py` (the trunk loop and arm S's calibration),
`tests/test_hip_int8.py`. Branch `feature/int8-linear`.

**Answer: rejected.** The pre-registered verdict: "rejected: int8 linears
with per-token and per-channel scales, because every head's largest logit
error is 1.8 to 4.3 times bf16's, with 7 actor argmax flips against bf16's
1".
- **Speed.** The int8 GEMM runs own2's four linears 2.0 times as fast as
  rocBLAS fp16.
- **Accuracy.** int8's rounding moves own2's logits several times further
  from fp32 than bf16's does. Both pre-registered arms fail the gate:
  plain per-token and per-channel scales (P), and SmoothQuant's migration
  (S). No single linear class on int8 passes it alone either.
- **What stays.** Nothing is wired; the kernels stay as the record. The
  whole-forward rate was not measured, since the prereg measures it only
  after a pass.

## The kernels (`wesnoth_ai/hip_int8.py`)

They are compiled by hiprtc at first use, launched through ctypes on
torch's stream and self-tested per device. `hip_int8_applies` follows the
pattern of `wesnoth_ai/hip_kernels.py`.

- **`gemm_i8`** computes `out = half((x_q . w_q^T) * row_scale * col_scale
  + bias)`, with an optional ReLU. The products are exact in int32 by
  `v_dot4_i32_i8` (`__builtin_amdgcn_sdot4`; the compiler emits the VOP2
  form `v_dot4c_i32_i8`).
  - **Tiling.** A block of 256 threads computes a 128 x 128 output tile.
    Each thread holds 8 x 8 int32 accumulators.
  - **K tiles.** K is walked in 32-byte tiles. Each thread loads one
    16-byte chunk of A's and B's tile rows into registers for the next
    tile, and stores the current one to shared memory transposed:
    `[K/4][128]` dwords, so a dword is four K values of one row.
  - **Column groups.** A thread's 8 columns are two groups of 4, 64
    columns apart. Its 16-byte reads of B's tile then cover consecutive
    dwords across threads, and its output rows go out in coalesced
    8-byte stores.
  - **The main loop** (llvm-objdump of the code object) is 512
    `v_dot4c_i32_i8` for 32 `ds_read_b128` per K tile, plus two
    barriers, two global loads and four `ds_write2st64_b32`.
  - **Resources.** 105 VGPRs, no spill, 8 KB of LDS a block.
- **`quant_rows`** turns fp16 rows into int8 rows with a scale per row:
  the row's largest magnitude over 127, each value rounded to nearest with
  ties to even. One wave32 handles a row. It is bit-exact against
  `quantize_rows_reference`.
- **`add_ln_q`** is `hip_kernels.add_ln` with its float32 output quantized
  for the next linear, in place of add_ln's fp16 copy. It costs what
  add_ln does.
- **Smoothing variants** (arm S) of both quantizers multiply each column
  by a factor first.
- **Bounds.** Every launch's grid follows the problem's size and every
  loop is bounded by K (at most 8,192), so no launch can run long.

## Rates (`gemm_rates.json`, `bench_int8_gemm.py`)

- **Setup.** M = 20,693 tokens (a 48-leaf batch), 60 rounds. Each kernel
  and `F.linear` fp16 run interleaved call by call, in an order reversed
  every other round, timed by CUDA events.
- **Contention.** The stage 1 generation run shared the GPU throughout,
  so the absolute rates read low. The ratios are the reading.
- **Medians.** These are the medians of the rounds. The fastest calls are
  not used: one int8 ff1 call timed at 0.29 ms, 83 TOPS, which is above
  the card's 35.7 TOPS peak and so a timing artifact.

| shape | fp16 ms | int8 ms | fp16/int8 (medians) | median of per-round ratios | fp16 TFLOP/s | int8 TOPS |
|---|---|---|---|---|---|---|
| qkv [M,384]x[384,1152] | 2.065 | 0.840 | 2.46 | 2.24 | 8.9 | 21.8 |
| out [M,384]x[384,384] | 0.802 | 0.376 | 2.13 | 2.16 | 7.6 | 16.2 |
| ff1 [M,384]x[384,1536], ReLU fused | 2.668 | 1.034 | 2.58 | 2.67 | 9.1 | 23.6 |
| ff2 [M,1536]x[1536,384] | 2.601 | 1.768 | 1.47 | 1.60 | 9.4 | 13.8 |
| **sum of the four** | 8.135 | 4.018 | **2.02** | | | |

- **An earlier run.** The same comparison before this record's (60
  rounds, quoted in commit 0ef0c88d) read 2.06 summed. Its fastest calls
  read 19.9 to 26.7 TOPS, 56 to 75% of the dot4 peak, but fastest calls
  are not the reading here.
- **ff2 is the weakest shape.** It has 486 blocks of 48 K tiles each, and
  one LDS buffer with two barriers per tile. No further tuning was done
  once the gate had failed.
- **What int8 adds** (medians):
  - quantizing the attention output costs 0.17 ms, and the ReLU output
    (ff2's input) 0.61 ms;
  - `add_ln_q` costs 0.61 ms against `add_ln`'s 0.72;
  - the fp16 path's separate ReLU, 0.69 ms, is fused away.
- **Derived, not measured.** A layer's linears, ReLU and quantizers would
  have taken about 4.8 ms against fp16's 8.8 (medians). With the linears
  at about 62% of a forward's GPU time, the forward would have gained
  roughly 1.3 to 1.4 times.

## Accuracy: the gate (`accuracy_P.json`, `accuracy_PS.json`, `int8_accuracy.py`)

- **Reference.** own2 on the 200 bench positions, each with the 64-slot
  initial memory, in batches of 24. The reference is the padded trunk in
  float32.
- **E.** The largest |X - R| over the largest |R| per head, over the real
  slots. **D** counts the actor argmax disagreements.
- **The bound** is bf16 measured in the same process.

**Arm P**, read first (`accuracy_P.json`):

| path | actor | type | target | weapon | value | D | mean actor TV |
|---|---|---|---|---|---|---|---|
| bf16 (the bound) | 0.0207 | 0.0197 | 0.0129 | 0.0154 | 0.0078 | 1 | 0.0025 |
| fp16 (serving now) | 0.0011 | 0.0015 | 0.0016 | 0.0014 | 0.0007 | 0 | 0.0003 |
| int8, arm P | 0.0675 | 0.0708 | 0.0554 | 0.0558 | 0.0143 | 7 | 0.0113 |

P fails all six checks: E runs 3.3, 3.6, 4.3, 3.6 and 1.8 times bf16's,
and D is 7 against 1.

**Arm S**, with P rerun in the same process (`accuracy_PS.json`):

| path | actor | type | target | weapon | value | D | mean actor TV |
|---|---|---|---|---|---|---|---|
| bf16 (the bound) | 0.0165 | 0.0130 | 0.0115 | 0.0154 | 0.0078 | 1 | 0.0024 |
| fp16 | 0.0020 | 0.0015 | 0.0017 | 0.0014 | 0.0007 | 0 | 0.0003 |
| int8, arm P | 0.0503 | 0.0601 | 0.0598 | 0.0621 | 0.0143 | 7 | 0.0113 |
| int8, arm S | 0.4480 | 0.4680 | 1.0514 | 0.7374 | 0.0491 | 5 | 0.0143 |

S fails all six checks: E runs 27 to 92 times bf16's on the policy heads.

**Why S failed.** It was run as pre-registered, alpha 0.5 with
SmoothQuant's reference clamp of the factors at 1e-5, calibrated on the
200 non-holdout positions of `calibration_states.json` (the prereg's
"Deviation").
- **The floor.** It bound 3,819 channel factors: inputs whose calibration
  range was zero or nearly so, presumably ReLU units that never fired on
  those positions.
- **The effect.** Such a channel's input is multiplied by up to 1e5. When
  it fires on a bench position, it takes over its token's scale and
  rounds the rest of the row to zero.
- **The evidence.** S on ff2 alone carries the whole failure (E 0.45 to
  1.05). S on the norms' outputs helps: qkv alone goes from 0.053 to
  0.039 on the actor head, ff1 alone from 0.078 to 0.021.

## Accuracy: the diagnostics (where the error comes from)

Each linear class alone on int8, the rest of the fp16 path unchanged, all
from the second process (`accuracy_PS.json`). Each row gives E over bf16's
E per head, then D.

| on int8 | actor | type | target | weapon | value | D (bf16: 1) |
|---|---|---|---|---|---|---|
| qkv only, P | 3.19 | 2.68 | 3.62 | 2.00 | 1.04 | 7 |
| out only, P | 1.46 | 0.93 | 0.98 | 0.76 | 0.41 | 2 |
| ff1 only, P | 4.72 | 2.17 | 3.20 | 1.61 | 0.90 | 4 |
| ff2 only, P | 0.70 | 0.60 | 1.21 | 0.82 | 0.46 | 2 |
| qkv only, S | 2.37 | 2.29 | 2.62 | 2.11 | 0.73 | 1 |
| out only, S | 1.55 | 1.16 | 1.04 | 0.83 | 0.49 | 1 |
| ff1 only, S | 1.25 | 0.99 | 1.27 | 1.76 | 0.58 | 1 |
| ff2 only, S | 27.07 | 36.23 | 91.22 | 48.03 | 6.44 | 2 |

The first process's P rows (`accuracy_P.json`) rank the classes the same
way: qkv 2.50 and ff1 2.17 on the actor head, out 1.14, ff2 0.96.

- **The worst inputs.** The error sits on the layer norms' outputs (the
  inputs of qkv and ff1), as in SmoothQuant's per-channel outliers. The
  ReLU's heavy-tailed output, predicted the worst, is among the mildest.
- **No configuration passes.** There are 10 int8 configurations, with
  15 readings over the two processes, and none passes the full gate, even
  with one linear class alone. The nearest is P's ff2 alone. It flips 2
  argmaxes against bf16's 1 in both processes. In the second it also
  reads 1.21 times bf16 on the target head.

## Run-to-run variation

The two processes read different E for the same paths.
- **Same in both.** The value head and D are identical across the two runs.
- **Different.** bf16's actor E is 0.0207 in one and 0.0165 in the other,
  int8's 0.0675 against 0.0503, and P's ff1 alone 0.0450 against 0.0780.
  A largest error over a quantized path moves with small changes of its
  input.
- **Within a process, identical.** In one process the encodings and the
  fp32 padded forward are bit-identical across two calls (checked on 24
  positions).
- **On the CPU, identical.** The encodings are identical across processes
  and across `PYTHONHASHSEED`.
- **Cause.** Not pinned. A plausible one, not checked, is rocBLAS
  choosing another kernel for other operand alignments in another
  process.
- **Effect on the verdict.** None: every comparison is within one
  process, and the arms fail by 1.8 times or more in both runs.

## The GPU

The 46,000-game stage 1 run owned the GPU during every measurement.
- **Kernels.** Each new kernel ran first in the tiny self-test problems
  (M up to 131, K up to 96), then at M up to 1,000, before any full-size
  launch.
- **Processes.** Each GPU process lasted under 2 minutes of GPU work:
  - the two accuracy runs: 17 to 28 s for the padded paths, 2 to 4 s
    for each packed path, 8 s of calibration;
  - each GEMM bench: about 5 s.
- **Resets.** The System log shows no event 4101 (display driver reset)
  on 2026-10-29. It was checked before the first GPU process, between the
  sessions and after the last.

## What is left

- **Nothing is wired.** There is no `infer_int8`, no `--infer-int8`, and
  no record field.
- **Kept as the record:** the kernels, their tests and these scripts.
- **The nearest lead:** smoothing on the norms' outputs only (qkv and
  ff1), plain P on out and ff2, and finer activation scales (one per
  32 or 64 channels of a token, with the GEMM rescaling its int32
  partial sums per group). It would need its own pre-registration.
  - From these diagnostics, even the best single classes sit near or
    above bf16 on some head, so an all-int8 trunk is unlikely to meet
    this yardstick.
  - A looser yardstick, such as a play check of an int8-served player
    against an fp16-served one, would be a different question for the
    user.

## Reproduce

```
set HIP_VISIBLE_DEVICES=1
%USERPROFILE%\venvs\wesnoth_rocm\Scripts\python.exe training/metrics/int8_linear_20261029/bench_int8_gemm.py ^
    --root . --out gemm_rates.json --rounds 60
%USERPROFILE%\venvs\wesnoth_rocm\Scripts\python.exe training/metrics/int8_linear_20261029/int8_accuracy.py ^
    --root . --dataset <repo>/replays_dataset_imitation --out accuracy_PS.json --diagnostics --arm-s
```

`accuracy_P.json` is the same script's first run, without `--arm-s`.
