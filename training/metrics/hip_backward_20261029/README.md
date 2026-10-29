# The packed trunk trains on the RX 6600 (2026-10-29)

Question: with a backward for the HIP attention kernel, how fast does `own2`
train on the local RX 6600? On 2026-10-28 its train step ran at about 3
positions a second (`training/metrics/local_gpu_20261028/README.md`), so any
training here was pointless: with a gradient wanted, the packed trunk fell
back to per-segment SDPA, and the trainers ran the padded trunk's math
attention.

**Answer: about 30 times faster.** On the same 64 positions and schedule,
the train step reads 2.96 positions a second in fp32 on the padded trunk,
and 88.5 in fp16 on the packed trunk with the new backward. Over all 200
bench positions it read 90 and 148 in two runs. Every number here was taken
while the stage 1 generation run served `own2` on the same GPU, so all of
them read low; one reading moved by up to 3x from run to run.

## What changed (branch `feature/hip-attention-backward`)

- **`wesnoth_ai/hip_kernels.py`**: a forward variant that also writes each
  row's log-sum-exp, float32 `[heads, total]`. The inference kernel's entry
  point and arithmetic are unchanged: its output equals the variant's bit
  for bit (tested).
- **`wesnoth_ai/hip_attention_grad.py`** (new): FlashAttention-2's backward
  in two HIP kernels, no atomics.
  - dq: a thread owns two query rows (one at head_dim 64). It walks the
    segment's keys in 16-key tiles, recomputes P from q, k and the saved
    log-sum-exp, and also writes D = dO . O.
  - dkdv: a thread owns two key rows (one at head_dim 64) and walks the
    queries in 8-query tiles.
  - The products are `v_dot2_f32_f16` into fp32 accumulators, as in the
    forward; P and dS enter as fp16 pairs.
  - Registers a thread: dq 201, dkdv 244 (head_dim 32); no kernel spills.
  - A `torch.autograd.Function` wraps them. It is self-tested on first use
    per device against a closed-form fp32 reference, with no autograd (see
    "A deadlock" below).
- **`wesnoth_ai/packed_trunk.py`**: `packed_attention` and the custom op's
  eager kernel take it on a ROCm build when the dtype is fp16, the head size
  is instantiated and a gradient is wanted. CUDA and every other case are
  unchanged.
- **`wesnoth_ai/model.py`**: the fused fp16 inference loop has no backward
  and reads a frozen weight copy, so it is skipped when a gradient is wanted.
  An eval-mode forward that is differentiated (the gradient pass) then runs
  `packed_trunk`.
- **`tools/pg_pass.py`**: `--fp16` (the device's float16 autocast with a
  loss scale, `tools/pg_loss_scale.py`) and `--packed-trunk` (the existing
  `PassConfig.packed`). Both are opt-in; defaults are unchanged.

## Correctness

### The kernels against per-segment fp32 SDPA autograd

19 segments of 1-692 tokens, inputs N(0, 4) (`tests/test_hip_kernels.py`).
Each figure is the largest absolute error over the reference's largest
magnitude:

| head_dim, heads | out | dq | dk | dv |
|---|---|---|---|---|
| 32, 12 | 2.9e-4 | 7.0e-4 | 6.4e-4 | 4.5e-4 |
| 64, 6 | 3.2e-4 | 5.6e-4 | 6.9e-4 | 5.4e-4 |

Largest absolute errors at head_dim 32: out 0.0025, dq 0.0099, dk 0.0107,
dv 0.0036.

One segment at a time (1, 2, 15, 16, 17, 33, 64, 65 and 692 tokens), the
worst of the four tensors is at most 9.0e-4 (dk, 2 tokens). In a 1-token
segment, dq and dk are exactly 0, as they should be. The tests bound these
errors at 5e-3, and the self-test at 1e-2.

### `own2`'s parameter gradients against the padded fp32 trunk

8 bench positions, eval mode, a bounded loss (each head's log-sum-exp over
the real slots), loss-scaled; `bench_train.py --check`:

| path | cosine | relative norm of the difference |
|---|---|---|
| fp16 packed, HIP backward | 0.999995 | 0.32% |
| fp16 packed, fp16 SDPA in its place | 0.999997 | 0.25% |
| fp16 padded | 0.999996 | 0.29% |
| fp16 packed, HIP against fp16 SDPA | 0.999995 | 0.31% |
| each path run twice | 1.0 | 0 |

So the HIP backward sits at fp16 autocast's own rounding.

- An earlier check summed the squared logits over the padded slots too, and
  read 6%. That came from the loss, not the kernel: the two trunks fill the
  padded slots differently, and unscaled squared logits underflow in fp16.

## Rates

### `own2`'s train step

Forward, backward and AdamW, batch 32, 64-slot memory, the squared-logit
surrogate of `bench_local.py`; `bench_train.py`:

| path | positions | positions/s | forward / backward / step, ms a batch |
|---|---|---|---|
| fp32, padded trunk (math SDPA) | first 64 | **2.96** | 5946 / 8455 / 57 |
| fp16, packed trunk, HIP backward | first 64 | **88.5** | 163 / 253 / 22 |
| fp16, packed trunk, HIP backward | all 200 | 90.5, 148.0 | 68-144 / 145-220 / 10-21 |

- The two fp16 runs over all 200 positions came 2 minutes apart; only the
  GPU's other load differed.
- Three earlier fp16 runs, with one key row a thread in dkdv, read 80.3,
  85.0 and 82.7.
- The 2026-10-28 fp32 reading was 3.0, so the baseline did not move.

### One layer's attention

48 segments of 141-692 tokens (20,693 tokens), 12 heads of 32, the
2026-10-28 reference problem; the best of 5 interleaved rounds:

| | ms | TFLOP/s |
|---|---|---|
| inference forward | 1.71 | |
| forward with log-sum-exp | 1.82 | 8.6 |
| backward (dq, then dkdv) | 9.0 | 4.3 (FlashAttention-2's count: 2.5 forwards) |

- These kernels compute S and dP twice, so the backward's arithmetic is 3.5
  forwards.
- Alone, dq took 4.7 ms. dkdv took 11.3 ms with one key row a thread and
  16-query tiles, and 5.0-6.7 ms with two rows and 8-query tiles.
- Three rows spill registers and run at 36-68 ms (the tiling comment in
  `hip_attention_grad.py`).
- Same-run noise: one call moved between 2.2 and 6.9 ms from round to round.

### The gradient pass (`tools/pg_pass.py --fp16 --packed-trunk`)

16 of the 128 training games of the 192-game pilot of 2026-10-28
(`pilot_fused_t1_j96`, served in fp16 by the fused loop; labels from
`pilot_post`), copied out read-only. `one_split_labels.py` put all 16 games
in split 0, so their 32 game-sides fill the pass's 32 streams; the pass
runs one split at a time. Default settings otherwise: 32 streams, windows
of 16, recompute on, 3 replay workers, no floor (`--floor-dtype off`).
Summary in `pass_fp16_16games.json`.

| | |
|---|---|
| decisions, windows | 5,411, 27 |
| wall | 188 s |
| rate past the 5 warm-up windows | **21.9 decisions a second** (2,925 in 133.6 s) |
| replay KL estimate | **9.8e-6** nats a decision (limit 1e-4; mean abs 0.0018, largest 0.026) |
| loss scale | 2^10 throughout, no overflow in 27 windows |

- **The rate understates a full pass.** The 32 game-sides end at different
  lengths, and a stream idles once its game-side ends. Past the warm-up, a
  step averaged about 8.3 rows (2,925 decisions over about 352 steps,
  22 windows of up to 16), against the 32 of a pass whose splits hold
  thousands of game-sides.
  - A step took about 0.38 s at that size. If a full step cost the same, the pass
    would run 84 decisions a second; a full step costs more, so the full
    rate lies between 21.9 and 84.
  - The CPU was contended as well: the pass's main process used about three
    cores, beside the generation run's workers.
  - Reading the full-batch rate needs a split large enough to keep the
    streams full: about a hundred games or more, tens of minutes of GPU.
    It is best run once stage 1 is done.
- **The fp16 pass reproduces the logs within the base limit.** The replay
  identity compares the logged log p (the server's fp16 fused loop) with
  the pass's (packed_trunk under fp16 autocast, the same HIP forward). It
  reads a tenth of its limit with no floor lift. The floor itself, the
  serving rounding against the packed fp32 trunk, was not read: its padded
  fp32 forward is the slow path here.
- **No fp32 pass was timed.** The fp32 train step runs at 2.96 positions a
  second, and the pass adds a recomputed forward, so an fp32 pass would run
  below that.

## A deadlock, found and fixed

The backward's first self-test ran through autograd. In the gradient pass,
the kernels' first use comes inside a forward under
`torch.utils.checkpoint(..., use_reentrant=False)` (the pass's recompute).
Two things then deadlocked it:
- the checkpoint's saved-tensor hooks took the self-test's graph;
- its backward recomputed the checkpointed forward, which asked for the
  kernels again under the build's non-reentrant lock.

The first pass sat for 5 minutes at 9 s of CPU, with no display-driver
reset. A script reproduced it: the old self-test never finished in 30 s,
and the new one finished.

The self-test now launches the kernels directly against a closed-form fp32
reference under `no_grad`. Two tests hold it:
- the closed form against SDPA autograd, on the CPU, so CI runs it;
- the first build inside a checkpointed forward, in a thread with a
  timeout.

## Display driver

No event 4101 (TDR) was logged from 09:25, before the first GPU run, to the
end of the session. Every kernel was first run on a 3-token segment, and
each block's loop is bounded by its segment's length.

## Reproduce

```
set HIP_VISIBLE_DEVICES=1
set PY=%USERPROFILE%\venvs\wesnoth_rocm\Scripts\python.exe
rem the train step, fp16 packed, with the gradient check and the attention timing
%PY% training/metrics/hip_backward_20261029/bench_train.py --root . --dataset <repo>/replays_dataset_imitation ^
    --modes fp16_packed --check --attention --out rates_fp16_packed.json
rem the matched comparison on the first 64 positions
%PY% training/metrics/hip_backward_20261029/bench_train.py --root . --dataset <repo>/replays_dataset_imitation ^
    --modes fp32_padded --states 64 --warmup 1 --reps 1 --phase-batches 1 --out rates_fp32_padded_64.json
%PY% training/metrics/hip_backward_20261029/bench_train.py --root . --dataset <repo>/replays_dataset_imitation ^
    --modes fp16_packed --states 64 --warmup 1 --reps 1 --phase-batches 1 --out rates_fp16_packed_64.json
rem the kernels' tests
%PY% -m pytest tests/test_hip_kernels.py
```

The gradient pass (a copy of the pilot's games and labels in `<dir>`):

```
python training/metrics/hip_backward_20261029/one_split_labels.py <dir>   # reads <dir>/labels_pilot.jsonl
%PY% tools/pg_pass.py --checkpoint training/checkpoints/own2.pt --games <dir>/games ^
    --labels <dir>/labels_16_one_split.jsonl --out <dir>/pass_fp16_16.pt --device cuda --workers 3 ^
    --fp16 --packed-trunk --floor-dtype off --fresh
```

## What is left, and the user's decisions

- **The imitation trainers have no fp16 option.** `tools/sequence_train.py`
  and `tools/supervised_train.py` still offer bf16 only, which runs at a
  fifth of fp16's matmul rate on this card. Adding fp16 means a GradScaler
  through each trainer's gradient clip and its non-finite-step rule:
  `sequence_train` stops after NONFINITE_LIMIT non-finite steps in a row,
  and a scaler's expected overflows would trip it. It also touches the
  telemetry's per-term backward passes and the optimizer variants. That is
  engineering, not a design conflict. But these trainers are imitation's,
  parked since 2026-10-04, and their packed trunk is "other arithmetic, its
  own acceptance". Whether to build it is the user's call.
  - `sequence_train --packed-trunk` also requires the bf16 autocast
    (`tools/sequence_args.py:163`). On this card that falls back to
    per-segment SDPA in bf16: the new backward serves fp16 only.
- **Stage C on this machine.** The local stage 1 prereg
  (`docs/selfplay_t1_local_prereg_20261028.md`, "What differs", item 3) says
  this GPU's rates price nothing, at 3 positions a second. They now do:
  - Stage C's design runs the pass in fp32 on the padded trunk, with `--bf16`
    as its low-precision option. `--fp16 --packed-trunk` is a third
    arithmetic, held to the replay identity like `--bf16`. Here it read a
    tenth of the limit against fp16-served games.
  - Whether stage C (C0, the pass, the arms) may run here in fp16, and at
    which full-batch rate, is the user's decision; the full-batch rate is
    unread.
- **Not measured:** the floor in fp16 (`--floor-dtype float16`), and any
  rate on an idle GPU.
