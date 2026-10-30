# Stage C's tools in fp16 on the RX 6600, and the GPU they wedged (2026-10-29/30)

## What changed (branch `feature/pg-fp16`, then `feature/pg-corrections`)

Stage C's forward-only tools ran fp32 on the padded trunk, 20 to 55
positions a second on this card. At that speed stage C's 42.5M held-out
forwards would take about two weeks (a subagent's mapping, 2026-10-29). So:

- `tools/pg_eval.py` gains `EvalConfig.fp16` (`time --fp16`, the factory's
  `fp16=1`): the packed trunk under the device's float16 autocast, with the
  outputs cast to float32 before the log softmax. On a CPU it runs the
  packed trunk in float32.
- **A bug the port exposed:** the evaluator's moved networks are deep
  copies, and they kept θ_k's frozen fp16 weight copy. Every candidate
  would have served θ_k's trunk. `WesnothModel.weights_changed()` now
  refreshes the copy, and a CPU test holds it.
- `tools/pg_plant_grad.py` (`pg_plant delta/readout --fp16`): the same
  autocast, with the pass's loss-scale rule. The weights are normalized to
  1 inside the loss, since a split's 1/(2N) would leave fp16 gradients
  about 2^13 smaller.
- `tools/pg_pass.py`:
  - its rate leaves the floor's forwards out (`floor_seconds`);
  - `--floor-chunk` bounds the floor's padded fp32 forward (about 3 s at
    64 rows here, the kind of call that reset the display on 2026-10-28).
- `WesnothModel.fold_biases` is off in every pg tool (`load_network`): see
  below.

The CPU tests pass. **The GPU side was never measured**:
- `accuracy.py` (fp16 against fp32 on 8 held-out games) never finished;
- the full-batch rates are unread.

## The wedge

The first GPU run of the fp16 evaluator (`accuracy.py`, 2026-10-29 23:42)
reset the display 38 s in. A probe (2 games) wrapped the fused loop's
kernels with a device sync and host checks:
- all 8 layers' kernels finished (0.1-0.2 ms each, cu_seqlens valid, 729
  rows);
- then the GPU hung after the trunk, before the next step's first
  attention.

After that, every new HIP context failed: hipMalloc returned 719
(hipErrorLaunchFailure) in a fresh process, and each attempt itself reset
the display. Nine resets in all, from 23:43 to 00:20. None of these fixed
it:
- waiting;
- keeping the display on (ES_DISPLAY_REQUIRED);
- the driver-restart hotkey;
- `PAL_DISABLE_SDMA=1`.

DirectX kept working. There is no admin right and no auto-logon, so only a
reboot by the user restores HIP (docs/autonomy_20261029.md, the log).

**Ruled out:**
- **Out-of-range indices in the data.** The same walk on the CPU (2 games,
  812 decisions) ran clean, replay identity −4.9e-5.
- **The Python-built layout.** It matches its concatenation on 3,000
  random CPU batches (a subagent's check).
- **Temporary pinned buffers.** Copies from temporaries survived 40 trials
  with queued GPU work.
- **A leaked memory budget.** The adapter counters show 273 MB dedicated.

**Not ruled out, in order:**
1. **The bias-fold loop on the Python-layout route.** Stage 1's 13 h of
   serving ran before the bias fold existed (the frozen fc560ab9). The fold
   has under an hour of server history. Every reset on record that involved
   the fused loop went through a Python-built layout: `bias_fold_ab.py`
   twice on 2026-10-29, and `pg_eval` here.
2. **ROCm 10.1.0's HIP runtime.** TheRock #9564 reports HIP built with
   10.1.0 stalling into repeated driver resets on Windows (gfx1151), which
   10.0.0 did not. Our torch is `+rocm10.1.0`.
3. Something in the heads or memory write under no_grad fp16. A code
   reading found nothing.

## The protocol for the next GPU session (the user present: a wedge costs a reboot)

1. **A health check:** a small matmul.
2. **The probe on 2 games:**
   - with the fold off (now the pg tools' default);
   - `HIP_LAUNCH_BLOCKING=1`;
   - a sync and a log line after each stage: trunk, heads, memory write,
     outputs, the next step's staging.
3. **If it hangs with the fold off:** try a ROCm 10.0.x torch build, which
   would point to the runtime.
4. **Once the probe is clean:** `accuracy.py` (fp16 against fp32 on 8
   games), then the rates:
   - `pg_eval time --fp16` on 200 held-out games;
   - `pg_pass --fp16 --packed-trunk` on 300 training games.
