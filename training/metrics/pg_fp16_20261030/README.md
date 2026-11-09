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

## The second reading (2026-10-30, 06:45-07:30, the lead)

The probe's hang lies between step t's last trunk kernel and step t+1's
first attention, the latter included (its sync never returned). That
stretch holds: the final norm, the gathers, the memory write, the four
heads, the cast to float32, `action_log_probs` and `class_masses`, the
`.cpu()` copies, then step t+1's staging (`stage_raws`, `stage_step`), the
memory router's batch, `embed_staged`, the in-projection, and the bias
variant of the attention.

**Every op in that stretch has run clean on this GPU in another tool,**
except the fold's kernels:
- **The server** (stage 1, 13 h, fc560ab9): `forward_embedded` on the fused
  fp16 loop without the fold, the gathers, the 64-slot memory write and all
  four heads with the value head. Stage 1's records carry v̂, on which the
  labels' α 0.830 was fitted.
- **The pass** (`tools/pg_pass.py`, fp16 with a gradient): the same
  `stage_raws`, `stage_step`, `embed_staged` and `PGMemoryRouter`, and
  `action_log_probs` on the GPU. It takes `packed_trunk`, not the fused
  loop.
- **The runtime** is the same torch `+rocm10.1.0` everywhere.

So the fold is the only code element unique to the runs that reset: the
two `bias_fold_ab.py` attempts and this walk.

**The fold's kernels, read for out-of-range access:**
- `bias_relu_f16` writes exactly `numel / 8` 16-byte chunks. The width is
  a multiple of 8 and the pointer 16-byte aligned (`_aligned`).
- `add_ln_bias_d*` reads `yb[c .. c+3]` for c < d and writes `rows × d`. A
  whole wave exits together on `row >= rows`, before the shuffles.
- The attention's bias variant reads q's bias at `hd·D`, and k's and v's at
  `(heads + hd)·D` and `(2·heads + hd)·D`. That is in range for the
  contiguous `[3, heads, D]` that `_check_bias` enforces.
  - The key loop's bound, `cu[seg+1] − cu[seg]`, is uniform across the
    block.
  - The early return precedes every `__syncthreads`.
- Every launch goes on torch's current stream, so temporaries freed after a
  launch are stream-ordered.

No kernel writes past its output, so nothing points to a corruption that
a later kernel would trip on. With the fold's kernels clean on this
reading, the two live hypotheses remain:
1. the fold's kernels hang or fault on the hardware in a way the reading
   cannot show;
2. an intermittent stall of the 10.1.0 runtime (TheRock #9564) that the
   fold's extra launches (three per layer) make likelier.

Both are answered by the same first GPU step: the probe with the fold off,
now the default everywhere. **The protocol below is unchanged.** If the
fold-off probe also hangs, the fold is cleared and the runtime is next.

## After the reboot (2026-10-30, 08:33-08:55)

The user rebooted at 08:33. HIP was healthy (a 4096-square fp16 matmul
loop). Logs are in `C:/Users/momom2/wesnoth_runs/gpu_probe_20261030/`.

| Run | Fold | Serialized (`AMD_SERIALIZE_KERNEL=3`) | Result |
|---|---|---|---|
| The 2-game probe (module hooks, a sync per fused kernel; `HIP_LAUNCH_BLOCKING=1`) | off | yes | clean: 812 decisions, both readings, 29 s |
| `accuracy.py`, 8 games | off | no | **launch failure 27 s in, at the laws walk's `nonzero`, with a display reset at 08:44:14**. HIP survived it |
| `accuracy.py`, 8 games | off | yes | clean |
| `pg_eval time --fp16`, 200 held-out games, 6 workers | off | yes | clean: **406 positions a second**, replay identity KL 7.9e-6 |

**So the fold is cleared.** The fault needs unserialized kernels, not the
fold's. The earlier probe's log (`tdr_probe_1.log`, fold on) ends right
after step 0's last trunk kernel: the hang came at that step's first
unsynced op or later.

On the CPU's side the stretch is the same code as the clean runs. Since
serialization hides the fault, the likeliest cause is an asynchronous
hazard on the evaluator's path. That path differs from stage 1's clean
server in two ways:
- its staging (`stage_raws`: a fresh pinned buffer and a non-blocking copy
  per step, plus `host_to_device` copies for the memory router and the
  step's batch), where the server staged through the Rust stage;
- its memory router.

The two HIP runtimes are one: torch's `amdhip64_7.dll` and the kernels'
ctypes handle are the same module.

**fp16 against fp32** (8 games, 2,500 decisions, a random direction at a
KL of about 6.7e-4 a decision):
- θ_k's log-probabilities differ by 0.0017 on average (0.015 at most);
- the KL readings by 0.75% (6.67e-4 against 6.62e-4);
- each game-side's log W by 0.036 on average, against 0.24 for log W
  itself (16 sides).

Stage C runs serialized (its prereg, departure 16). Next, when a reset is
affordable (the user present, or automatic sign-in): the same 8-game run
unserialized with the copies blocking. That separates the copies from the
kernels.

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
