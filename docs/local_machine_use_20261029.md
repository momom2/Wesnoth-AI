# The desktop as a compute source: what it does, and the rules to decide (2026-10-29)

The user gave the desktop's GPU to the project on 2026-10-28 and asked to
settle rules of use together. This note collects the measurements and
the options. The rules themselves are the user's.

## The machine

| | |
|---|---|
| CPU | Ryzen 5 7600, 6 cores, 12 threads |
| Memory | 63 GB |
| GPU | Radeon RX 6600: RDNA2, gfx1032, 8 GB, about 224 GB/s, no matrix units, slow bf16. It drives the display. |
| GPU software | ROCm for Windows, torch 2.14.0+rocm10.1.0 in `%USERPROFILE%\venvs\wesnoth_rocm` |

## What it runs, measured

| Job | Here | On a rented 4090 box |
|---|---|---|
| Self-play generation, `own2` at temperature 1 | 457-470 decisions a second at 96 games in flight (about 4,300 games an hour) | about 2,900 decisions a second |
| A strength match: 800 decisive games at the reference decode | about 15 minutes (estimated from the generation rate, about 476 decisions a game) | about 17 minutes, $0.12 |
| Stage 1 of the training design | about 11.5 hours, $0 | about 3.1 hours, $1.8 |
| The full test suite, both tiers | 11.5 minutes | CI |
| The forward of `own2` | 535 positions a second (fp16, the HIP kernels) | |
| A training step of `own2` | 3 positions a second (fp32, padded math attention) | 1,477 positions a second |

The training rate is what the backward-kernel work on branch
`feature/hip-attention-backward` set out to change (TO FILL with its
result).

Sources: `training/metrics/local_gpu_20261028/README.md` and
`docs/selfplay_t1_local_prereg_20261028.md`.

## What a run costs the user

- **The GPU.** A run takes the whole GPU. Windows cannot split an AMD
  card's compute between processes, though a run's memory share can be
  capped. A game or a 3D program then competes with it, and the run slows
  it down.
- **The display.** The card drives the screen, so a kernel that stalls the
  GPU past 2 s resets the display driver (a TDR, System event 4101). That
  happened once, on 2026-10-28 at 17:01, with the old padded path. None
  happened since the kernels, and the runner stops on one.
- **The CPU.** About 2-3 of 12 threads, at below-normal priority: the
  user's programs come first.
- **Sleep and restarts.**
  - `--keep-awake` holds the machine awake while a run lives.
  - Windows Update's automatic restart killed the first stage 1 launch at
    23:59; the run resumed in the morning. Only the user can set active
    hours or pause updates.
- **Electricity: unmeasured.** At a guessed 180 W for the whole machine
  under load and €0.25 a kWh, about €0.045 an hour, €0.5 for stage 1.

## Options for the rules

1. **A monitor on the motherboard's video output.** The RX 6600 then runs
   headless: a run can no longer blank the screen, and desktop work stops
   competing with it. The CPU's iGPU drives the display. This costs a cable
   move, and is the single most useful change.
2. **Runs only in agreed windows**, overnight or when the user is away: a
   STOP file ends a run between steps, and every run resumes.
3. **Yield to the user's GPU use (built, off by default).** Windows
   reports GPU use per process (`\GPU Engine(*)\Utilization Percentage`;
   on 2026-10-29 it showed the server at 88-90% compute and nothing else).
   - `--yield-gpu PCT` on both local scripts (`scripts/gpu_yield.py`,
     `local_machine.run_match`) watches every process outside the run once
     a minute.
   - When one has used at least PCT% of a GPU engine for two samples in a
     row, the match stops. The run waits until no such process has been
     seen for `--yield-quiet-min` minutes (10), then runs the match again.
     The match resumes, losing only the games in flight, one to two
     minutes of work.
   - Typing and browsing stay in single digits; a game takes most of an
     engine.
   - Not yet exercised against a real game.
4. **A fixed share.** Half the GPU at all times cannot be had on Windows
   with an AMD card, since compute does not split. The nearest is option 3
   with a schedule.

## Proposed changes to the standing rules (CLAUDE.md)

- "Compute on rented boxes only, proposed with cost first" would become:
  generation, strength matches and tests on the desktop under the user's
  rules of use; rented boxes for training and for runs the desktop cannot
  finish in time, proposed with cost first.
- A strength match served in fp16 records `infer_fp16` and never pools
  with bf16 matches. Whether matches on the desktop chain onto box matches
  is a decision: the numerics differ the way bf16's batched numerics
  already do.
