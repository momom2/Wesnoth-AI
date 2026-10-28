# Stage 1 on the local GPU (pre-registration, 2026-10-28)

Branch `feature/local-gpu-stage1` (stages 0, 1 and C from
`feature/pg-integration`, `main` merged in, and the local GPU path below).
Runner `scripts/selfplay_t1_local.py`. The design is
`docs/training_design_20261019/current_design.md`, sections 4-6, stage 1
("no training"); its box script is `scripts/selfplay_t1_box.sh`. This run
trains nothing and changes no network weight.

Tags as in the design: **[M]** measured (record named), **[D]** derived
(inputs named), **[U]** unknown before this run.

## Why here and not on a box

The user gave the desktop's GPU to the project on 2026-10-28 ("you're free
to run tests and training runs at your leisure"). Stage 1 waited on the
user's approval of $1.65 (at most $3.75) on Vast; here it costs nothing but
electricity. Earlier that day the machine ran `own2` at 20-55 positions a
second and a self-play pilot reset the display driver
(`training/metrics/local_gpu_20261028/README.md`). Three commits on this
branch made it a generation box:

- a HIP varlen attention kernel compiled by hiprtc at run time
  (`wesnoth_ai/hip_kernels.py`): ROCm on Windows has no fused attention
  for gfx1032;
- fp16 serving (`--infer-fp16`), since this card's bf16 matmuls run at a
  fifth of fp16's rate;
- a fused residual add and layer norm in the model's ROCm fp16 loop.

`own2`'s forward on the 200 bench positions went from 42 positions a
second (the padded fp32 trunk) to 535 [M]. A 192-game self-play pilot at 96
games in flight served 457 decisions a second at steady state [M,
`C:/Users/momom2/wesnoth_runs/pilot_fused_t1_j96`]. No display-driver reset
was logged during the pilots [M, System event log, event 4101].

## What is the box design's, unchanged

- **The players and decodes.** Self-play: `own2` against itself at
  temperature 1 on both seats, no end_turn offset, 64 memory slots, labels
  `own2a` and `own2b`, every decision logged. D0: `own2` at temperature 1
  (`own2t1`) on side A against the reference at its decode
  (`raw:t0+eo-1.5`, 64 slots) on side B. The memory report: `own2` at
  temperature 1, 64 slots (`own2m64`) against 0 slots (`own2m0`).
- **The games and seeds.** Self-play: 30,000 games in six matches of 5,000,
  seeds 800000-829999, index below 20,000 trained on and the rest held out.
  D0: 12,800 decisive games from 830000, capped slots replaced at seed +
  k x 1,000,000, up to 6,400 replacements. Memory: 3,200 decisive games
  from 845000, up to 1,600 replacements.
- **Acceptance.** A short match is played again once. A self-play match is
  accepted with at most 10 games missing, and a failed one stops the
  self-play. D0 and the memory report are accepted on run_elo_batch's exit
  0 or 4.
- **The readouts.** `tools/selfplay_rows.py`, `tools/pg_labels.py`
  (labels.jsonl, fits.json), `tools/selfplay_readout.py` (readout.json:
  the census, E[R²] by correction, the sharpening slope, and the gates
  I-sample, legal mass, I-count, dice control and I-V);
  `tools/elo_collect.py` fits for D0 (anchor `own2`) and the memory report
  (anchor `own2m0`).
- **The decisions per outcome.** Those of the design's section 6:
  - a red gate is fixed before stage C;
  - corrections enter by section 4's rule;
  - a cap rate above 0.009 is the user's ruling on cap scoring.

## What differs, and why

1. **Numerics.** The server autocasts to fp16, not bf16. The packed trunk
   runs on the HIP kernels and the fused fp16 loop, not flash attention on
   a 4090. Every game records `infer_fp16: true`, and `elo_collect` never
   pools it with bf16 games.
   - fp16 is the finer rounding. Against the padded fp32 trunk on the
     bench positions, the logits differ by at most 0.04 over the real
     slots, with actor argmax 200/200 [M].
   - D0 and the memory report are read in this numerics, both sides alike.
2. **The pool.** 96 games in flight, eight per worker process, one server.
3. **No stage C rates.** They price a box's trainer. This GPU trains at
   about 3 positions a second in fp32 [M], so its rates price nothing. The
   box that runs stage C measures them, as C0 does.
4. **No deadline sizing.** No dead-man's switch, so every planned match is
   played. A STOP file in the output directory ends the run between steps,
   and a new start resumes from the markers.
5. **CI.** The branch cannot be pushed from this machine yet: GitHub
   refused the only account signed in, 2026-10-28. Instead, the full
   suite (`pytest -m ""`, both tiers) ran locally on Windows at 115be5d1,
   the code this run serves (the runner and the records came after it).
   - Result: 2,011 passed, 124 skipped, 2 failed, in 11.5 minutes [M].
   - The two failures fail identically on `main` (0e69087e) on this
     machine, so they come from the machine, not the branch:
     - `test_core_source::test_line_ends_and_files_the_build_does_not_read_leave_it`:
       the checkout's autocrlf already gives lib.rs CRLF line ends;
     - `test_secret_guard::test_the_shell_prefix_redacts_both_streams_and_keeps_the_status`:
       the guard's shell prefix prints nothing under Git Bash.

     Neither touches generation.
   - The GPU tests of the kernels (`tests/test_hip_kernels.py`) ran under
     the ROCm venv: 8 passed.
   - The rule that a campaign launches from a green CI run is met when the
     branch is pushed. A red CI run then voids the run, unless its failure
     is shown not to touch generation.
   - The runner was smoke-tested end to end (`--smoke`: 2 x 24 self-play
     games, D0 and the memory report of 24, walk, labels, readout and fits
     in 2.5 minutes), and a second start resumed past every finished step.
6. **Records.** Everything stays in the output directory. At the end,
   the small records (matches.txt, readout.json, fits.json, the two fits,
   the server stats) are copied to
   `training/metrics/selfplay_t1_local_20261028/`. The games (about
   1.5 GB) stay on the machine until the user decides where they go.

## Expected wall [D]

| Step | Games | Rate | Time |
|---|---|---|---|
| Self-play | 30,000 | 4,300 games an hour (the pilot's 457 decisions a second, 378.6 decisions a game) | 7.0 h |
| D0 | 12,800 decisive | [U] games longer at argmax on side B (475.7 decisions a game at argmax, [M] census) | about 3.5 h |
| Memory report | 3,200 decisive | | about 0.8 h |
| Walks, labels, readout | | | about 0.3 h |
| **In all** | | | **about 11.5 h** |

Cost: $0 on Vast. A step past twice its estimate plus 15 minutes is cut
by the runner, and is played again once.

## Kill criteria

- The run stops if the display driver resets: System event 4101 while a
  match runs. The pilots logged none.
- It stops if the GPU path fails (a server's fatal) twice in one match.
- It stops on the user's word: the STOP file, or ending the runner.

None of these is a verdict on stage 1. They decide only where it runs.

## Results

(Written when the run lands.)
