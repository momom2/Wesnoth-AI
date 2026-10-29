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
   - Addendum, 2026-10-29: a HIP attention backward since raised the
     train step to 88.5-148 positions a second in fp16
     (training/metrics/hip_backward_20261029). Stage C is still a box's
     to run and to price, so the run's plan is unchanged.
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

## Incidents

- **2026-10-28 23:59: an automatic restart.** Windows Update restarted
  the machine (System event 1074: svchost, on behalf of NT AUTHORITY)
  47 minutes into the first match, with 3,488 of its 5,000 games played
  at about 470 decisions a second. Every process of the run died; the
  machine then slept at 00:30, with no session holding it awake.
  - Fix (faafcced): the runner's `--keep-awake` holds the system awake
    for its process's lifetime.
  - Not prevented: a restart. Only the user can change Windows Update's
    active hours or pause it.
- **2026-10-29 09:04: the resume refused its own games.** run_elo_batch's
  resume pre-scan expected `infer_bf16` on cuda, while the fp16 games
  record it False. The first match's two attempts each refused at once,
  the runner marked it failed, and D0 began.
  - Fix (fc560ab9): `precision_wants`, with a test.
  - The runner was stopped, the false failure marker removed by hand (both
    noted in runner.log), and the run resumed from fc560ab9 at 09:11. The
    code that plays the games did not change: 115be5d1's.
  - D0's 339 games from that interval stay in its directory; they were
    played under the same flags.
- **2026-10-29 09:17-09:25: a pause** for an A/B of one against two
  inference servers, 192 games each, seeds 872700. One server served 443.5
  decisions a second at steady state, two 454.7: +2.5%, at the noise
  level. One server was kept, and the run resumed at 3,868 of the first
  match's games.

- **2026-10-29 14:06 and 14:18: two display-driver resets, from a
  benchmark run beside the match.** At 14:06:10 and 14:18:09 (System event
  4101), during match s02, a kernel benchmark in another checkout
  (`feature/hip-bias-fold`, training/metrics/hip_bias_fold_20261029)
  queued twelve 48-leaf forwards on the same GPU without waiting for any.
  The benchmark's own process failed both times.
  - The run lost no game: s02 played its 5,000. Its server served about a
    fifth of its rate in the 30 s around the first reset, then recovered.
  - The kill criterion ("the run stops if the display driver resets")
    fired as written: the runner stopped after s02, at 15:02:59, and wrote
    STOP.
  - The cause was outside the run:
    - both resets fell inside the benchmark's two unsynced attempts;
    - the same benchmark waiting after each forward ran five times without
      one;
    - the run had served about 13 hours on the kernels without one.
  - So the run resumed at 15:33:49, from the same frozen checkout (noted in
    runner.log). Meanwhile the GPU served an idle A/B for the next code
    (training/metrics/pipe_buffer_20261029); that code is not the run's.
  - Lesson, for the rules of use (docs/local_machine_use_20261029.md): a
    second heavy GPU program beside a run can reset the display.

## Results

- **D0, landed 2026-10-29 12:31** [M, `d0.fit.json`]: `own2` at temperature
  1 (64 slots, no offset) against the reference at its decode
  (`raw:t0+eo-1.5`) reads **-415.2 +- 5.5 Elo** (PURE, decisive games only,
  the reference at 0). That is 12,800 games, all decisive, none capped,
  with no replacement used, seeds 830000-842799. The match took 9,997 s
  from its resume at 09:44, about 4,600 games an hour; 339 of its games
  were played between 09:04 and 09:10, before the pause.
  - The record it sharpens: on 2026-09-04 the September seed at argmax beat
    the same weights sampling by +412 +- 97 Elo over 40 games
    (docs/raw_argmax_control_20260904.md).
  - The decision it feeds (the design's section 4, "D0"): the debt the
    rounds' claims start from.
- **The run ended 2026-10-29 20:27:18**, every step done
  (`training/metrics/selfplay_t1_local_20261028/`):
  - 30,000 self-play games, D0's 12,800 and the memory report's 3,200;
  - every game decisive, none capped, no failed slot, no replacement used.
  - From the resume at 09:11 to the end took 11.3 hours, three pauses (41
    minutes) included. The 3,488 games played in the 48 minutes before
    the restart of 2026-10-28 were kept. So the run's own time was about
    11.4 hours, against the 11.5 estimated.
  - Self-play served 450-467 requests a second on an uncontended machine
    (s01, s04, s05) [M, `server_stats.txt`], about 4,300 games an hour as
    estimated.

### The census [M, `readout.json`]

| | Temperature 1 (this run) | Argmax (design §4) |
|---|---|---|
| Decisions a game | 378.3 | 475.7 ± 3.2 |
| Decisions a side-game | 189.1 | 237.9 |
| Side-turns a side-game | 16.34 | 20.16 |
| Turns a game | 17.09 | |
| Shares: move, attack, recruit, end_turn | 0.579, 0.237, 0.097, 0.086 | 0.579, 0.237, 0.099, 0.085 |
| Refused recruits a game | 0.0 | |
| Cap rate | 0 of 30,000 | 2 of 6,402 |

- At temperature 1 a game takes 0.795 of argmax's decisions and 0.81 of
  its side-turns, with the same action shares.
- So costs that scale with decisions drop by about a fifth [D].
- The cap rate is below 0.009, so cap scoring needs no ruling.

### The gates: all five pass [M, `readout.json`]

The 18 statistical tests share a family-wise false-alarm rate of 0.01
(|z| ≤ 3.45 each).

- **I-sample**, over 11,348,812 decisions:
  - end_turn χ² 11.3 on 12 dof (p 0.50), attack 7.0 (p 0.86), recruit 7.9
    (p 0.79);
  - the sampled action's cdf position χ² 18.6 on 19 dof (p 0.48);
  - the temperature score z = 1.46 (p 0.14).
- **Legal mass:** the largest deviation from 1 is 4.0e-7, against a
  tolerance of 1e-3.
- **I-count:** no game with a problem.
- **Dice control:** separate code reproduces the luck slope to 8.4e-15. The
  labels' z on β_S reads slope 0.998 ± 0.019.
- **I-V:** all twelve means are within the gates' z; the largest |z| is
  1.75 (b against recruit).

### Label noise and the corrections [M, `readout.json` "label_noise"]

Per side-game:
- E[z²] = 1;
- E[(z − b)²] = 0.983 ± 0.002;
- E[R²] = 0.851 ± 0.004.

Per decision, cross-fitted over two folds by game:

| Label | E[R²] | Cut from luck alone | Share | Enters (2 SE and ≥ 5%) |
|---|---|---|---|---|
| Luck alone | 0.880 ± 0.005 | | | |
| − α·v̂ (the value head) | 0.643 ± 0.005 | 0.237 ± 0.003 | 26.9% | **yes** (α 0.744, 0.757 by fold) |
| − κ·U (the opponent's score) | 0.869 ± 0.005 | 0.010 ± 0.001 | 1.2% | no: below 5% |
| Both | 0.637 ± 0.005 | 0.242 ± 0.003 | 27.5% | U adds 0.005 over v̂ alone |

- So **v̂ enters the labels as a baseline and U does not**. This is the
  admission the design's §4 left to this reading ("Value head as baseline:
  admitted only on stage 1's reading").
- The design's argmax figure, E[R²] 0.699, came from round 9's residual
  script. Whether its estimator is this one's is not checked here.
- In the held-out halves' fits the cells' shrinkage constant reads k ≈
  8.5e8 (132 over all held-out games): the between-cell variance is at its
  floor. So each half's cell means shrink to zero, and the half-fits'
  baseline is the map, the pair and β alone [M, `fits.json`].

### The sharpening slope [M]

The mean over side-games of R · Σ_t (log p_t + H_t) is **+2.64 ± 0.08 (z =
32.4)**: resolved. Sampling colder raises the outcome at first order. D0
shows the same direction in play, though its argmax side also carries the
end_turn offset.

By the design's §4 rule (slope ≥ 3 SE), the link match runs in stage P
(+$0.33 on a box).

### P1's targets [M]

These are Σ R(1[a ∈ X] − p(X)) per game, which stage C's probe gradients
must reproduce:
- end_turn −0.107;
- attack −0.039;
- recruit +0.027.

### The memory at temperature 1 [M, `memory.fit.json`]

`own2` at 64 slots against the same weights at 0 slots, both at temperature
1, reads **+1.3 ± 6.1 Elo** over 3,200 decisive games (1,606 to 1,594).
At temperature 1 the memory costs nothing either, as at argmax in round 2
(+13.0 ± 12.3).

### What it decides (the design's §6, stage 1)

- **Gates:** green, so there is nothing to fix.
- **Corrections:** v̂ enters and U does not.
- **Cap rate:** 0, so there is no ruling to make.
- **The link match:** it runs in P.
- **Next is stage C.** It waits on the user's decision about these
  fp16-served games and their upload
  (docs/local_machine_use_20261029.md, decision 3).
