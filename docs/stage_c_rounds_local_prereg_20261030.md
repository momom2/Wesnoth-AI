# Stage C and the rounds on the local GPU (pre-registration, 2026-10-30)

Branch `feature/pg-corrections` (on `feature/pg-fp16`). Written during the
user's 10-day absence (docs/autonomy_20261029.md), while the RX 6600's HIP
compute was wedged (training/metrics/pg_fp16_20261030). Nothing here has
run on the GPU. It is the plan for the first session after the user's
reboot, with the user present.

The design is `docs/training_design_20261019/current_design.md`, §1, §5 and
§6. The user approved stage C on the local stage 1's fp16-served games
(2026-10-29, docs/local_machine_use_20261029.md, decision 3).

## Order

1. **The GPU's health**, then the hang protocol of
   training/metrics/pg_fp16_20261030 ("The protocol"). Nothing else runs on
   the GPU until the 2-game probe is clean and `accuracy.py` reads the fp16
   evaluator against fp32.
2. **The full suite** (`pytest -m ""`) at the commit that launches, in place
   of CI (the branch cannot be pushed). Its prereg addendum names this
   machine's failures. The GPU tests of `tests/test_hip_kernels.py` run under
   the ROCm venv.
3. **Stage C:**

       HIP_VISIBLE_DEVICES=1 <rocm venv>/python scripts/stage_c_local.py \
           --stage1 C:/Users/momom2/wesnoth_runs/selfplay_t1_local --out C:/Users/momom2/wesnoth_runs/stage_c_local --keep-awake

   Its parts:
   - the corrections fitted on stage 1's held-out games;
   - C0: the pass on 200 games with the full floor, and the forwards on 100
     held-out games;
   - the plan;
   - the pass on 20,000 games;
   - the driver;
   - the plant;
   - the verdict.
4. **P and the link match**, by design §6, if stage C's verdict says
   rounds. Neither is built yet.
5. **The rounds:**

       HIP_VISIBLE_DEVICES=1 <rocm venv>/python scripts/rounds_local.py \
           --stage1 C:/Users/momom2/wesnoth_runs/selfplay_t1_local \
           --stage-c C:/Users/momom2/wesnoth_runs/stage_c_local --out C:/Users/momom2/wesnoth_runs/rounds_local --keep-awake

## What differs from the design's box run, and why

1. **Numerics.**
   - The passes run fp16 on the packed trunk with the HIP backward
     (`--fp16 --packed-trunk`, loss-scaled), and the forwards run fp16 on
     the fused loop (`fp16=1`).
   - The fused loop's bias fold is **off** in every pg tool. That route
     hung the GPU with the fold on, and the unfolded loop is the one that
     served stage 1's games.
   - The replay identity holds the trainer to the logs at the fp16 floor
     (limit about 1.1e-4, docs/stage1_local_handoff_20261029.md §4).
   - fp16 rounds a log probability near −5 by about 0.002 nats. So
     independent rounding of two laws lifts a KL reading by about 2e-6 a
     decision, under 1% of ε (0.0021). `accuracy.py` measures both before
     any run.
2. **The labels' per-decision corrections, built here, enter.** A decision
   weighs R − α·v̂_t − γ·βᵀx_t (`tools/pg_corrections.py`,
   `pg_pass --corrections`).
   - Stage 1 found that v̂ enters (E[R²] 0.880 → 0.643). The luck dealt so
     far enters beside it (0.596; training/metrics/luck_baseline_20261030).
   - On stage 1's 10,000 held-out games: α 0.830 and γ −0.876, both
     entering. The luck term's share is 5.08%, just above the rule's 5%.
   - I-V passes for both (|z| ≤ 0.76 on held-out games, ≤ 1.28 on all
     games).
   - The box script's stage C (`scripts/stage_c_box.sh`) predates the
     corrections. A box run of it would train on per-side R.
3. **The floor:**
   - the main and planted passes read 100 game-sides, not 400 (C0 still
     reads all of its 400);
   - the floor's padded fp32 forward runs 8 rows at a time;
   - at 100 sides the limit moves from about 1.09e-4 to 1.10e-4.
4. **No dead-man's switch, no uploads.** A STOP file and display resets end
   a run between steps, and a new start resumes.
5. **The gate's prices are the box's** (`PRICE_PER_GAME`, `ROUND_FIXED`):
   the design's yardstick for a round's size, not this machine's costs.
6. **The rounds:**
   - round 1 is stage C's decision taken as read (`pg_round --adopt-from`);
   - later rounds play N 20,000 + H 10,000 games of the policy on seeds from
     1,000,000 + 100,000·k;
   - claims run every third round (12,800 decisive games against the
     reference at temperature 1, as D0);
   - the pause and transfer stops follow §6 (`scripts/rounds_local.py`);
   - b and β are fitted on each round's own held-out games, which it does
     not train on. The design's text has them from the previous round's;
     both are fixed before the games they label.

## Expected walls [U]

Every rate is unmeasured on the GPU. C0 measures the pass's and the
forwards' rates, and the plan prices the rest. From the train-step bench
(150 positions a second, batch 32) and the forward (574 positions a
second), stage 1's 378 decisions a game give these estimates:

| Part | Estimate |
|---|---|
| Stage C's pass | 14 to 26 h |
| The driver's forwards (11.25 forward-equivalents over 3.8M held-out decisions) | 20 to 25 h |
| The plant | 8 to 12 h |
| A later round | 30 to 45 h: about 6 h of play (pipelined), 14 to 26 of pass, 6 to 10 of step |

Stage C would then take about 2 to 2.5 days, and each round about 1.5 to 2
days. The rounds' size and count are stage C's gate's to set.

## Kill criteria

- A display reset during any GPU step: the run stops (STOP), and the cause
  is found before anything resumes.
- The replay identity, the floor's cap or the path check failing (pg_pass
  exits 3, 4, 5): the part fails and the chain stops.
- Stage C's verdict "neither": no rounds; the user decides.
- The rounds' pause or transfer stop (§6): the chain stops for the user.

## Decisions per outcome

Design §6, unchanged:
- the gate on round 1's candidate;
- the optimizer by §2's criterion;
- P before rounds;
- the link match iff the sharpening slope is at least 3 SE. It reads
  +2.64 ± 0.08, so it runs.
