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
4. **P and the link match**, by design §6, if stage C's verdict says P
   (built 2026-10-30, `scripts/p_local.py`):

       HIP_VISIBLE_DEVICES=1 <rocm venv>/python scripts/p_local.py --stage1 C:/Users/momom2/wesnoth_runs/selfplay_t1_local \
           --stage-c C:/Users/momom2/wesnoth_runs/stage_c_local --out C:/Users/momom2/wesnoth_runs/p_local --e 0.02 --keep-awake

   - **P:** the planted step θ′ (stage C's `planted.pt`) against π₀, 5,000
     games at temperature 1. X per side-game must rise on each seat by 2
     SE, and every game-side θ′ played must carry θ′'s weights.
     - A seat is the game's side, 1 or 2. run_elo_batch keeps θ′ on label
       A and alternates the sides, so θ′'s count on side s is read over the
       games it held s, and π₀'s over the games π₀ held s. The first build
       keyed the seats by label, so in that layout it could never have
       passed (review of 2026-10-30).
   - **The link:** `own2` at T = 1/1.02 against `own2` at T = 1/0.98,
     12,800 games, against the forecast 2e × 2.64 ± 2e × 0.08 =
     **0.106 ± 0.003** in E[z]. It holds within 2 combined SE.
     - It is played only if stage 1's sharpening slope is resolved at 3 SE
       (design §6). It reads 32 SE, so it is played; otherwise `link.json`
       records "not run" and the rounds do not wait on it.
     - Its mean z counts wins, losses and the game's own caps (turns,
       actions, no progress) at 0. A game killed on the wall clock
       (`timeout_kill`) ended on the machine's speed, not on the play, so
       it is left out and counted.
   - **Why e = 0.02, proposed here for the next session to seal before the
     match:**
     - the forecast is exact to second order, so a small e keeps the third
       order negligible;
     - the match's SE over 12,800 games, about 0.009, still resolves the
       forecast at about 12 SE;
     - a miss of 2 combined SE is about 17% of the forecast.
5. **The rounds** run on stage C's gate (`stage_c.json` "gate"). "resolved"
   plays N 20,000 a round, and "rounds_at_size" plays the gain curve's
   `round_size`. "neither", or a gate stage C did not read, plays no round,
   round 1 included, and `--no-p-gate` does not waive this. |H| stays
   10,000 either way. Round 2 waits on stage P's two verdicts
   (`--p-out`):

       HIP_VISIBLE_DEVICES=1 <rocm venv>/python scripts/rounds_local.py \
           --stage1 C:/Users/momom2/wesnoth_runs/selfplay_t1_local \
           --stage-c C:/Users/momom2/wesnoth_runs/stage_c_local --p-out C:/Users/momom2/wesnoth_runs/p_local \
           --out C:/Users/momom2/wesnoth_runs/rounds_local --keep-awake

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
   - later rounds play N (20,000, or the gate's size) + |H| 10,000 games of
     the policy on seeds from 10,000,000 + 1,000,000·k, k ≤ 39;
   - claims run every third round (12,800 decisive games against the
     reference at temperature 1, as D0);
   - the pause and transfer stops follow §6 (`scripts/rounds_local.py`);
   - b and β are fitted on each round's own held-out games, which it does
     not train on. The design's text has them from the previous round's;
     both are fixed before the games they label;
   - when both §6 stops fire after a claim, STOP names both. The claims'
     upper tail is logged, and the rounds go on (design §6: "reported").
7. **P before rounds at the gain curve's size too.** The design's stage C
   row names P after "resolved" only. P asks whether first-order labels
   price play (§6's P row), which round 1's verdict does not decide, so
   stage C's verdict sends "rounds_at_size" through P as well. It costs
   17,800 games of play.
8. **The bias fold is off by default everywhere** (`WesnothModel.fold_biases`,
   since 2026-10-30), not only in the pg tools. Its route hung the GPU, the
   cause is not found, and a hang costs a reboot. The stage 1 games were
   served unfolded. This gives up the fold's share of the serve A/B's +17%
   (training/metrics/pipelined_serve_20261029).
9. **The serve loop is the pipelined one** (`--inference-pipelined`, +9.2%
   decisions a second on the GPU), not the serial loop that served stage 1.
   It changes how forwards are batched, so their fp16 rounding can differ.
   Each pass's replay identity bounds that against the trainer.
10. **P's played-law check is the weights' SHA-256** at θ′'s memory slots
    under `raw:t1`, in place of a law identity. The served numerics are held
    to the trainer by each round's replay identity.
11. **The corrections' coefficients:** whether a correction enters is read
    cross-fitted (two folds of H), and α and γ are then fitted on all of H.
    A missing or non-finite v̂ counts as 0 and is counted (`v_nonfinite` in
    the pass's position counts and in `corrections.json`).
12. **P1 under corrections** holds the probe gradients against the logs'
    Σ R_t(1[c] − p(c)) with R_t the corrected label. That is the design's
    identity with its label corrected, checked in the CI tier
    (tests/test_pg_pass.py).
13. **The human holdout cross-entropy, design §6's per-round crash barrier,
    is not computed.** `tools/holdout_ce.py` exists, but it reads the network
    at 0 slots, through the padded fp32 forward in batches of 64. On the RX
    6600 that is about 3 s a call, the kind of call that reset the display
    on 2026-10-28. Its cost on the CPU is unmeasured. A run of it is a
    follow-up, measured before it is scheduled.
14. **The seeds** are laid out so that no two matches share one (a match's
    replacements step by 1,000,000 a generation):
    - stage 1: 800,000-848,199, its measures' replacements under 10M;
    - the rounds' play: 10M + 1M·k;
    - P: 50M, and the link: 60M;
    - round k's claim: 100M + 10M·k.

    The first build put the claims at 2,000,000 + 100,000·k, inside round
    10's play block and later ones.
15. **Stage C checks stage 1's split:** every label's game index (its seed
    less 800,000) is under 30,000 and held out iff it is at least 20,000.
    The planted labels are cut at the same index. Stage 1's labels pass:
    30,000 games, 10,000 of them held out.

## Expected walls [U]

Every rate is unmeasured on the GPU. C0 measures the pass's and the
forwards' rates, and the plan prices the rest. From the train-step bench
(150 positions a second, batch 32) and the forward (574 positions a
second), stage 1's 378 decisions a game give these estimates:

| Part | Estimate |
|---|---|
| Stage C's pass | 14 to 26 h |
| The driver's forwards (9.3 forward-equivalents over 3.8M held-out decisions when reduced AdamW wins, 11.25 at worst; 2.0 h each at 528 a second with the fold off: training/metrics/stage_c_forwards_20261030) | about 19 h, 22.5 at worst |
| The plant | 8 to 12 h |
| A later round | 30 to 45 h: about 6 h of play (pipelined), 14 to 26 of pass, 6 to 10 of step |

Stage C would then take about 2 to 2.5 days, and each round about 1.5 to 2
days. The rounds' size and count are stage C's gate's to set.

## Kill criteria

- A display reset during any GPU step: the run stops (STOP), and the cause
  is found before anything resumes.
- The replay identity, the floor's cap or the path check failing (pg_pass
  exits 3, 4, 5): the part fails and the chain stops.
- Stage C's gate "neither" or unread: no rounds (`rounds_local` refuses
  before round 1); the user decides.
- The rounds' pause or transfer stop (§6): the chain stops for the user.

## Decisions per outcome

Design §6, unchanged:
- the gate on round 1's candidate;
- the optimizer by §2's criterion;
- P before rounds;
- the link match iff the sharpening slope is at least 3 SE. It reads
  +2.64 ± 0.08, so it runs.
