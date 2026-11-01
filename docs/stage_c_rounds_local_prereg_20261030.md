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
   - **e = 0.02, sealed 2026-10-31** (the morning review: P starts on its
     own after stage C, so it must not play an unsealed forecast). **Why:**
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

16. **Every GPU tool runs with its kernels serialized** (`AMD_SERIALIZE_KERNEL=3`,
    set on the runner, which its tools inherit; from the planned restart
    on, set in code for every GPU tool, `run_tool` in
    scripts/local_machine.py, e940ab2e). Added 2026-10-30, after the
    reboot (training/metrics/pg_fp16_20261030, "After the reboot"):
    - the fp16 evaluator's 8-game walk faulted unserialized with the fold
      off (a launch failure and a display reset, at 08:44);
    - serialized, the same run passed, as did the 2-game probe, and 200
      held-out games read 406 positions a second with the replay identity
      holding.
    The cost is unmeasured until the cause is found; C0 reads the rates
    under it.
17. **The full suite last ran at 1252ecba** (2,177 passed, 1 failed: this
    machine's `test_turn_gap`). The merges since then ran their own test
    files: the pass split (107 passed) and the gain curve's warm start (31).
    The rest is records. Stage C launched from eb1594c5 at 08:55 on
    2026-10-30.
18. **A planned restart after the pass** (2026-10-30 10:31; the 10:13
    review). A planned STOP ends the runner after the pass. A watcher
    (`C:/Users/momom2/wesnoth_runs/tools/relaunch_after_stop.py`) then
    relaunches it on the worktree's code, f070d548, which skips C0 and the
    pass on their done markers. That code adds two things:
    - serialization set in code, with each step's commit logged;
    - the driver's readings cached on disk by key (tools/pg_eval_cache.py),
      so a restarted driver resumes.
    The cache changes no reading. A reading is read back only for the same
    checkpoint, held-out data, forward-changing configuration, direction
    bytes and step, and the code is not in the key: nothing is merged into
    the worktree while stage C lives. Its tests passed: 67 with the
    runner's. The full suite on f070d548's tree passed before the restart
    (2026-10-30 22:36: 2,181 passed, 0 failed, 139 skipped).
19. **P and the rounds run from a pinned worktree, on `fix/handoff-safety`'s
    code** (2026-10-31; the 2026-10-31 morning review). The stage1 worktree
    keeps stage C's code until stage C exits. P and the rounds run from
    `..\wesnoth_ai-run`, a worktree detached at 934f137a, with own2.pt copied
    in and its SHA-256 checked. Its code differs from stage C's in five safety
    checks and the cache, and no reading or decision rule changes:
    - the reading cache's key holds the code (wesnoth_ai/, tools/, the core's
      binary), the device and torch's version;
    - the rounds' ladder passes `cache=`, so a restarted ladder resumes;
    - a hung display-reset check is asked again;
    - STOP says when a step had returned 0;
    - P and the rounds refuse a stage C without ALL_DONE, or with a STOP or a
      failed part.
    The full suite on 934f137a: 2188 passed, 139 skipped, 5 warnings in 802.48s (0:13:22).
20. **A launcher chains stage C, P and the rounds** (scripts/chain_local.py,
    2026-10-31).
    - P starts once stage C has finished and the pinned worktree and the
      checkpoint check out.
    - The rounds start only when P returns 0, with `--rounds 3`: round 1 is
      stage C's decision, rounds 2 and 3 play, and round 3 claims
      (`k % 3 == 0`).
    - Nothing is retried, and every refusal is written to
      `C:/Users/momom2/wesnoth_runs/chain.log`.
    - The window's final match is pre-registered apart
      (docs/final_match_local_prereg_20261031.md).
21. **Each round's pass takes the head maps' gradient alone**
    (`tools/pg_pass.py --heads-only`, rounds_local `--heads-only-pass`;
    `feature/heads-only-pass` 6faf2162, 2026-11-01).
    - **Why:** stage C chose heads_only (+0.0973 ± 0.0079 over reduced_adamw
      on half A, z 12.3). That arm reads the pass's g and v on the four head
      maps alone (`pg_step.direction`), so the full pass's gradient through
      the trunk, most of a 25 h pass, is computed and thrown away.
    - **What it does:** the heads-only pass freezes every other parameter.
      The trunk and the memory run without a graph, no step is recomputed,
      and the other parameters' sums are zero.
    - **Why the direction is unchanged:** a head map's gradient does not
      depend on whether the layers below it learn. Checked three ways:
      - the tests (tests/test_pg_pass.py) hold, on the suite's games, the
        heads' sums equal to the full pass's, the forward's readings equal
        to rounding, and the heads_only and probe_span directions equal;
      - on 3 of stage 1's games, on the CPU, the head maps' sums agree to
        4e-6 of their scale, every other sum is 0, the decisions and
        windows are the same, and the replay KL agrees (5.497e-5 against
        5.511e-5);
      - the CPU rate went from 4.1 to 7.5 decisions a second.
    - **Speed on the GPU is unmeasured.** The GPU's full pass also
      recomputed each step (two forwards and a backward), so the gain
      should be larger than the CPU's. At worst the pass runs at the full
      pass's speed. The pass's cut stays 48 h.
    - The other arms refuse such a pass. It changes no estimate and no
      decision rule.
    - **Found while testing:** the actor head's bias gradient is rounding,
      because the bias shifts every actor's logit alike. Reduced AdamW
      turns it into a unit-sized entry of the heads_only direction, in
      either pass, which moves no probability.
22. **The rounds run to a deadline instead of a count** (rounds_local
    `--end-by`, f7cbe1e9, 2026-11-01).
    - The chain runs `--rounds 9 --end-by "2026-11-07 10:00"`. Round k >= 3
      starts only when 1.2 times the longest round so far fits before the
      deadline; otherwise the rounds end there, done.
    - The final match then starts by its 2026-11-07 12:00 limit. Claims
      stay every third round.
    - The full suite on f7cbe1e9: 2195 passed, 139 skipped, 5 warnings in 905.76s (0:15:05).

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

**Measured by C0, serialized (2026-10-30 09:51, c0_plan.json).** The
trainer reads 76 positions a second and the forwards 352. The plan prices:
- the pass at 28.1 h (cut 42.4 h);
- the driver at 34.4 h (the 11.25 forward-equivalents of the worst case;
  about 28 h at the expected 9.3);
- the plant at 19.9 h;
- 82 h in all.

Expected: the pass ends about Oct 31 14:00 (corrected 22:20: its splits
take 3 h 02 to 3 h 08 each, 85-86 decisions a second, so about 10:40), the driver about Nov 1 18:00,
the plant about Nov 2 14:00, then P. At about 28 h of pass a round, about
two rounds fit before the window ends, so no claim falls on "every third
round". One PURE match of the window's last candidate against own2 is to
be pre-registered at the end.

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
