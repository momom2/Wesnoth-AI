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
23. **A rung whose importance weights fail half B's check is read by a
    cross-play match** (`feature/cross-play-rung` f9deb26c, registered
    2026-11-01 04:56, before any game of it is played).
    - **What happened:** stage C's ladder on heads_only (stage_c.json,
      2026-11-01 03:43) chose its 2x rung on half A:
      - the rungs on half A: KL 0.00267 a decision, side KL 0.50, ΔĴ
        +0.154 +- 0.012; the 1x rung +0.110 +- 0.008; the 1/2x +0.078 +-
        0.005.
      - On half B the 2x rung's weights fail the check: ESS share 0.098
        against the floor 0.185. Its exact reading is +0.103 +- 0.036, its
        first order +0.152 +- 0.009.
      - pg_step.decide keeps pi_0 ("the rung needs a cross-play match"), so
        stage C's gate reads "neither". As registered (kill criteria), the
        chain would stop at P and the GPU would idle until the user returns.
    - **The design's rule for this case:** the checks table says "Heavy-tailed
      ratios (a rare action raised): that rung is read by a cross-play
      match", and the design's §1 "Range" bullet (current_design.md:70)
      sizes such a match at 10,000 games against pi_k (corrected 2026-11-01
      by the morning review; first written "§3 'Range'").
      The code had implemented the case as "keep pi_k", and the gate as
      "neither". The design rule is now built and applied to every round.
      The user is away; the decision is the lead's, under the standing
      delegation of 2026-10-30.
    - **The match:**
      - The chosen rung's network theta_k + s d (pg_round
        `--chosen-network`; round 1: s 0.04015 along heads_only) plays pi_k
        at temperature 1, both at 64 memory slots, with no end_turn offset.
        This is stage 1's decode, the one ΔĴ reads.
      - XP_GAMES 10,000 games, no replacements, seeds 70M + 100k k.
      - It is read as the link is read: a win 1, a loss -1, a cap 0, and a
        wall-clock kill left out. Its mean z is in E[z], ΔĴ's unit, since
        pi_k against itself reads 0.
      - A game whose label A is not the rung's network (by SHA-256), not at
        64 slots, or not raw:t1 on both sides is refused. Any refusal, or
        fewer than 9,000 games read, fails the part and the chain stops.
    - **Decision rules, sealed here:**
      - Adoption is decide's rule with the match in half B's place: the rung
        is adopted unless the match reads below -2 SE; otherwise pi_k stays
        and the radius halves. round.json keeps the importance decision
        under "importance".
      - Stage C's gate, for round 1 only: the rounds go on at N 20,000
        (the gate's "resolved") iff the match's lower 2-SE bound is above 0.
        Otherwise the rounds wait for the user, as "neither" did. The final
        match still plays round 1's rung if the rung was adopted (the
        chain's rule for a rounds wait).
      - A later round's match changes that round's decision only. The
        pause and transfer stops read its mean z as that round's confirmed
        reading.
    - **Forecast** (from half A, which selected the rung, so biased
      upward): +0.15. Half B's first order reads +0.152 +- 0.009, and its
      exact reading +0.103 +- 0.036. The match's SE is about 0.01.
    - **Order:** P runs first, as for "resolved". The stage C runner's code
      (5a045b33) predates the rule, so the chain reads stage C's verdict
      again with the new code before P (`refresh_verdict`; the runner's
      verdict is kept as verdict_stage_c.json). Then round 1's step, its
      match (about 3 h at 3,450 games an hour), and the rounds. As first
      built the match had no time budget and would have stopped at 55 min
      (departure 24).
    - The tests: the case and the reading (tests/test_pg_cross_play.py);
      the rounds' flow, both stops of round 1's match, and a short match
      (tests/test_rounds_local.py); the verdict and its second reading;
      the chain; pg_round's rung network on the miniature stage 1.
    - The full suite on f9deb26c (the 38 failures below): 38 failed, 2164 passed, 139 skipped, 5 warnings in 1237.72s (0:20:37); every failure is a bash box test (test_boxlib 34, test_rent_box 4), code the change does not touch, run while stage C's driver loaded the CPU (20.6 min against 15). The two files alone after: 77 passed and 1 failed, a wall-clock test (test_the_switch_charges_the_stages_running_time), which then passed alone twice.

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
  before round 1); the user decides. The one exception, departure 23: a
  "neither" whose ladder left its rung to a cross-play match is decided by
  that match.
- The rounds' pause or transfer stop (§6): the chain stops for the user.

## Decisions per outcome

Design §6, unchanged:
- the gate on round 1's candidate;
- the optimizer by §2's criterion;
- P before rounds;
- the link match iff the sharpening slope is at least 3 SE. It reads
  +2.64 ± 0.08, so it runs.
24. **Every match P and the rounds play has a time budget within its cut,
    and is played again once when it ends short** (`fix/match-time-budget`
    96be1a81, merged into `feature/cross-play-rung` at 2287f570; registered
    2026-11-01 11:00, before P or any match of the rounds has played).
    - **What was wrong** (the 2026-11-01 09:43 review, H1): run_elo_batch
      stops a match at `--time-budget-min`, 55 min by default, and exits 3
      with games left to play. P's planted match (5,000 games, 70-87 min at
      stage 1's rates), the link (12,800), the cross-play match (10,000) and
      the claims (12,800) passed no budget, and their callers read 3 as a
      failed part. P would have failed about 55 min in, and the chain
      stopped with the GPU idle; the final match plays only after the rounds
      return 0 or 3.
    - **The rule:**
      - Each match's budget is its cut less 15 min, for the games in flight
        to drain before the cut: P's matches and the cross-play match 345
        min, a claim 585.
      - A match that ends short (a failed game, the budget or the cut: exit
        1, 3 or 124) is played again once from the games on disk, as
        scripts/match_local.py plays the final match. A replayed slot keeps
        its seed (run_elo_batch derives seeds from the game index). A second
        short end fails the part, and a display reset still stops the run.
      - No decision rule changes.
    - The tests: tests/test_p_local.py (the budget, a match played again, a
      second short end failing the part); tests/test_rounds_local.py (the
      cross-play match's and the claim's budgets); tests/test_chain_local.py
      (the liveness query's OSError, fixed in the same commit).
    - The full suite on 2287f570: 38 failed, 2165 passed, 139 skipped, 5 warnings in 1254.63s (0:20:54); the 38 are the bash box tests that failed on f9deb26c (test_boxlib 34, test_rent_box 4), under stage C's CPU load, code the change does not touch.
25. **The rounds play N = 5,000 training games a round, not 20,000**
    (rounds_local `--n-train`, passed by chain_local `--rounds-n-train`:
    `feature/rounds-n-train` f8ad2f97, merged into `feature/cross-play-rung`
    at 8af92eb6; registered 2026-11-01 18:50, after stage C's driver wrote its
    critical batches and its gate at 17:12, before P or any round has played).
    - **Why:** design §2 sizes a round at N* = √(c_f·B/c_g). Stage C's arm is
      heads_only, and the gate reads B in the winner's metric: 280.9 ± 29.9
      games, resolved (reduced AdamW's reads 44.3 ± 6.4).
    - **The prices, in this desktop's hours** (the one GPU's hours bound the
      rounds; the autonomy log, 2026-11-01 12:41):
      - c_g is 0.00093-0.00148 h a game: play at 0.84 s a game, the full
        pass at 4.48 s.
      - c_f is 6.5-11 h a round, up to about 16 h with the rounds' 9.5-h
        step.
      - N* is then 1,111-2,198 games, and 1,223-2,421 at B + 2 SE. At the
        box's prices (the gate's: $7.76e-5 a game, $2.42 a round) it is
        2,960.
    - **The rule:** N = clamp(ceil(N*), 5,000, 20,000). The floor is the
      smallest size the gain curve measured (2 of the pass's 8 splits). N
      stays 20,000 if B is unresolved. So **N = 5,000**.
    - **The gain curve agrees.** A step built from 5,000 games reads 1.062 ±
      0.039 of the whole pass's first-order gain; the model
      √[(N/(N+B))/(20,000/(20,000+B))] reads 0.98 at B 281. A step from
      10,000 games reads 1.058 ± 0.037 (model 0.99). Both are read in
      reduced AdamW's metric.
    - **What it buys:**
      - play of 15,000 games a round, not 30,000 (3.5 h, not 7);
      - a pass over 5,000 games: at most 6.2 h at the full pass's rate,
        against 25;
      - the step unchanged, since it reads the 10,000 held-out games.

      That is about 19 h a round against 41, so about 5-6 rounds by Nov 7
      10:00 instead of 2-3.
    - **Unchanged:**
      - N_HELD 10,000;
      - the gate's reading, the ladder and the claims;
      - the seeds: round k's are 10M + k·1M, and N + N_HELD stays under 1M;
      - departure 23's cross-play case, which goes on at N too.

      A round's N is recorded when its games start (its `n_train.txt`), so
      a re-arm finishes a begun round at the size it began with.
    - The tests: tests/test_rounds_local.py (N in the resolved and cross-play
      cases, a begun round keeping its size), tests/test_chain_local.py (the
      flag passed through).
    - The full suite on 8af92eb6: 38 failed, 2166 passed, 139 skipped, 5
      warnings in 1234.29s (0:20:34). The 38 are the bash box tests above,
      under stage C's CPU load.
    - **Amended 2026-11-01 evening (the 21:43 review, L2):** "agrees" is too
      strong. The curve is consistent with the model, but it is no test of
      it at B ≈ 281: it was read at reduced AdamW's B (44), where the model
      predicts 0.997 whatever B holds for heads_only, and its 5,000-game
      direction used the whole pass's second moment. What carries the choice
      is B, which was measured on heads_only, and the decision is robust: even
      at B 2,000 the model keeps 0.89 of the gain.
26. **The ladder neither searches nor reads a rung past the range**
    (`fix/ladder-skip-past-range` c3d2286b, merged into
    `feature/cross-play-rung` at 97298a01; registered 2026-11-01 18:50,
    before any round's ladder has run).
    - **Why** (the 2026-11-01 09:43 review, L1): a rung whose KL per
      side-game is past K = ½ is never selected, yet the ladder ran its KL
      search and read its ΔĴ on half A. That costs about 1-2 h of a round's
      step once the radius reaches the range's edge. Stage C's 2ε rung was
      just inside the range (ε 0.00133), so after the first adopted 2ε step a
      round's 2ε rung is likely past it.
    - **The rule:** such a rung is recorded with `in_range` false, `s` and
      `kl` null, and no halves. Every consumer reads only the chosen rung,
      which is always in range, so every decision is the one the old ladder
      made.
    - **Not taken:** the design's cross-play match for a 2ε candidate past
      the range (current_design.md §1.2, "Range"). It costs a 10,000-game
      match a round (2.3-2.9 h here, more than the reading it replaces), and
      the rounds' code has no path that selects on it. The ladder stays capped
      at K = ½, as it was.
    - The test:
      tests/test_pg_step.py::test_a_rung_past_the_range_is_neither_searched_nor_read.
      The ladder, round, cross-play, stage C and rounds tests: 45 passed
      before it was added.
    - The full suite on 97298a01: 38 failed, 2167 passed, 139 skipped, 5
      warnings in 1196.04s (0:19:56). The 38 are the same bash box tests
      (test_boxlib 34, test_rent_box 4); the one more passed is the new test.
    - The chain was re-armed at 97298a01 with `--rounds-n-train 5000` on
      2026-11-01 19:06 (pid 14296), while it still waited on stage C's plant.
27. **The first rung past the range is read at its edge**
    (`fix/ladder-edge-rung` d622278d, merged into `feature/cross-play-rung`
    at 118d7a23; registered 2026-11-01 22:55, before any round's ladder has
    run).
    - **Why** (the 2026-11-01 21:43 review, M2): stage C set ε at K ¼ a
      side-game (0.25 / 187.35 decisions), so its 2ε rung sat at K ½ to float
      precision. If round 1 adopts that rung, round 2's ε is 0.0026687, and
      its ε rung lands on the edge again. Whether it is read then turns on
      π₁'s decisions per side-game: games slightly longer put it past the
      range, so only the ε/2 rung (K ¼) is read. Adopting that rung halves the
      radius, and the step alternates between K ½ and ¼ from round to round.
      In stage C, half A read ΔĴ 0.110 at K ¼ against 0.154 at K ½.
    - **The rule:** the first rung past SIDE_KL_RANGE is searched and read at
      the edge (target KL ½ / decisions per side-game). It is skipped when
      the edge is less than EDGE_MIN_RATIO = √2 above the rung below (the
      rungs' geometric midpoint), since its reading would then repeat that
      rung's. Every other rung past the range is neither searched nor read,
      as in departure 26.
      - An adopted rung's next radius is its target KL, the edge for a
        clipped rung, in pg_step.decide and pg_cross_play.decide_by_match.
        For an unclipped rung this equals the old multiplier × ε.
      - A known point at ε (stage C's arm reading) is not reused for a rung
        moved to the edge.
    - **Unchanged:** the range itself (K ½); the selection on half A; the
      confirmation and the revert on half B; departure 23's cross-play case;
      round 1, which takes stage C's ladder as it stands (its chosen rung is
      unclipped, target 0.0026687 = 2ε).
    - The tests: tests/test_pg_step.py::test_the_first_rung_past_the_range_is_read_at_its_edge
      and tests/test_pg_cross_play.py::test_a_rung_read_at_the_ranges_edge_sets_the_next_radius_to_the_edge.
      - Three tests that put a rung past the range move from ε 0.3 (0.006 at
        50 decisions) to 0.4 (0.008), so their far rung stays past the edge
        and unread under the new rule; their assertions are unchanged.
      - The rounds test's rung gains the `target_kl` that every real summary
        carries.
      - The ladder, cross-play, round, stage C, end-to-end, rounds, chain, P
        and evaluator tests: 83 passed.
    - The full suite on 118d7a23: 38 failed, 2169 passed, 139 skipped, 5
      warnings in 1173.70s (0:19:33). The 38 are the box tests, which fail
      only when pytest is launched from PowerShell: its PATH lacks Git's
      coreutils, so boxlib.sh's bash finds no dirname, mkdir, rm or date.
      From Git Bash the two files pass (78 passed, 20 skipped). The
      earlier departures' "under stage C's CPU load" was that artifact.
    - The chain was re-armed at 118d7a23 on 2026-11-01 23:02 (pid 12536),
      while it still waited on stage C's plant.
28. **The ladder reads each half in one walk, the heads-only rungs on one
    trunk** (`feature/shared-trunk-ladder` 005ac4cb, merged into
    `feature/cross-play-rung` at d9c000f2; registered 2026-11-02 01:40,
    before any round's ladder has run).
    - **Why** (the 2026-11-01 21:43 review, M4): under the heads-only arm
      a step moves only the heads' final linear maps (pg_step.HEAD_MAPS),
      which read nothing but the trunk's rows. The memory is written from
      the trunk's output (model.py:798-802), so θ_k and every rung share the
      trunk and the memory along each game-side. Yet each rung walked its
      half with full forwards, and θ_k's full reading was a walk of its own.
      Stage C's driver_cache times a full walk at 146 min and a half's
      reading at about 74.
    - **The change:**
      - The ladder searches every rung's step before reading any. The
        searches read only the KL subsample, so no number changes.
      - It then reads the rungs and θ_k on half A in one walk, and the
        chosen rung and θ_k on half B in another (`NetworkEvaluator.prefetch`,
        `base_logp_on`). θ_k's full reading is never taken.
      - When the direction moves the head maps alone (`EvalContext.shares_trunk`),
        such a walk runs θ_k's trunk once a time step, carries one memory,
        and applies each rung's heads to the trunk's rows
        (`WesnothModel.keeping_head_inputs`, `heads_from`).
      - A direction that moves the trunk (the other arms) reads each half in
        one walk with one forward a network, as `evaluate` always did.
    - **What changes in the numbers:** only θ_k's reading. It now comes from
      the half's own walk, batched with that half's positions, not from the
      full walk. On the CPU in float32 the two agree to 4.8e-7 at most (the test games, padded and packed trunks) (test
      below). On the GPU in fp16, θ_k and a rung read in one walk share the
      trunk's rounding, so their log ratio carries the heads' rounding alone.
      The old path subtracted readings from walks batched differently.
    - **Expected wall:** a round's step loses the full walk (146 min) and
      all but one of the half-A readings (74 min each, two or three rungs):
      about 3.7-4.9 h of the 7.5-9 h step. This is an estimate from stage C's
      times, unverified until round 2's step; that step's log times it.
    - **Unchanged:** the rungs, the KL searches, the selection on half A,
      the confirmation and the revert on half B, departures 23 and 26-27,
      and every reading's cache key format (a new code key makes old
      readings a miss).
    - **Fallback:** if round 2's step fails on the GPU, re-arm the chain at
      118d7a23 (the old ladder); the round resumes at its step.
    - The tests:
      - tests/test_pg_eval.py::test_steps_that_move_the_heads_alone_share_theta_ks_trunk
        (bit-equal readings, padded and packed trunks, one trunk counted);
      - tests/test_pg_eval.py::test_a_prefetch_reads_a_halfs_steps_beside_theta_k_in_one_walk;
      - tests/test_pg_step.py::test_the_ladder_searches_every_rung_before_one_prefetch_a_half.
      - The evaluator, ladder, stage C, end-to-end, cross-play, rounds, round
        and plant tests: 74 passed.
    - The full suite on 005ac4cb: 2211 passed, 139 skipped, 5 warnings in 833.93s (0:13:53), launched from Git Bash, so the box tests ran and passed; d9c000f2's tree is 005ac4cb's.
    - The chain was re-armed at d9c000f2 on 2026-11-02 01:38 (pid 3464),
      while it still waited on stage C's plant.

29. **After the final match, the tail match and the continued rounds; a
    resume keeps the finished rounds' walls** (`feature/tail-match`
    58e713fc, merged into `feature/cross-play-rung` at ec0146d7;
    registered 2026-11-02 03:30, before any round has run; the design is
    docs/tail_local_prereg_20261102.md).
    - **Why:** the final match's forecast is near 0 at argmax, so the
      trained player's own gain may go unmeasured (the 2026-11-01 09:43
      review). The GPU also idles 24-36 h after the final match (the
      21:43 review, M3).
    - **The tail:** once the final match returns 0, its candidate plays
      own2 with both sides sampling at temperature 1 and no end_turn
      offset (`match_local --t1`, raw:t1). The match is 12,800 decisive
      games from seed 950,000 into `tail_match/`.
    - **The continuation:**
      - It runs once the tail returns 0, and only if the rounds returned 0.
      - The rounds resume (rounds_local, the same arguments) with
        `--rounds 39` and `--end-by "2026-11-08 12:00"`, under this
        prereg as it stands.
      - A round adopted after the final match's gets the standard match
        into `final_match_cont/` (12,800 games from 900,000, paired with
        the final match's).
    - **The resume fix:** rounds_local's `timed` rewrote a finished
      round's end at every restart, so a resume read each finished
      round's wall as reaching the restart, and `--end-by` refused the
      next round. Now the first start and the first end of each round
      stand. This also bears on any re-arm during the rounds, the
      departure-28 fallback included.
    - **Unchanged:** the rounds, their stops and claims, P, the final
      match and its seeds, every number a round reads.
    - The tests:
      - tests/test_match_local.py::test_t1_samples_both_sides_without_the_offset_and_relabels_the_reference;
      - tests/test_match_local.py::test_a_t1_match_samples_both_sides_and_fits_against_the_relabelled_reference;
      - tests/test_chain_local.py::test_the_tail_match_plays_the_final_candidate_at_t1_once_the_final_match_passes;
      - tests/test_chain_local.py::test_the_continuation_resumes_the_rounds_then_measures_a_newer_round;
      - tests/test_chain_local.py::test_no_continuation_after_a_failed_match_or_unfinished_rounds;
      - tests/test_chain_local.py::test_the_command_line_holds_the_tail_and_the_continuation_together;
      - tests/test_rounds_local.py::test_a_resume_keeps_the_finished_rounds_walls_and_goes_on_to_a_later_end_by
        (with the old `timed` it fails: "round 5 would need about 26.4 h").
      - The chain, match and rounds tests: 68 passed.
    - The full suite on 58e713fc: 1 failed, 2218 passed, 139 skipped, 5 warnings in 1010.18s (0:16:50), launched from Git Bash. The failure is tests/test_boxlib.py::test_the_switch_charges_the_stages_running_time, the wall-clock test on record since Nov 1: a 3-s sleep entry read 8 s used under this machine's load, and it fails alone too; boxlib.sh and its tests are unchanged on this branch. ec0146d7's tree is 58e713fc's.
    - The chain was re-armed at ec0146d7 on 2026-11-02 03:42 (pid 13656),
      while it still waited on stage C's plant.

30. **Round 1 alone while the plant is debugged** (registered 2026-11-02
    10:30, before round 1's step; no code change, ec0146d7 as run).
    - **Why:** stage C ended at 10:13 with the plant failing on both seats
      (`PG_PLANT_READOUT fail`: seat A measured 0.0007 against a forecast
      of 0.0261, miss -6.4 SE; seat B 0.0021 against 0.0273, miss -6.1 SE;
      resolutions 8.5 and 8.3). The rule "debug before P and rounds"
      stopped the chain at P (rc 3, 10:17), and the GPU idles.
    - **What runs:** `rounds_local --rounds 1` into `rounds_local/`, with
      the chain's own arguments (`--heads-only-pass --end-by "2026-11-07
      10:00" --n-train 5000 --p-out p_local`). Round 1 consults neither P
      nor the plant: its step is stage C's ladder decision taken as read
      (heads_only, the rung at K ½), and departure 23's 10,000-game
      cross-play match at temperature 1 against π₀ reads it in play
      (seeds 70,100,000 on). That match is independent of the gradient
      machinery the plant tests.
    - **Its reading:** as departure 23 (adopt iff the match's lower 2-SE
      bound is above 0). It is not a strength claim against the reference.
    - **Unchanged:** round 2 still waits on P (rounds_local's `p_gate`), and
      P on the plant. A later chain resumes past round 1 from its markers.
    - **While it plays:** the plant's miss is diagnosed. The planted pass
      carries the plant at its expected size (its projection on P'u reads
      276 against 290 expected), and the finite difference along P'u
      matches the gradient forecast to 4 digits. Yet the change measured
      along the pass's own direction d misses the first order predicted
      along d (0.0248 on seat A), and the trunk carries 0.0225 of that.

31. **The rounds start without P** (`feature/chain-skip-p` 687cd3d9,
    merged into `feature/cross-play-rung` at 6f86c363; registered
    2026-11-02 10:42, before round 2).
    - **Why:** P's played check cannot pass at the plant's size, so as
      registered no round past round 1 could ever start (the 10:13
      review, H2). `tools/pg_planted_play.py` `planted()` compares θ′'s
      and π₀'s attack counts per side-game across different games
      (unpaired). Those counts have an SD of about 25 (400 stage-1
      games), so a seat's SE at 2,500 against 2,500 is about 0.71 and the
      rule needs a rise above 1.42. The forecast rise is the plant's
      teacher-forced change, about 0.026 (z about 0.04). The plant's δ was
      sized for the teacher-forced readout (paired, 10 SE), never for a
      played check.
    - **What changes:** `chain_local --skip-p` runs no P (neither the
      planted match nor the link) and passes `--no-p-gate` to rounds_local
      in place of `--p-out`. The chain is re-armed with it once round 1 has
      ended.
    - **What stands in for P:**
      - round 1's cross-play match, which reads the evaluator's forecast
        of the step the rounds take in play: stage C's ladder reads the
        chosen rung (heads_only, ×2, s 0.0401) at ΔĴ +0.153 ± 0.009 on
        half A by first order, and +0.152 ± 0.009 on half B;
      - the plant's own decomposition. The planted pass carries δu at
        its expected size, and the miss is in the trunk's part of the
        full-network direction (seat A: the matrices carry 0.0216 of the
        first order 0.0248, the heads 0.0023). The heads-only rounds move
        neither the trunk nor the memory.
    - **Kill:** if round 1's match reads below its forecast by more than 3
      combined SE, the rounds stop after round 1 for diagnosis (STOP in
      `rounds_local/`), whatever departure 23's adoption rule says.
    - The link match (own2 at T 1/1.02 against 1/0.98) is not played. The
      cross-play match tests the same link, on the step itself.
    - The test:
      tests/test_chain_local.py::test_skip_p_runs_no_p_and_waives_the_rounds_p_gate.

32. **The heads-only arm's own direction shares θ_k's trunk**
    (`fix/shares-trunk-zeros` cbae242f, merged at 8eb24002; registered
    2026-11-02 10:42, before round 2's step).
    - **Why:** `pg_step.heads_only` keeps every parameter name, with zeros
      off the heads, and `EvalContext.shares_trunk` required every name to
      be a head map. So departure 28's one-trunk walk never engaged: every
      rung ran full forwards (the 10:13 review, M1). Its saving was
      unmeasured and its fp16 path never ran on the GPU.
    - **What changes:** a name off the heads whose tensor is all zeros
      counts as unmoved. The test builds the direction through
      `pg_step.heads_only` and reads it on one trunk, bit for bit the head
      names' reading.
    - **Deployed only after a GPU check.** Before the chain is re-armed,
      `shared_trunk_gpu_check.py` (wesnoth_runs/tools) reads stage C's
      heads-only direction at θ_k and two rungs on one trunk and alone,
      on 24 held-out games, under the rounds' config (fp16,
      `AMD_SERIALIZE_KERNEL=3`). It passes when the differences are at
      most 1e-3 in log probability and class mass, with one trunk read.
      On a pass the chain runs at 8eb24002; otherwise at 6f86c363, without
      the fix. The CPU smoke run of 1 game: equal bit for bit, one read,
      18.7 s against 46.1 s for the three alone.
    - **The fallback** written in departure 28 (re-arm at 118d7a23) is
      void (the 10:13 review, M2): 118d7a23 knows neither the tail's
      flags nor the resume fix. The fallback is now 6f86c363.

33. **The rounds go on past a design stop** (`feature/after-stop`
    786c453a, merged into `feature/cross-play-rung` at a0e2b56b;
    registered 2026-11-02 23:12, before round 3's claim).
    - **Why:**
      - A design stop (rounds_local exit 3: the pause, or a claim below its
        forecast) ends the chain after the final and tail matches, with no
        continuation. The stop asks the user to choose a magnet or a pool,
        and the user is away until about Nov 8.
      - Round 3's claim comes about Nov 3 11:00, and a stop there would
        idle the GPU for about 3.5 days (the 21:45 review, item 3).
      - The transfer test is tight: its combined SE is about 10 Elo against
        a forecast of about +150 after round 3.
    - **Why the trunk** (the review's item 1, and the window check):
      - Under `--heads-only` the gradient is window-free by construction.
        The trunk arms' gradient is not.
      - On 8 training games of stage C's pass, the full-network gradient
        windowed at 16 decisions reads, against one window per game-side,
        a cosine of 0.62 in stage C's reduced-AdamW metric. By group:
        encoder layers 0.77, the memory 0.25-0.47, and the heads exactly
        1.0 (the control). On the CPU the one-window pass costs 3% more.
      - Stage C compared its arms on the windowed gradient: heads_only beat
        reduced_adamw by 0.097 on half A, and the class probes alone carry
        86% of that gain.
      - Open: whether the window biases the expected gradient or only each
        game's. The 64-game split reading (`window_check64`, due about Nov
        3 05:00) is recorded beside this departure and does not gate it.
    - **What runs.** Only after a design stop at round K0, and once the
      final and tail matches returned 0 (`wesnoth_runs/tools/after_stop.py`,
      a watcher started now):
      1. **The paired arm reading**, when it and one trunk round fit before
         Nov 8 02:00 (13 h + 19 h). First, a full-network pass in one window
         per game-side on round K0's training games, at the policy that
         played them (`pg_pass --window 10000000`, with the rounds' fp16
         flags). Then `pg_round --arm reduced_adamw` at round K0's radius,
         on the same halves. Its files go beside round K0's (`trunk_*`),
         and its candidate is not adopted.
      2. **The rounds go on** from round K0's policy (`chain_local
         --after-stop K0`: rounds to 39, ending by Nov 8 02:00), rebased on
         round K0:
         - the stops of rounds K0 and before are not read again;
         - the pause sums only the confirmed readings after K0;
         - a claim is tested against round K0's claim in place of D0, with
           the forecast from the readings after K0.
         The arm is reduced_adamw in one window iff its best half-A reading
         in step 1 is at least heads_only's best on the same half. It is
         heads_only (with the heads-only pass) otherwise, or when step 1
         fails or does not fit.
      3. **The standard match** (`final_match_after_stop/`, seeds 900,000,
         12,800 games) on a round adopted after K0; none otherwise.
    - **Guards:**
      - Step 1 is the one-window pass's first run on the GPU. If it fails,
        a GPU health check (a 1,024-element sum) runs before anything
        else, and on a failure nothing more runs on the GPU (never two
        faults in a row).
      - The rounds after K0 play on their own seed blocks (10M + k·1M, and
        the cross-play and claim blocks by k), which no round has used,
        since none ran past K0.
      - The STOP is moved aside to `STOP_roundK0` and kept.
    - **The readings:**
      - Each round after K0 reports its halves and its claim as before.
      - A trunk round's reading against the heads-only rounds' (0.14-0.16
        a round at ε about 0.0025) is reported, not gated.
      - Confounds: the trunk line changes the arm and the window at once.
        Step 1 isolates the arm at one window on the same data (the same
        θ, ε and halves), and stage C's reading isolates the arm at 16.
      - The after-stop line's final match is a strength claim against the
        reference, like the window's other final matches. The rest is not.
    - **The tests:** tests/test_rounds_local.py covers the arm and the
      window reaching the pass and the step, and the rebased stops.
      tests/test_chain_local.py covers the after-stop flow: the stop moved
      aside, the flags, a match only after K0, and the refusals. 36 passed.
      The full suite runs at a0e2b56b.

33a. **Amendment to departure 33** (registered 2026-11-03 10:27, before
    round 3's claim reads; the 2026-11-03 review's findings 1 and 2).
    - **Step 1, the paired arm reading, is withdrawn.**
      - pg_pass's `--window` counts time steps across all 32 streams
        (wesnoth_ai/sequence_streams.py, `StreamSchedule.window`). So
        `--window 10000000` puts a whole split in one graph: about 270,000
        decisions (about 8,500 steps), whose checkpointed inputs and
        memories do not fit the card's 8 GB.
      - The reduced-AdamW metric's second moment is the mean over windows
        of each window's squared gradient (wesnoth_ai/pg_step.py,
        `second_moment`). With one window a split it would be read from 8
        split-level squares, so the arm's metric would change with the
        window. Step 1 would not have isolated the arm as registered.
    - **What runs after a design stop:** step 2 with the heads-only arm and
      the heads-only pass, as in the rounds before the stop. Step 3 is
      unchanged.
    - **Round K0's end:** at a design stop rounds_local returns before it
      stamps round K0's end in times.json, and the resume would stamp it
      hours late. The round's wall would then bound `--end-by` too long and
      end the rounds 1-2 rounds early. So the watcher stamps the end at the
      STOP file's time before it arms the rounds.
    - The watcher (`wesnoth_runs/tools/after_stop.py`) restarted with this
      amendment at 10:27, before the claim reads.

34. **|H| = 5,000 from round 4** (`feature/held-size` 2949d267, merged
    into `feature/cross-play-rung` at e48a96c7; registered 2026-11-03
    10:45, before round 3's claim reads; the 2026-11-03 review's advice).
    - **Why:**
      - H only measures. Its halves choose the rung (A) and confirm it (B),
        and the corrections are fitted on it; the pass trains on the N =
        5,000 training games alone. At |H| = 10,000, H holds 2/3 of each
        round's games and costs about half of its wall. In round 3, H took
        about 148 of the play's 222 min, and its halves 234 of the step's
        291 min (the KL reads took 47).
      - The ladder's range caps each round's step: round 3's x2 rung was
        past the range and unread, and ε went 0.00267, 0.00244, 0.00229.
        So the lever is rounds per day, not each round's precision.
      - Halving H saves one chunk of play (about 74 min) and about half of
        each half's reading (about 117 min): about 3.2 h of round 3's
        11.4 h before its claim, 28%.
    - **Costs:**
      - Half B's SE grows by sqrt(2), from 0.0117-0.0174 to about
        0.017-0.025, against readings of 0.11-0.16 (z of 4.5 or more).
      - Half A's rung choice is noisier.
      - The corrections (α, γ, β) are fitted on 5,000 held-out games in
        place of 10,000.
      - The transfer test's combined SE grows, since each round's SE
        enters it times 347.4: from about 10.6 Elo at round 3 to about 13.
      - The ESS check is unchanged: its floor depends on the step's KL
        alone (wesnoth_ai/pg_step.py, `ess_floor`).
    - **What runs:** the rounds whose games start from now play N + 5,000
      games, the last 5,000 held out (`rounds_local --held 5000`, recorded
      in each round's held.txt). Rounds 1-3 keep 10,000.
      - If round 3's claim stops the rounds, departure 33a's after-stop
        chain carries `--rounds-held 5000`.
      - Otherwise a REDEPLOY stop, written before round 3's claim ends,
        ends the chain at round 4's start (rounds_local reads its STOP
        between steps only, so the claim plays to its end). The watcher
        then re-arms the chain at e48a96c7 with `--rounds-held 5000`
        (arm_chain.ps1 -Held), and the rounds resume from their markers.
      - The seed blocks are unchanged: a round's 10,000 games stay inside
        its 1M block.
    - **Kill:** back to 10,000 at the next round boundary if either holds:
      - half B's ESS check fails in two rounds in a row (each failure plays
        a 2 h cross-play match, which eats the saving);
      - a half B's SE exceeds 0.035 (a reading of about 0.1 would then be
        under 3 SE).
    - **The readings:** each round's wall and half B's SE, reported against
      rounds 2 and 3.
    - **The tests:**
      - tests/test_rounds_local.py: |H| reaches the play and the pass, a
        round begun before held.txt keeps N_HELD, and a resume without
        --held keeps a round's recorded |H|.
      - tests/test_chain_local.py: --rounds-held reaches rounds_local.
      - 37 passed; the neighbouring local-script files, 59 passed. The full
        suite at e48a96c7 runs.
      - The watcher's REDEPLOY and design-stop paths were tested with the
        processes, git and PowerShell stubbed.

35. **The after-stop line goes on at a later design stop, and --end-by
    reads the current rounds' walls** (`fix/end-by-rebased` aa87740a, merged
    into `feature/cross-play-rung` at d045e17c; the watcher
    `C:/Users/momom2/wesnoth_runs/tools/after_stop_line.py`; registered
    2026-11-03 22:45, during round 4's pass; the 2026-11-03 21:45 review,
    M1 and M2).
    - **Why (M1):** after the rebase on round 3, the claims at rounds 6, 9
      and 12 test claim_k - claim_3 against 347.4 times the sum of the
      half-B readings after round 3 (rounds_local.py `stops`). The readings
      compare the T=1 policies, while the claim plays against the reference
      at argmax, and on today's data the gain against the argmax reference is
      0.88 +- 0.06 of the T=1 gain (118.4 against 134.1). The shortfall grows
      with the forecast while the combined SE grows as the square root of the
      rounds, so the stop can fire again: the review puts it at about 9% at
      round 6 and 20-25% over the three claims if the gap is real. A stop
      ends the line after its final match, and no watcher re-arms it: the GPU
      would idle up to two days.
    - **Rule (M1):** a later design stop (STOP or PAUSE after a claim round
      K > K0) is acted on as departure 33 acted on round 3's. The line plays
      its final match on round K's candidate (as registered, into its
      final-out), then the watcher stamps round K's end at the STOP's time
      and re-arms the line rebased on K, its final match into
      `final_match_after_stop_kKK`. Each such stop costs about 3.7 h for its
      final match, which reads round K at the reference's decode. The stops'
      texts and the registered tests are unchanged; the stop is still
      reported to the user.
    - **Why (M2):** --end-by starts a round only when 1.2 times the longest
      round so far fits. Round 3's wall is 14.2 h (|H| 10,000, its claim and
      the diagnostics beside it), so no round would start after Nov 7 09:00,
      while the rounds at |H| 5,000 run about 7.8 h (10.3 h with a claim).
      The line would end about one round early, and the GPU would idle 8-13 h
      before Nov 8 02:00.
    - **Rule (M2):** the walls are those of the ended rounds whose held.txt
      records the current |H|, once one has ended; otherwise every round's
      (rounds_local.py `no_time`). A first version read the rounds after
      --rebase-after, but a re-arm at a later stop would have read round 3's
      wall again.
    - **Deployment:** a REDEPLOY stop, written during round 4's pass, ends
      the rounds at the next step boundary (rc 2: no final match). The
      watcher checks out d045e17c in the run tree and re-arms the line
      (arm_after_stop.ps1, now with -FinalOut and an appended console log)
      with --after-stop 3, |H| 5,000 and the same final-out. The REDEPLOY
      text names "after round 3", which chain_local --after-stop 3 requires,
      and it moves that text aside as STOP_round03_2. The rounds resume from
      their markers, with round 4's start kept.
    - **Kill:** none for the rule, which changes no reading. If the re-arm
      fails, the heartbeat re-arms the line by hand at d045e17c (corrected
      2026-11-04 on the 09:43 review, L3: e48a96c7 as first written would
      bring back M2's walls).
    - **Tests:**
      - tests/test_rounds_local.py: a resume at |H| 5,000 after a 14 h round 3
        goes on to round 7 by --end-by; so does a re-arm after round 6's stop.
        Both fail on the old code. The file: 21 passed. The rounds, chain and
        match files: 80 passed.
      - The watcher was stub-tested (processes, git and PowerShell stubbed)
        on five cases: REDEPLOY then a stop at round 6 then the end; a
        REDEPLOY not naming K0; a stop whose final match failed; a stop with
        no time left; the base stop.
      - The arming script's command line was printed by a dry copy, for
        K0 3 and K0 6.

36. **|H| = 2,500 from round 7** (no code change: the line watcher
    restarted with `--held 2500`, the line re-armed at d045e17c with
    `-Held 2500`; registered 2026-11-04 10:37, during round 6's games; the
    2026-11-04 09:43 review, M2).
    - **Why:**
      - The step adopts unless half B reads below -REVERT_SE SE
        (wesnoth_ai/pg_step.py `decide`), so a noisier half B costs no
        rounds. H's halves choose the rung (A) and confirm it (B), and the
        corrections are fitted on it; the pass trains on the N = 5,000
        training games alone.
      - The x1 rung has sat at the range's edge since round 3
        (SIDE_KL_RANGE over the decisions per side: next eps 0.00229,
        0.00224, 0.00218), and the x2 rung is past it. Each round's step is
        capped, so the lever is still rounds per day (departure 34's reason).
      - At |H| 5,000, rounds 4 and 5 took 7.2 and 7.4 h: play 2.6 h (10,000
        games), the pass about 2.5 h, the step 2.3 h. At |H| 10,000, round
        3's step took 4.9 h, so the step scales with H. At 2,500, play takes
        about 2.0 h and the step about 1.2 h: rounds of about 5.8 h, 8.6 h
        with a claim.
      - Under --end-by 2026-11-08 02:00, the line at 5,000 ends after round
        14 (15 if its rounds stay under about 7.47 h; games lengthen, 223 to
        230 decisions a side). At 2,500 it ends after about round 17, about
        Nov 7 21:00: three more rounds, at about +0.085 on half B each (rounds
        4 and 5: 0.0896 and 0.0875), about 30 Elo at T=1 by 347.4 Elo a unit.
    - **Costs:**
      - Half B's SE grows by sqrt(2), from 0.019 to about 0.027, against
        readings of about 0.09 (z about 3.3).
      - Half A's rung choice is noisier. In round 4, x0.5 read 0.073 and x1
        0.106 on half A; the two share their games, so the difference reads
        tighter than either. A pick of x0.5 halves the next radius, and the
        next round's x2 rung reads it back at the edge.
      - The corrections are fitted on 2,500 held-out games. In round 5 the
        value term entered at 39 SE and the luck term over it at 11 SE, with
        a share of 6.8% against the rule's 5%. At 2,500 they read about 28
        and 8 SE, and the share's noise is about 0.009, so in an occasional
        round the luck term drops out (coefficient 0), about 7% more E[R^2]
        in that round's pass.
      - The transfer test's combined SE grows: about 17 Elo at round 9,
        against about 12.7 at round 6.
      - The ESS floor is unchanged: it depends on the side KL alone
        (pg_step.py `ess_floor`).
      - The first round at 2,500 extrapolates --end-by from every round's
        wall, round 3's 14.2 h included (departure 35's rule, no round of
        the current |H| yet ended): harmless before Nov 7. Until round 9's
        claim ends, the 2,500 walls hold no claim round. That matters only
        near the end, where every claim round has been read.
    - **What runs:** the rounds whose games start after the redeploy play
      N 5,000 + 2,500 games, the last 2,500 held out. Round 6 keeps 5,000
      (its held.txt). The seed blocks are unchanged.
    - **Deployment:** the line watcher re-arms with its own `--held`
      (after_stop_line.py `arm`), so it is restarted first with `--held
      2500` (K0 3, the same head and final-out). Then a REDEPLOY stop naming
      "after round 3" ends the line at round 6's next step boundary (after
      its games, about 12:40; rc 2, no final match). The watcher re-arms the
      line at d045e17c with `-Held 2500`; chain_local --after-stop 3 accepts
      the text and moves it aside, and round 6 resumes from its markers at
      5,000. A design stop at round 6's claim re-arms at 2,500 by the
      departure 35 path.
    - **Kill:** back to 5,000 at the next round boundary (the watcher
      restarted at `--held 5000`, then a REDEPLOY) if any of these holds:
      - half B's ESS check fails in two rounds in a row (each failure plays a
        2.4 h cross-play match);
      - a half B's SE exceeds 0.045 (a reading of 0.09 under 2 SE);
      - a round at 2,500 without a claim takes over 6.8 h (the saving is not
        there).
    - **The readings:** each round's wall, half B's estimate and SE, and the
      corrections' entries, against rounds 4 and 5.
    - **Tests:** no code changes. The |H| paths are departure 34's: |H|
      reaches the play and the pass, and a resume keeps a round's recorded
      |H|. MIN_HELD (4) is far below 2,500. The full suite runs at the
      stage1 tip (code equal to d045e17c, apart from
      scripts/window_20261029/) during round 6 (the review's L2).
    - **Corrected 2026-11-04 22:40 (the 2026-11-04 21:43 review; records only, nothing deployed changes):**
      - **The line's end.** After round 9's claim, --end-by's need is 1.2 x the longest wall at |H| 2,500. That is a claim round's (about 9 h), so every later round needs about 11 h (rounds_local.py:294), the claimless ones included. The line ends after round 15 or 16, not about round 17: ROUNDS_DONE about Nov 7 17:00-21:00, the final match about Nov 7 21:00-Nov 8 01:00. This costs about one round, about +14 Elo at T=1. The "only near the end" above understated it.
      - **The PAUSE bar.** The PAUSE needs the last three half-B readings' sum above 2 SE. At |H| 2,500 that bar is about 0.073, against 0.061 for rounds 4-6.
        - Half B falls: 0.138, 0.109, 0.090, 0.088, 0.059. A geometric fit projects sums of about 0.12 for rounds 7-9 (clear), 0.06-0.08 for rounds 10-12 (about even) and 0.03-0.05 for rounds 13-15 (a likely PAUSE).
        - A PAUSE is a design stop: a final match (about 3.9 h), then a re-arm when 6 h remain. Window-end amendment 1 covers the case where the re-armed rounds find no time.
        - This cost is missing from the list above.
      - **Half B's SE** at 2,500 reads about 0.021, from round 6's 0.0147 at 5,000, not 0.027.
      - **The KL subsample.** `kl_fraction` 0.05 (pg_eval.py:115) reads about 250 game-sides at 2,500. The 1x rung's edge is placed from it, so its step can be a few percent off. Not listed above; small.
