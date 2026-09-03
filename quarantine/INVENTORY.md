# Quarantine inventory (2026-09-03)

Every separately switchable mechanism above the basic self-play
loop (play games in the simulator -> build policy/value targets ->
gradient step), enumerated from the code as of tag
`pre-restart-20260903`. Each entry: where, mechanism, default,
evidence, category, coupling. Verdicts are the documents' own;
"never measured in isolation" means exactly that.

Categories: CORE (part of the basic loop), INSTRUMENT (measurement
only, no effect on training), OPS-SAFETY (tripwires, escrow,
watchdogs, config guards), TRICK-VALIDATED (measured to help the
quantity it targets), TRICK-REFUTED (measured to hurt or do
nothing), TRICK-UNTESTED (never isolated).

## 0. Evidence ledger (legs and arms cited below)

Strength numbers are Elo from decisive-only Bradley-Terry fits;
"vs seed" = vs the imitation seed `2516k-b-294k-l4-0k` (step
2,809,659, board +211 +- 67, re-gauged +223 +- 70).

| Run | Stack (beyond the loop) | Result |
|---|---|---|
| Tier-a 72h campaign (2026-07-28..31, 5M net) | Gumbel MCTS-32 completed-Q targets, replay, C51, draw tiebreak, aux head, draw_value_weight 0.25, mini/midgame/fogless mix, spool workers | +133 +- 57 in-lineage (with search); 0-0-30 vs RCA (raw policy); mini draws 0/24 -> 9/24 over 112k decisions |
| Seed measurements (2026-08-20, 08-28) | imitation checkpoint, no self-play | seed+MCTS-32 vs seed raw 9-0-1 (+321 +- 153) |
| Handoff leg 1 (tier_b_handoff, 2026-08-11) | Gumbel teacher, lambda 0.9, value anchor, jitter, tripwires, actor pool | human CE 3.207 -> 3.65..3.78 plateau; value AUC 0.76-0.79; no Elo taken |
| F1 arm (leg-2 arm of the handoff) | as leg 1 with policy anchor v1 instead of lambda | human CE held <= +0.27 over 356k steps; K median 2-4.5 on ladder; RCA 0/28 |
| TCS leg 1 (tier_b_tcs, 2026-08-14) | TCS teacher, no anchor | 7 iterations; CE 3.207 -> 3.874, aborted by the CE tripwire; K 14-17 |
| Legs 2-3 (tier_b_tcs2, 2026-08-15..17) | TCS + GBC + policy anchor, lambda 1.0, opponent frame, project none, mix 0.6/0.2/0.2 | leg 3: K median 10 -> 2, draws 0.68-0.75; 3.37M ckpt 0-0-18 vs new_2p52M (-321 +- 150); tcs2-558k -276 +- 126 (re-gauged -490) |
| Leg 4 (tier_b_l4, 2026-08-18..20) | leg 3 + linear link beta 5 + lambda 0.9 (launcher drift) + value-coef 1.0, value anchor OFF | 495k steps; 0-16 vs seed, 1-24 vs 2516k, -309 +- 105; 0-20 vs seed at sims 0 |
| Leg 5 (tier_b_l5, 2026-08-21..25) | from seed; lambda 1.0, mover frame, K-tripwire 10, policy anchor v2, value anchor ON, project none | 122k steps; killed by the value-AUC tripwire (instrument later found unsound; trunk rotation onto unit count, AUC 0.338 on one game) |
| Leg 5 resume (2026-08-25..26) | same, from escrow | to +251k steps; pin 9-0-31 vs seed (seed +208 +- 64) with every proxy healthy; 2x2: seed+TCS vs seed+MCTS 9-0-31 |
| Arm T (2026-08-29) | leg-5-resume config + fixed GBC labels; TCS teacher | 24-game probes vs seed: -28, -137, -61, -478, -104, -255, -201, -104, -382; K 17-22 |
| Arm M (2026-08-29) | arm T with `--no-turn-search` (Gumbel MCTS-32 teacher) | K-median tripwire at iteration 10 (~124k); -162, -374, ~-60, -374; final -263 +- 90 |
| Arm G (2026-08-29) | arm T with `--no-gbc` | -201, -85, -263 |
| Arms V1/V2 (2026-08-30..31) | arm T + value memory 20 iterations (full unfreeze / head-only) | K-collapse at ~5 / ~3 iterations; V1 +56 then -182 |
| Arm VG (2026-09-01) | arm T minus memory + project reval + value grounding (rollout 1.0 / consist 0.25) + aux, moves-left detached | K-tripwire at iteration 4; 6-0-18 at 59k (~-190) |
| Arm VG2 (2026-09-02) | VG + Gaussian consistency (b, sigma2 estimated) + trust region from lambda 1.0 | K median 1 at iteration 1 |
| Arm VG3 (2026-09-02..03) | VG2 + gate-frame capture + measured lambda0 12.1 + checkpoint continuation metadata | K 12 -> 8-10 over 6 iterations; pins 6-0-18 (~-190) at 53k, 5-0-19 (~-230) at 83k; consistency term 99-100% of the update |

Offline instruments referenced: rung-1 TCS probe (2026-08-14),
E1/E2/E3/Q7 and the gate bake-off (2026-08-20..21), the TCS
collapse probe (2026-08-31), signal-profiler rounds 1-6
(2026-08-31..09-01), label-calibration harvests (2026-09-02).

---

## 1. CORE

### 1.1 Simulator self-play game loop
- Where: `tools/sim_self_play.py::run_iteration`, `play_one_game`; `tools/wesnoth_sim.py`; `tools/scenario_pool.py`.
- Mechanism: plays N games per iteration in the bit-exact simulator with random ladder map, random factions and leaders, PvP economy defaults (`--starting-gold/--village-gold/--village-support/--exp-modifier`, `--no-map-settings`), uniform advancement, a recruit-bounce retry loop, per-game JSONL telemetry.
- Default: on (24 games/iteration on the box, 4 at the CLI).
- Evidence: combat 731/731 strike parity; replay corpus 24,796/24,796 bit-exact. Not a trick; the substrate.
- Category: CORE.
- Coupling: every producer (in-process, actor pool, spool) runs it.

### 1.2 Transformer model with C51 value head
- Where: `wesnoth_ai/model.py::WesnothModel` (value head, `cliffness = std(Z)`), `wesnoth_ai/trainer.py::_categorical_value_loss`, `_project_returns_to_atoms`.
- Mechanism: 51-atom categorical value distribution on [-1, 1]; the scalar V is its mean; the value loss is cross-entropy against the projected label.
- Default: on (no flag; architecture).
- Evidence: signal-profiler rounds 4-6 (teacher_arms_findings): the value term owns ~99% of the applied update direction at every checkpoint including the seed (norm 3.07 vs policy 0.25); winner/loser gradients anti-parallel (cos -0.83..-0.95) with the bulk on turns 1-20 where labels are aleatoric. `docs/az_minimal_spec.md` claims a scalar squared-error loss would charge a confident head in proportion to the miss; that alternative is now switchable (6.26) and has not been run in this lineage. The tier-a pre-C51 comparison (`docs/mcts_vs_reinforce_eval.md`) was 10/10 draws and uninformative.
- Category: CORE.
- Coupling: value-label smoothing, draw tiebreak z, draw_value_weight, every value-side term; `fresh_ce_floor` telemetry assumes the categorical form.

### 1.3 MCTS core (PUCT, chance nodes, transposition table, virtual loss)
- Where: `tools/mcts.py::mcts_search`, `MCTSConfig` (`c_puct` 1.5, `chance_nodes` True, `use_transposition_table` True, `virtual_loss` 1.0, `batch_size` device-aware via `--mcts-batch-size`).
- Mechanism: AlphaZero search with network leaf values; stochastic edges re-fork the sim and keep one child per outcome state; a per-search transposition table; batched leaf evaluation under virtual loss.
- Default: on under `--mcts` (CLI default since 2026-08-10).
- Evidence: as a PLAY procedure at 32 sims, measured to help on the seed: seed+MCTS-32 vs seed raw 9-0-1, +321 +- 153 (BACKLOG 2026-08-28). Transposition hit rate measured ~0.4% at 20 sims (techniques.md 1.6). Chance nodes / TT / virtual loss never isolated for strength.
- Category: CORE.
- Coupling: Gumbel root (3.x), FPU, tree reuse, exact enumeration, playout cap, draw tiebreak terminal, aux/moves-left search consumers.

### 1.4 Distillation trainer step
- Where: `wesnoth_ai/trainer.py::_trainer_step_mcts` (`Trainer.step_mcts`), `_mcts_factored_policy_loss`, `TrainerConfig` (`learning_rate` 1e-4, `weight_decay` 1e-4, `grad_clip` 1.0, `value_coef` 0.5, `value_clip` 1.0, `max_transitions_per_step` 4000, `train_batch_size` device-aware, `vectorized_mcts_policy_loss` True).
- Mechanism: factored cross-entropy of the four policy heads against the stored target tuples plus `value_coef` x categorical value loss, one AdamW step per call, gradient-norm clip.
- Default: on.
- Evidence: the loop's fixed part; the only measurement is the gradient-share profile above (policy ~1% of the update direction at beta 5, teacher_arms_findings round 4-5). `--value-coef 1.0` was passed on every tier-b leg (launcher) and never isolated.
- Category: CORE.
- Coupling: every loss term below plugs in here; per-game/per-side weights (5.15) and value_weight censoring (5.11) enter every term.

### 1.5 Terminal-outcome labels (z) with honest draws
- Where: `tools/mcts_policy.py::finalize_game` (z = +-1 from the side-to-move view; z = 0 on winnerless games unless `--train-draw-tiebreak`).
- Mechanism: every recorded state of a game gets the game's result as its value label.
- Default: on.
- Evidence: the 2026-07-10 comparison that set honest z=0 as default is item 4.10. Round 6 of the profiler: outcome labels on turns 1-20 are largely aleatoric and carry 3-7x the gradient mass of the informative late game.
- Category: CORE.
- Coupling: draw_value_weight (5.11), train_draw_tiebreak (4.10), value memory (4.5), holdout probe.

### 1.6 Imitation seed (warm start)
- Where: `--checkpoint-in`; seed `seed_imit_tierb_start.pt` (HF `tier-b/a3/`), built by `tools/supervised_train.py --imitation-config` with `--reinit-value-head`.
- Mechanism: self-play starts from the behaviour-cloned 15M checkpoint instead of random init.
- Default: on for every tier-b leg (a bare CLI invocation is random init).
- Evidence: seed board +211 +- 67 / +223 +- 70, the strongest checkpoint ever measured; every self-play leg from it measured below it (ledger). `--reset-decision-step` (weights-only warm start) is inert while the combat-oracle alphas are 0 (7.4).
- Category: CORE.
- Coupling: policy anchor cache (4.1) and value anchor cache are built from the same corpus; the human-holdout probe's t0 is the seed's CE.

### 1.7 Execution topology (thread workers / actor pool / spool workers)
- Where: `--workers`, `--actor-pool` + `--actor-max-batch` + `--pool-drain-grace` (`tools/actor_pool.py`), `--spool-workers` + `--spool-worker-device` + `--spool-cuda-workers` + `--spool-dir` (`tools/selfplay_worker.py`, `SpoolWorkers` ingest in sim_self_play).
- Mechanism: who generates games and where forwards run; the actor pool ships leaves to a central batched server (no weight sync), the spool runs whole games per process and pickles them.
- Default: serial in-process; tier-b legs used `--actor-pool` (F3 ruling 2026-08-10, "activated without a fresh A/B").
- Evidence: spool saturated a 4090 at 99% at tier-a (techniques.md 7.4); actor-pool server measured ~200 req/s ceiling at tier-a; A6 postmortem ~54 fwd/s shared by ~24 games (turn_search comment). Cross-actor batching breaks bit-determinism. Instrument gaps on the pool path: boundary pairs read n=0 for two campaigns, TCS/distill stats not aggregated in leg 3 (leg4 doc R2). Never compared for strength.
- Category: CORE (execution; affects which telemetry is live).
- Coupling: every flag the workers build targets with must be forwarded (symmetry contract); `--pool-drain-grace` fixes leg 3's 30-60% discarded games.

### 1.8 Inference precision, compile, and CPU thread cap
- Where: `--infer-bf16`, `--infer-compile` (`wesnoth_ai/transformer_policy.py`); eval defaults cuda-auto; `--torch-threads` (CPU-only auto-cap to 4, measured ~1.3-2.3x on CPU).
- Mechanism: bf16 autocast and `torch.compile` on the inference copy only; intra-op thread cap on CPU devices.
- Default: OFF for training (2026-08-29: an in-process compile deadlocked the spool e2e); ON for eval on CUDA.
- Evidence: 3060 bench (BACKLOG 2026-08-28): eager bf16 1.15x, compile 1x, compile+bf16 2.0x; bf16 alone NO-GO on the 3060 at batch 1 (2026-08-05). `docs/teacher_arms_20260829.md` states arms T/M ran "the new compile+bf16 inference default"; BACKLOG states the training default was rolled back to OFF the same day so the arms would not carry it. The two statements disagree; the sim_self_play banner logs the resolved value.
- Category: CORE (throughput; numerics change flagged as an attribution confound).
- Coupling: E2 control-leg caveat in `docs/redesign_1000x_20260828.md`.

### 1.9 REINFORCE path with shaping rewards (legacy alternative loop)
- Where: `--reinforce` / `--no-mcts`; `wesnoth_ai/trainer.py::Trainer.step` (`gamma` 0.99, `entropy_coef` 0.001, `normalize_advantages`); `wesnoth_ai/rewards.py::WeightedReward`, `--reward-config`, `configs/reward_selfplay.json`.
- Mechanism: policy gradient with a value baseline and entropy bonus on per-step shaped rewards (gold killed, village delta, damage, per-turn penalty, unit-type and turn-conditional bonuses).
- Default: OFF (production is `--mcts`; `--reward-config` with `--mcts` is refused at startup since F5).
- Evidence: structurally inert under `--mcts` (`MCTSPolicy.observe` is a no-op, `uses_step_rewards=False`; autonomous_run cycle 29: "a whole prior investigation was aimed at a dead channel" -- the `weight_gold=0` non-fix). The only MCTS-vs-REINFORCE comparison (`docs/mcts_vs_reinforce_eval.md`) predates the C51 head and was 10/10 draws. No REINFORCE-era strength number exists on the current net.
- Category: CORE (legacy alternative); the shaping terms are TRICK-UNTESTED on the production path.
- Coupling: only the REINFORCE step consumes it; draw tiebreak (5.9) is the one live shaping seam under MCTS.

---

## 2. INSTRUMENT

### 2.1 Frozen self-play holdout probe
- Where: `--holdout-size`, `--holdout-per-game-cap`; `MCTSPolicy.offer_holdout_game`, `holdout_metrics`, `save_holdout`/`load_holdout` (`<ckpt>.holdout` sidecar), `Trainer.eval_value_loss`.
- Mechanism: diverts whole games (<= 64 sampled states each) out of training until 512 states are held; logs the net's value CE on them each iteration; persisted across relaunches.
- Default: 0 (off) at the CLI; 512 on every tier-b leg.
- Evidence: 2026-07-07 (2-game holdout measured idiosyncrasies -> per-game cap); 2026-07-18 (per-restart resampling jumped levels 0.44 <-> 0.88 -> sidecar). Feeds the stall tripwire (3.2). Read as tripwire, not success gauge (A5 ruling).
- Category: INSTRUMENT.
- Coupling: removes ~512 states from training; index-basis stamp ties it to the relevant-set flag.

### 2.2 Fresh-batch value metrics (`fresh_value_ce` family)
- Where: `MCTSPolicy.train_step` pre-update probe, `Trainer.eval_value_metrics`: `fresh_value_ce`, `fresh_ce_floor`, `fresh_ce_std`, `fresh_pred_entropy`, `fresh_decisive_ce`, per-decade `fresh_{ce,floor,auc,n}_{d1_10..d61p}` (2026-09-01).
- Mechanism: value CE on <= 256 of this iteration's incoming game states before any gradient step, with the state-blind marginal floor; grounding experiences excluded.
- Default: on whenever the replay buffer is on.
- Evidence: the project's declared success metric (read floor-relative). Arm VG: grounding states leaked into it and read 13+ before exclusion. Per-decade split produced the round-6 finding (seed AUC 0.40 on turns 1-10, 0.96 on 21-30).
- Category: INSTRUMENT.
- Coupling: requires `--replay-buffer`; the stall tripwire reads it.

### 2.3 Target-composition telemetry (`z_*_frac`, `z_*_frac_w`, `value_signal_states`)
- Where: `MCTSPolicy._attach_z_composition`; `TrainStats.value_signal_states`.
- Mechanism: win/loss/draw share of incoming labels, raw and game-weight-weighted; count of states carrying value weight.
- Default: on.
- Evidence: 2026-07-22 misread (0.19 raw draws = ~5% weighted); leg-4 R3 used `value_signal_states` 2048 -> 1038 to expose halved value supervision.
- Category: INSTRUMENT.
- Coupling: draw_value_weight, censoring.

### 2.4 Boundary-consistency probe (`boundary_sum`)
- Where: `MCTSPolicy.harvest_boundary_pairs`, `_attach_boundary_sum` (`BOUNDARY_SAMPLE_K` 64).
- Mechanism: mean V(s_pre) + V(s_post) over sampled side-switch pairs; zero-sum play predicts ~0.
- Default: on.
- Evidence: +0.4..+0.65 on the fogged 2026-07 lineage, ~0 fogless; seed -0.19 (autonomous_run cycle 13); the batch-mean consistency penalty (T1-F) was built and then killed by its own gate (cycle 12: live campaign read -0.14 / -0.02). Dark (n=0) on the spool path in 2026-07-29 and on the pool path through leg 4 (harvest not wired into the pool drain until the leg-4 fixes). VG2 calibration re-measured the head-vs-truth optimism at +0.42.
- Category: INSTRUMENT.
- Coupling: harvested only for training games; pairs must come from one game's ordered experiences.

### 2.5 Distillation-target telemetry
- Where: `tools/mcts.py::extract_gumbel_policy_target` stashes, `MCTSPolicy.drain_distill_stats`, `TurnCommitPolicy.drain_distill_stats`: `distill_sharpen_top`, `distill_prior_entropy`, `distill_prior_top80`, `distill_et_prior/target`, `distill_kl_prior`, `link_clip_frac`, per-category target mass.
- Mechanism: per-decision comparison of the taught target against the net's own prior.
- Default: on (in-process and pool drain; dark on the spool path).
- Evidence: leg 4: target flatter than the prior at 26/26 iterations, `sharpen_top` -0.032 at iteration 0 matching the computed -0.033 (leg4 doc); leg 3: `K * et_prior = 1.035 +- 0.086`; round 4-5: KL(target||prior) median 0.002-0.006. Leg 3's stats were dark on the pool path during the collapse.
- Category: INSTRUMENT.
- Coupling: lambda, link, beta, rescale floor.

### 2.6 Search diagnostics (`search_q_spread`, `search_overturn_frac`, depth, reuse, end_turn context)
- Where: `MCTSPolicy._note_search_diag`, `pop_search_diag`; games.jsonl `engagement.search`.
- Mechanism: per-decision root child-Q spread, whether search overturned the prior's argmax, tree depth, reuse hits, unspent-MP context at chosen end_turns.
- Default: on (Gumbel/classic path; empty under TCS).
- Evidence: empty in all 28 leg-4 rows (TCS path); no verdict rests on it.
- Category: INSTRUMENT.
- Coupling: MCTS path only.

### 2.7 TCS planning telemetry
- Where: `TurnCommitPolicy.drain_tcs_stats`: `tcs_plans`, `tcs_accepts_per_plan`, `tcs_replans_per_plan`, `tcs_projections`, `tcs_gate_shorten_per_plan`, `tcs_gate_flips`, `tcs_blind_coord_frac`; `TRACE` hook in `turn_search.py`.
- Mechanism: per-iteration accept/re-plan/shortening rates of the turn search.
- Default: on (added to the CSV field list after leg 4; INFO-only before).
- Evidence: VG2 collapse read shorten/plan 0.10 -> 0.58 and accepts/plan 0.15 -> 0.65 in one iteration; VG3 accepts/plan 0.17 -> 0.27 with K 12 -> 8. Not in the CSV during legs 3-4 (leg4 doc R2).
- Category: INSTRUMENT.
- Coupling: TCS only; pool drain only (no spool channel).

### 2.8 Plan-tournament telemetry (`pt_*`)
- Where: `tools/plan_tournament.py::PT_DRAIN_KEYS`, `drain_tournament_stats` (certification rate, beta percentiles, forwards/turn, half-turn cap hits).
- Mechanism: per-iteration certify/abstain accounting.
- Default: on when `--plan-tournament` is on; not aggregated from spool workers (loud warning).
- Evidence: never produced a live row (PT never ran a leg).
- Category: INSTRUMENT.
- Coupling: 5.28.

### 2.9 In-loop signal telemetry (`sig_*_norm`, `dv_consult`)
- Where: `--signal-telemetry`; `tools/signal_telemetry.py::signal_grad_norms`, `consult_values`, `dv_stats`; `MCTSPolicy._attach_signal_telemetry`.
- Mechanism: 4 extra unclipped backward passes on <= 128 states (policy / game-value / rollout-value / consistency norms) and the mean |dV| the applied update moved on <= 64 grounding-captured states.
- Default: norms OFF (ruling 2026-09-02); `dv_consult` always on when grounding states exist (the trust-region controller reads it).
- Evidence: recorded the VG collapse live (consist norm 25-42 vs game 3.8-6.5, dv 0.03-0.17); VG3's first in-loop rows were invalid until the value_weight surgery fix (4a360e5); VG3 held-out dv 0.035-0.063 under lambda 14-35.
- Category: INSTRUMENT.
- Coupling: grounding (4.4-4.6) for the consulted states; trust region (4.7) consumes dv.

### 2.10 VG2 estimate telemetry
- Where: `MCTSPolicy._vg2_prepare/_vg2_finish`: `trust_lambda`, `consist_bias_hat`, `consist_sigma2_hat`, `consist_pair_n`, `consist_var_diff`, `consist_roll_noise`, `consist_sigma2_point/se`, `consist_label_minus_truth`, `consist_head_minus_truth`.
- Mechanism: the paired-label estimates and controller state, logged each iteration.
- Default: on when consistency pairs exist.
- Evidence: VG3: sigma2 0.10 -> 0.067 -> floor, lambda 12 -> 267, head-minus-truth +0.53 -> -0.03 -> +0.17.
- Category: INSTRUMENT (the same numbers also drive training; see 4.6/4.7).
- Coupling: 4.6, 4.7.

### 2.11 Human-holdout probe loop (CE, stratified value AUC)
- Where: `scripts/holdout_probe_loop.py` (hourly `supervised_train.py --imitation-config --eval-only`), `tools/supervised_train.py` value-AUC block; CSV escrowed to HF.
- Mechanism: human-play policy CE and winner-vs-loser value AUC on the imitation manifest holdout (8 pairs/game across ~150 games since 2026-08-25, between-game SE).
- Default: on on every box leg.
- Evidence: leg 1 CE plateau (+0.5 nats); TCS leg 1 monotone erosion; leg 3 CE flat while K collapsed ("blind to search-driven passivity"); leg 5's 1,200-pair pooled AUC was 60% one game and the SE wrong by ~10x (leg5 doc M2) -> stratified rebuild; leg-5 resume: CE 2.48-2.60 and AUC 0.68-0.73 healthy while -200 Elo eroded. Per-faction split (technique review) not implemented.
- Category: INSTRUMENT (its tripwire role is 3.4).
- Coupling: policy anchor cache excludes these holdout games.

### 2.12 Trainer history CSV, games.jsonl, engagement telemetry, validation exports, profiling hooks
- Where: `_TrainerHistoryCSV` (~140 columns, `--trainer-history-csv`), `--game-log-dir`, `eng_*` columns, `--validate-export-every`/`--validate-export-dir` (`tools/validation_exports.py`), `--prof` (`WESNOTH_PROF=1`, `tools/prof_report.py`), root cliffness log.
- Mechanism: per-iteration and per-game records; every Nth game exported as a Wesnoth-loadable replay for strict-sync checks; per-component timers.
- Default: on (exports every 100th game at the CLI, every game on the box).
- Evidence: the 2026-08-04 export sweep found 538/574 clean with every failure root-caused; six known-bad midgame Aethermaw exports on HF predate the fix. Cliffness calibration (`docs/cliffness_calibration.md`) was only ever run on a pre-C51 checkpoint.
- Category: INSTRUMENT.
- Coupling: none on training.

### 2.13 Offline probes and profilers
- Where: `signal_profiler/` (gradient/update trees, label calibration, `run_label_calibration.py`), `tools/policy_shape_probe.py` (E1), `tools/target_channel_icc.py` (E2), `tools/turn_counterfactual_probe.py` (rung 1, bake-off), the TCS collapse probe (`eval_games/tcs_collapse_probe/`), `tools/recruit_prior_drift.py`, `tools/mini_anatomy.py`, `tools/run_elo_batch.py`/`elo_eval_game.py`/`elo_catalog.py`, `tools/eval_vs_builtin.py`.
- Mechanism: state-set probes on frozen checkpoints; Elo matches.
- Default: run by hand.
- Evidence: rounds 1-3 of the profiler measured random-init nets by mistake (missing checkpoints silently random-init; guard shipped); the 2x2 pin+TCS arms played the opponent frame while the pin trained mover (BACKLOG 2026-08-27 qualification); every RCA eval self-blinded our side under fog while RCA saw all (fixed, unmeasured since).
- Category: INSTRUMENT.
- Coupling: none.

---

## 3. OPS-SAFETY

### 3.1 Decisive-rate tripwire
- Where: `--abort-decisive-rate`, `--abort-window` (exit 4), sim_self_play main loop.
- Mechanism: stop when the trailing-window fraction of decisive games falls below the floor.
- Default: off at the CLI; 0.35 over 20 on every box leg.
- Evidence: designed for the tier-a all-draws shape; never fired on a tier-b leg (24/24 decisive typical; leg 3's draws rose to 0.75 but the leg was paused by hand first).
- Category: OPS-SAFETY.
- Coupling: none.

### 3.2 Holdout-stall (memorization) tripwire
- Where: `--abort-holdout-stall`, `--abort-holdout-min-delta` (exit 5); reads `last_fresh_rel_ce`.
- Mechanism: stop when floor-relative fresh CE makes no new best for N iterations.
- Default: off; 60 on box legs.
- Evidence: motivating 2026-07-02 Kaggle run (train value loss 3.8 -> 1.15, holdout flat 3.1); the raw-CE version mis-fired twice in the 72h run -> floor-relative (A5 ruling). Never fired on a tier-b leg.
- Category: OPS-SAFETY.
- Coupling: needs `--replay-buffer`.

### 3.3 K-median (turn-length collapse) tripwire
- Where: `--abort-k-median` (exit 7); `k_median_of(outcomes)`.
- Mechanism: stop when median actions per side-turn is below the bar for 3 consecutive iterations.
- Default: off at the CLI (explicit-pass ruling); 10 on leg 5 and all arms.
- Evidence: fired on arm M (iteration 10) and arm VG (iteration 4); leg 3 collapsed to K 2 with every other guard green (motivation); VG3 hovered 8/9/10 without a third strike. The one tripwire `docs/az_minimal_spec.md` keeps.
- Category: OPS-SAFETY.
- Coupling: none.

### 3.4 Value-AUC redraw tripwire and qualify gate
- Where: `scripts/holdout_probe_loop.py` (`PROBE_AUC_FLOOR` 0.60, up to 3 independent redraws, `redraw_verdict`; `--qualify` with `QUALIFY_AUC_MIN` 0.60 wired into `scripts/vast_onstart.sh` 2026-08-25).
- Mechanism: abort the leg if the human-holdout value AUC reads below the floor on three independent sample redraws; refuse to launch from a checkpoint that fails the floor.
- Default: on on box legs.
- Evidence: the pre-redraw version killed leg 5 at 122k on the unsound 1,200-pair statistic (leg5 doc); the redesigned probe never fired on the resume. Leg-5 resume qualify PASS at 0.765.
- Category: OPS-SAFETY.
- Coupling: 2.11.

### 3.5 Human-CE abort tripwire (REMOVED)
- Where: was in `scripts/holdout_probe_loop.py` (abort at t0 + 0.5 nats, 3 consecutive points).
- Mechanism: stop when human-play policy CE rose past the bar.
- Default: removed by ruling 2026-08-25 ("learning to play better does not necessarily mean learning how humans play").
- Evidence: killed TCS leg 1 at 7 iterations (3.207 -> 3.874); leg 4 finished 0.006 nats under the bar while losing ~520 Elo (leg4 doc).
- Category: OPS-SAFETY (retired).
- Coupling: 2.11.

### 3.6 Dead-iteration guard
- Where: `DEAD_ITER_LIMIT = 5` in the main loop (exit 3).
- Mechanism: stop after five iterations that rolled zero games.
- Default: on, hardcoded.
- Evidence: 2026-05-09 job "completed" 999 empty iterations.
- Category: OPS-SAFETY.
- Coupling: none.

### 3.7 Fork guard smoke (`SIM_FORK_GUARD=1`)
- Where: `tools/mcts.py` (`deep_state_fingerprint` around every search), launcher smoke stage (A6), covers TCS/PT since 2026-08-27 as a BaseException.
- Mechanism: assert the live state is bit-identical before and after every search fork.
- Default: off in steady-state; one smoke iteration per launch.
- Evidence: caught three aliasing bugs invisible to `state_key` (village ownership `fa95da5`, Aethermaw morph, `first_time_only` latch).
- Category: OPS-SAFETY.
- Coupling: none.

### 3.8 Box supervision: relaunch loop, stall watchdog, stop-on-abort, HF escrow, atomic checkpoints
- Where: `scripts/vast_onstart.sh` (10-try relaunch, ABORTED_* markers), `scripts/stall_watchdog.py`, `scripts/box_stop_on_abort.py` (`.vast_api_key`, `.instance_id`), HF escrow sweep every 30 min (checkpoint + sidecar + CSV + probe CSV + games log), `transformer_policy.save_checkpoint` (`.tmp` -> `.bak` -> replace; loader falls back to `.bak`), holdout sidecar carry, `--time-budget`, graceful-cancel sentinel, CUDA OOM demotion retry, `--pool-drain-grace`.
- Mechanism: keep a paid box producing or stop it; make every identity-critical artifact survive a box death.
- Default: on on box legs.
- Evidence: a fresh box reseeded itself from escrow unattended 3x incl. one cross-machine migration; arm VG idled ~8h post-tripwire before stop-on-abort existed; leg-3 discarded 30-60% of some iterations' games before the drain grace; the imitation run hung silently ~2 days (session-bound watcher lesson).
- Category: OPS-SAFETY.
- Coupling: none on the loss.

### 3.9 Config-drift guards
- Where: checkpoint-sticky heads (`aux_score`/`moves_left`/`gbc` peek-and-OR in main), relevant-set basis inheritance + contradiction halt, spool index-basis tripwire (`SystemExit(6)`), `training_meta` recipe fingerprint refused on mismatch (`MCTSPolicy.apply_training_meta`), `--reward-config` + `--mcts` refusal (F5), mix-ratio sum validation, `configs/leg_l5.json` file-mode launch, `--turn-boundary-frame` must be asserted explicitly.
- Mechanism: make silently-different runs fail at startup.
- Default: on.
- Evidence: motivations are all incidents: leg 3 ran a [60,200] cap from a launch-env omission; leg 4 ran lambda 0.9 from an unconditional launcher emit; `--no-gbc` resume of a gbc checkpoint attached no labels; VG3's lambda0 was silently unapplied in the first launch.
- Category: OPS-SAFETY.
- Coupling: none on the loss.

---

## 4. TRICK-VALIDATED (for the quantity each targets; none has a measured strength benefit)

### 4.1 Policy-head human anchor (F1)
- Where: `--human-anchor-policy-file/-updates/-batch`; `tools/policy_anchor.py::anchor_policy_step`, `sample_pairs_game_normalized` (cache v2, game-normalized, holdout-excluded); called in `run_iteration` before `train_step`.
- Mechanism: 4 extra gradient steps per iteration of the four-head imitation CE on 128 winner-side human pairs (no mask, no value gradient).
- Default: OFF at the CLI; ON on every leg from leg 2 through VG3 (the F1 "one protection per leg" ruling; anchors ruled default-OFF 2026-08-26 as "symptom control").
- Evidence: FOR its target: the F1 arm held human-holdout CE <= +0.27 over 356k steps while TCS leg 1 without it eroded monotonically to the abort bar; leg 3 with it held CE 3.46-3.52 through the K collapse. AGAINST as a strength lever: every anchored leg still eroded (legs 3, 4, 5, arms T/G/V/VG*); leg-4 review refuted the anchor as that leg's cause (its class weights are a restoring force for attack/recruit); leg-3 R5 names the anchor's end_turn slot pinning as an untested reason K looked green. Round 5: policy channel is adequately sized after Adam but starved of target content, so the anchor competes with an ~empty target.
- Category: TRICK-VALIDATED (human-CE proxy only).
- Coupling: measured under cache v1 (pair-uniform) for the F1 arm and leg 2, v2 from leg 3; excludes probe games; interacts with lambda (only one prior protection per leg by ruling).

### 4.2 Value trust region (VG2/VG3)
- Where: `TrainerConfig.trust_lambda`, `trust_delta` 0.08; `MCTSExperience.v_anchor`; `_trainer_step_mcts` trust term; `MCTSPolicy._vg2_prepare/_vg2_finish` (proportional multiplicative dual ascent on `dv_consult`, factor bounded [1/2, 4], lambda in [1e-3, 1e3]); lambda0 from `signal_profiler/run_label_calibration.py` (p90(dv)/delta); `training_meta` continuation.
- Mechanism: penalise lambda x (V - V_at_iteration_start)^2 on search-consulted states; lambda tracks the measured per-iteration movement toward 2 C51 atoms.
- Default: 0 (off) unless grounding states exist; lambda0 1.0 in VG2, measured 12.1 in VG3.
- Evidence: FOR: VG3 held per-iteration movement on gate-frame states at ~0.1 (held-out 0.035-0.063) vs 0.83 unregulated; continuation metadata restored lambda 26.6 on a real resume. AGAINST: VG2's unprotected first iteration (lambda 1) moved consulted values 0.31 and K went to 1; VG3's controller saturated (dv 0.23 at lambda 267) once the consistency precision hit its floor; VG3 pins -190/-230 — rate control did not change the direction. Never run without the consistency term.
- Category: TRICK-VALIDATED (rate control only).
- Coupling: needs grounding captures (4.4) for anchors; the mixture it regulates (4.6) set the direction.

### 4.3 Gate-frame (mover pre-flip) grounding capture (VG3)
- Where: `turn_search.plan_turn(capture=)` offering stage-1 boundary sims; `TurnCommitPolicy._make_ground_capture`; `value_grounding._handed_over` (labels derived by ending the turn first).
- Mechanism: capture, anchor and measure the states the TCS gate actually compares (mover frame, pre-end_turn) instead of the post-flip projection pairs.
- Default: on whenever `--value-ground` is on (since VG3).
- Evidence: FOR: VG3 calibration showed head-vs-truth disagreement of ~0.10 on this frame vs 0.4-0.7 on the post-flip frame ("largely artifacts of training the other frame"); VG2's collapse was the post-flip training generalizing across the flip with amplification (0.31 trained -> 0.71 on the gate frame). AGAINST: VG3 still eroded. Never isolated from the rest of VG3.
- Category: TRICK-VALIDATED (frame artifact measured and removed).
- Coupling: 4.4-4.7, projection depth.

### 4.4 TCS two-stage acceptance and placebo separation
- Where: `turn_search.two_stage_accept` (`--turn-reval-salts` 3, `--turn-min-delta` 0.01); `tools/turn_counterfactual_probe.py` placebo arm.
- Mechanism: the argmax alternative is re-graded against the incumbent at fresh salts and accepted only above 2 sigma / sqrt(3).
- Default: on within TCS.
- Evidence: rung-1 probe: revalidated accept 0.64/0.46, placebo 0.13/0.18, naive -> reval 0.84 -> 0.67; bake-off separation 3.7-3.8:1 under both frames. E2: sighted ICC 0.958-0.995 (the gate's channel is not salt noise). Gate-level only; the target pass grades at one salt (leg4 doc R2 distinction).
- Category: TRICK-VALIDATED (gate noise control, offline).
- Coupling: part of 5.20; does not touch the distill target.

### 4.5 Mover boundary frame (`--turn-boundary-frame mover`)
- Where: `TurnSearchConfig.boundary_frame`; `turn_search.materialize(mover_frame=)`, `batch_boundary_values`.
- Mechanism: grade candidate turns on the pre-end_turn state from the mover's own information set instead of the post-flip fogged opponent view.
- Default: `opponent` in the dataclass; `mover` asserted on leg 5 and all arms.
- Evidence: FOR: opponent frame graded 4 different candidates bit-identically on no-contact fogged turns; bake-off revalidated accept 0.90 vs 0.67 with stage-1 verdicts that survive (gap 0.04 vs 0.17); E2 blind fraction 8% -> 0. AGAINST: Q7 measured a mover-frame pass bias of +0.147 +- 0.021 (3.7 atoms) — the head overrates passing; TCS collapse probe: under mover frame without projection, end_turn alternatives win 30/40 accepted gates on the seed. Strength never isolated (every mover-frame leg also changed other things).
- Category: TRICK-VALIDATED (gate discrimination offline; carries a measured pass bias).
- Coupling: projection (5.21) is the named counter to the pass bias; `mover_mp0` variant (5.22) falsified.

### 4.6 Gumbel q-transform normalization fix (c_scale 0.1, min-max rescale)
- Where: `MCTSConfig.gumbel_c_visit` 50, `gumbel_c_scale` 0.1, `gumbel_rescale_q` True; `mcts._rescale_q`, `_gumbel_sigma`.
- Mechanism: the sigma(q) tilt uses the reference implementation's scale and [0,1] rescale instead of raw q with c_scale 1.0.
- Default: on since 2026-07-28.
- Evidence: before the fix Q differences were multiplied ~50-80x and the target was near one-hot (recruit target mass 0.000-0.002 across independent searches vs prior 0.16); after it the tier-a lineage measured +133 +- 57 in-lineage and 0-0-30 vs RCA. A bug fix, listed here because it is switchable.
- Category: TRICK-VALIDATED (target shape; strength unchanged externally).
- Coupling: shared byte-for-byte with the TCS `exp` link.

---

## 5. TRICK-REFUTED

### 5.1 Distillation prior discount (`--distill-prior-discount` 0.9)
- Where: `MCTSConfig.distill_prior_discount`; applied in `extract_gumbel_policy_target` and `tcs_target_distribution` (`base = p ** lam`).
- Mechanism: the target uses lambda x log(prior), so the prior is a decaying memory of search evidence.
- Default: 1.0 (off); 0.9 on handoff leg 1 (A1) and, by launcher drift, leg 4.
- Evidence: FOR (stage 1, 2026-08-05, exp link, 3 games): sharpen_top +0.130 -> +0.030, end_turn re-teaching +0.124 -> +0.020. AGAINST: leg-4 postmortem final verdict — killed by lambda 0.9 alone under the linear link: the decay acts on all ~386 legal actions while the boost touches <= 5, net-flattening per step in every regime (computed sharpen_top -0.033 vs measured -0.032); E1 on frozen states: pin +1.03 nats more entropic, top80 3.5x lower; leg 4 -309 +- 105, 0-20 vs seed at sims 0. The A1 ruling was never re-derived for the linear link. Handoff leg 1's CE plateau (+0.5 nats) was under lambda 0.9 with the exp link.
- Category: TRICK-REFUTED (under the linear link; the exp-link version has only the 3-game stage-1 read).
- Coupling: target link (5.5); one prior protection per leg ruling vs 4.1.

### 5.2 Gumbel completed-Q distillation target as the teacher
- Where: `MCTSConfig.gumbel_root` True, `gumbel_m` 16; `mcts._gumbel_root_search`, `_completed_q`, `extract_gumbel_policy_target` (softmax(lambda log prior + sigma(completed_q)) over ALL legal actions, unvisited at v_mix).
- Mechanism: Gumbel-top-16 candidates, sequential halving, target = prior tilted by completed Q; play the argmax.
- Default: on under `--mcts` without `--turn-search`.
- Evidence: as teacher: arm M K-collapsed from a healthy seed in 10 iterations with end_turn prior mass inflating through distillation (0.212 -> 0.255), final -263 +- 90 — "teacher-intrinsic to Gumbel-MCTS distillation"; the F1-arm self-play policy played K median 2-4.5 vs the seed's ~12 (turn truncation acquired under this teacher, tcs_spec rung 0); tier-a: +133 in-lineage, 0-0-30 vs RCA; autonomous_run cycle 29: the "tried-and-cut tax" (edges sampled and cut grade below v_mix). As play procedure: +321 on the seed (1.3). Never isolated from replay (16 updates), anchors, GBC, aux, C51 at tier-b.
- Category: TRICK-REFUTED (as target producer at tier-b; retained as play procedure).
- Coupling: 5.1, 5.3, 5.4, playout cap (6.7), tree reuse; `docs/az_minimal_spec.md` proposes visit-count targets instead.

### 5.3 Search-budget levers: raising `--mcts-sims`, `gumbel_m` 16 -> 8
- Where: `--mcts-sims`, `--mcts-gumbel-m`.
- Mechanism: more simulations per decision; fewer Gumbel candidates.
- Default: 50 at the CLI, 32 on every box leg; m 16.
- Evidence: cycle 34: two independent 512-sim searches on the same state differ by TV 0.85-1.00; the target concentrates rather than converges; visited edges pinned at m=16 at every N; "32 is at or near compute-optimal". m 16 -> 8 measured the wrong direction on midgame recruit mass (-0.0135 vs m=16). Both are "permanent do-not-retry" entries (technique review).
- Category: TRICK-REFUTED (as improvement levers).
- Coupling: 5.2.

### 5.4 Turn-Commitment Search (TCS)
- Where: `--turn-search` (default True); `tools/turn_policy.py::TurnCommitPolicy`, `tools/turn_search.py::plan_turn` (`record_spine`, `materialize`, `gumbel_top_k_alternatives`, `build_coordinate_target`); knobs `--turn-alt` 4, `--turn-rounds` 3, `--turn-fast-rounds` 1, `--turn-max-spine` 40, `--turn-full-prob` 0.25.
- Mechanism: sample a whole side-turn from the policy, hill-climb it by single-coordinate substitutions graded by the value head at the turn boundary, execute the committed plan (re-plan on divergence), and distill per-coordinate targets from the evaluated alternatives.
- Default: ON since 2026-08-14.
- Evidence: FOR (offline): rung-1 probe accept 0.64, median accepted delta ~2 atoms, placebo-separated 5:1; its KL gate failed as pre-registered (user ruled proceed). AGAINST (games): as a play procedure on identical weights, seed+TCS lost 9-0-31 to seed+MCTS-32 (~-200; not compute-matched, ~113 vs ~384 forwards/side-turn); as a teacher, every TCS leg eroded (TCS leg 1 CE abort; leg 3 K collapse; leg 4 -309; leg 5 -208; arm T mean ~-200 with oscillation) and profiler rounds 4-5 measured its targets at KL 0.002-0.006 from the prior (0.3% away) carrying ~1% of the update, with the systematic component pushing toward passivity (attack -0.003..-0.006, end_turn +0.002..+0.008). The findings doc exonerates the teacher procedure as the erosion ROOT cause ("the value function's off-distribution behavior is"); it never K-collapsed on arm T (K 17-22) where arm M did. `docs/tcs_collapse_mechanism_20260831.md`: boundary-only grading cannot price tempo; the gate accepts end_turn alternatives whenever value deltas dip.
- Category: TRICK-REFUTED (as shipped: play-time cost measured, no transfer measured; not shown to be the erosion cause).
- Coupling: 4.4, 4.5, 5.5, 5.21, 5.22, force-inclusion (6.20), grounding (4.3-4.7), GBC per-decision tracing, TCS telemetry.

### 5.5 Linear target link at beta 5 (`--turn-target-link linear --turn-target-beta 5`)
- Where: `turn_search.tcs_target_distribution` (prior^lam x max(0, 1 + beta(q - LOO mean)), evaluated-block mass renormalized); `TurnSearchConfig.target_link/target_beta`; derivation in `docs/design_constants.md`.
- Mechanism: evaluated actions get a multiplicative factor linear in their advantage over the other evaluated actions; unevaluated keep factor 1; exposure-invariant under a null grader.
- Default: linear, beta 5 (ruling 2026-08-17); `exp` = the Gumbel sigma tilt.
- Evidence: FOR the link's stated property: leg-3 R2 measured the exp link's exposure ratchet (+0.068 expected mass for the always-evaluated end_turn under pure noise vs +0.002 decoy); leg-4 E2 refuted grader-null (the channel carries signal). AGAINST beta 5 as shipped: the profiler measured targets 0.3% from the prior at every checkpoint (KL median 0.0018-0.0062, TV 0.03-0.045) — "homeopathic" — and this is "the measured reason a +320-Elo teacher signal does not transfer". W2 (beta 15) proposed, never run.
- Category: TRICK-REFUTED (at beta 5: measured to move the policy ~nothing).
- Coupling: 5.1 (lambda multiplies through), 5.4, `link_clip_frac` telemetry.

### 5.6 Value memory (`--value-memory-iters`)
- Where: `MCTSPolicy._value_memory_ingest`, `value_memory_step` (value head only since arm V2), `--value-memory-batch` 256, `--value-memory-states-per-game` 32; `Trainer.step_value_from_raw`.
- Mechanism: one extra value-only gradient step per iteration over game-uniform samples from the last N iterations' outcomes.
- Default: 0 (off); 20 on arms V1/V2.
- Evidence: V1 (full unfreeze) K-collapsed at ~5 iterations, V2 (head-only, parameter-exact freeze) at ~3, distill targets healthy throughout; V1 probes +56 then -182. Round 4: value_memory was 2-3x larger on arm checkpoints than on the seed, a second value channel on a value-dominated update. The collapse probe attributes V1/V2's collapse to boundary-only grading (project none) opening the end_turn door, not to a trunk gradient; "arm V3" (memory + projection) was proposed and never run.
- Category: TRICK-REFUTED (collapsed twice; with the collapse-probe caveat).
- Coupling: TCS without projection; interacts with 5.4's gate.

### 5.7 Search-consistency (bootstrap) value labels
- Where: `value_grounding.build_grounding_experiences` (`label_kind="consist"`, `--value-ground-consist` 8/game, `--value-consist-weight` 0.25 in VG); VG2+: `TrainerConfig.consist_bias/consist_sigma2`, Gaussian NLL term in `_trainer_step_mcts`, estimates in `_vg2_prepare` (EMA 0.8; sigma2 upper 90% CI since ad5c87f; rollout noise from 2 playouts since 9d04c2f).
- Mechanism: label a consulted state with the search's own depth-H projected value; VG2+ weights it by an estimated precision after subtracting an estimated bias.
- Default: off; on in VG, VG2, VG3.
- Evidence: VG: consistency norms 25-42 vs game 3.8-6.5 (coherent push, not a categorical blow-up per the 2026-09-02 correction); K-collapse in 4 iterations. VG3: precision estimator (small difference of two ~0.95 quantities) floored; the term took 98.8% then 100.5% of the update direction, game outcomes 2.6% -> 1.2%, rollout truth 0.8% -> -0.9%; cos(winner, loser gradient) +0.82 (outcome no longer in the gradient); fresh CE on real states 0.51 -> 1.70 above the floor; pins -190, -230. "The head is being trained to agree with its own lookahead."
- Category: TRICK-REFUTED (self-distillation measured at three precisions; the two estimator fixes are committed but unrun).
- Coupling: 4.2, 4.3, 5.8; `is_grounding_experience` keeps it out of the fresh probe.

### 5.8 Rollout-truth value labels inside the VG mixture
- Where: `value_grounding.rollout_outcome` (`label_kind="roll"`, `--value-ground-rollouts` 4/game, `--value-ground-weight` 1.0, `rollout_max_halfturns` 120, `rollouts_per_state` 2).
- Mechanism: play the captured consulted state to the end with the raw policy (closed loop, forked) and label it with the outcome; censored if capped.
- Default: off; on in VG/VG2/VG3 alongside 5.7.
- Evidence: VG: rollout norms 14-25 (inflated by the same coherence); VG2: "the pusher was the rollout-truth term (16-33), coherent by nature", head optimism +0.28 overshot to -0.23 in one iteration; VG3: 0.8% then -0.9% share (pushing against the net). Rollout-label bias vs true search-play outcome never measured. The rollout-only variant (VG4b, `--value-ground-consist 0`) has never run.
- Category: TRICK-REFUTED in the mixture as run; rollout-only is TRICK-UNTESTED (6.24).
- Coupling: 5.7 (the pairs it feeds), 4.2.

### 5.9 GBC event-supervision heads
- Where: `--gbc` (default True), `--gbc-coef` 0.1; `wesnoth_ai/gbc.py::GBCHeads`, `labels_for_game_states`, `gbc_loss_for_output`; labels attached in `finalize_game`, per-decision tracing via `note_observation` (since 2026-08-27); `TrainerConfig.gbc_coef`.
- Mechanism: small heads on the trunk predict fog-censored "dies within k" / "village flips within k" (k in {1,2}) from hindsight labels; BCE at coef 0.1.
- Default: ON since 2026-08-14 (checkpoint-sticky).
- Evidence: FOR the premise: rung 0d — cumulative observed events predict the outcome at AUC 0.794 while the head's event-orthogonal turn movement is AUC 0.527. AGAINST as a training signal: arm G (`--no-gbc`) oscillated identically (-201, -85, -263) — "GBC exonerated"; profiler rounds 4-5: gradient-inert (norm ~1% of value's, |du| ~0.01, dv ~0.0001). Labels were structurally degraded under TCS for every leg through leg 5 (wrong turn stamps, stale death hexes, fogless censoring); fixed 2026-08-27, so arm T onward is the first leg with working labels; fixed-vs-broken never isolated. The rung-0b w-grounding and completion seams are dead by pre-registered rule (R^2 0.112 < 0.15).
- Category: TRICK-REFUTED (measured to do nothing as a gradient).
- Coupling: adds heads (checkpoint stickiness); `peek_checkpoint_arch` dropped gbc heads in every arm eval until round 2 ("5 unexpected keys").

### 5.10 Material-z draw labels (`--train-draw-tiebreak`)
- Where: `MCTSPolicy.finalize_game` (`draw_tiebreak_z` on winner == 0).
- Mechanism: label drawn games' states with the material differential instead of 0.
- Default: OFF since 2026-07-10.
- Evidence: with it on, "predict material" became the dominant lesson (~93% of ladder games were draws at the time): human-corpus late-game AUC 0.88 -> 0.60 in ~80 iterations, r_material/r_outcome 1.28 -> 2.18. User ruling 2026-08-05b: stays OFF.
- Category: TRICK-REFUTED.
- Coupling: 6.10 (search-side tiebreak stays on).

### 5.11 No-progress stalemate rule (`--no-progress-turns`)
- Where: `tools/wesnoth_sim.py` progress tracker; `ended_no_progress` telemetry.
- Mechanism: end a game as a draw after N full turns without damage, kill, recruit or village change.
- Default: 0 (off); the tracker logs would-fire stats regardless.
- Evidence: F2 offline read over the first 112 handoff-leg games: zero would-fire events at any K — "priced irrelevant for this regime" (CLAUDE.md 2026-08-11). User 2026-08-17: unsatisfied with the detection criteria (chip damage resets the clock; reversible village trading counts as progress).
- Category: TRICK-REFUTED (would never have fired in the measured regime).
- Coupling: draw labeling (censoring).

### 5.12 `mover_mp0` boundary frame
- Where: `TurnSearchConfig.boundary_frame = "mover_mp0"`; `materialize(mover_mp0=)`.
- Mechanism: mover frame with the mover's current_moves zeroed and has_attacked set at the boundary encode, so plans compare on position not unspent potential.
- Default: off (probe apparatus).
- Evidence: TCS collapse probe: neutralizing MP/acted flags made truncation WORSE (all heads to committed median 1).
- Category: TRICK-REFUTED.
- Coupling: 4.5.

### 5.13 Detector-advice conditioning (REMOVED 2026-08-10)
- Where: was `--mcts-advice`, `wesnoth_ai/model.py` cross-attention graft, `tools/swap_detector.py` (kept as diagnostic).
- Mechanism: actor tokens attended to advice tokens built from swap-detector opportunities via a zero-init gate.
- Default: removed from the tree (X1).
- Evidence: autonomous_run cycle 32: placebo control (cross-state permutation of advice content) reproduced the `advice_out_norm` growth identically; three instruments agree the channel carries no information.
- Category: TRICK-REFUTED (deleted).
- Coupling: none now.

### 5.14 `FORBID_IDLE_END_TURN` mask gate (REMOVED)
- Where: was `wesnoth_ai/constants.py` + `action_sampler` gate (X3).
- Mechanism: make end_turn illegal while any unit can still move or a recruit is affordable.
- Default: removed.
- Evidence: bound on 92.4% of decisions and forbade 69% of HUMAN end_turns in the corpus; counter-doctrinal for a BC-warm-started lineage.
- Category: TRICK-REFUTED (deleted).
- Coupling: none now.

---

## 6. TRICK-UNTESTED

### 6.1 Distillation target temperature (`--distill-target-temp`)
- Where: `MCTSConfig.distill_target_temp`; both target builders.
- Mechanism: divide the summed target logits by T (linear link: scales beta down).
- Default: 1.0. Evidence: never run in a leg or probe. Category: TRICK-UNTESTED. Coupling: 5.1, 5.5.

### 6.2 Gumbel rescale floor (`--mcts-gumbel-rescale-floor` 0.04)
- Where: `MCTSConfig.gumbel_rescale_floor`; `mcts._rescale_q`.
- Mechanism: below one C51 atom of completed-Q spread the sigma tilt fades toward the prior.
- Default: 0.04 since 2026-08-12 (legacy 1e-8).
- Evidence: diagnosis only — at 1e-8, KL(target||prior) was independent of value-noise level (every low-signal root got the full ~5-logit tilt). tcs_spec's "K curing itself under the floor fix" reading was a cross-lineage pooled artifact (review finding A). No A/B.
- Category: TRICK-UNTESTED. Coupling: 5.2, `exp` link.

### 6.3 Hierarchical Gumbel root (`--mcts-hierarchical-gumbel`)
- Where: `MCTSConfig.gumbel_hierarchical`; `_gumbel_root_search` two-level pick.
- Mechanism: actors compete with their full prior mass, then edges within actors.
- Default: off. Evidence: motivating measurement at random init (end_turn is the fattest edge on ladder, 2.9x median); the pre-registered A/B (mini decisive rate, end_turn slot share, ladder Elo) never ran. Category: TRICK-UNTESTED. Coupling: 5.2.

### 6.4 Classic AlphaZero root (`--mcts-classic-root`, Dirichlet noise, visit temperature)
- Where: `MCTSConfig.dirichlet_alpha` 0.3, `dirichlet_eps` 0.25, `add_root_noise`, `temperature` 1.0, `temperature_decisions` 30, `root_fpu_reduction` 0; `mcts.sample_action`, `extract_visit_counts`.
- Mechanism: root noise + visit-count sampling for the first 30 decisions; visit-count targets.
- Default: off (Gumbel root is default). Evidence: the pre-Gumbel campaigns ran it (2026-05..06) before the C51 head and the imitation seed; no comparison against the Gumbel root on the current net exists. `docs/az_minimal_spec.md` proposes this form. Category: TRICK-UNTESTED (on this lineage). Coupling: 1.3.

### 6.5 First-play urgency (`--mcts-fpu-reduction` 0.25)
- Where: `MCTSConfig.fpu_reduction`; `_puct_select`.
- Mechanism: unvisited interior edges score at parent value minus 0.25 instead of 0.
- Default: on. Evidence: motivation (one-visit sweep at small budgets); technique review "clean positive audit"; never isolated. Category: TRICK-UNTESTED. Coupling: 1.3.

### 6.6 Subtree reuse across decisions (`--mcts-no-tree-reuse` to disable)
- Where: `MCTSConfig.tree_reuse`; stash in `MCTSPolicy.select_action`.
- Mechanism: reuse the played edge's subtree iff the live state key matches the searched child.
- Default: on. Evidence: `reuse_frac` telemetry only; never isolated. `docs/az_minimal_spec.md` proposes off. Category: TRICK-UNTESTED. Coupling: 1.3, playout cap (n_simulations contract).

### 6.7 Exact combat-outcome enumeration (`--mcts-no-exact-outcomes` to disable)
- Where: `MCTSConfig.exact_outcome_enumeration`; `tools/combat_outcomes.py` DP.
- Mechanism: once observed children cover the exact outcome mass, select among them by exact probability with no sim fork.
- Default: on. Evidence: 1,725/1,725 fights enumerated with zero fallbacks (mechanically works); adaptive outcome bucketing (X5) deleted as never measured. Strength never isolated. Category: TRICK-UNTESTED. Coupling: 1.3.

### 6.8 Playout-cap randomization (`--mcts-playout-cap`, `-prob` 0.25, `-fast-sims`)
- Where: `MCTSConfig.playout_cap_*`; `MCTSPolicy.select_action` (`n_sims_override`); TCS analog `--turn-full-prob` 0.25.
- Mechanism: only a random quarter of decisions (turns under TCS) run the full budget and record a target; the rest run n_sims//4 (1 round under TCS) and record nothing.
- Default: ON at the training CLI since 2026-08-05 (library off).
- Evidence: KataGo-cited 3-10x games/GPU-hour; never measured here for throughput or strength. Consequence measured in leg 3: only ~25% of side-turns emit experiences, so distill telemetry is a full-turns-only measurement. Cycle-34 note: a full-move N=128 at matched cost "roughly halves target class bias" — untried. `docs/az_minimal_spec.md` proposes off.
- Category: TRICK-UNTESTED. Coupling: 5.2/5.4 (target density), value labels (only recorded states get z).

### 6.9 Aux-head value bonus in search (`--mcts-aux-value-bonus`) and moves-left utility (`--mcts-moves-left-utility`)
- Where: `MCTSConfig.aux_value_bonus`, `moves_left_utility`; `mcts._aux_adjusted`, `_puct_select`.
- Mechanism: add bonus x predicted material margin to every leaf value (also shapes the Gumbel target); nudge PUCT toward shorter winning lines.
- Default: 0 / 0. Evidence: motivations only (zero village captures in 100-turn games, 2026-07-11; 42/66 tier-a eval games dying to the action cap); moves-left utility parked "indefinitely" by user; never run. Category: TRICK-UNTESTED. Coupling: needs the aux / moves-left heads (6.10).

### 6.10 Auxiliary material-margin head and moves-left head
- Where: `--mcts-aux-score` (+`--mcts-aux-coef` 0.15), `--mcts-moves-left` (+`-coef` 0.1); `model.py` heads (detached from the trunk since 2026-09-01), targets in `finalize_game` (next recorded state's margin; remaining turns / 200), losses in `_trainer_step_mcts`.
- Mechanism: extra regression heads on the global token; the aux target is one-step material change.
- Default: off at the CLI; `--mcts-aux-score` emitted unconditionally by the launcher on every tier-b leg (and the tier-a campaign); heads checkpoint-sticky.
- Evidence: profiler round 5 (first measurement): aux norm 0.32-0.74, 4-6% of the update direction, modest dv — "a real trunk regularizer, not a dominant channel"; leg-5 review ruled out aux-material axis capture as the rotation cause; moves-left ~0.03% of the gradient. Detached by ruling (telemetry-only; validated 100% of the aux gradient now in its own head). Never isolated for strength. D2 (technique review) found the imitation lineage carried never-measured head tensors.
- Category: TRICK-UNTESTED. Coupling: draw tiebreak weights define the margin; value_corpus aux targets use END-of-game margin (do not mix).

### 6.11 Search-side draw tiebreak (`--draw-tiebreak-cap` 0.3, `--draw-tiebreak-config`)
- Where: `tools/draw_tiebreak.py::DrawTiebreakConfig`, `configs/draw_tiebreak.json` (village fraction 2.0, unit value 0.0167, gold 0.0, score_scale 5); `mcts._terminal_value` at turn-cap leaves.
- Mechanism: a capped terminal inside the search scores cap x tanh(material differential) instead of 0.
- Default: on. Evidence: motivating chicken-and-egg (100% draws, zero value gradient) from the tier-a era; `weight_gold` 0 since 2026-07-20 (hoarding 2.8x vs SL baseline with gold priced). Leg-4 R3 notes the search scores cap terminals by material while the trainer censors those games — two rulings that disagree. Never isolated under the current regime (24/24 decisive games rarely reach the cap in-search).
- Category: TRICK-UNTESTED. Coupling: 5.10 (labels off), 6.10 (margin definition), censoring (6.12).

### 6.12 Winnerless-game value censoring (`--draw-value-weight` 0)
- Where: `MCTSPolicy.finalize_game` (`value_weight = draw_value_weight if winner == 0`); `TrainerConfig.draw_value_weight`; every value-side term multiplies `value_weight`.
- Mechanism: capped/stalled games keep their policy targets but carry no value label.
- Default: 0 since the 2026-08-17 truncation ruling (was 1.0; 0.25 on the tier-a campaign).
- Evidence: 2026-07-10 diagnosis (71% draws flattened the head even with honest z=0 and a rehearsal anchor); leg-3 R4: the draw flood at weight 1.0 preceded the K collapse; technique review: "decent case for 0 but never run as an arm". `docs/az_minimal_spec.md` keeps discarding. Never isolated.
- Category: TRICK-UNTESTED. Coupling: 1.5, 2.3, value memory skips censored states.

### 6.13 Value label smoothing (`--value-label-smoothing` 0.02)
- Where: `TrainerConfig.value_label_smoothing`; `_categorical_value_loss` (train only).
- Mechanism: mix 2% uniform mass into the projected C51 target.
- Default: 0 at the CLI; 0.02 on every box leg. Evidence: motivating 2026-07-07 entropy collapse (Z entropy 1.86 -> 1.13 over 46 replay iterations); HL-Gauss named as the shape-correct alternative; never isolated. Category: TRICK-UNTESTED. Coupling: 1.2.

### 6.14 Value-loss weight override (`--value-coef` 1.0) and arm W
- Where: `TrainerConfig.value_coef` (0.5); launcher passes 1.0; `docs/arm_w_spec_20260901.md` W1 (0.1) / W2 (beta 15).
- Mechanism: scale of the value term.
- Default: 0.5 library, 1.0 on every box leg. Evidence: profiler: value owns ~99% of the update direction at coef 1.0 including on the seed; W1/W2 never launched. Category: TRICK-UNTESTED. Coupling: 1.4.

### 6.15 Experience replay + multi-epoch updates (`--replay-buffer`, `-updates` 16, `-minibatch` 128, `-capacity` 24000, `-min-size` 512)
- Where: `tools/mcts_policy.py::ReplayConfig`, `MCTSPolicy.train_step`.
- Mechanism: bounded FIFO of experiences; 16 minibatch gradient steps per iteration sampled from it.
- Default: ON at the CLI since A4 (library off); every tier-b leg used updates 16 / capacity 24000.
- Evidence: motivating 2026-06-15 diagnosis (one-pass value head stuck at the ~uniform floor, val loss 3.56 vs ln 51 = 3.93; overfit probes needed ~80-100 steps); 2026-07-02 Kaggle memorization (train 3.8 -> 1.15, holdout flat 3.1). `docs/az_minimal_spec.md` claims 16 updates "multiplied every systematic tilt by 16" — a claim, not a measurement. No replay-vs-one-pass strength comparison exists.
- Category: TRICK-UNTESTED. Coupling: `fresh_value_ce` and the stall tripwire only exist on this path; value memory and the VG estimates run at the drain.

### 6.16 Per-game / per-side gradient normalization and the midgame weight floor
- Where: `MCTSExperience.game_weight = 1/(2 x side_weight_divisor(n_side, midgame))`, `MIDGAME_GW_FLOOR` 8; consumed by every term in `_trainer_step_mcts`; `policy_weight` (PT beta).
- Mechanism: every game contributes equal total weight, split equally between sides; human-continuation stubs floored at 1/8 game.
- Default: on (no flag). Evidence: motivations (190-turn draw vs 10-turn mini 19:1, 2026-07-12; winner/loser 54/46, 2026-08-05); T1-D gate (cycle 6): the per-(game,side) mechanism could not account for the boundary bias it was proposed to fix. `docs/az_minimal_spec.md` proposes removal. Never isolated. Category: TRICK-UNTESTED. Coupling: 1.4, 2.3.

### 6.17 Human-corpus value anchor (A2, `--human-anchor-file`)
- Where: `tools/build_human_anchor.py` cache; `run_iteration` value-only steps (`Trainer.step_value_from_raw`, 4 x 128 per iteration, trunk unfrozen); `tools/value_corpus.py`.
- Mechanism: rehearse clean +-1 human outcome labels each iteration.
- Default: off at the CLI; launcher default ON at the time (confirmed ON for legs 1 and 5, confirmed OFF for leg 4 with no ruling recorded; legs 2-3 not stated explicitly), OFF on every arm and by ruling since 2026-08-26.
- Evidence: motivating erosion 0.88 -> 0.60 late-game AUC in ~80 iterations (2026-07-10); leg 1 value AUC recovered 0.42 -> 0.76-0.79 under it; leg 4 without it read 0.912 on human play while leg 5 with it rotated to 0.338 — X4 (which of A2 / lambda flips the rotation) never run; leg-3 R5 names its full-unfreeze value steps as a candidate trunk-drift driver (untested). Never isolated.
- Category: TRICK-UNTESTED. Coupling: 1.6, 2.11, 4.1 (one owner for value training ruling).

### 6.18 Scenario mix (`--midgame-ratio`, `--fogless-ratio`, `--mini-ratio`, `--ladder-ratio`), `--mini-maps`, `--mini-random-tod`, `--forced-faction`
- Where: `tools/scenario_pool.py` (`roll_mix`, `validate_mix`, `FORCED_FACTION = "Knalgan Alliance"`, `MINI_MAP_SCENARIO_IDS`), `tools/midgame_starts.py` (value corpus), `WESNOTH_MINI_RANDOM_TOD`.
- Mechanism: which starting positions self-play experiences: human mid-game continuations, fog-off games, small maps; every game has a Knalgan side.
- Default: 100% fogged ladder at the CLI; the launcher's default 0.6/0.2/0.2 (ladder/midgame/fogless), mini 0, dates from 2026-07-28 and ran on legs 2-5 and all arms; Knalgan forced on.
- Evidence: tier-a: mini passivity drift (0/24 -> 9/24 draws) confined to the 3 fixed-ToD mini maps; midgame exports produced engine-illegal moves before a fix; the technique review prescribed 100% ladder for leg 1 (whether leg 1 overrode the launcher default is not recorded in the docs read), legs 2+ inherited 60/20/20 ("40% of training games off the eval distribution", leg4 R3); the Knalgan confound on the all-faction human probe is noted, not measured. `--mini-random-tod` never activated. None isolated.
- Category: TRICK-UNTESTED. Coupling: midgame games use the human value corpus and the weight floor; fogless games change the boundary_sum baseline.

### 6.19 Turn-cap jitter (`--max-turns-min` 60, `--max-turns` 100)
- Where: `sim_self_play._roll_max_turns` (training only).
- Mechanism: each game's cap is uniform in [60, 100] so end-of-game banking is unreliable.
- Default: on (code default since 2026-08-17). Evidence: motivating fixed-cap banking (2026-07-20); leg 3 accidentally ran [60, 200]; never isolated. Category: TRICK-UNTESTED. Coupling: censoring (capped games carry no value label), draw tiebreak.

### 6.20 TCS end_turn force-inclusion
- Where: `turn_search.gumbel_top_k_alternatives(et_idx)` (end_turn always among the evaluated alternatives).
- Mechanism: keep "stop here" representable at every coordinate.
- Default: on. Evidence: leg-3 R2: real target defect under the exp link (P(evaluated) = 1.0 vs ~0.3 for an equal-prior decoy), sign-conditional, reversed in leg 3's late phase, "amplifier at most"; under the linear link the exposure lottery is dead by construction (2026-08-17 ruling); leg-4 review: the ratchet null model reproduces the surplus curve with zero free parameters. Never toggled in a leg. Category: TRICK-UNTESTED. Coupling: 5.5.

### 6.21 Multi-turn projection (`--turn-project reval|all`, `--turn-project-halfturns` 1, `--turn-project-max-actions` 40; deprecated `--turn-reply`)
- Where: `turn_search.project_value`, `plan_turn` (`use_proj`, `proj_all`).
- Mechanism: grade a candidate turn by the value H closed-loop half-turns past the boundary (one sampled line; `reval` gates stage 2 only, `all` also drives selection and targets).
- Default: none (off); `reval` on arms VG/VG2/VG3.
- Evidence: FOR the mechanism: Q7 — depth-1 projection removes the +3.7-atom mover-frame pass bias (residual t = 1.1 vs depth 2); collapse probe — with project=all the seed's end_turn accepts fall 30 -> 3, committed length ~= spine. AGAINST as implemented: bake-off gate projection separation 1.0:1 (placebo == real; 82% of proposals rejected) — "REJECTED as implemented", variance reduction needed (k averaged rollouts ~= k x cost); `all` placement is on the leg-4 "DEAD, do not relitigate" list. In vivo it only ever ran together with value grounding (VG*, all collapsed or eroded); never with grounding off. Cost ~2x game generation.
- Category: TRICK-UNTESTED (in isolation; offline results conflict by placement and metric).
- Coupling: 4.5, 5.4, grounding captures ride the stage-1/stage-2 sims, `rollout_max_actions`.

### 6.22 TCS budget knobs (`--turn-alt` 4, `--turn-rounds` 3, `--turn-fast-rounds` 1, `--turn-max-spine` 40, `--turn-reval-salts` 3, `--turn-min-delta` 0.01, `--turn-full-prob` 0.25)
- Where: `TurnSearchConfig`.
- Mechanism: alternatives per coordinate, hill-climb rounds, spine cap, gate salts, accept floor, recorded-turn fraction.
- Default: as listed. Evidence: `min_delta 0` arm is on the leg-4 dead list; the rest never varied in a leg (probe ran defaults). Category: TRICK-UNTESTED. Coupling: 5.4.

### 6.23 Plan tournament (`--plan-tournament` and every `--pt-*` knob)
- Where: `tools/plan_tournament.py::PlanTournamentPolicy`, `run_tournament`, `certify`, `accept_rule` (`_T_CRIT`), `TournamentConfig` (challengers 6, depths (1,3), redraws 1, cert depth 3 x 3, budget 900 forwards, band 0.08, beta_max 0.25, margin_ref 0.32); certified one-hot targets at `policy_weight = beta`, per-side certified-mass renormalization in `finalize_game`.
- Mechanism: the policy's own sampled turn is the incumbent; challengers are ranked by paired projection margins under sequential halving; a challenger plays only if replicated certification clears an n-aware t-test and a 2-atom band; otherwise the policy plays its own turn and the state trains value-only.
- Default: off. Evidence: built and adversarially reviewed (39 rounds, ~270 defects fixed, 2026-08-26..27); the pre-registered step-1 equal-compute match vs Gumbel MCTS-32, the branch-audit instrument, and any leg never ran (no `eval_games/` artifact, no live `pt_*` row). The 2x2 matrix that motivated it (seed+TCS -200) and the seed+MCTS +321 result (2026-08-28) redirected the program to the teacher arms.
- Category: TRICK-UNTESTED. Coupling: takes precedence over `--turn-search`; reuses `record_spine`, projection; PT telemetry pool-only.

### 6.24 Rollout-only value grounding (VG4b) and the committed estimator fixes
- Where: `--value-ground` with `--value-ground-consist 0`; `GroundingConfig.rollouts_per_state` 2 (9d04c2f); sigma2 = point + 1.28 SE (ad5c87f); recalibrated seed required (`seed_imit_tierb_start_vg3cal.pt` carries the point sigma2).
- Mechanism: rollout-truth labels on gate-frame consulted states with the trust region and no bootstrap term.
- Default: off. Evidence: "the variant that has never run, and the one the profile points at" (VG3 verdict). Category: TRICK-UNTESTED. Coupling: 4.2, 4.3, 5.8, 6.21.

### 6.25 Prior hardcoded bias machinery (combat oracle alphas, anneal, mini end_turn bias)
- Where: `wesnoth_ai/constants.py::COMBAT_TARGET_ALPHA/COMBAT_TYPE_ALPHA` (0.0), `COMBAT_ANNEAL_HORIZON` 1M, `action_sampler.combat_alphas_at` (threaded through `decision_step`, `--reset-decision-step`), `WESNOTH_PRIOR_BIAS_END_TURN_MINI`, `wesnoth_ai/combat_oracle.py`.
- Mechanism: additive hand-designed nudges on target/type logits before masking, annealed by decision count.
- Default: all 0 (ruling 2026-08-06: every hand-placed prior nudge defaults OFF); the anneal is live code multiplying zero ("dead code in this era", collapse probe).
- Evidence: alphas were 0.1 until 2026-07-16 on the tier-a lineage; cycle 29 lists them as "dead machinery" for the gold question; no isolated measurement. Mini end_turn bias never activated. `--reset-decision-step` therefore has no effect today.
- Category: TRICK-UNTESTED (inert). Coupling: decision_step plumbing is also what re-plans/bounces roll back.

### 6.26 Scalar squared-error value loss (`TrainerConfig.value_loss_form = "mse_mean"`)
- Where: `wesnoth_ai/trainer.py::TrainerConfig.value_loss_form` ("c51" | "mse_mean"), `_trainer_step_mcts` (uncommitted working-tree change present at inventory time, added for `tools/az_loop.py`; `tests/test_value_loss_form.py`).
- Mechanism: train the C51 head's MEAN prediction by squared error against the result instead of categorical CE on the projected label; the head architecture is unchanged.
- Default: "c51". Evidence: motivation only — profiler rounds 4-6 (value term ~99% of the update direction; categorical loss charges a confident head heavily for an aleatoric label). Never run. Category: TRICK-UNTESTED. Coupling: 1.2, 6.13 (label smoothing has no effect under mse_mean), `fresh_ce_floor` telemetry stays categorical.

### 6.27 Relevant-set hex encoding (`--relevant-set-hexes`)
- Where: `wesnoth_ai/encoder.py` (`relevant_set_hexes`), `wesnoth_ai/visibility.py`; basis guards (3.9).
- Mechanism: encode only reach + villages + castles + visible units (~0.30 of the board) instead of the whole board; changes the action index basis.
- Default: off (inherited from checkpoint). Evidence: forward speedup 4.3-4.8x, <= ~3.4x end-to-end; warm-start value MAE 0.217 then 0.351 ("NOT-A-WARM-START"); the recovery leg (F4, "finish it") never ran after the handoff. Category: TRICK-UNTESTED (throughput lever). Coupling: checkpoints, buffers and holdouts are not interchangeable across the flag.

---

## 7. Counts

| Category | Items |
|---|---|
| CORE | 9 (1.1-1.9) |
| INSTRUMENT | 13 (2.1-2.13) |
| OPS-SAFETY | 9 (3.1-3.9) |
| TRICK-VALIDATED (proxy only) | 6 (4.1-4.6) |
| TRICK-REFUTED | 14 (5.1-5.14; two of them already deleted from the tree) |
| TRICK-UNTESTED | 27 (6.1-6.27) |
| Total | 78 |

Also removed from the tree before this inventory, listed for
provenance only: cliffness search consumers (`cliffness_bootstrap_alpha`,
`adaptive_sim_budget`, X2, never scheduled), adaptive outcome
bucketing (X5, never measured), drill scenarios and `--drill-ratio`
(X4, "declared unusable, cost money twice"), scripted openers
`--opener-spec` (F6, unreachable on both production topologies),
`--replay-pool` (X6, parsed and ignored). Designed but never built:
reanalyze target regeneration, human co-training B, skirmish/fight-window
curriculum, opponent-sampling league, CGR-32, exact-chance control
variate, CRN keying (Q8: median 0 shared fight identities — dead),
luck ledger (Q3: rho^2 0.03-0.05 — dead), hindsight credit
(measurement idea only).

## 8. Confounds (mechanisms that only ever ran together)

No single-item verdict exists for anything in this block; every
tier-b Elo number is a verdict on the whole stack it ran in.

- Every tier-b leg and arm (ledger) carried, unvaried: replay buffer 16 updates / capacity 24000 / minibatch 128; `--value-coef 1.0`; `--value-label-smoothing 0.02`; `--mcts-aux-score` (trunk-attached through arm V, detached from VG on); C51 value head; per-side game weights; holdout 512; draw tiebreak cap 0.3 (search side); turn-cap jitter 60-100; playout cap 0.25 (Gumbel) or turn_full_prob 0.25 (TCS); `--mcts-sims 32`; forced Knalgan faction; actor-pool topology; 24 games/iteration. None of these has an on/off arm.
- Legs 2-5 and all arms: scenario mix 0.6/0.2/0.2 (ladder/midgame/fogless), the launcher default since 2026-07-28. Leg 1 was prescribed 100% ladder; its realized mix is not recorded in the docs read. Mix never varied within a teacher.
- Legs 2 through VG3: policy anchor (v1 for the F1 arm and leg 2, v2 after) ON; GBC ON except arm G. The "anchor holds CE" finding is therefore never separated from GBC, and vice versa, except by arm G (GBC) — the anchor was never switched off after leg 1.
- GBC label fix (2026-08-27): every erosion number before arm T (legs 1-5) was measured with broken event labels; fixed-vs-broken labels were never isolated.
- lambda 0.9 ran only with the exp link (leg 1, 3-game stage-1 probe) or with the linear link (leg 4); the linear link ran with lambda 1.0 only from leg 5 on, always with mover frame + K-tripwire + policy anchor v2.
- Mover frame (leg 5+) always with lambda 1.0, K-tripwire, policy anchor v2; projection (`reval`) only with value grounding (VG*), and value grounding only with projection `reval` and mover frame; the trust region only with the consistency term; VG3's gate-frame capture and measured lambda0 only together.
- Value anchor (A2) ON on legs 1, 2, 3, 5 and OFF on leg 4 and all arms; the only leg without it (4) also ran lambda 0.9 and value-coef 1.0 for the first time, so A2 vs the leg-5 trunk rotation (X4) is unresolved.
- Teacher (TCS vs Gumbel) was isolated once (arm T vs arm M), on top of the full stack above; the play-procedure comparison (2x2) was not compute-matched and the pin+TCS cells played the opponent frame.
- Value memory (V1/V2) ran only on TCS without projection; the collapse probe attributes the collapse to the missing projection, so the memory step's own effect is unresolved.
- Inference numerics: arms T/M/G are documented both as running compile+bf16 and as not (1.8); Rust kernels (`wesnoth_core`) from VG2 on; `wesnoth_src/data/core/macros` were untracked on every bare-clone box until 2026-08-30 (arm V's first ~2 iterations ran without them; arms T/M/G side-loaded them).
- Tier-a campaign (the only in-lineage positive, +133): Gumbel target + replay + C51 + draw tiebreak + aux head + draw_value_weight 0.25 + mini/midgame/fogless mix + spool workers + relevant-set OFF, on a 5M net from a self-play checkpoint, not the imitation seed; its external number (0-0-30 vs RCA) was taken with our side self-blinded under fog.
- Instruments dark during specific legs: distill and TCS telemetry not aggregated from the actor pool in leg 3; boundary pairs n=0 on the spool path (2026-07-29) and the pool path through leg 4; `tcs_*`, `search_*`, `gbc_loss`, `aux_loss` absent from the CSV in leg 4; the 1,200-pair value-AUC probe in leg 5; profiler rounds 1-3 on random-init nets; VG3's first in-loop signal rows.
