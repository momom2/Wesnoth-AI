# Credit assignment for the value-head restart (leg 4) — design review

*Produced 2026-08-17 by a 19-agent Opus 5 workflow (6 family
researchers -> 12 proposals -> adversarial cost/correctness review
(11 recommend/conditional, 1 rejected) -> synthesis), user-directed.
Binding constraints: the value head restarts from random weights
(human-corpus warm-start allowed); compute effectiveness first-class;
proposals priced against this project's asymmetries (fast forkable
sim, expensive forwards, resamplable combat RNG, side-turn macro
structure, draw-prone caps). Verbatim synthesis output below.*

---

# Credit assignment for the value-head-restart leg (leg 4)

**Synthesis of 12 proposals × adversarial review, 2026-08-17.** Reads against `docs/leg3_passivity_rootcause_20260817.md` (M0–M6 ladder), `docs/tcs_spec.md`, `docs/gbc_spec.md`.

---

## 0. The answer in one paragraph

Nothing in this review is a better *credit-assignment estimator* than what TCS already is. TCS already does the thing the literature would tell us to do: it treats the side-turn as the semi-MDP macro-step and computes **exact counterfactuals by rewind** rather than estimating them from a learned model (COMA's marginal, measured instead of learned — `docs/tcs_spec.md` §3). What broke in leg 3 was not the estimator, it was (i) **the grader** the estimator queries (turn-scale value movement at AUC 0.527 in `docs/gbc_spec.md:47-58`; 0.434 in vivo, below chance from leg entry) and (ii) **the label composition** feeding that grader (`z=0` at `draw_value_weight=1.0`, `wesnoth_ai/trainer.py:205`, over 40–65% of the value gradient). So the leg-4 credit-assignment plan is: **do not add an estimator; repair the grader before the leg and repair the label during it, and route every proposal that amplifies fidelity to the grader into stage (c) behind a measured value-AUC gate.** Concretely — seed the fresh head offline from the human corpus, fix the draw label, match the value head's training distribution to its query distribution (turn boundaries), keep GBC on as the value-free event supervision, and ship *zero* mechanisms whose teacher signal is a difference of two outputs of the head we just randomized.

---

## 1. The recommended plan

Prices are in **box-hours** on the $0.35/h 4090 (one leg-3 iteration ≈ 1 box-hour). "$0" means laptop or the already-rented probe box.

### (a) Ships BEFORE the restart leg launches

| # | Item | Price | Pre-registered success | Blocking? |
|---|---|---|---|---|
| **A1** | **Dashboard + entry gate.** Add `value_auc` to `_abort_check` (`scripts/holdout_probe_loop.py:140` currently trips on `ce` only, while `value_auc` is already in `_COLS:65-67`), stratified by turn bucket, plus ECE/Brier and frozen-state `p(end_turn)`. Split it into two instruments: an **entry-qualification gate** (a checkpoint must clear a bar to launch) and a **drift tripwire** (relative to that leg's own t0, the existing `PROBE_T0`/`PROBE_ABORT_DELTA` semantics). | $0, 23 s/probe cycle | Backtest over leg-3 stored checkpoints reproduces `value_auc` 0.309 at step 3,111,037 and *refuses the launch* | **YES** |
| **A2** | **M0 telemetry** exactly as the postmortem specifies. `drain_tcs_stats` is now wired (`tools/turn_policy.py:190`), so add: truncation- vs extension- vs same-length accepts, the `n=1` deterministic-shortcut rate, the two threshold components of `two_stage_accept` separately (`tools/turn_search.py:364-372`), completed-Q spread vs the 0.04 floor, `q_et − v_root`, and `actions_per_turn` split full/fast. | $0 | Every column non-NaN in iteration 0 | **YES** |
| **A3** | **Human-corpus value seed, frozen trunk.** `tools/value_pretrain.py --freeze-trunk` **already exists** (`:90-102`); do not write `tools/value_seed.py`. Rebuild the anchor with AlphaGo hygiene (`build_human_anchor.py` + a new `--max-states-per-game 8` across all 19,367 games → ~155k states, split **by game**), cache `global_ctx` once (`model.py:419` feeds the value head only the 384-d global token), then fit **both** a linear probe (384→51) and the full head (`model.py:303-306`) on cached features at zero further forwards. | **2–4 box-hours** ($1.5–3.5); reconstruction is the dominant term and wants a many-core box | Pooled outcome AUC ≥ **0.68 + 2 SE** (0.68 = the documented frozen-trunk plateau, recorded verbatim at `tools/value_finetune.py:5-8`); 526-state frozen self-play holdout AUC ≥ 0.60; trunk param hashes bit-identical; frozen-state `p_et` moves < 0.005. **Readout that matters: linear ≈ full ⇒ the trunk is the cap** (stage-2 unfreeze is a separate project, not approved here) | **YES** — this is the leg's grader |
| **A4** | **Counterfactual-probe confirmation of A3.** Re-grade the ~300 stored leg-3 coordinates through `tools/turn_counterfactual_probe.py` under {random head, leg-3 entry head, seeded head}. This is the estimand TCS actually consumes (a *counterfactual* ΔV between two forked boundaries), **not** the temporal ΔV that scored 0.527. | few CPU-hours, ~$1 | Revalidated-accept-vs-placebo separation (probe baseline 0.640/0.460 vs placebo 0.130/0.180) **widens** under the seeded head | **YES** |
| **A5** | **Zero-architecture label/target levers**, each one flag: (i) `--draw-value-weight` down from 1.0 — see (b) for the number; (ii) **force-inclusion off** at both call sites (`tools/turn_search.py:532` selection, `:622` target pass) — M6.2, and per the P1 review this one-liner supplies *100%* of the claimed benefit of a dedicated β head; (iii) `ended_by` split so `max_turns`, `no_progress` (`tools/wesnoth_sim.py:819`), `max_actions` (`:1300-1302`) and mutual elimination stop sharing one weight. | $0 code, measured in (b) | — | Recommended |
| **A6** | **Infrastructure prerequisites** (postmortem §4): raise/fix the actor-pool deadline (17/21 leg-3 iterations discarded 30–60% of games — the corpus was survivorship-filtered *on planning cost* every iteration, which alone makes leg 3 partly uninterpretable); escrow learner-side `trainer_history.csv` per leg. | ~0 | Games kept/iteration stable across iterations | **YES** |
| **A7** | **$0 gates for the stage-(c) candidates**, run now so (c) is decidable later: luck-ledger exactness audit + human-corpus ρ²; g-hat feature-resolution P0; leg-3 drawn-game margin IQR. See §5. | $0 | See §5 | Recommended |

**One correctness patch worth landing regardless (A5b′):** under `--turn-project all`, `materialize`'s boundary forward is consumed *only* by the `math.isnan(m.value)` check at `tools/turn_search.py:540`/`:628`, and `isnan(value)` is true iff `invalid` (`:328`) — so that forward is provably redundant. Change the guards to `if m.invalid:` and thread a `skip_value` flag. ~15 lines, zero risk, removes a redundant forward from a forward-bound pipeline.

### (b) The restart leg itself — credit assignment while the head is weak

The leg's flag set, and *why each one is safe under a fresh head*:

| Component | Setting | Value-head dependence | Rationale |
|---|---|---|---|
| **Turn granularity** | TCS **on** (default), `project=none` | The macro-step is structural, not learned | The side-turn is the real semi-MDP step (~12 actions, no enemy reply). Intra-turn credit is delegated to exact rewind; inter-turn credit to the terminal label. Keep it. |
| **Grader** | The **A3-seeded head**, on the C51 scale, no recalibration at read points | It *is* the head — but seeded, and gated by A3/A4 | If a 2-parameter temperature/bias affine is ever fitted, it goes into config and touches **reported metrics and absolute gates only**. Measured: temperature scaling does **not** preserve the ranking of `E[V]` over an ordered atom support (11.8% of logit pairs reorder in `E[Z]` at T:1→3), and `_rescale_q`'s scale-invariance holds only above the 0.04 spread floor (`tools/mcts.py:1244-1247`), while `min_delta=0.01` (`tools/turn_search.py:54`) is not scale-invariant at all. |
| **Event supervision** | **GBC on**, `--gbc-coef 0.1` (already default) | none | Events predict outcomes at AUC 0.794 on the same substrate where the head's turn-scale delta is chance. It is the leg's only dense, value-free trunk signal. |
| **Draw label** | `--draw-value-weight` **0.1–0.25**, with the sign pre-registered from the free arithmetic in §5-Q6 | none | The z=0 flood at weight 1.0 is the documented poison (`trainer.py:198-205` carries a prior incident: human-corpus late AUC 0.88 → 0.64 in 51 iterations). Note honestly that this **discards** gradient mass rather than using it, and does **not** address R4 (non-conversion). |
| **Boundary coverage** | **All-boundary value rows** (P2 stage A) — reverses TCS spec deviation 1 (`docs/tcs_spec.md:38-43`) | none | Boundary states are ~1/K of recorded rows; the head is queried *only* at boundaries. Cost is zero forwards, zero sim steps. **Conditional on the plumbing**: dedup against the existing coordinate-0 rows on full turns; fix the **policy** normalizer `total_gw` (`trainer.py:1061`, applied `:1192`) — the value side already has its own `total_value_w` at `:1112-1114`; exclude the new rows from `harvest_boundary_pairs` (`mcts_policy.py:989-1011`) or `boundary_sum` is corrupted; raise replay capacity to hold the ~5-iteration buffer horizon. Be honest that this is a **reweighting**, not new within-game target variation. |
| **`end_turn` target channel** | **Force-inclusion off** at both sites | none | Removes the R2 exposure ratchet, which under a *noise* grader has a measured non-zero attracting fixed point (p_et ≈ 0.19–0.25). |
| **Turn-length lever** | `prior_bias_end_turn` (`action_sampler.py:685-703`) promoted from env-var/mini-only to `configs/`, **symmetric** with the trainer re-forward at `trainer.py:759-763` | none | This is the config-first turn-length knob. It requires a user ruling on whether `tcs_spec.md §5.3` ("no aggression priors, ever") covers it — but the user has already personally sanctioned this exact mechanism (2026-08-06). |
| **Label variance** | **Optional**: luck ledger (`z' = z − α·L`), *only* if the A7 gate passes | none — genuinely, this is the one estimator in the review whose "value_head_dependence: none" survived audit | Exact martingale control variate from `enumerate_attack_outcomes`. **If it ships, the C51 support must be widened first** (`VALUE_V_MIN/MAX = ±1.0`, `VALUE_N_ATOMS = 51`, `model.py:57-59`) with atom count raised to hold width at 0.04 — and **the restart is the only moment that is free**. Draw/decisive gate must be computed on **raw** `z` (`trainer.py:1113-1115`, `:1206-1210`), or every decisive state silently reclassifies as a draw. |

**Explicitly NOT in the leg** (each is an amplifier of, or a teacher trained by, the freshly-randomized head): multi-turn projection at `all` placement, a dedicated β termination head, λ-return bootstrapping, CRN event-key synchronization, online soft-value targets, adjudicated cap labels.

**Leg price:** unchanged from leg 3 modulo A6 — call it **20–30 box-hours** ($7–11) for a 20-iteration leg, plus whatever the deadline fix buys back in kept games.

**Leg-level pre-registered success** (all read on frozen state sets, per M1's discriminator):
1. `value_auc` on the human holdout stays **above 0.60** for 3 consecutive probes at any point after iteration 3 (leg-3 baseline: 0.434 mean, 9/10 below chance, 0.309 at entry).
2. Counterfactual-probe revalidated-accept vs placebo separation, re-measured at leg end, **≥** the entry value.
3. `actions_per_turn_mean` stays in **[8, 20]** — a *symmetric* tripwire, not a floor.
4. `fresh_value_ce` read **floor-relative** (`trainer.py:291/305/315`) does not degrade. If the label semantics change (luck ledger, boundary rows), `fresh_ce_floor` must be **re-derived and re-based before the leg**, or the project's default success metric silently changes definition mid-experiment.

### (c) Piloted during / after — everything gated on a measured baseline

| Item | Gate to unlock | Price | Metric |
|---|---|---|---|
| **Projection** (`--turn-project reval --turn-project-halfturns 1`), with the **net** playout | M5 says the sign flips **and** the seeded head clears a level-AUC bar. Second gate the original proposal omitted: **paired sd of the projected delta must not exceed 1.3× the unprojected sd** — `two_stage_accept`'s `max(1.155·sd, 0.01)` otherwise exceeds the probe's median accepted Δ of 0.070 and accepts collapse regardless of grading quality | 4–6 box-hours offline; +108 forwards/side-turn in production (games/hour halves) | median `q(end_turn) − mean(q_alt) < −0.04` under projection while ≥ −0.04 without |
| **Heuristic playout (HPP)** | Projection passes with the *net* playout first. Then price the real artifact: RCA's move phase (`get_villages`, `retreat`, `move_to_targets`, leader safety), the exposure term and `leader_threat` that `tools/neutral_ai.py` explicitly does **not** implement (`:152-160`, and `run_neutral_side_turn:204-227` is a pure adjacent-attack loop). Measured price is **17–93 ms/decision**, not 5 | multi-hundred-line Wesnoth-fidelity artifact + $1–2 pilot | Spearman ρ ≥ 0.60 vs net-projection grades; deterministic (no `rng`) |
| **Continuation labels for cap games (CRB)** | Needs a playout **and** a ground-truth validity test the original pilot lacked: truncate *decisive* games before the leader kill, run continuations, score AUC + calibration against the hidden true outcome. Cap the label at \|z\| ≤ 0.3 (matching `draw_tiebreak`'s derived cap) unless calibration justifies more | $1 stage-0 (raw policy as continuation player), ~$2–4 later | AUC ≥ 0.75, no sign bias toward the materially-ahead side; must beat `--draw-value-weight 0.1` **and** `--train-draw-tiebreak` as declared arms |
| **g-hat as a blend grader (TRD-G)** | Only if A3 fails, i.e. the seeded head does *not* clear the turn-scale bar. Then it must pass P0 (≥50% of candidate substitutions move a feature) and fix `v_root` (`turn_search.py:228` uses `_value_for`, **not** `boundary_value` — the "dispatch in one place and every caller inherits it" claim is false, and leaving it re-arms R2 with a *systematic* rather than coin-flip sign) | $0 fit, 3.7–5× generation speedup if it works | Turn-scale AUC ≥ 0.60 **and** g-hat fit including draws at 0.5, validated on leg-3 drawn states |
| **Time-awareness** (Pardo's *other* prescription) | Cheap and never tried: add remaining-turns-to-cap as a 7th global (`encoder.py:199 GLOBAL_FEAT_DIM = 6`, `:1248-1255`). Requires a `global_proj` graft (copy 6 columns, zero-init the 7th) and a sentinel for uncapped human games | ~1 box-hour arm | Control arm for the whole "cap is unobservable" family |
| **CRN event-keying** | Only after the head earns AUC; its product is *fidelity* to the grader. Kill first with the 30-min instrumentation in §5-Q8 | $0 kill test | median count of truly-shared downstream RNG events ≥ 1 |
| **Per-game live RNG salt** | Independent hygiene item; ship **with** a blocked eval estimator | $0 | — |

---

## 2. Ranked proposal table

| Rank | Proposal | Verdict | Mechanism (one line) | V-head dep. (honest) | Cost | Single biggest risk |
|---|---|---|---|---|---|---|
| 1 | **Seed-and-Watch** (dashboard + frozen-trunk human seed) | **Adopt, conditioned** | Manufacture a non-random grader offline; measure the statistic TCS consumes, every iteration | none for the monitor; **moderate on the trunk representation** for the seed (already measured to cap at ~0.68) | $0 + 2–4 bh | The trunk, not the head, is the cap — which the linear-probe arm measures for free |
| 2 | **Censored-not-drawn** (draw/cap label repair) | **Adopt the cheap half** | Stop treating a turn cap as a point mass at 0; split by `ended_by`; set-NLL as the softening | none | ~$0 | Adjudication Goodharts into camping; the pilot as written is unmeasurable on the human corpus (median game = 12–14 turns; **zero** games reach turn 60) |
| 3 | **P2 stage A** (all-boundary rows + `ended_by`) | **Adopt, conditioned** | Match the value head's training distribution to its query distribution | none | ~0 | It adds *states*, not within-game target *variation*; boundary coverage already tripled during the collapse (1/K: 8%→28%), which argues against causation |
| 4 | **Luck Ledger** (AIVAT/MIVAT control variate) | **Adopt if the free gate passes** | Subtract the exactly-zero-mean combat residual from `z` | **none — verified** | ~0.014% of a side-turn | Effect size unmeasured; and `material_margin` is **cost-only**, so the residual is a *kill* residual that discards all HP-attrition luck |
| 5 | **HPP** (forward-free rollout in `project_value`) | **Conditional, stage (c)** | Interpose exact dynamics between commitment and evaluation, forward-free | tolerant claimed; **actually zero value at t=0** — it is a *delta* objective needing level discrimination at the projected state | `reval` survivable; `all` is 20–108 s CPU/side-turn — **drop it** | Converts an on-distribution value query into an off-distribution one — exactly backwards for a fresh head |
| 6 | **CRB** (continuation rollouts for cap games) | **Conditional, stage (c)** | Replace the cap draw with a measured continuation outcome | none | 0 forwards; but **on the actor's critical path** at the end of the longest games | Rewards coasting: raises the leading side's cap payoff from 0 to +shrink·\|z\|, i.e. reduces the only pressure to convert — R4's exact seed |
| 7 | **TRD-G** (frozen g-hat grader) | **Conditional, contingency only** | Move the return predictor off the transformer onto exact features; grade turns with it | none for the label; **`v_root` still comes from the value head** | 3.7–5× *cheaper* generation if it works | Feature blindness: pure repositioning gives Δg exactly 0, so material greed becomes structural rather than a calibration error |
| 8 | **HCA hindsight head** | **Conditional, stage (c)** | `log P(a\|s,win) − log P(a\|s,loss)` as an additive logit in the distill target | none for the credit; **couples via `_gumbel_sigma`'s fixed 5.5-logit span** — under a random head, `κ·c` becomes the *whole* target | $0 laptop pilot | Its likely whole effect is a scalar `end_turn` bias reproducible by `--et-target-bias` at $0 |
| 9 | **SOV** (soft-outcome value distillation) | **Offline arm only** | Train the value head on `2·ĝ−1` instead of `z` | "none" is **relocated, not removed** — total dependence on an unmeasured teacher | $0.2–1.75 | Draw-only default trains the head that reaching the cap is *good* — the forbidden tempo-blindness mode, on the population where the honest signal was removed |
| 10 | **CRN event-keying** | **Conditional, stage (c)** | Key search-fork seeds by event identity, not a global counter | claimed tolerant; **is actually amplification of the grader** | free (stage 1) | Lowers the accept threshold exactly where the grader is least trustworthy; and it is largely inert under projection, which samples through an unsalted `rng` (`turn_search.py:552-556`) |
| 11 | **P2 stage B** (λ-return bootstrapping) | **Blocked** | Bootstrap at turn boundaries; bootstrap the cap instead of absorbing | requires a good baseline | small | `boundary_sum` measured **+0.4…+0.65** in fogged play — every bootstrap hop is poisoned, not just the cap. Its own gate (\|boundary_sum\| < 0.1) is currently shut |
| 12 | **P1** (dedicated β termination head) | **Reject** | Factor `end_turn` out of the actor softmax into an option-termination function | tolerant claimed; **actively destructive under a weak head** | free | **Falsified by simulation**: the BCE label is independent of β, so its minimiser is the mean label = 0.5 exactly for symmetric Δ ⇒ noise fixed point p_et ≈ 0.5, K ≈ 2. Twice as bad as the channel it replaces |

---

## 3. Design insight

**The macro-step is already right; stop looking for a better estimator.** Every family in this review, when you strip the vocabulary, proposes one of three things: measure the counterfactual instead of learning it (rollout family), redistribute the terminal number onto a denser predictor (RUDDER family), or remove the exogenous noise from the label (AIVAT family). This project already does the first, at turn granularity, exactly, with the engine's own dynamics — that is a stronger position than the literature usually gets to occupy, and the reason is the two asymmetries the brief names: the sim is bit-exact and forkable, and the natural macro-step is a whole side-turn with no interleaved reply. What TCS *cannot* do is grade the resulting boundary. That single scalar is where all of leg 3's failure lives, and it is where the entire leg-4 budget should go. The uncomfortable corollary is that most of the sophistication on offer is **fidelity amplification**: CRN sync makes the accept gate more faithful to the grader; projection makes the grader query a state further from the commitment; a β head re-parameterizes a target the grader produces. Under a randomly-initialized grader, faithfulness is a *cost*, not a benefit, and the review found at least two proposals (P1, CRN) whose noise behavior is strictly worse than the incumbent's.

**"Measure vs learn" is the right axis, but the measurement has to bottom out somewhere.** The rollout family's pitch — interpose exact dynamics, then ask the learned evaluator only for what it measures well — is genuinely the strongest structural idea here, and it is the one the postmortem itself names (M5/M6.4). But every version of it terminates in a learned value at the projected boundary, so it converts a *level* query into a *delta of two levels one half-turn further apart*, and that helps only if the head has real level discrimination at the projected state. Two of its variants make this worse rather than better: HPP's heuristic playout walks the projected state off any distribution the net has ever seen, and CRB's continuation needs a *converting* policy to produce ground truth — which under TCS means it needs the value head. The one branch that truly bottoms out in measurement rather than in a learned quantity is the luck ledger, because `enumerate_attack_outcomes` supplies the *true* conditional expectation, not an estimate. That is why it is the only "value_head_dependence: none" in the review that survived audit, and why it is worth a $0 gate even though its effect size is unknown.

**Turn granularity and label granularity are different questions, and the project has been conflating them.** TCS assigns credit at turn granularity. The value head is trained at *coordinate* granularity on a *game*-granularity label. So every state in a game shares one bit, boundary states get ~1/K of the training mass while receiving 100% of the queries, and within-game target variation is exactly zero. Adding boundary rows fixes the coverage mismatch but not the variation problem — and this matters, because a turn-scale *ranking* metric is precisely a within-game statistic. The two mechanisms in this review that actually create within-game target variation are the luck ledger (a forward-sum martingale, so siblings differ) and λ-returns (blocked on `boundary_sum`). Everything else moves mass around. This is the sharpest under-appreciated structural point in the whole review: **`n_eff` for the value head is games, not states, and no amount of coverage engineering changes that.** AlphaGo's own remedy — one position per game across 30M games — is why the A3 anchor rebuild must cap states-per-game and split by game, and why the current whole-game batch assembly in `tools/value_finetune.py:176-203` is a real (if overstated) hazard.

**The draw label is the seed and it is not primarily a variance problem.** R4 says leg 3 stopped converting won fights inside the cap; the z=0 flood then degraded the only grader TCS has. Four proposals attack this and they differ in *how much they assume*: down-weighting (assumes nothing, discards the signal), `ended_by` splitting (assumes only that a cap and a stalemate are different events), set-NLL (assumes the true value lies in a set — Tobit, honest), adjudication/continuation (assumes we can say who was winning). The assumption ladder maps exactly onto the risk ladder, and the top of it re-creates R4: *any* mechanism that pays the materially-ahead side something positive at the cap reduces the marginal incentive to convert from 1.0 to (1 − shrink). `draw_tiebreak.py`'s cap of 0.3 was derived precisely so a real win always dominates; two proposals silently exceeded it. Take the bottom two rungs now, put the top two behind a ground-truth calibration test, and note that Pardo's *other* prescription — time-awareness in the observation — was never tried, costs a 7th global feature, and changes the objective not at all.

**The scarce resource may not be what the brief says, and this is the largest un-costed lever in the review.** Two reviewers independently measured a single-thread forward at **~1.0 s** on real ladder maps (1,050–1,190 hex tokens, 8 layers, O(n²) attention at d=384), against the brief's 10–16 ms; leg-3's own log serves 194,065 forwards in 3,600 s across 57 pool processes = 1.06 process-seconds per forward, while `infer` accumulates 3,912 thread-seconds ≈ 194,065 × 20 ms. Those reconcile if the inference server is the serial ceiling at ~54 forwards/s and the 57 actors are ~97% blocked. If that reading is right, then: (i) the "sim is free, forwards are dear" asymmetry holds *between* subsystems but **inverts inside TCS**, where the inner loop is O(K) sim steps per forward; (ii) TRD-G's 3.7–5× is real but is a *cost* argument, not a credit-assignment one; and (iii) the single highest-leverage change in this entire document is not algorithmic — it is that `materialize`'s 48 independent candidate boundary evaluations are issued **one at a time, blocking** (`tools/turn_search.py:537` → `:341`), when `model.forward_batch` exists and the pool already batches across actors (`tools/actor_pool.py:561`). Vectorizing that loop is a round-trip/latency win of unknown but plausibly large size, carries **zero** objective risk, and must be measured before any cost argument for any proposal here is quoted. `tools/box_bench.py` already has the harness.

**Finally, the meta-lesson leg 3 actually taught.** `value_auc` sat below chance from leg entry, in a column that was already being written, for 21 iterations, and nobody looked. Three separate proposals in this review pre-registered success metrics that could not distinguish their own hypothesis (an aggregate credit sum dominated by the state term a softmax discards; a "reliability" battery with no accuracy test; a clip-fraction gate that is arithmetically unreachable). The discipline that generalizes: **pre-register on the estimand the production code consumes, not on a correlated cousin of it** — counterfactual ΔV between two forked boundaries, not temporal ΔV along a played trajectory; within-coordinate centered credit, not a turn-summed one; calibration co-primary with AUC whenever the change can rescale the value axis.

---

## 4. Rejected — do not relitigate

- **P1, dedicated β termination head** — BCE on σ(ΔV/τ) has noise fixed point **exactly 0.5** (K≈2), twice as bad as the channel it replaces; the "accepted-sign" variant flips between K≈11 and K≈2 depending on whether the `n=1` deterministic shortcut fires (`turn_search.py:556-559`). Its claimed benefits are both purchasable by a two-token edit (force-inclusion off) plus promoting `prior_bias_end_turn` to config.
- **`--turn-project all` placement** — 20–108 s CPU/side-turn with a heuristic playout; and under `all` the projected values *become* the distill targets (`turn_search.py:604-607`, `:630`), so an RCA aggression score would shape the training signal directly, violating the `tcs_spec.md §5.3` ruling through the target channel.
- **CRN sync as a leg-4 arm** — its product is fidelity to the grader; deploying it before the grader is measured is backwards. Also largely inert under projection.
- **P2 stage B / λ-returns** — gated shut by its own criterion: `boundary_sum` = +0.4…+0.65 poisons every hop, not just the cap.
- **SOV's online draw-only override** — trains the head that reaching the cap is good, on the population where the honest signal was removed. The offline human-corpus arm survives.
- **Temperature/affine recalibration at TCS read points** — does not preserve `E[Z]` ranking (11.8% pair reordering) and interacts with the 0.04 spread floor and `min_delta=0.01`. Reported-metrics-only.
- **`--turn-min-delta 0.0` arm** — already refuted in the postmortem (refutation 9).
- **Raising `--mcts-sims`** — refuted 2026-07-31; the Gumbel target concentrates rather than converges.
- **`leg2_histories/` as a no-collapse baseline** — it is May-2026 REINFORCE data (postmortem refutation 10). No leg-2 baseline exists.

---

## 5. Open questions → pilot mapping

| # | Question | Pilot | Price | Decision rule |
|---|---|---|---|---|
| **Q1** | Is the trunk or the head the cap on value discrimination? | **A3** linear probe (384→51) vs full head on the same cached `global_ctx` | included in A3 | linear ≈ full within 0.03 ⇒ trunk is the cap; stage-2 unfreeze becomes its own re-scoped project (run M2 first) |
| **Q2** | Does a seeded head actually improve the estimand TCS consumes? | **A4** counterfactual probe, {random, leg-3, seeded} | ~$1 | placebo separation widens, else the grader must change (→ TRD-G contingency) |
| **Q3** | Is combat luck a material share of outcome variance? | **A7a**: replay ~500 human games with `enumerate_attack_outcomes` at every attack; regress z on standardized forward-luck L, game-level split. Also report an HP-weighted covariate (the shipped `material_margin` is **cost-only**, so it is a kill residual), the mid-game advancement bail rate, and the induced shrinkage E[ẑ]/E[z] | **$0, laptop** | out-of-sample ρ² < 0.05 on decisives for *both* covariates ⇒ luck is not what decides these games; ledger dies for free. ρ² ≥ 0.15 and shrinkage < 10% ⇒ ship in (b) with widened C51 support |
| **Q4** | Can a feature-based grader even *resolve* candidate turns? | **A7b / P0**: compute the ~40-dim feature vector at incumbent and every materialized candidate boundary on the 300 stored coordinates; report the fraction of substitutions that move *any* feature, split by substituted action type | **$0** | < 25% ⇒ reject the feature set outright; ≥ 50% ⇒ TRD-G stays a live contingency. Free byproduct: fraction clearing `min_delta=0.01` and the 0.04 `gumbel_rescale_floor` |
| **Q5** | Do leg-3 drawn games actually contain large mid-game asymmetries? | **A7c**: margin distribution at mid-game on stored drawn games | **$0** | IQR < 0.3 in win-probability ⇒ every "you were ahead here" mechanism (SOV, CRB, adjudication) is relabelling noise |
| **Q6** | Which way should the cap-draw weight move? | **A5-sign**: recompute from the 22 rows of leg-3 `trainer_history.csv` what the value-gradient composition would have been at w ∈ {0, 0.1, 0.25, 1.0}, checking `value_signal_states` against the starvation watch (`trainer.py:306-309`) | **$0, minutes** | Pre-registers the sign *before* code is written |
| **Q7** | Does projection flip the truncation sign, and can it be measured at n=3? | **M5** at `project=reval, halfturns=1` on the M3 coordinates, both checkpoints | 4–6 bh offline | Primary: median `q_et − mean(q_alt)` crosses −0.04 under projection but not without. **Secondary (the gate the original omitted): projected paired sd ≤ 1.3× unprojected**, else accepts collapse regardless |
| **Q8** | Is there any RNG left to couple? | **CRN kill test**: instrument `materialize` to log, per accept round, whether incumbent/candidate consume the same *number* of synced-RNG events, and the count of downstream events whose identity key matches in both | **$0, ~30 min** | median matched-downstream = 0 ⇒ CRN family dies without writing `_next_seed` code. (Prior: RNG-consuming coordinates are a stable 13.6–20.5% of all coordinates across all 22 leg-3 iterations) |
| **Q9** | What does a forward actually cost on the box, and is `materialize`'s candidate loop latency-bound? | `tools/box_bench.py` + a one-off timing of 48 serial `boundary_value` calls vs one `forward_batch(48)` | ~0.3 bh | If the batched form is ≥2× faster, vectorize `materialize` before quoting *any* cost argument from this document |
| **Q10** | Is a human-warm-started head still turn-*blind*? | The A3/A4 pair is exactly this control — level AUC 0.951 was measured on human data while turn-scale movement scored 0.527 on the same substrate. These are different statistics and the project has quoted them interchangeably | included | If A3 clears level but A4 fails, the grader must change and TRD-G moves from (c) to (b) |
| **Q11** | Is a hindsight credit head anything more than a scalar `end_turn` bias? | Laptop HCA pilot with the **free `--et-target-bias` control arm** run first | $0 + ~$1 | If the scalar reproduces the effect on K, the head is not doing the work |

---

## 6. Interaction with the M0–M6 ladder

The credit-assignment plan and the collapse forensics **share their instruments**, and that is a feature — every measurement below does double duty.

- **M0 ↔ A1/A2.** Same dashboard. The postmortem asks for `value_auc` on the abort dashboard because it was dark all leg; this plan additionally requires it as an **entry-qualification gate**, because leg-3's entry value (0.309 at step 3,111,037) means a naive tripwire fires at iteration 0 of every leg. Those are two instruments with two thresholds. `drain_tcs_stats` is now called (`turn_policy.py:190`), so the accept-composition columns are reachable.
- **M1 ↔ A3's acceptance test.** M1's frozen-state `p(end_turn)` (weights-drift vs state-drift) is *literally* one of A3's success criteria ("frozen-state `p_et` moves < 0.005 under a head-only fit"). Run M1 first as the postmortem says; A3 then reuses the same frozen sets and the same two forward passes, and the postmortem's infrastructure prerequisite (c) — run the M1 probe *every iteration* — becomes a standing column in A1's dashboard.
- **M2 ↔ stage-2 unfreeze.** M2 (policy-free value updates → Δ`p_et`, minutes of GPU) is the gate on ever unfreezing the trunk during value fine-tuning. This plan **does not approve** the unfreeze; if Q1 says the trunk is the cap, M2 is the mandatory precondition for re-scoping it.
- **M3 ↔ A4 and A5(ii).** Same ~300 stored coordinates, same probe harness. M3 settles R2's sign; A4 measures whether the seeded head grades those same coordinates better. Force-inclusion-off (M6.2) is a panel in M3 *and* a leg-4 flag here — run it as a panel first, ship it second.
- **M4** is untouched and stands: the exchangeable-null residual is the honest `end_turn`-specific effect size. Note that the P1 review independently reproduced the null's non-zero fixed point (0.247–0.254 in its construction vs the postmortem's 0.19), which strengthens M4's premise and, by the same simulation, kills P1.
- **M5 ↔ Q7.** Same measurement, with one added pre-registered gate (projected paired sd ≤ 1.3× unprojected) that the postmortem did not specify and that plausibly dominates the grading-quality question.
- **M6 ordering, amended.** The postmortem ranks: (1) R4 conversion probe, (2) force-inclusion off, (3) draw-value-weight, (4) projection. This plan keeps that order but inserts **A3 (value seed) ahead of all of them**, because M6.1's conversion probe and M6.2's null are both read through a grader; running them under a randomly-initialized head measures the grader, not the hypothesis. M6.3 (`--draw-value-weight 0.1`) moves from "a training arm" into the leg's baseline config, with its sign pre-registered by Q6. M6.4 stays last and stays gated on M5.
- **Infrastructure prerequisites (a)/(b)/(c)** from the postmortem are **hard blockers here too**, and one of them is now load-bearing for a credit-assignment reason: with 17/21 iterations discarding 30–60% of games at the deadline, and the discarded tail concentrated on the *longest* games, any mechanism that adds end-of-game work (CRB especially) selectively discards exactly the class it exists to label.

---

## 7. Honest uncertainty register

- **The forward price is unresolved** (Q9). The brief says 10–16 ms; two independent re-measurements say ~1.0 s single-thread on ladder maps, with leg-3 telemetry consistent with a serial ~20 ms/forward server at a 54 fwd/s ceiling and 57 actors ~97% blocked. Cost arguments in §2 that depend on the forward:sim ratio (TRD-G's 3.7–5×, HPP's "spend the idle CPU") should be re-quoted after Q9.
- **A3's bar is set against a single prior number** (~0.68 late-game AUC, `value_finetune.py:5-8`) that was measured on a *different* trunk with a *warm* head. It is the most relevant prior available and it is not strictly transferable; the linear-probe arm is what makes the result interpretable regardless.
- **Whether the leg-3 turn cap was actually jittered 60–100** is not confirmed from the captured log (`--max-turns` defaults to 200, `--max-turns-min` to None), and `mean_turns` reached 102.4 at iter 19. If the cap was unjittered, the "unobservable latent variable" framing weakens considerably and the Pardo case-2 reading changes. Confirm before quoting it.
- **The claim that TCS never grades against an unsalted rollout** (which the luck ledger's unbiasedness silently depends on) held under inspection — every accept/reject is made on salted materializations (`turn_search.py:520-521`, `:551-578`) — but it is undocumented. If the ledger ships, write it into `docs/tcs_spec.md` as an invariant.
- **Stage (b) removes label poison; it does not fix non-conversion.** R4's seed — material advantage that cannot be converted inside the cap — is untouched by everything in stage (b). The postmortem's M6.1 conversion probe remains the highest-information experiment in the project and this plan only re-orders it behind the grader repair, it does not replace it.
---

## Q3 ANSWERED (2026-08-17, tools/luck_probe.py, 500 human games)

**The luck ledger is DEAD by the pre-registered rule.** OOS rho^2:
L_hp 0.0305, L_cost 0.0499 -- both under the 0.05 kill bar (L_cost
exactly at the boundary; neither near the 0.15 ship bar). Max
achievable label-variance reduction 5.6%: not worth the C51-support
surgery. Luck is real (+0.24 in-sample correlation, ~5 sigma at
n=500) and small: human ladder games are ~95% decided by
non-dice factors. Caveats recorded: DP bail rate 16.9%
(advancement-possible fights -- important fights, so total luck is
somewhat underestimated; doubling it still fails the ship bar), and
this is the human regime (re-runnable on self-play games at zero
design cost if a future policy's fighting style differs). Side
finding: side 1 won 293/500 = 0.586 -- a first-move/map advantage
to remember when reading self-play win splits. Per-game data:
training/metrics/luck_probe.csv.

## Q5, Q6 CLOSED AS MOOT (user ruling 2026-08-17)

The truncation ruling (capped games are no-result: value weight 0,
excluded from eval fits) killed the entire pay-the-ahead-side
family (adjudication, continuation labels, soft outcomes) that Q5
gated, and superseded the draw-weight sign measurement Q6 fed.
Neither measurement gates anything anymore. Q5's instrument
(mid-game margin distribution on stored games) stays trivially
buildable if the family ever revives.

## Q8 ANSWERED (2026-08-17, tools/crn_kill_probe.py): CRN family DEAD

76 incumbent-vs-single-edit pairs from human midgame positions,
instrumented at the sim's single RNG-allocation point
(WesnothSim._next_seed): median downstream RNG events per incumbent
turn = 0 (mean 0.55); matched downstream fight identities median 0
under BOTH the strict key (ids+hexes+weapon) and the loose key (ids
only); only 28% of pairs share even one. A turn edit changes which
fights happen -- event-keyed seeds would have nothing to reuse.
Pre-registered rule (median strict >= 1 keeps it alive): DEAD.
Caveat: measured under the collapsed short-turn leg-3-end
checkpoint; re-runnable in one command under a healthy policy, but
the margin (0 vs the >= 1 bar) survives any plausible regime shift.

## A3/A4 ANSWERED (2026-08-19): the leg-4 judge exists, and the
## imitation trunk wins everything

A3 (frozen-trunk seeds, 16,824 games, cached features): holdout
outcome-AUC seeded-imit 0.824 / seeded-tcs2 0.798; linear probe
within 0.007/0.025 of the full head on both trunks => Q1: THE TRUNK
IS THE CAP. Both clear the 0.60 launch gate and the 0.68 plateau.

A4 (counterfactual probe, 200 states + 200 placebo per judge, GPU):
  judge                 accept  placebo  separation  medDelta  K_gen
  seeded imitation      0.620   0.090      6.9:1      0.079    12.6
  seeded leg-3 trunk    0.810   0.240      3.4:1      0.074     9.5
  unseeded leg-3 (base) 0.730   0.230      3.2:1      0.093     9.7
The A4 gate (separation widens vs the rung-1 baseline 4.9:1) PASSES
for the seeded-imitation judge and fails for both leg-3-trunk
judges -- their high raw accept rides a 2.6x higher noise
(placebo) acceptance. rho(delta,survival) ~ 0 everywhere. KL gate
False for the imit judge (same disputed magnitude proxy as rung-1;
precedent: not blocking). Caveat: each judge probes states its own
policy generates (K 12.6 vs 9.5 regimes), so cross-judge rows carry
a state-distribution confound; the within-judge real-vs-placebo
separation is the clean per-judge statistic, and it is not close.

RECOMMENDATION (decision = user's): restart leg 4 from the
imitation checkpoint with the A3-seeded head
(tier-b/a3/seed_imit_tierb_start.pt, escrowed). Every instrument
agrees: AUC, noise rejection, counterfactual separation, generation
turn structure, Elo history, trunk-feature parity.
