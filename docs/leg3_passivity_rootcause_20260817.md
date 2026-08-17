# Leg-3 passivity — root-cause report

*Produced 2026-08-17 by a 17-agent Opus 5 workflow (6 lens finders →
10 mechanisms → adversarial verification (8 refuted, 2 plausible) →
synthesis), user-directed, under the binding constraint that any
explanation must name a symmetry-breaking mechanism (self-play is
zero-sum; "the policy fights badly" stories are inadmissible).
Forensics inputs: leg-3 train.log, trainer_history.csv,
holdout_probe.csv, repo code. Verbatim synthesis output below.*

---

**Scope:** synthesis over 10 finder mechanisms × adversarial verification (8 refuted, 2 plausible), plus my own checks on `leg3_train.log`, `trainer_history.csv`, `holdout_probe.csv` and the code. Everything below that says "measured" I reproduced myself unless attributed.

---

## 0. First: the explanandum was mis-stated, and that matters

**Both sides did not stop fighting.** Per-*game* combat is essentially flat across the whole leg (iters ≤6 vs ≥15 means):

| | iters 0–6 | iters 15–20 |
|---|---|---|
| `eng_attacks_pg` | 101.8 | 93.4 |
| `eng_damage_pg` | 1386 | 1340 |
| `eng_kills_pg` | 22.4 | 18.7 |
| `eng_contact_rate` | **1.000** | **1.000** (every single iteration) |
| `action_attack_pct` | 10.6% | **13.0%** (rose) |
| `action_move_pct` | 77.1% | 57.7% |
| `mean_turns` | 39.7 | 84.4 |
| `actions_per_turn_mean` | 12.80 | 4.41 |
| `eng_unused_mp_s1/s2` | 0.361 / 0.368 | 0.723 / 0.738 |
| `mean_end_gold_s1` | 137 | 564 (peak 931) |

The same quantity of fighting is smeared over 2–3× the turns (attacks per *side-turn* 1.28 → 0.56). The class that collapsed is **moves**; attacks and recruits lost *share* to nothing — they gained it. So this is **turn truncation + game inflation → cap → draw**, not aggression aversion. Any mechanism whose payload is "the grader can't see that attacking is good" is aimed at a fact that did not occur; several of the refutations below turn on exactly that.

**And the explanandum reduces to one scalar.** `record_spine` samples each coordinate from the policy prior (`tools/turn_search.py:229-235`, `:251-257`) and **breaks on `end_turn`** (`:246-247`), so turn length is a geometric stopping time in the net's own `p(end_turn)`. Measured: `K · distill_et_prior = 1.035 ± 0.086` over 21 iterations, `corr(K, 1/et_prior) = 0.988`, and executed medians track the geometric median iteration by iteration. Two independent adversaries reproduced this. Corollaries, both load-bearing:

- `action_end_turn_pct ≡ 1/K` arithmetically (I get `K·et_pct = 0.992 ± 0.004`). It is **not** independent evidence and should never be quoted as a second signal.
- Executed turns are **not shorter** than the prior implies (product ≥ 1). **The hill-climb was not net-truncating.** A fix confined to the acceptance gate cannot restore K.

So the whole question is: **what raised `p(end_turn)` from 0.073 to 0.299?**

---

## 1. Ranked causes

### R1 — ROOT (structural, confirmed): `end_turn` is an absorbing action sampled from the prior
`turn_search.py:246-247` + `:251-257`. `end_turn` is the unique member of the action set that terminates the coordinate sequence: prior mass on it is a **hazard rate**, prior mass on an attack is a within-turn preference. A 4× inflation of `p(end_turn)` cuts K ~4×; a 4× inflation of one attack's prior changes K by ~0.

*Symmetry:* property of the action class and the sampling loop, identical for both sides; says nothing about who benefits from fighting. **Confidence: high** (code-true + the K = 1/p identity).

*Status:* this is the **transmission arm**, not the seed. It contains no inflation driver. Its value is that it collapses the search space of explanations to a single scalar and rules out the acceptance gate.

### R2 — AMPLIFIER (plausible, sign-conditional): `end_turn` force-inclusion exposure ratchet
`gumbel_top_k_alternatives` overwrites the lowest Gumbel pick with `end_turn` at **every** coordinate (`turn_search.py:358-360`), and it is the incumbent at the terminal coordinate — so **P(end_turn evaluated) = 1.0 at literally every recorded coordinate**, versus ~0.30 for an equal-prior decoy and ~0.037 for an average legal action. `tcs_target_distribution` gives evaluated actions their own boundary Q and shelters everything else at `v_mix` (`:396-399`), then `_gumbel_sigma` min-max-rescales and multiplies by `(50+5)×0.1 = 5.5` logits (`mcts.py:1245`, `:1286`, `:314-315`).

Independently reproduced by the adversary with a **decoy control** the finder lacked: flat value head → target ≡ prior exactly (so force-inclusion alone inflates nothing); value noise sd 0.08 → `Δ(end_turn) = +0.068` with force-inclusion vs `+0.002` for an equal-prior decoy, and `+0.009` with force-inclusion off. The **target-construction defect is real.**

*Symmetry:* action-class exposure, applied identically by both sides at their own coordinates. Valid.

*Why it is an amplifier and not the root — three hard limits:*
1. **Sign-conditional.** The sign flips at an `end_turn`-specific value offset of ≈ **−0.04 (one C51 atom)** and at a pre-state/boundary gap of ≈ +0.2. Truncating forfeits your remaining units' moves, so the physically expected offset is negative and sits inside the flip band. The premise `q_et − v_mix > 0` was **never measured** (`search_q_spread` NaN, `boundary_pairs_n` 0 in all 22 rows).
2. **Self-limiting.** `v_mix` is *prior-weighted*, and `end_turn` owns most of the prior mass inside the 5-action evaluated set, so it largely anchors its own shelter: `q_et − v_mix ≈ (q_et − v_root)/6`. A null model with `end_turn`'s Q drawn *exchangeably* reproduces the entire measured surplus curve with no free parameters, crossing zero at `p_et ≈ 0.19` against the measured **0.198**.
3. **It reverses while the collapse accelerates.** `distill_et_target − distill_et_prior` = **+0.022** (iters 0–6), **+0.038** (7–14), **−0.024** (15–20, negative in 5/6). During the final third, the distill target was pushing `end_turn` *down* by 3–6 points per coordinate while `p_et` rose 0.224 → 0.299. Lagged `corr(surplus_t, Δet_prior_{t+1}) = +0.055` — no per-iteration transmission.

**Confidence: medium.** The defect exists and should be fixed regardless; its promotion to *driver* is unproven, and the late phase is provably not it.

### R3 — PERMISSIVE CONDITION (high confidence on the fact): TCS's only ranking signal is at or near chance
TCS grades a whole candidate turn with one scalar forward pass (`turn_search.py:135-143`), with `project="none"` all leg (`:73`, launch line `reply=none`). Evidence that this scalar cannot rank turns:

- **`holdout_probe.csv value_auc`: 0.309, 0.401, 0.364, 0.450, 0.456, 0.655, 0.484, 0.461, 0.386, 0.371** — mean **0.434, 9/10 below chance, n_value = 1200 each — from decision_step 3,111,037, i.e. at leg entry.** This series sat unread all leg while every other anchor metric looked healthy.
- `docs/gbc_spec.md:47-58` (0d attribution, 694 rows): the value head's **turn-scale movement predicts outcome at AUC 0.527 ≈ chance**; only its level scores 0.796. TCS grades exactly the turn-scale delta.
- Frozen 526-state self-play holdout `holdout_value_loss` 0.465 → 1.024 (adversary argues this is recentering onto a genuinely drawish distribution, i.e. calibration not discrimination — direction unresolved, but it does not *help*).

*Symmetry:* one shared head, both sides, all states. Valid and unavoidable.

**This is the condition that makes R2's sign a coin flip and lets any structural bias in target construction walk the policy unopposed.** It is *not* itself directional: it explains why nothing corrected the drift, not why the drift pointed at `end_turn`. Confidence that the grader is near-noise: **high**. Confidence that this *is* the collapse's cause: low, by construction.

### R4 — OPEN, and it is what actually seeds the timeline: non-conversion → draw flood
The onset is **not** in the K series. `z_draw_frac` runs 0.00–0.10 for iters 0–6, then **0.417 at iter 7 and 0.508 at iter 8 while K is still 12.16 and 11.19** (inside the iters 0–6 range 11.0–14.6). K only exits its band at iter 9 (8.04). `train_value_loss` steps 0.42–0.56 → 0.58–0.78 at the same point. **Draws come first.**

And iters 7–8 are the *bloodiest* of the leg: kills/game 24.5 and 25.2, damage/game 1625 and 1894, `mean_turns` 44.9 → 63.3. So the initiating failure is **"cannot convert won fights into a leader kill inside the jittered 60–100 cap"**, which then floods the value target with `z = 0` at full weight (`mcts_policy.py:585`, `trainer.py:205` `draw_value_weight = 1.0`, `:1206-1214` where-clause is a no-op) — compounding R3.

*No lens investigated this.* Every finder took the K collapse as the primary event and inherited a timeline in which their mechanism arrives late. **This is the highest-value target for leg-4 forensics** and I would rank it above R2 as the thing to explain.

### R5 — OPEN: the late-phase overshoot has no measured driver
`p_et` runs from 0.224 to 0.299 in iters 15–20 while (a) the distill target sits *below* the prior (R2 point 3), (b) the policy anchor's own rehearsal mix implies 0.101 (a restoring force at 0.30), and (c) the exposure ratchet's null fixed point is 0.198. **Nothing measured is pushing `end_turn` up in the final third.** Two named candidates, both untested:

1. **State drift, not weights.** `distill_et_prior` is measured on the run's own drifting state distribution — increasingly late-game, MP-unspent, no-legal-improvement states where `p(end_turn)` is *legitimately* high. Corroborating: the human-holdout probe held CE 3.46–3.57 and actor@1 0.47–0.49 across the entire leg; a genuine 4× state-independent inflation of one actor slot should have cost ~0.2–0.3 nats there. Part of the "collapse" may be self-fulfilling distribution shift.
2. **Shared-trunk drift from policy-free value updates.** `step_value_from_raw` (`trainer.py:1309-1324`) runs **4 × 128 value-only updates per iteration with the trunk fully unfrozen and *no policy loss***, alongside 16 self-play updates whose value component is 55–65% `z = 0`. Nothing constrains the actor head during those steps. A trunk being reshaped to say "this is a draw" on late-game high-unspent-MP states can move the `end_turn` logit as a side effect. Zero-cost to test (see M2).

---

### Explicitly NOT a cause, but a real data-quality hazard: actor-pool censoring
**17 of 21 iterations blew the 3600 s wall-clock deadline and discarded all outstanding games** (`leg3_train.log` — iters 4/5/18 kept only **13/7/9** games; iter 5 spent 126k forwards to keep 7). TCS forward cost is linear in K (measured `forwards/side-turn` 113.6 at K = 12.3 → 30.5 at K = 3.6, `corr(K, fwd/side-turn) = 0.76`) and sim-step cost is quadratic, so the corpus is survivorship-filtered on planning cost every iteration.

**But the measured direction opposes the collapse:** over iters 0–8, `corr(games_kept, K) = −0.529` — the most heavily censored iterations (4, 5, 6) kept the games with the *highest* K of the leg (11.9, 14.6, 14.5). Decisive games end early and are cheap; capped games are expensive. So censoring favored the healthy regime. It is not the seed — but the training corpus composition was uncontrolled for the whole leg and must be fixed before leg 4 is interpretable.

**Also:** only ~25% of side-turns emit training experiences at all (`turn_policy.py:130-135` `recorded = bool(target)`; targets are `None` unless `full`, `turn_full_prob = 0.25`). Verified: iter 0 = 4,434 exps ≈ 0.25 × 1660 side-turns × 12.25; iter 19 = 3,205 ≈ 0.25 × 3482 × 3.62. All distill telemetry is a **full-turns-only** measurement while `actions_per_turn_mean` covers all turns — do not compare them as if they share a denominator.

---

## 2. The single narrative

**Iters 0–6 (healthy, K 11.0–14.6).** All the structural biases are already fully armed — force-inclusion, `lam = 1.0`, `project="none"`, anchor v2, `z = 0` draws, the 5.5-logit gain. Nothing moves: `et_prior = 0.0809 ± 0.0110`, trend slope +0.00002/iter (p = 0.99). Note this refutes every "constant structural bias" mechanism as the *seed* — they all predict a smooth effect from iteration 0. The one thing that is already broken at entry is the grader: `value_auc = 0.309` on the first probe (R3).

**Iters 7–8 (the seed, R4).** Combat peaks (kills 24.5/25.2, damage 1625/1894) but stops converting: `mean_turns` 44.9 → 63.3, draws 0.00–0.10 → 0.417 → 0.508 — **with K still at 12.2 and 11.2**. The `z = 0` flood begins at full weight; `train_value_loss` steps up; the value head starts recentering (frozen holdout loss begins its climb toward 1.23). The grader, already near-chance at turn scale, gets worse.

**Iters 9–15 (the slide).** With the grader unable to rank candidate turns, whatever structural bias exists in target construction walks the prior. R2 is the only measured directional bias in this window: mean surplus +0.038, peaks +0.081/+0.095 at iters 8–9 — exactly where K breaks 12.2 → 8.0 → 5.3. K = 1/p_et (R1) converts each point of prior into hyperbolic turn shortening. Shorter turns → more unspent MP (0.36 → 0.72) → more turns per game (63 → 86) → more caps → more draws (0.43–0.53) → more `z = 0` → grader worse. The loop closes. Honest caveat: the per-iteration lagged correlation for the R2 → prior step is null (+0.055), so R2 is the *best-supported* driver of this window, not a demonstrated one.

**Iters 16–20 (the floor, R5).** The target channel **reverses** (surplus −0.024, negative in 5/6 iterations) and the policy anchor pulls down from 0.101, yet `p_et` climbs 0.224 → 0.299 past the exposure ratchet's own fixed point (0.198). Draw share of the value gradient peaks at 0.647. Gold banks (137 → 564, peak 931), unused MP 0.72–0.82, games run to the 60–100 cap. **The late phase is driven by something not in the distill channel** — state drift and/or unconstrained trunk drift from policy-free value updates.

**Seeds:** R4 (non-conversion → draw flood). **Compounds:** R2 (directional bias, iters 8–14) × R1 (hyperbolic transmission) × R3 (no corrective force). **Unexplained:** the iters 15–20 overshoot (R5).

---

## 3. Refuted — do not re-litigate

1. **Forced-inclusion + stale shelter as *cause*** — the surplus it predicts is a deterministic function of `p_et` (null model crosses zero at 0.19 vs measured 0.198) and goes *negative* in iters 15–20 while `p_et` rises; the mechanism survives only as R2's amplifier.
2. **One-ply commitment horizon / unpriced payoff** — `boundary_value` plays no plies; it is a learned outcome predictor, and `turn_policy.py:32-37` puts boundary states in the value head's training distribution with full-outcome labels. No horizon truncation exists in the value sense.
3. **Fog-invisible truncation (exact Q ties)** — its own predicted correlation has the wrong sign (`corr(et_target/et_prior, iteration) = −0.33`, `corr with K = +0.44`); the precondition measurement was n = 1 game with a metric that returned negative hidden counts (scenery counted as visible enemies).
4. **Unpriced tempo + missing draw tiebreak** — every candidate turn is exactly one side-turn long and both boundaries share a turn number; `turn_search.py:139`'s `tiebreak=None` is reachable only under `sim.done`, i.e. ≤1 boundary per game (~0.5%). Also `encoder.py:1249` *does* feed `turn_number/TURN_NORM`, so "V has no clock input" is false.
5. **Value-resolution collapse silencing the accept gate** — `max(2·sd/√n, min_delta)` is **scale-invariant** under value contraction (mean and sd scale together); only the `min_delta = 0.01` branch is scale-sensitive and is refuted by the probe-era median accepted Δ of 0.070. The `n=1` deterministic-pair shortcut (`:557-559`) makes the gate *easier* as attacks per side-turn fall.
6. **Anchor v2 (game-normalized rehearsal)** — the arithmetic is right (end_turn share 0.0782 → 0.1007, K 11.78 → 8.93; I reproduced it independently) but v2 was live from iter 0 and `p_et` sat at 0.0809 ± 0.0110 for nine iterations, statistically *below* v2's mix and indistinguishable from v1's. The run then overshot to 3× the v2 target, where the anchor is a restoring force.
7. **Draw-sink lock-in via CE predictability** — the premise "decisive states carry an irreducible ln 2" is false in this leg (`fresh_decisive_ce` = 0.25–0.60 routinely); and `_rescale_q` (`mcts.py:1226-1245`) is scale- and offset-invariant, so uniform value compression leaves the target *exactly* unchanged unless spread drops under the 0.04 floor — and `distill_kl_prior` rose (0.416 → 0.529), so it did not.
8. **Anchor mode-split / "false health"** — requires the head to retain decisive calibration on human states; `value_auc` mean 0.434 says it does not. The *monitoring* half of this claim survives and is important (see M0).
9. **`min_delta` gating aggression** — refuted by the finder's own arithmetic; do not spend a `--turn-min-delta 0.0` arm.
10. **`leg2_histories/` as the no-collapse baseline** — **it is not leg-2 data.** All nine CSVs are timestamped 2026-05-13…05-21 with the 26/47-column pre-TCS REINFORCE schema (no `distill_*`, `mean_turns` pinned at 201, 100% draws). **No leg-2 baseline exists in any supplied artifact**, and leg 1 (7 iterations, K 14–17) sits entirely inside leg-3's own healthy window (iters 0–6, K 11.0–14.6) — it discriminates nothing.

---

## 4. Leg-4 discriminating measurements, cheapest first

**M0 — free, and non-negotiable before any leg launches.**
`drain_tcs_stats` was defined and never called on any path (`turn_policy.py:186-188`); `search_q_spread`/`search_overturn_frac` are NaN and `boundary_pairs_n` = 0 in all 22 rows. Now that accept/projection telemetry reaches the log, log per plan: accepts, **truncation-accepts vs extension-accepts vs same-length**, the `n=1` deterministic-shortcut rate, the two threshold components separately (`2·sd/√n` vs `min_delta`), completed-Q spread vs the 0.04 floor, and `q_et − v_root`. Also: `actions_per_turn` **split full vs fast turns** (they have different denominators today), and per-arm fog/fogless distill telemetry. Add `value_auc` to the abort dashboard — it was below chance from entry and nobody looked.

**M1 — minutes, CPU. Weights vs state drift. Run this first.**
Score mean `p(end_turn)` for the **leg-3 entry checkpoint** (`tier_b_tcs2.pt`, step 3,111,037) *and* the paused end checkpoint (~3,541,899) on the **same frozen state sets**: the 526-experience frozen holdout (bucketed by turn number <25 / 25–50 / >50), ~2,000 iter-0 self-play coordinates, and the 1,200 human-probe pairs.
→ *`p_et` rises 0.07 → ~0.30 on frozen iter-0 states*: genuine weight inflation; R1 transmission confirmed and the search space is the target/trunk channels.
→ *`p_et` stays ~0.07 on frozen states while live is 0.30*: **the collapse is largely state drift**, the causal arrow is not prior → K, and this reframes the entire investigation. (The flat human-holdout CE weakly favors this branch.)
→ Simultaneously resolves whether the inflation is state-conditional (self-play only) or global.

**M2 — minutes, GPU. Tests R5(ii), nobody has run it.**
From the entry checkpoint, apply **only** `step_value_from_raw` (`trainer.py:1309-1324`) — 36 updates × 128, the exact pre-break budget — with no policy gradient at all, and measure Δ`p(end_turn)` on the frozen holdout every 4 updates. Run the same with the self-play value loss on a draw-heavy batch.
→ *Policy-free value updates move `p_et` up materially*: the shared-trunk leak is a live driver and the fix is architectural (head-local value fine-tune, or a policy-KL trust region during value-only steps).
→ *No movement*: R5(ii) is dead, leaving state drift.

**M3 — a few CPU-hours. Settles R2's sign; this is the measurement that was dark all leg.**
Offline re-grade of ~300 stored leg-3 coordinates (turns ≥ 8, post-contact) with `tools/turn_counterfactual_probe.py`, under **both** checkpoints, dumping per coordinate: `q_et`, `v_root`, `weighted`, `v_mix`, `rank(q_et)` within the evaluated set, `end_turn`'s prior share of `Σp_ev`, spread `hi−lo` vs 0.04, and the signed `target(end_turn) − prior(end_turn)`. Panels: force-inclusion on/off (`end_turn_idx=None`), fog vs fogless.
Pre-registered decision rule: **the ratchet is live only if median `q_et − mean(q_alt) > −0.04` AND median spread > 0.04 AND median `(v_root − mean q) < +0.2`.**
→ *median `q_et` bias ≤ −0.04*: the head **does** price truncation as a loss, R2 is sign-reversed and cannot be the driver — go straight to R4/R5.
→ *spread < 0.04 at most coordinates*: sigma was damped, the 5.5-logit story never fired.
→ *paired entry-vs-end comparison*: if the truncation preference exists **already at entry** (when K was 12.3 and stable), it predates the collapse and cannot be its cause.

**M4 — 10 minutes, no checkpoint.** Overlay the exchangeable null curve (`tcs_target_distribution` verbatim, `end_turn`'s Q drawn exchangeably, sweep `p_et` over 0.066 → 0.299) on the 21 measured `(et_prior, surplus)` pairs. The **residual** above the null curve is the honest `end_turn`-specific effect size — the number any R2 argument must be made from. My expectation: the residual is small.

**M5 — projection, now that `--turn-project reval|all` shipped.** On the M3 coordinates, re-grade with `project=reval, halfturns=1` and check whether `q_trunc − q_full` changes sign. Do this **offline before any training arm** — and only if M3 shows a truncation preference at all; otherwise projection is treating a symptom. If it does flip the sign, projection is the cheapest structural fix (it is the only change that gives the grader information about the cost of leaving MP unspent).

**M6 — training arms, only after M1–M3, ranked by expected information:**
1. **R4 probe (highest value, cheapest to set up):** 3–5 iterations from the **iter-6** checkpoint with instrumentation on *why games fail to convert* — leader-kill attempt rate, leader HP trajectories, turn at which material advantage first exceeds 2:1 and whether it ever converts. The draw flood at iters 7–8 with peak combat is the least-understood event in the leg and seeds everything.
2. **Force-inclusion off** at both call sites (`turn_search.py:532` selection and `:622` target pass), seeded identically to a control. Pre-register: `actions_per_turn_mean` and frozen-state `p_et`. Note leg-3 already shows 7 straight iterations of target inflation failing to move the prior, so a null here is likely and informative.
3. **`--draw-value-weight 0.1`** (or `--train-draw-tiebreak`, which is already configured at `draw_tiebreak_cap=0.3` and inert in TCS). This does not address the seed but removes the `z = 0` gradient flood that degrades the grader; `trainer.py:198-205` carries a prior incident note on exactly this failure (human-corpus late AUC 0.88 → 0.64 in 51 iterations).
4. **`--turn-project reval --turn-project-halfturns 1`**, last, and only if M5 says the sign flips.

**Infrastructure prerequisites for leg 4 to be interpretable at all:** (a) fix or raise the actor-pool deadline so iterations stop discarding 30–60% of their games — the corpus composition was uncontrolled for the entire leg; (b) escrow the **learner-side** `trainer_history.csv` per leg (the leg-2 baseline is simply gone, and the folder that claimed to be it is May-2026 REINFORCE data); (c) run the M1 frozen-state `p_et` probe **every iteration** — it is two forward passes and it is the one number that distinguishes "the policy changed" from "the states changed", which was ambiguous for this entire investigation.
