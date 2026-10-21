# Current design (after round 11)

Supersedes the round-10 design. The user's rulings of 2026-10-21 (seed §8) withdrew its core: the near-tie coins are not a training mechanism, nothing beside the network decides play, and the player samples at temperature 1. What carries over is the outcome label with the dice's luck subtracted, the census of decisions, the prices, the claim machinery and the planted-signal discipline. The new core is on-policy policy gradient on the player's own sampled games, applied as one certified step per round.

Tags: **[M]** measured (record or script named), **[D]** derived (inputs named), **[U]** unknown (measured before anything relies on it, or nothing relies on it), **[C]** a choice whose consequences are derived. Scripts are in `analysis/`; *plan* is `round11_synthesis_plan.py` and its `.out`, § for its sections. Prices are at $0.577 an hour and at the argmax game length. The temperature-1 game length is [U] and scales every price; stage 1 reads it.

## 1. The update

Each round k:

1. **Generate.** π_k plays itself at temperature 1: no end_turn offset, 64 memory slots, both seats, the 21-map pool with fog and uniform factions. It plays N games to train on and |H| held-out games. π_0 is `own2`.
2. **Label.** Every decision of side s in game g carries R_{g,s} = σ_s(z_g − b(cell) − βᵀS_{g,s}).
   - σ_s orients the result to the side.
   - b is the shrunk map × factions × seat mean.
   - S holds the luck walk's seven exact attack-luck sums (`tools/match_luck.py`).
   - b and β are fitted on games the round does not train on and then frozen: round 1 uses stage 1's held-out set, later rounds the previous round's.
   - A game that reaches the turn cap scores z = 0.
   - Two per-decision corrections enter only if stage 1 measures them paying (§4): the frozen value head's reading, −α·v̂_t, and the opponent's surprisal, −κ·U_g, where U_g sums log p + H over the opponent's decisions.
3. **Gradient.** One pass of the sequence trainer over the N games accumulates ĝ = (1/2N) Σ_g Σ_s R_{g,s} Σ_{t∈s} ∇log π_θ(a_t|h_t) at θ_k.
   - log π is the masked joint log-probability the player sampled from (`legal_log_probs`, `wesnoth_ai/own_loss.py`). The signed weight is applied outside `own_policy_loss`, whose `prob > 0` filter would turn negative weights into winners-only imitation.
   - The gradient flows through the recipe's 16-decision memory windows.
   - No value, belief or imitation loss.
   - The pass also keeps eight split accumulators (by game index), each window's squared gradient, and the probe gradients (§3.4).
4. **Direction.** d = Pĝ, with P the metric stage C picks. The default is the reduced AdamW: ĝ divided by the root of the pass's mean squared window gradient, with no momentum, weight decay or clipping.
5. **Step.** Three candidates θ_k + s·d sit at mean per-decision KL ε/2, ε and 2ε from π_k, set on held-out positions with forwards only. Two estimators read each candidate on H, from the same forwards:
   - **Ĝ, the first-order gain:** the mean over game-sides of Σ_t (r_t − 1)R, where r_t = π′(a_t|h_t)/π_k(a_t|h_t).
   - **ΔĴ, the exact gain** of one seat switched to the candidate against π_k: the mean of (W − 1)R, with W = Π_{t∈s} r_t.

   The candidate with the highest ΔĴ on half A of H is read on half B. The round adopts it unless half B reads below −2 SE. In that case π_k stays and ε halves. The next round starts from the adopted radius.
6. **Claim.** Every third round, π_k at temperature 1 plays the reference (§3.2).

**Why one step per round.**
- **Same move.** Where a round's gradient is mostly noise, as the one measured noise scale suggests (§4), a minibatch pass with stable Adam statistics moves the weights by the same summed, normalized gradient to first order.
- **The bench is the update.** The round's change depends on its games alone, so stage C compares optimizers on the update itself, from one gradient.
- **Nothing to tune.** There is no learning rate, warm-up, schedule, momentum or bias correction: KL sets the length and held-out games certify it.
- **Each game is used once.** Reusing data (PPO's epochs) refits one batch's noise and repays training, 0.67 of a game's all-in cost [D: 5.162e-5/(2.598e-5 + 5.162e-5), *plan* §1].

### 1.1 Bias and noise

- **Unbiased.** E[ĝ] = ∇J(π_θ against π_k) at θ_k under three conditions:
  - actions are drawn from the logged law (I-sample and the replay identity check this, §5);
  - b, β, α and κ are fixed before the games they label;
  - each correction has zero mean given the history before its event.

  The dice's law does not depend on the policy, so the luck terms keep zero mean under fog, memory and any reweighting ([Burch et al. 2018](https://arxiv.org/abs/1612.06915)).
- **Departures, each checked.**
  - The memory's gradient is truncated at 16 decisions [M, `tools/sequence_setup.py`].
  - The generator's and trainer's arithmetic differ (the replay identity).
  - Refused recruits are sampled actions and must enter the sum (I-count).
- **Noise.** The label is nearly independent of each decision's score, so a game's gradient noise is about E[R²]·F_game, the policy's Fisher summed over the game [D, `round11_challenge.md` §2]. The actions are sampled from the model, so this is the true Fisher, and the empirical-Fisher objection ([Kunstner et al. 2019](https://arxiv.org/abs/1905.12558)) does not apply.
  - E[R²] at argmax is 0.699 [M, `round9_simplify_residual.out`]; at temperature 1 it is [U → stage 1].
  - z is a sum of zero-mean increments: setup, dice, and both sides' decisions. One side's total choice signal per side-game is therefore at most E[R²]/2 [D, `round11_provenance.md` §1.2].
- **The two readings.**
  - Ĝ is unbiased for the first-order gain under any baseline, since E[r − 1 | h] = 0.
  - ΔĴ reads the whole gain, including how the step changes the positions reached, at variance about E[R²](e^{2K} − 1)/2 per game, where K is the per-side-game KL. At K ≤ ½ its SE is at most 0.93 of a same-size match's [D, `round11_provenance_derived.out` §5].

### 1.2 The radius

The record shows why the step must be read on the true objective. Recruit rule R gained +18.0 Elo at first order and lost 12.3 ± 4.5 to its own second-order term [M, round 10].

In the quadratic step model, a round at k times the best radius keeps 2√k − k of the best gain [D, `round11_challenge_budget.out`].

| Rule, 12 stationary rounds from k₀ = 1/16 | Mean share of the best gain |
|---|---|
| Fixed step | 0.438 [D] |
| Ratio rule (`round11_challenge.md`), z_G 1–5 | 0.43–0.60 at F₁ 10 Elo; 0.39–0.83 over F₁ 5–40 Elo [D, `round11_challenge_critique_rule.out`] |
| This ladder, z* 0.5–3, reading correlation 0–0.9 | 0.62–0.90; rounds at k ≥ 4 at most 0.02 [D, *plan* §6] |

z* is the z of the best step's half-A reading. At the best step the first-order gain is twice the true gain, so z_G ≈ 2z* [D]. The ladder does better because ΔĴ's noise shrinks with the step, while a match's does not.
- **Range.** ΔĴ is used up to K = ½ per side-game, ε = 0.0021 per decision at 237.9 decisions [D, *plan* §3]. A 2ε candidate past that is read by a 10,000-game cross-play match against π_k ($0.88 [D]).
- **ε₀** comes from stage C's ladder.

## 2. The optimizer

Under this update the optimizer's only job is the metric P: KL sets the step and held-out games certify it.

| Candidate | Evidence | Status |
|---|---|---|
| Reduced AdamW (default) | It trained `own2`. In the noise-dominated regime its second moment estimates the noise's diagonal, a square-root diagonal natural gradient [D]. [Zhang et al. 2024](https://arxiv.org/abs/2402.16788): SGD falls behind Adam on transformers when Hessian blocks are heterogeneous (supervised) | Stage C arm |
| SGD | [Zhang et al. 2019](https://arxiv.org/abs/1907.04164), the noisy quadratic model: preconditioners raise the critical batch (abstract; that optimizers differ little far below it is from the body, not re-read) | Stage C arm |
| Muon (built, update RMS matched to AdamW's) | [Ruan et al. 2026](https://arxiv.org/abs/2607.16169): v1, single seed, 0.290 → 0.546 success; v4, a full-budget RMS-matched control "removes the late-success gain". [Fan et al. 2026](https://arxiv.org/abs/2605.19282): collapse in RLVR, attributed to low SNR and per-head specialization. [Liu et al. 2025](https://arxiv.org/abs/2502.16982): fine-tuning an AdamW-pretrained model with Muon shows no significant advantage, by the authors' reading (Table 6: 3 of 4 benchmarks ahead; Table 7: 0 of 4) | Stage C arm, as one orthogonalized step on the summed gradient |
| Heads-only, probe span | Fewer noisy dimensions at the cost of the signal's share | Stage C arms |
| SOAP | No RL evidence found | Built for this update iff reduced AdamW beats SGD by 2 paired SE |
| Damped natural gradient (conjugate gradient) | ACKTR: 2–3× sample efficiency ([Wu et al. 2017](https://arxiv.org/abs/1708.05144), abstract). Fisher-vector products through the memory are new code | Iff SOAP beats reduced AdamW by 2 paired SE |
| Schedule-free ([Defazio et al. 2024](https://arxiv.org/abs/2405.15682)) | No schedule exists here. Its gradients are taken at one point and play at an average, so on-policy generation must pick one (`round11_simplify.md` §2.1) | Rejected |

**The criterion** (stage C) runs on one gradient from 20,000 games. Each arm's direction is scaled to the same KL and read by ΔĴ on the same held-out games (half A), paired by game. An arm replaces the default only beyond 2 paired SE [C].

This judges each arm as one step on a summed gradient. That is how this update applies it, but it is not how Muon runs as a minibatch optimizer: orthogonalizing each noisy step is nonlinear.

**The built bench** (16 imitation passes from scratch, `scripts/optimizer_bench_box.sh`, $2.50) reads high-signal supervised training: speed, memory and stability. It does not read this update. It waits for a future imitation pass.

**The round size.** N* = √(c_f·B/c_g), the cost-optimal batch of [McCandlish et al. 2018](https://arxiv.org/abs/1812.06162) with a fixed per-round cost:
- c_f is the held-out set, its forwards and the claim share, $1.23–2.42;
- c_g is a generated and trained game, $7.76e-5;
- N* is 12,600–17,700 games at B = 10⁴ and 39,800–55,800 at B = 10⁵ [D, *plan* §5].

For a KL-normalized step, B = tr(PFPΣ)/(gᵀPFPg), not B_simple in the P metric (`round11_provenance_critique.md` §2.1). Stage C reads it from the KL of the eight single-accumulator directions against their sum. E[dᵀFd] is gᵀPFPg + tr(PFPΣ)/n for one accumulator and the same with n × 8 for the sum: two equations, two unknowns. The forwards run on a 5% subsample.

**The [LessWrong post](https://www.lesswrong.com/posts/9gqjdRyJxSM6kS7zi/i-trained-gpt-2-with-full-batch-gradient-descent-so-you-don)** has one run per setting, a learning rate tuned for minibatches and a Muon-family optimizer (ANVIL), by its author's caveats. Three of its findings bear on this update.
- **A reused set generalizes worse.** Validation was 3.521 after full-batch training against 3.248 for minibatches on the same 800M tokens. Hence each game is used once.
- **Exact-gradient steps at a fixed rate ride the edge of stability.** The consecutive-gradient cosine was −0.945, the loss rose on a third of steps, and the top Hessian eigenvalue went 48, 82, 113. Here the step is set by KL and certified, and the cosine between consecutive rounds' directions is logged. Without momentum it reads oscillation directly.
- **Fresh data at 50M tokens a step matched minibatches** (3.230 against 3.248) with 18.8× the tokens in 6.8× fewer steps [D, `round11_simplify_budget.out`]. Large batches buy steps, not data. The cost here is games, so N sits near N*.

## 3. Prior, yardstick, opponent, memory, probes

### 3.1 The human prior

The human prior is the starting point only. Each round moves at most 2ε of KL and is certified, and nothing anchors it [C].
- An anchor toward `own2` would tax the sharpening that sampling needs.
- The anchor evidence is from human-facing play ([Jacob et al. 2022](https://arxiv.org/abs/2112.07544); [Bakhtin et al. 2023](https://arxiv.org/abs/2210.05492)).
- The human holdout cross-entropy is reported each round as a crash barrier, never as a target.

The risk is cycling: mirror descent without an anchor need not converge in a two-player zero-sum game. Magnetic mirror descent converges to a regularized equilibrium ([Sokota et al. 2023](https://arxiv.org/abs/2206.05825)). The claims are the alarm, and a lower-tail miss stops the chain for the user's choice between a magnet toward `own2` and a pool of past iterates.

### 3.2 The yardstick

The reference is `own2` at `raw:t0+eo-1.5` with 64 slots [M, `configs/reference_player.json`, adopted 2026-10-21]. The seed's `own1` is superseded.

Claims put π_k at temperature 1 against it, 12,800 decisive games ($1.13 [D]). The SE is 3.1 Elo at a 0.5 win rate and 5.4 at 0.09 [D, *plan* §3]. D₀, `own2` at temperature 1 against the reference, is the debt the user expects self-play to repay. It contains the offset's worth as well as the sampling loss.

### 3.3 Opponent and memory

- **Opponent.** The current self on both seats. A pool game costs 3.41× and trains one seat [D].
- **Memory.** Trained by the same gradient on the histories it creates: the remedy the record validated (own-trajectory training cut the memory's cost from 105 Elo to none within `own2` [M, CLAUDE.md]). On-policy, past actions are the agent's own interventions ([Ortega et al. 2021](https://arxiv.org/abs/2110.10819)). Stage 1 reads 64 against 0 slots at temperature 1 as a report.
- **Auxiliary losses.** None. Under a KL-limited step an auxiliary gradient spends the step on policy changes not aimed at the result, and the record's value gradient owned 94–99% of the trunk's updates [M, seed §3]. One step cannot price an auxiliary loss's benefit to representation, so no bench arm is built for it.

### 3.4 Probes

**P1 (an identity).** Scalars on the end_turn, attack and recruit logits enter the trainer's forward. Their gradients must equal the logs' Σ R(1[c] − p(c)) to bf16 tolerance. This catches alignment, sign and label errors.

**The sharpening slope** comes from stage 1's logs alone: the mean over side-games of R·Σ_t (log p_t + H_t). log p + H is the score of the player's own joint inverse temperature, so this is the first-order value of sampling colder against a temperature-1 opponent: the user's hypothesis at first order. If it is resolved at 3 SE, the antisymmetric match `own2` at T = 1/(1 + e) against `own2` at T = 1/(1 − e) checks it in play. By symmetry f(e, −e) = 2e·∂f + O(e³) (`round11_simplify_critique.md` §2.3), so the forecast is exact to second order. It is a measurement match only.

**The head-temperature floor.** Scaling a head's final linear map (`actor_head`, `type_head`, `target_q_proj`, the weapon head's last layer; `wesnoth_ai/model.py` lines 210–223) scales its logits: a direction in the network's own weights. From the slopes s and the Fisher F_s along these four directions:
- gᵀF⁻¹g ≥ sᵀF_s⁻¹s by Cauchy–Schwarz;
- so √(2K·sᵀF_s⁻¹s) is a measured lower bound on the first-order gain of the best step at KL K [D].

The bound's noise lives in four dimensions, so it resolves where the full gradient may not. The probe span is also a stage C arm.

## 4. Measurements first

**Settled by records**

| Quantity | Value | Tag, source | Decision it feeds |
|---|---|---|---|
| Base and reference | `own2` +22.7 ± 4.4 Elo over `own1`, 6,400 decisive games; reference since 2026-10-21 | [M] `own2_base_20261020/reading.json`; `configs/reference_player.json` | π₀; the yardstick |
| Census at argmax | 475.7 ± 3.2 decisions a game, 237.9 a side-game, 20.16 side-turns; move 0.579, attack 0.237, recruit 0.099, end_turn 0.085 | [M] `round11_provenance_census.out` | Prices; K per decision |
| Caps at argmax | 2 of 6,402 (`own2`–`own1`); 0 of 1,600 self-play | [M] census; `round11_simplify_rows.out` | z = 0 for caps; the rate re-read at temperature 1 |
| Prices | $2.598e-5 a pool game, $8.848e-5 a two-network game; trainer 1,477 positions a second | [M] `round9_simplify_labels.out`; `trainer_calibration_20261015` | Every cost [D, *plan* §1–2] |
| Label noise at argmax | E[R²] 0.699 (0.692 on `own2`) | [M] `round9_simplify_residual.out` | SEs until stage 1 |
| Luck coefficients by setup | +0.0007 ± 0.0002 of variance | [M] `round11_invent_luck_strata.out` | No chance head |
| Forks | Turn-start R² 0.346 (0.298–0.395) bounds a fork at ≤ 1.53× on post-fork decisions; at K = 8, 1.19× (1.13–1.25) over a value-head baseline | [M] `round11_provenance_turn_signal.out`, [D] `round11_provenance_derived.out` §3; [M, transferred from `obs8`] `round11_challenge_forks.out` | Forks rejected |
| Value head as baseline | Explains 0.40 (0.29–0.51) of var(V); label 0.854 → 0.725 (`obs8`, human positions) | [M, transferred] `round11_challenge_forks.out` | Admitted only on stage 1's reading |
| One outcome-gradient noise scale | 43-column recruit rule on coin labels: B_simple 11,593 (7,813–22,457) games, r_eff 18.9 | [M] `round11_simplify_noise.out`, `round11_simplify_critique_rank.out` | A report; another parametrization |
| Departures from argmax at temperature 1 | ≥ 36.0 a side-game (0.151 of decisions) | [D] `round11_provenance_census.out` §2 | The signal exists beyond the band |
| Sampling-cost indicators | September seed: argmax +412 ± 97 over sampling (40 games); `obs8`: a temperature-1 turn −0.068 z (does not add over a game) | [M] `docs/raw_argmax_control_20260904.md`; `round11_provenance_turn_signal.out` B | Nothing; D₀ is measured |
| Second-order loss on record | R: +18.0 first order, −12.3 ± 4.5 second | [M] round 10 | Steps are read on the true objective |
| TRPO's bound | Certifies only below 4.7e-6 nats a decision at a 230-decision horizon | [D] `round11_simplify_budget.out` | Measurement replaces the bound |

**Still to measure**

| Quantity | Cheapest way | Cost [D, *plan*] | Decision it feeds |
|---|---|---|---|
| Game length, decisions, cap rate and price at temperature 1 | Stage 1 | inside $1.65 | Every price; cap scoring if above 0.009 [D: one claim SE / 347.4] |
| E[R²] under luck alone, + v̂, + U, cross-fitted | Stage 1 logs | $0 | A correction enters iff it cuts E[R²] by 2 SE and by ≥ 5% [C: 5% of about $70 of rounds is $3.5] |
| D₀ | Stage 1, 12,800 decisive one-network games | $0.33 | The debt; the claims' origin |
| Memory at temperature 1 | Stage 1, 64 against 0 slots, 3,200 games | $0.28 | Report; read beside a claim miss |
| Sharpening slope | Stage 1 logs | $0 | The hypothesis at first order; whether the link match runs |
| I-sample, legal mass, I-count | Stage 1 logs | $0 | Gates |
| Replay identity; forward-only speed | C0 | inside C | Gate; C's arm list |
| B in the step's metric; the gain curve Ĝ(N) at 5,000, 10,000, 20,000 | C | inside C | N; whether rounds resolve |
| Best metric; ε₀; head-temperature floor | C | inside C | The optimizer; round 1's radius; the floor |
| Planted recovery | C offline, P played | $0.65 | Gate before rounds |
| The link at temperature 1 | Antisymmetric match, iff the slope ≥ 3 SE | $0.33 | Whether first-order labels price play |
| Second-order loss against the radius; transfer to the reference | Each round's ladder; claims | inside rounds | The radius; continue or stop |
| Exploitability; human level | None | — | Nothing rests on them |

## 5. Checks and plants

| Check | Prediction | Failure reading |
|---|---|---|
| I-sample (stage 1) | Binned χ² of 1[a ∈ c] against logged p(c) for end_turn, attack, recruit; the sampled action's rank uniform under its logged law | A decode that differs from the logged law (a kept offset, a stale temperature) |
| Legal mass | The server's Σ prior over the legal list is 1 at every decision | If not, the loss needs log L_t: a masked factor outside the enumeration |
| I-count | Logged decisions = trained decisions, refused recruits included | A dropped score term |
| Replay identity (C0) | The trainer's π_k reproduces logged log p_t; the mean KL is below ε/20 [C: a tenth of the smallest rung] | Memory carry, masks or rejection replay differ |
| P1 | Probe gradients = logs' Σ R(1[c] − p(c)) | Alignment, sign or label |
| Dice control, I-V | Separate code reproduces the fitted luck slope on z; each correction's mean against 1[a ∈ X] − p(X) is 0 within 3 SE | Join or orientation; a correction read after its event |
| CI toy game | The production estimator on an enumerable two-decision game equals the exact J difference within 3 SE; a seat swap flips R; the KL search hits its target | Estimator code |
| **Planted round** | z′ = z + δΣ_t σ_t(1[a_t ∈ X] − p_t(X)) in side A's frame, X = attack, written into a copy of the result table and read by the production join, b and β. δ is set from C's noise for 10 SE [C]. The expected gradient is exactly δΣ_t∇p(X\|h_t). **C (offline):** the teacher-forced change of Σ_t p(X) on H, per seat, within 3 SE of its first-order forecast from the step and the exact planted gradient, by separate code. **P (played):** the planted step against π₀, 5,000 games; X per side-game rises on each seat; the played p(X) matches the trainer's θ′ on the same positions | One seat reversed: σ_s. Neither moves: the join or the hand-off. Wrong size: direction or step |
| Importance weights (C, rounds) | The effective-sample share of W on H is at least half of e^{−2K} [C] | Heavy-tailed ratios (a rare action raised): that rung is read by a cross-play match |
| Gain-curve model | Ĝ(N)/Ĝ(20,000) = √[(N/(N + B))/(20,000/(20,000 + B))], B from the KL readings | Noise not shaped as the model assumes; N is then taken from the curve itself |

The memory's credit path is the recipe's back-propagation, validated by the own-trajectory rounds' measured gains. No memory-specific plant is built (§9).

## 6. Regimen

| Stage | Cost [D, *plan* §2] | Readout | Decision per outcome |
|---|---|---|---|
| **0, CI** | $0 | The signed-weight loss; accumulation; the direction transforms; the KL search; Ĝ and ΔĴ on the toy game; P1 on a CPU forward | Red: fix before any box |
| **1, no training** | $1.65 | 30,000 `own2` temperature-1 self-play games: 20,000 to train on, 10,000 held out. Logs: log p_t, H_t, p(end_turn), p(attack), p(recruit), legal mass, v̂_t, refused recruits; z, cell, caps, S. D₀ (12,800); memory 64 against 0 (3,200). Offline: the census, E[R²] by correction, I-sample, the sharpening slope | Gates red: fix. Corrections by the rule of §4. Cap rate above 0.009: the user rules on cap scoring |
| **C, nothing played or kept** | $2.73–5.71 | C0. The gradient pass. Arms at N 20,000, each at K ¼ on half A: SGD, reduced AdamW, Muon, heads-only, probe span. The winner on the ladder K ∈ {⅛, ¼, ½}. Reduced AdamW at 5,000 and 10,000. B; the floor; the planted offline readout | **Round 1's candidate** (best arm and radius on half A, read on half B): lower 2-SE bound above 0: P, then rounds, with the measured gain per dollar. **Unresolved, but the curve gives z ≥ 2 at a round size costing ≤ $18** [C: a quarter of the $73.42 left for rounds at the forward bound]: rounds at that size. **Neither:** no rounds; the user chooses the $1,000 case, sized from the curve, or a stop. **Optimizer:** by §2's criterion. If C0's forwards run at the bound, heads-only and the 5,000-game point are dropped ($5.04) |
| **P** | $0.65; link match +$0.33 iff the slope ≥ 3 SE | The planted step played; the antisymmetric temperature match against its sealed forecast | Planted miss: debug. Link outside its forecast ± 2 combined SE: first-order labels do not price play at temperature 1; stop and diagnose before rounds |
| **Rounds** | $2.78–3.97 at N 20,000, \|H\| 10,000 | §1. Each round logs Ĝ and ΔĴ at three radii, the adopted reading on half B, entropy, cap rate, the human cross-entropy, the cosine with the last direction | **Revert:** half B below −2 SE (0.023 at a zero gain [D]). **Pause:** after every third round, if the last three confirmed readings' sum is not above 0 by 2 SE. A null pauses at 0.977; a gain of 3.8 Elo a round passes at 0.8 at K ¼ [D, *plan* §4]. The user sees the measured Elo per dollar |
| **Claims**, every third round | $1.13 | claim_k − claim_0 against the sum of confirmed readings since round 0; SE 8.6–11.1 Elo at rounds 3–12 at a 0.09 win rate [D, *plan* §4] | **Lower tail at 2 SE:** gains do not transfer; stop; the user picks a magnet or a pool. **Upper tail:** reported |

**Budget** [D, *plan* §2d]:
- $81.18 is left [M: the Vast account's spendable balance, read 2026-10-21 with no instance running].
- Stages 1, C and P cost $5.03–8.01.
- That leaves 18–27 rounds at N 20,000.
- A round takes 4.8–6.9 box-hours at N 20,000 [D, *plan* §2e].
- The ceiling for rounds is the user's.
- The $1,000 case is argued only from adopted rounds' measured gain per dollar.

## 7. What is new, and the nearest work

Searched 2026-10-21; finding nothing does not prove absence. The update itself is REINFORCE with exact baselines. Its nearest relative is AlphaGo's RL policy: outcomes ±1 from a supervised start, a pool of past iterates, about 80% wins over its start ([Silver et al. 2016](https://www.nature.com/articles/nature16961); from secondary sources, unverified).

1. **A step certified offline on the true objective.** Three radii are read by trajectory importance weights with the chance-law luck terms inside, selected on one half of the held-out games and confirmed on the other. Neighbours: TRPO's in-sample line search on its surrogate ([Schulman et al. 2015](https://arxiv.org/abs/1502.05477)), which tracks true gains poorly ([Ilyas et al. 2020](https://arxiv.org/abs/1811.02553)); HCPI's held-out tests ([Thomas et al. 2015](https://proceedings.mlr.press/v37/thomas15.html)); POIS's Rényi-bounded importance objective ([Metelli et al. 2018](https://arxiv.org/abs/1809.06098)). The assembly was not found.
2. **The optimizer chosen by paired held-out gain at matched KL, from one gradient.** The nearest is Ruan et al.'s RMS-matched control, which compares inside training runs.
3. **The critical batch of a KL-normalized step,** read from the KLs of split-accumulator directions. McCandlish et al. read B_simple from pairs of batch sizes.
4. **A measured floor on attainable gain before training,** from realizable temperature directions by Cauchy–Schwarz in the Fisher metric. The temperature score itself is standard (`round11_challenge_critique.md` §2.1).
5. **AIVAT's chance terms as a training control variate.** Neighbours: input-dependent baselines ([Mao et al. 2019](https://arxiv.org/abs/1807.02264)) and hindsight baselines ([Mesnard et al. 2021](https://arxiv.org/abs/2011.09464)).

## 8. Weakest points

1. **The signal per game.** Nothing on record bounds B from above. If C finds no resolvable round at an affordable size, $5.03–8.01 is spent and the honest result is a stop.
2. **Distance.** Repaying D₀ may need the policy to move far. At most 2ε of KL per round is certified offline, so progress per round is bounded by √(2K·gᵀF⁻¹g), which is unknown. Past K = ½ the ladder needs cross-play matches.
3. **Transfer.** Gains against the predecessor need not reach the reference. The claims see a cumulative shortfall of about 17 Elo at 2 SE by round 3 [D, *plan* §4].
4. **Prices at temperature 1** are unmeasured until stage 1.

## 9. Decision record

**Rejected this round:**
- **Several AdamW steps per round with a fixed rate per segment** (simplify). Fresh moments make each segment's first steps lr·sign(g), and one line-search step does not calibrate many (`round11_simplify_critique.md` §2.1).
- **The ratio rule for the radius** (challenge). It freezes at low z and overshoots under its ×4 cap (`round11_challenge_critique_rule.out`). The ladder keeps 0.62–0.90 of the best gain [D, *plan* §6].
- **Adoption only on a confirmed gain** (provenance). It is underpowered: a 3-Elo step is adopted 9% of the time (`round11_provenance_critique.md` §2.3). Here a step is reverted only on confirmed harm, and claims check the sum.
- **A stage-C prediction from B_simple in the P metric.** That formula is wrong for a KL-normalized step; it is replaced by the KL-based B.
- **"Preconditioning boosts noise, so Muon and Adam lose"** (provenance). The noisy quadratic model's abstract says preconditioning raises the critical batch, not that it hurts below it; the transformer evidence cuts the other way. The arms decide.
- **Training on the cross-play games' new seat** (challenge). It saves about $0.07 a round [D: 5,000 side-games of generation] for a second data path.
- **The exact-law action walk** (invent). Its costly columns need expectations over every legal action, which the generator does not compute. Only its free columns (v̂, U) are tested.
- **The chance head** (invent). Setup-varying coefficients are worth +0.0007 of variance.
- **The parity memory plant** (challenge). Parity is a known failure case for gradient learning ([Shalev-Shwartz et al. 2017](https://arxiv.org/abs/1703.07950)), across a 16-decision window. A plant on a bit shown once and hidden later is open.
- **Caps scored −1 for both sides** (simplify). It is not zero-sum.
- **Dropping capped games.** It conditions on an outcome.
- **The belief loss at the recipe's weight** (challenge, invent). It has the shape of the 94–99% failure.
- **The bandit-converted imitation arm** (invent). At B = 10⁵ its chord carries a signal-to-noise of about 0.15 (`round11_invent_critique_checks.out` §5).
- **The 16-arm imitation bench for this decision.**
- **Schedule-free, PPO epochs, a KL anchor from the start, a pool from the start, forks, bootstrapped critics.**

**Withdrawn by the user's rulings:**
- the near-tie coins as training;
- the recruit rule Bd and every decode-time head;
- argmax play;
- rounds 8–10's screens and seals.

Their records stay as measurements: S0(a), the coin plant and match L's +15.3 ± 1.8 Elo.

**Kept from earlier rounds:**
- the label's luck correction, with frozen, cross-fitted b and β;
- dice control and I-V;
- single-decision comparisons only;
- one-sided claim stops;
- fresh-seed confirmation, now the half-B reading;
- PURE claims;
- nothing distilled before a match certifies it.
