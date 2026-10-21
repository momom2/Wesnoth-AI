# Round 11, provenance stance: certified policy gradient at temperature 1

The update by which `own2` learns from its own sampled games at temperature 1, and the optimizer that applies it, ordered so that no stage spends money on an unmeasured number.

Tags: **[M]** measured, **[D]** derived (inputs in §4), **[U]** unknown, **[C]** a choice. Scripts, each with its `.out` in `analysis/`: `round11_provenance_census.py` (*census*), `round11_provenance_turn_signal.py` (*turn signal*), `round11_provenance_derived.py` (*derived*).

## 1. The algorithm

### 1.1 Labels and update

**Play.** π_k plays itself at temperature 1 over the joint of legal actions, no end_turn offset, 64 memory slots; π_0 is `own2`.

**Logs.** Per decision, the server's probability p_t of the sampled action. Per game, z (±1), the cell (map, faction pair, seat) and the luck walk's sums S (every attack's realized minus expected material, from the exact enumerator).

**The label.** Each decision of side s in game g carries

    R_{g,s} = σ_s z_g − b_k(cell, s) − β_kᵀ S_{g,s},

with b_k and β_k fitted on earlier iterations' games and frozen. The update is the policy gradient of J(π_θ against π_k), averaged over seats:

    ĝ = (1/N) Σ_g Σ_s R_{g,s} Σ_{t ∈ side s} ∇ log π_θ(a_t | h_t)

**Code.** The sequence trainer's policy loss, with the sampled action as target, the signed label as weight and label smoothing 0 (the recipe's 0.05, [M] `wesnoth_ai/imitation_loss.py`, adds a term of non-zero mean). No value or belief loss: under a KL-normalized step (§2) an auxiliary gradient spends the step's budget, and on record the value gradient owned 94–99% of the trunk's update.

### 1.2 Credit across a game's decisions

Every decision carries its side's result (critics and search priced decisions no better on record). Two derived facts set the price.

- **Noise is uncorrelated across decisions.** With raw z, z² = 1, so E[s_t s_uᵀ] = 0 for t ≠ u: the later score has zero mean given its history. A game's gradient noise is σ_R² Σ_t F_t. This is exact for raw z and approximate after the control variates.
- **The signal has a ceiling.** The result is a sum of martingale increments: z = V(start) + Σ dice luck + Σ over both sides' decisions of A_t. So E[V₀²] + D + Σ_t E[A_t²] = Var(z) = 1. The baseline and the luck columns explain at most E[V₀²] and D. One side's tabular signal I_side = Σ_{t ∈ side} E[A_t²] is therefore at most σ_R²/2 per side-game, and at most 1/2.

**Indicator.** On human mid-game positions (obs8, temperature-0.5 continuations), a temperature-1 turn's value varies by I_turn = 0.0204 (0.0064–0.0394) a side-turn [M, *turn signal* A]: 0.41 over 20.16 side-turns [M, *census*], 0.0017 a decision. Each label is about 99.8% noise; credit is resolved across games, by what the network generalizes, which the critical batch B (§1.4) measures and no record holds.

### 1.3 Forks, rejected

Monte-Carlo values from intermediate states, as in VinePPO ([Kazemnejad et al. 2024](https://arxiv.org/abs/2410.01679)), shrink a post-fork label's noise to (1 − R²)K/(K − 1) of the single path's, with K forks.

R² at turn starts is 0.346 (0.298–0.395) [M, *turn signal* C]: the factor is 0.87 at K = 4 and never below 0.654 [D, *derived* §3], for post-fork decisions only, and temperature 1 lowers R². *Rejected: forking the server's per-game memory for at most 1.53× on part of the decisions;* reopened only if C shows noise binding by less than that.

### 1.4 Bias and variance

**Bias.** ĝ is unbiased for ∇J(π_θ against π_k) at θ_k when the actions came from π_k (up to the server–trainer arithmetic gap C0 measures), b and β were fixed before the game, and a capped game scores −1 for both sides (2 of 6,402 at argmax [M, *census*]). Truncated back-propagation through the memory (16 decisions, [M] `tools/sequence_setup.py`) biases the memory's long-range credit, not the label.

**Variance.** In the update's metric P, the critical batch in games is B_P = σ_R² tr(PF)/(gᵀPg). In the natural metric, gᵀF⁻¹g ≤ I_both ≤ σ_R², so B ≥ rank(F) games [D]: the noise fills every Fisher direction, while one game holds at most σ_R² of signal.

### 1.5 Steps and certification

**The step.** θ' = θ_k + s·d: the optimizer gives d = P(ĝ), and s sets the mean per-side-game KL(π_k ‖ π') on held-out positions to K ∈ {1/16, 1/4, 1/2} [C]. **The gain** is read on held-out games H (from π_k, unused by ĝ) by per-trajectory importance sampling:

    ΔĴ = (1/2|H|) Σ_g Σ_s (W_{g,s} − 1) R_{g,s},   W_{g,s} = Π_{t ∈ side s} π'(a_t | h_t) / p_t

- **Unbiased:** the opponent's ratios are 1, and the cell and luck terms have zero mean given the history before their event.
- **Variance:** σ_R²(e^{2K} − 1)/2 per game [D]. Its SE is 0.57 of a match's over the same games at K = 1/4 and 0.26 at K = 1/16 [D, *derived* §5]; a match's noise does not shrink with the step. At K = 1 it is 1.79× a match's, hence the cap at 1/2, where the effective sample share e^{−1} = 0.37 is checked against the measured share to catch heavy tails.
- **The step size drops out:** the gain grows as sqrt(2K · I_net · cos²), cos² being the step's share along the true gradient, and the noise as sqrt(2K).

**Adoption.** Candidates (direction, K) are ranked on half of H. The best is read on the other half and adopted iff that reading minus 2 SE is above 0. Under a null, false adoption is 0.023, whatever the first half picked. Claims against `own1` (§3) catch gains that do not transfer.

### 1.6 The human prior

Kept by the starting point and the step size (at most 1/2 nat a side-game per step; the KL to `own2` logged). No anchor term [C]: against a fixed yardstick in a zero-sum game an anchor caps the gain, needs a coefficient no record derives, and costs a frozen forward per trained position. The evidence for anchors is from human-facing play ([Jacob et al. 2022](https://arxiv.org/abs/2112.07544); [Bakhtin et al. 2023](https://arxiv.org/abs/2210.05492)); AlphaStar's ablation was not verified. A lower-tail claim miss, the signature of cycling, adds a magnet ([Sokota et al. 2023](https://arxiv.org/abs/2206.05825)) or AlphaGo's pool of past iterates.

### 1.7 The memory

The same gradient trains the memory on the histories it creates, through truncated back-propagation: the remedy for the failure on record, an imitation memory continuing its own history's patterns. Temperature 1 may revive it: a memory that sees its own sampled mistakes may infer a weak player and keep playing like one, the auto-suggestive delusion of [Ortega et al. 2021](https://arxiv.org/abs/2110.10819), whose remedy (own actions as interventions) is on-policy training. B2 measures it before any training, 64 slots against 0 at temperature 1, against the argmax reading +13.0 ± 12.3 [M, CLAUDE.md], and is re-read at every claim.

### 1.8 Why more signal per dollar

Every decision is labelled at one forward-backward (training costs 1.99× generation per game [D]). The signal is the policy's own choice variance, up to σ_R²/2 a side-game, where argmax play had none outside the coin band: a temperature-1 player leaves `own1`'s argmax at least 36.0 times a side-game, 0.151 of its decisions [D, *census* §2]. Certification costs 0.26–0.57 of a match's noise with no new games. The alternatives were measured or derived not to pay: critics, search (−124 ± 58), forks (≤ 1.53×).

## 2. The optimizer

**Under a KL-normalized step the optimizer supplies only the direction.** The learning rate, warm-up and schedule disappear, so schedule-free has nothing to replace. In agentic RL, Muon's gain over AdamW vanished once update RMS was matched ([Ruan et al. 2026](https://arxiv.org/abs/2607.16169), abstract).

**The regime decides the direction.** Under N ≪ B the noise covariance is σ_R²F, so a preconditioner that boosts low-Fisher directions boosts noise. The noisy quadratic model predicts that Adam and K-FAC help only near or above the critical batch ([Zhang et al. 2019](https://arxiv.org/abs/1907.04164)).

- **Undamped natural gradient: excluded,** since B ≥ rank(F).
- **Muon: predicted to lose while noise dominates.** It sets every singular value of the noisy gradient to 1, and it is reported to collapse in RL with verifiable rewards ([Fan et al. 2026](https://arxiv.org/abs/2605.19282), abstract).
- **Default [C]:** the recipe's AdamW reduced to one step per batch:
  - the batch gradient divided by the root of its per-window second moment, an estimate of σ_R² diag F;
  - no momentum across updates, which would mix stale policies' gradients;
  - no weight decay and no clip, since the KL bounds the step.
- **Paired candidates in C:** SGD and Muon (the built code).
- **Built only if C shows signal dominance:**
  - damped natural gradient by conjugate gradient, TRPO's form (ACKTR: 2–3× sample efficiency, [Wu et al. 2017](https://arxiv.org/abs/1708.05144));
  - SOAP, as a within-batch Kronecker preconditioner.

**The built bench** (imitation from scratch, 1,000,000 positions, full-distribution targets) is low-noise and curvature-dominated. It settles each implementation's throughput, memory and stability, and a future imitation pass's optimizer, not this update. **This update's criterion is paired certification** (stage C, about the bench's price): one gradient batch, each candidate scaled to the same K, ΔĴ on the same held-out games with the SE from per-game pairs; pre-registered pairs are the default against each candidate at N = 17,500, K = 1/4; a candidate replaces the default only beyond 2 SE.

**Batch.** N* = sqrt(c_f B / c_g), the cost-optimal batch of [McCandlish et al. 2018](https://arxiv.org/abs/1812.06162)'s model with a fixed per-step cost c_f ≤ $0.58 (held-out games, four candidates' forwards) and c_g = $7.76e-5 a game [D]: 8,654 games if B = 10⁴, 27,366 if B = 10⁵, illustrations until C measures B. The step is the K ladder.

**The [LessWrong post](https://www.lesswrong.com/posts/9gqjdRyJxSM6kS7zi/i-trained-gpt-2-with-full-batch-gradient-descent-so-you-don)** (full-batch GPT-2, under-tuned by its author's account): one fixed set repeated over 350 steps lost about 0.2 nats of validation, so here each game is used once; full-batch steps were curvature-limited (consecutive gradients at cosine −0.945, the loss up on a third of steps), which the KL cap and certification guard, with successive directions' cosine logged; fresh large batches took about 7× fewer steps at about 19× the tokens, so with data as the cost the batch sits near N*, not at its maximum.

## 3. Regimen

Measurements come first. Costs are at $0.577 an hour [D, *derived*], priced as argmax pool games, so each scales with the game length at T1, which B3 measures.

| Stage | Cost | Readout | Decision per outcome |
|---|---|---|---|
| **A**, offline, done | $0 | Census, turn signal, prices | §4 |
| **B**, no training; builds p_t and entropy into the record | $0.89 | **B1:** `own2` T1 against its reference decode, 3,200 decisive games (SE 7–10 Elo): the sampling cost Δ_s. **B2:** 64 against 0 slots at T1. **B3:** 20,000 T1 self-play games: length, caps, σ_R², entropy | Δ_s is the claw-back target. B2 2 SE below +13.0: the memory is a primary target. B3 feeds C |
| **C**, after the discussion; nothing played or kept | ≈ $2.75 | **C0:** trainer log-probabilities against p_t; the null step's ΔĴ. **C1:** 8 accumulators over 17,500 of B3's games: B_P from cross-pair products. **C2:** the default at N 2,187 to 17,500 and K 1/16 to 1/2, SGD and Muon, certified on 2,500 games (SE 3.3–4.0 Elo). **D1:** plants | **C0 off:** fix first. **Prediction:** C1's B_P fixes C2's curve, ΔĴ(N)/ΔĴ(17,500) = sqrt[(N/(N+B))/(17,500/(17,500+B))]; a miss is a bug. **Resolved:** the cost to Δ_s, (Δ_s/ΔĴ)(N* c_g + c_f), goes to the user. **Unresolved:** the bound prices the run; above the ceiling, stop at about $3.6 |
| **D2**, training on a modified game | 3 (N* c_g + $0.58) | Three self-play steps on the planted reward | Within 2 SE of D1's efficiency, or debug |
| **E**, iterations | N* c_g + c_f a step; per 5 adoptions a claim ($1.13) and B2 ($0.28) | Each step's ΔĴ against sqrt(2K I_net cos²) from the last; each claim (12,800 games, SE 3.07) against the certified sum | **Two misses at N\*:** double N once, then stop. **Claim's lower tail:** stop; the user picks a magnet or a pool. Cost re-derived each step |

**Plants.** The label adds δΣ_{t ∈ side}(f(a_t) − E_π f), f the recruit indicator. Its gradient is a second accumulator in C over 4,375 games, beside the real one over the same games, so any δ is a linear combination chosen after C1–C2. The readout ΔE[Σf] is computed exactly from π' and π_k at held-out positions, free of outcome noise. With V_f = Σ_t Var_π(f_t), the plant's critical batch is B_f = B_P I_net/(ρ_f² δ² V_f).
- **D1-large** (δ set so that B_f ≤ N/20) reads ρ_f, the followed share of the noise-free bound sqrt(2K V_f).
- **D1-mid** (δ set so that B_f = N) must read sqrt(2K V_f) ρ_f/√2 within 2 SE.
- **Each seat** must read above 0 separately: a sign error on one seat shows.

**CI tests:** the production estimator on an enumerable toy game equals the exact J difference within 3 SE (a ratio over the wrong seat fails it); the loss on a one-decision bandit has gradient R(e_a − π); the KL search hits its target; luck and cell terms have zero mean given the pre-event state (I-V).

## 4. Provenance table

| Number | Value | Tag and source | What rests on it |
|---|---|---|---|
| Decisions per game; per side-game, at the reference decode | 475.7 ± 3.2; 237.9 | [M] *census* §1, 6,402 `own2`–`own1` games | Prices; per-decision share |
| Side-turns per side-game; decisions per side-turn | 20.16; 11.80 | [M] *census* §1 | The I_side indicator |
| Decision kinds | move 0.579, attack 0.237, recruit 0.099, end_turn 0.085 | [M] *census* §1 | The plant's V_f (measured again in C) |
| Not ended by a leader kill, argmax | 2 of 6,402 | [M] *census* §1 | Cap scoring minor; T1 rate [U] → B3 |
| Generation price | $2.598e-5 a game at $0.577 an hour | [M] `round9_simplify_labels.out` | Every cost |
| Two-network claim game | $8.848e-5; 12,800 games $1.13, SE 3.07 Elo | [M] same; [D] current_design | Claims, B2 |
| Trainer | 1,477 positions/s, configuration B, imitation loss | [M] `training/metrics/trainer_calibration_20261015`, BACKLOG | Training price; PG is the same loss with signed weights, re-timed in C |
| Training per game; ratio to generation | $5.16e-5; 1.99 | [D] 475.7/1,477 s × $0.577/h | §1.8; c_g = $7.76e-5 |
| Forward-only speed | ≤ a trained position | [D] bound; [U] → C | c_f ≤ $0.58, an upper bound only |
| Game length at T1 | — | [U] → B3 | Scales every cost; nothing bought before B3 |
| Sampling cost of `own2` | — | [U] → B1 (SE 7–10 Elo, [D] *derived* §2). Indicators: September seed +412 ± 97 [M] `docs/raw_argmax_control_20260904.md`; obs8 T1 turn −0.068 (−0.126 to −0.016) z [M] *turn signal* B, which does not sum over a game (−1.37 exceeds the bound 1 [D]) | E's target and size, after B1 |
| Memory at T1 | — | [U] → B2; argmax +13.0 ± 12.3 [M] CLAUDE.md | Whether the memory is a primary target |
| σ_R² at T1 | ≤ 1 | [D] z² = 1. At argmax 0.699 [M] `round9_simplify_residual.out`; at T0.5 the lucks explain 0.072 [M] `docs/turn_value_prereg_20260925.md`; [U] → B3 | Every SE, computed at 0.70 and at 1 until B3 |
| I_side ceiling | ≤ σ_R²/2 ≤ 1/2 | [D] martingale identity, §1.2 | B_nat ≥ rank(F); excluding undamped natural gradient |
| I_turn; I_side indicator | 0.0204 (0.0064–0.0394); 0.41 (0.13–0.79) | [M] *turn signal* A (obs8, human positions at turn ≥ 7, T0.5 continuations); [D] × 20.16 | Nothing: motivation only |
| Per-decision signal share | 0.0017 | [D] 0.41/237.9 | Nothing: motivation only |
| Turn-start R² | 0.346 (0.298–0.395) luck-corrected; 0.318 raw | [M] *turn signal* C | Rejecting forks (factor ≥ 0.654 [D] *derived* §3) |
| Argmax departures at T1 | ≥ 36.0 a side-game (0.151) | [D] *census* §2, `own1` coin gaps b00 and b06 | §1.8 only |
| B, critical batch in games | — | [U] → C1, cross-checked by C2 | N*, the run's cost, the optimizer's regime |
| I_net, cos²(N), ρ_f, V_f | — | [U] → C2, D1 | Predictions for E's steps and the plants |
| IS SE; effective sample share | σ_R sqrt((e^{2K}−1)/(2\|H\|)); e^{−2K} | [D] *derived* §5 | K cap 1/2; \|H\| = 2,500; adoption |
| K ladder | 1/16, 1/4, 1/2 | [C]; cap [D] (K = 1: SE 1.79× a match's) | Steps |
| Cost-optimal batch | N* = sqrt(c_f B/c_g) | [D] McCandlish et al.'s model with a fixed per-step cost | E's batch, after C |
| False adoption | 0.023 | [D] one-sided normal tail at 2 SE | Adoption |
| Elo per unit of E[z] near 0 | 347.4 | [D] | Conversions |
| BPTT window; label smoothing | 16 decisions; 0.05 in imitation | [M] `tools/sequence_setup.py`, `wesnoth_ai/imitation_loss.py` | The memory's bias; smoothing set to 0 |
| Stage costs | B $0.89; C ≈ $2.5 (gradients $1.03, candidates ≤ $1.42); D1 $0.23; claim $1.13 | [D] *derived* §1–2 | Order: B now, C after the discussion |
| Budget left | $82.34 if the $5 window counted | [M] `current_design.md` | The user's ceiling for E |

## 5. What is new, and the nearest work

Searched 2026-10-21; a short search proves no absence.

- **Nearest: AlphaGo's RL policy** ([Silver et al. 2016](https://www.nature.com/articles/nature16961)): REINFORCE from imitation on ±1 outcomes against a pool of past iterates, about 1.28 M games (secondary quote, unverified), over 80% against its supervised policy. New here: luck-corrected labels, certified KL-normalized steps, the memory.
- **The ceiling I_side ≤ σ_R²/2, hence B_nat ≥ rank(F).** The martingale decomposition is AIVAT's ([Burch et al. 2018](https://arxiv.org/abs/1612.06915)). Its use to bound the critical batch of outcome policy gradients was not found.
- **Certified KL-normalized steps for any preconditioned direction,** with exact chance control variates inside the estimator and a signal-to-noise independent of K. TRPO ([Schulman et al. 2015](https://arxiv.org/abs/1502.05477)) scales a natural-gradient step to a KL and checks it in-sample; HCPI ([Thomas et al. 2015](https://proceedings.mlr.press/v37/thomas15.html)) tests on held-out trajectories by importance sampling. The assembly was not found.
- **An RL optimizer bench by paired certification at equal KL.** The nearest is Ruan et al.'s RMS-matched control.

## 6. The weakest point

Everything after stage C rests on B, the games the network needs to resolve its own gradient, and no record bounds it from above. If C finds B in the millions for every direction, outcome policy gradient from `own2` cannot move within this budget, and the design stops at about $3.6 of measurements.

Further: the indicators are obs8's at temperature-0.5 continuations (B1–B3 replace them first); certified gains are against the predecessor, so non-transitivity shows only at claims.

## 7. Questions for the optimizer discussion

1. **The yardstick's decode:** `own1` at argmax with offset −1.5, at temperature 1, or both? Against argmax, claims read negative until the claw-back is done; B1 says by how much.
2. **Stage C before training is approved?** Its gradients and candidate steps never play and are not kept.
3. **The step rule:** a certified KL-normalized step, or a learning rate with a schedule? Evidence: Ruan et al. 2026, the post's curvature-limited steps, TRPO, HCPI.
4. **The direction:** the reduced AdamW against SGD and Muon in C; damped natural gradient and SOAP built only on signal dominance. Evidence: Zhang et al. 2019, Fan et al. 2026, Wu et al. 2017.
5. **The 16-arm bench ($2.50):** skip, a two-arm smoke (about $0.3), or all of it for a future imitation pass?
6. **The human prior:** no anchor with a magnet or pool on a claim miss (proposed), a KL anchor to `own2`, or a pool from the start? Evidence: piKL and Diplomacy are human-facing; MMD's last-iterate convergence; AlphaGo's pool.
7. **The ceiling for E,** derived after C, out of $82.34.
