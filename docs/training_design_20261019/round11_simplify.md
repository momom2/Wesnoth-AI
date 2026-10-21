# Round 11, simplify stance: plain REINFORCE on the player's own sampled games

**Proposal.** The network plays both sides at temperature 1. After each batch of games, one AdamW step raises the log-probability of every sampled action in proportion to its side's luck-corrected result. There is no critic, clipping, data reuse, KL anchor or opponent pool. Each step measures its own signal from two halves of its batch and forecasts its gain in Elo before a match checks it. Three measurement stages, about $9, come before the first real training step.

Tags: **[M]** measured, **[D]** derived, **[U]** unknown (measured where stated), **[C]** a choice with derived consequences. The scripts are in `analysis/`: *noise* is `round11_simplify_noise.out`, *budget* is `round11_simplify_budget.out`, *rows* is `round11_simplify_rows.out`. Section 4 lists every number.

## 1. The algorithm

### 1.1 What is learned, and by which update

- **Learned:** the policy heads, trunk, encoder and memory of `own2`. The value and belief heads get no loss, and nothing in play reads them.
- **Behaviour:** `own2` at temperature 1, no end_turn offset, both seats, the ladder pool with fog and uniform factions.
- **Label:** one number per side-game j, R_j = s_j (z − b(cell) − βᵀS).
  - s_j orients the result to side j.
  - b is the shrunk map × factions × side mean.
  - S is the seven exact attack-luck sums of `tools/match_luck.py`.
  - b and β are cross-fitted on S0's games, then frozen. This is the coin rounds' label without the coin.
  - A capped game scores −1 for both sides [C]: 0 would reward stalling a lost game.
- **Update:** ĝ = (1/N) Σ_j R_j Σ_{i∈j} ∇θ log πθ(a_i | o_≤i), over N side-games, accumulated over the batch and applied as one AdamW step.
- **Code:** the own-trajectory loss of `wesnoth_ai/own_loss.py`, with a one-hot target on the sampled action and a signed weight per game-side.
- **Normalization and clipping:** normalized per side-game, so gains come out in result units. Clipping is off, since a clip that fires biases the step; a barrier counts norms above 100× the median.

### 1.2 How credit reaches one decision

By randomization alone.
- A side-game holds 230.4 decisions [M], all sampled.
- Since E[∇log π(a_i|h_i) | h_i] = 0, decision i's term has expectation Σ_a ∇π(a|h_i) Q(h_i, a). Everything else in R averages out of it.
- This is a factorial experiment over about 460 decisions a game, as the coins were over about 155 band decisions. The shared weights pool positions.
- At a band decision (top two within 0.35 nats), temperature 1 plays the runner-up with conditional probability 0.41–0.50. That is a treatment variance of at least 0.242, against the coins' 0.1875 [D]. Every other decision is randomized too.

### 1.3 Bias and variance

- **Unbiased.** E[ĝ] = ∇J exactly, where J is the per-side-game expected result against the current self, the opponent held fixed. Any b fixed before the batch cancels. Each attack's luck term has zero mean given the position and the attack, so βᵀS cancels for any fixed β ([Burch et al. 2018](https://arxiv.org/abs/1612.06915)).
- **Three departures from exactness:**
  1. The memory's gradient is truncated at the trainer's 16-decision window [M].
  2. The server's arithmetic differs from the trainer's; S0 measures it as a per-decision KL.
  3. Refused recruits are sampled actions and must enter the sum (I-count).
- **Variance** per side-game is about E[R²]·Σ_i E|∇log π_i|².
  - At argmax the corrections leave 0.699 of Var(z) (0.692 on `own2`) [M].
  - At temperature 0.5 the lucks explained 0.072 [M].
  - At temperature 1 it is [U, S0].
- **The one measured outcome-gradient noise scale on this game:** a 43-column recruit preference on batch 1's coin labels, the direction match L confirmed in play.
  - B_simple = tr Σ/|G|² is 11,593 games (2-SE band 7,813–22,457).
  - |G|² is resolved at z 4.13 over 60,000 games; split halves of 5,000 games each read z 1.29 [M, *noise*].
  - No decision rests on it: it is the scale S0 reads against.

### 1.4 The human prior

Released: the prior is the initialization only, with no KL term and no corpus in the mix. A human behaviour should survive iff it wins (ruling). The first gains expected at temperature 1 come from removing the human-mistake mass now sampled, which a KL to `own2` would tax.

### 1.5 The memory

The memory is trained by the same loss: truncated back-propagation over 32 streams × 16 decisions [M], carried across windows without gradient. On-policy training removes the train/play history mismatch behind the 105-Elo memory cost, since the memory reads and is graded on its own sampled history.

### 1.6 Why this is the cheapest signal

- **Training dominates cost.** A trained game costs $8.65e-5 [D]: generation 0.30, training 0.58, CPU encoding 0.12.
  - Data reuse (PPO's epochs) pays full training price to save at most 0.30, and pays only above the critical batch.
- **Fork labels** buy one decision per pair of playouts and still need distilling. A game randomizes about 460 decisions and trains on them directly.
- **Critics.** None a fair player can use beats material, and value gradients took 94–99% of the trunk's updates [M, seed §3]. The seed's value head as a baseline would be worth 1.19× games [D, an old head on human games: a report].
- **Search distillation** produced the prior or a truncation of it [M, seed §3].

### 1.7 What each omission costs

| Omitted | Cost | Read by |
|---|---|---|
| Value baseline | Up to about 1.19× games | S0 refits the factor at temperature 1 |
| KL anchor | Noise drifts unconstrained directions | KL to `own2` per step |
| Opponent pool | No guard against cycling; a pool game costs 3.41× and trains one seat [D] | Claims against `own1` |
| Belief and value losses | The memory's hidden-unit tracking loses its support | Claims |
| Asynchronous generation | At most 0.29 of the wall [D] | — |

## 2. The optimizer

### 2.1 The choice

**AdamW, the recipe's:** betas (0.9, 0.999), weight decay 1e-4, fresh moments, a constant rate per segment.
- It trained `own2`, and its path is bit-identical (`wesnoth_ai/optimizers.py`).
- The rate comes from S1, and changes between segments only on a readout.

Against the alternatives:
- **Muon.**
  - Under GRPO it drove accuracy toward zero by whitening low-SNR policy gradients ([Fan et al. 2026](https://arxiv.org/abs/2605.19282); one preprint).
  - Its positive agentic-RL result vanished under an RMS-matched control ([arXiv 2607.16169](https://arxiv.org/abs/2607.16169)).
  - Muon fine-tuning of AdamW-pretrained models showed no significant advantage ([Liu et al. 2025](https://arxiv.org/abs/2502.16982), Table 6).
- **SOAP.** Kronecker statistics from noisy gradients; I found no RL evidence.
- **Schedule-free** ([Defazio et al. 2024](https://arxiv.org/abs/2405.15682)) takes gradients at y and plays at an average x. Generating at x makes every gradient off-policy; generating at y forfeits the average. A weight EMA has the same conflict.
- **K-FAC natural gradient** ([ACKTR](https://arxiv.org/abs/1708.05144), claimed 2–3× over A2C). It needs new code for a packed, masked, factored transformer.
- **Trust regions.** TRPO's bound certifies improvement only below 4.7e-6 nats of maximum KL per decision at a 230-decision horizon. Moving one binary choice from 0.50 to 0.51 is 2.0e-4 [D]. Measurement replaces the bound.

### 2.2 Batch and step

- **Batch.** Under [McCandlish et al. 2018](https://arxiv.org/abs/1812.06162), (S/S_min − 1)(E/E_min − 1) = 1. The cheapest batch is then B* = √(B_noise · c_step / c_game) [D].
  - c_game is 0.533 s [D]. c_step is 69 ± 30 s [M] plus a trainer restart [U, S0].
  - At B_noise 11,593, B* is 1,225–2,328 games for c_step of 69–249 s.
  - Within 2× of B* costs ≤ 6% more, within 4× ≤ 27% [D].
  - B is re-derived each segment, since the noise scale grows during training.
- **Telemetry.** Two gradient buffers by game parity give, at every step:
  - |G|² = g_A·g_B and tr Σ;
  - the expected first-order gain lr_k(β₁ m_{k−1}ᵀP(g_A+g_B)/2 + (1−β₁) g_AᵀP g_B), with P = diag(1/√v̂) (v̂'s own-batch weight is 0.001). It is ×347.4 for Elo [D], with its SE from quarter-batch products;
  - the cosine of consecutive updates.

### 2.3 The LessWrong post

On a fixed 800M-token set, full-batch descent reached validation loss of about 3.50 against minibatches' 3.248. Fresh data at 50M tokens a step matched minibatches with 18.8× the tokens in 6.8× fewer steps [D]. The author did not retune the full-batch rate.

Our data is fresh every step, so the fixed-set overfitting does not arise; its analogue is data reuse, excluded above. What transfers is the trade between data and steps. Our cost is games, so batches stay near B*. Its edge-of-stability oscillation (consecutive-gradient cosine −0.945) is what the cosine telemetry watches for.

### 2.4 What the built bench can and cannot settle

- **Can:** speed, memory and stability per optimizer in from-scratch imitation.
- **Cannot:** behaviour at policy-gradient SNR, from an AdamW start, under the on-policy constraint.
- **The criterion this update needs:** planted gain per 10,000 games at matched per-step KL, paired by step. It can run as a Muon arm of S2 for $1.99 [D].

## 3. The regimen

Costs are at $0.577/h. From $81.44, S0–S3 cost $21.35, leaving about $60, or about 694,000 trained games [D].

| Stage | Cost | Readout and prediction | Decision |
|---|---|---|---|
| **S0** measurements, weights unchanged | $3.77 | 40,000 `own2` T1 self-play games [C], logging action key, log-probability, p(end), p(attack), p(recruit) per decision: cost, decisions, caps, E[R²], luck share [U]. Gap: `own2`@T1 against `own1`@reference, 6,400 games [U], the chain's origin. **P3:** the logs' slope along the end_turn logit, Σ(1[end] − p)R, forecasts a 6,400-game measurement match of `own2`@T1 with offset −1.5 against `own2`@T1 (forecast SE ≤ 8.3 Elo, match 4.34 [D]). Gradient pass: B_simple, \|G\|² against a sign-flip null; P1; I-logp; I-count | P1 fails: fix. P3 signs disagree, both resolved: stop. \|G\|² at z ≥ 2: S1. Otherwise the gradient is below the coin analog's strength; the user chooses a larger S0 or a stop |
| **S1** line search | $3.40 | d from half A (Adam's preconditioning from S0's second moments); θ₀ + η_k d, η_k ∈ {1, 2, 4}·η₁, η₁ set for a 12.3-Elo forecast from half B [D]. Each plays θ₀, 12,800 games. Sealed: the smallest step's gain equals its forecast | Below by 2 combined SE: forecasts do not transfer; the user decides. Else lr is set so 20,000 games of steps travel the largest η still within 2 SE |
| **S2** planted loop | $1.99 | 10 steps × 2,000 games, R replaced by δ Σ_i(1[a_i ∈ X] − p_i(X)), whose expected gradient is exactly δ Σ ∇p_i(X); δ from S0 for z ≥ 5 a step. Predicted: Σ p(X) moves between batches as each step forecasts; batch k+1's logged log-probabilities match the trainer's θ_{k+1} | Wrong sign: reward or orientation. No move: stale weights. Wrong size: the forecast. Weights discarded |
| **S3** segment 1 | $12.19 | 100,000 games [C], then 12,800-game claims (SE 3.07 [D]). Sealed: θ_K against θ₀ = Σ forecasts; against `own1`, a transfer ratio | Inside: continue. `own1` flat: non-transitivity; the user rules on a pool. Below: halve lr. Σ forecasts unresolved: the user decides on the rest |
| **S4…** | ≈ $60 | As S3. Adoption: `own1` claim above the chain's origin by 2 SE, one-sided | As S3 |

**Checks.**
- **I-logp:** server against trainer log-probability of each sampled action; mean KL 10× below S1's step KL.
- **I-count:** decisions logged = decisions trained, refused recruits included.
- **Label path:** the existing dice control, plus z′ = z + δ Σ s_j(1[a_j ∈ X] − p_j(X)) recovering slope 1 (wrong sign −1).
- **P1:** the trainer's gradient on probe scalars added to the end_turn, attack and recruit logits equals the logs' Σ R(1[c] − p(c)), with p(c) the probability each probe perturbs, to bf16 tolerance. It catches misalignment, sign and R errors.
- **P2:** the optimizer, the weight hand-off and the forecast.
- **P3:** the credit estimator on real signal.

## 4. Provenance

| Number | Tag | Source |
|---|---|---|
| Generation $2.598e-5 a game at $0.577/h | M | `round9_simplify_labels.out` [prices], 50,000 coin games in 8,105 s; at T1 [U, S0] |
| Two-network decisive game $8.848e-5 | M | Same file, 6,400 games in 3,533 s |
| 230.4 decisions a side-game | M | `memory_own_round2_20261011/candidates.log`: 1,107,818 over 4,808 own game-sides at argmax; T1 [U, S0] |
| Trainer 1,476.9 positions/s | M | `trainer_calibration_20261015/memory.json`, configuration B |
| Encode 7,003 positions/s | M | `candidates.log` |
| Luck walk 0.291 CPU-s a game; 20.08 turns a game; 0 caps in 1,600 self-play games | M | *rows* |
| Match fixed 69 ± 30 s; 155.0 ms a game; GPU busy 0.77 | M | `round10_provenance_r_link.out` §4 |
| $8.65e-5 a trained game; shares 0.30 / 0.58 / 0.12; 0.533 s a game | D | *budget* |
| $81.44 left; 941,103 games (1,292,896 at $0.42/h) | D | $82.34 (`current_design.md` §4, if the $5 window counted) less L's $0.9 (BACKLOG 0.30.5) |
| Stages $3.77, $3.40, $1.99, $12.19; $60.09 left; 694,406 games | D | *budget* |
| B* 1,225–2,328 at B_noise 11,593; ≤ 6% within 2×, ≤ 27% within 4× | D | *budget*, McCandlish eq. 2.11 |
| B_simple 11,593 (7,813–22,457) games; z 4.13 at 60,000; halves of 5,000 z 1.29; count column 4,151 (z 1.93) | M | *noise* on `round10_invent_recruit_rows_*` |
| Residual 0.699 (own2 0.692) | M | `round9_simplify_residual.out` |
| Lucks explain 0.072 at T 0.5 | M | `docs/turn_value_prereg_20260925.md`, 36,209 playouts |
| Value baseline 0.838, 1.19× | D | *budget*, `value_head/seed_by_phase.json` (seed network, human games) |
| Treatment variance ≥ 0.242 | D | q = e^{−0.35}/(1 + e^{−0.35}) = 0.413, q(1 − q) |
| TRPO certifies only below 4.7e-6 nats; 0.50 → 0.51 is 2.0e-4 | D | *budget*; TRPO Theorem 1, ε ≤ 2, γ for a 230-decision horizon |
| 347.4 Elo per result unit; claim SE 3.07, $1.13 | D | *budget* |
| End_turn forecast SE ≤ 8.3, match SE 4.34 | D | *budget*; approximate: R independent of the end-sampling noise, E[R²] ≤ 1, 20.08 turns |
| Window 16 × 32 streams | M | `memory_own_round2_20261011/train.log` |
| Pool price ratio 3.41 | D | $8.848e-5 / $2.598e-5 |
| Async gain ≤ 0.29 | D | 0.155 / 0.533 |
| Post: 18.8× data, 6.8× fewer steps; cosine −0.945 | D / reported | The post's runs |
| S0 40,000 games; S3 100,000; S1's 12.3 Elo (4 match SE) | C | Consequences derived after S0 |
| own2@T1 gap; T1 cost; E[R²]; B_simple at θ₀; trainer restart; server/trainer KL; end_turn slope | U | S0 |
| Curvature; lr | U | S1 |
| Transfer to `own1`; games to a resolved claim | U | S3 |
| AlphaGo RL: 10,000 × 128 games, 80% over SL | literature | Secondary sources; [Silver et al. 2016](https://www.nature.com/articles/nature16961) not read directly |

## 5. What is new, and the nearest work

I searched on 2026-10-21; finding nothing does not prove absence.
- **Nearest:**
  - AlphaGo's RL policy: REINFORCE on ±1 results from a supervised start, an opponent pool, and a baseline of 0 in its first round.
  - In RLHF, REINFORCE-style methods outperform PPO ([Ahmadian et al. 2024](https://arxiv.org/abs/2402.14740)).
  - Deep policy-gradient estimates correlate poorly with the true gradient ([Ilyas et al. 2020](https://arxiv.org/abs/1811.02553)): the case for per-step measurement.
- **AIVAT's chance term as a training-time control variate.** The searches found AIVAT used only for evaluation. [Mao et al. 2019](https://arxiv.org/abs/1807.02264)'s input-driven baselines are the nearest, and they need an action-independent input process. This per-attack form does not.
- **Per-step Elo forecasts from split-half gradients, sealed before each claim.** McCandlish et al. split batches only to measure noise. [Cheng et al. 2019](https://arxiv.org/abs/1810.06509) forecast gradients, not gains.
- **The centred planted reward,** whose expected gradient is exactly δ Σ ∇p(X), with no state-distribution term.
- **Nothing here is a new learning rule.** The novelty is the checking around plain REINFORCE.

## 6. Weakest point

The signal per game.
- The one measured outcome-gradient here needs about 11,600 games for half a step to be signal, and reached z 4.1 only at 60,000 games.
- The budget buys 0.7–0.9 million games, a few hundred steps. AlphaGo used 1.28 million on a game without dice or fog.
- A null run is a real outcome. The design finds it at S0 (about $4) or S3 (about $21), not at the end.
- Next weakest:
  - Without a pool, gains against the current self may not reach `own1`.
  - No check isolates credit through the memory.

## 7. Questions for the optimizer discussion

1. **Which optimizer applies the update?**
   - AdamW (proposed): it trained `own2`, and the evidence at low SNR favours it.
   - Muon as a second S2 arm (+$1.99): mixed evidence (§2.1).
   - SOAP: no RL evidence.
   - Schedule-free: conflicts with on-policy play.
   - K-FAC: new code.
2. **The bench** settles imitation only. Run it with an imitation retrain, or now as a measurement ($2.50)?
3. **Step control.** A fixed rate per segment from S1 (proposed), a KL-targeted controller, or a final cooldown ([Hägele et al. 2024](https://arxiv.org/abs/2405.18392))?
4. **Batch.** B* from S0 (proposed; within 2× costs ≤ 6%), or larger batches for less wall time? A million games is about 6 days serial [D].
5. **The human prior.** Released (proposed), or a KL anchor to `own2` (AlphaStar, piKL), which taxes removing sampled mistakes?
6. **The opponent.** The current self (proposed), or a pool, as in AlphaGo, if S3 shows non-transitivity?
7. **Auxiliary losses.** Off (proposed), or keep the belief loss for the memory?
8. **The yardstick and the money.** Does `own1` at its reference decode stay the yardstick? Are S0–S2 (≈ $9.16) approved as one box, with S3 onward gated on their readouts?
