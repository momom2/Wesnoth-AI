# Round 11, challenge (certified steps): critique

**Verdict: repairable.** The update (REINFORCE at temperature 1, unbiased free control variates) and its held-out first-order forecast Ĝ are sound. As written: no stop for an update that does not learn; two planted checks fail on correct code, the third passes with some wrong code; the step rule rests on two unmeasured quantities. Every repair uses data the plan already generates. *rule* = `analysis/round11_challenge_critique_rule.py` and its `.out`.

## 1. Provenance audit

| Number | Tag | Check |
|---|---|---|
| $2.598e-5 pool game, $8.848e-5 two-network game | M | `round9_simplify_labels.out` lines 20–21 (argmax games); T = 1 length unknown, stated |
| 1,477 positions/s; 7,003 encodings/s | M, transferred | BACKLOG; `candidates.log` 1,107,818 in 159 s = 6,967/s |
| 475.7 / 237.9 decisions | M, argmax | `round11_provenance_census.out` |
| 0.928, 0.854, 0.725; forks 1.19×, 2.81× | M, transferred | `round11_challenge_forks.py` rerun, identical; 1.19× is an upper side (v̂_t tracks later states, the fork mean does not) |
| +18.0, −12.3 ± 4.5; +22.7 ± 4.4; +9.3; $81 | M | current_design §2; BACKLOG (match L about $0.9) |
| GPT-2: 3.248, 3.521, 3.230, 19×; Muon 0.290 → 0.546 | external | Read: match; single seeds in both |
| 347.4; 3.47; 2.40, 5.36; 0.023; 2√k − k; ρ = ½; 1.99×; 0.58; stage and round costs | D | Redone; reproduce ($3.906 per 20,000-game round by hand) |
| E(r − 1)² ≈ 2ε in the forecast SE | D, assumption | Fails when a step raises a rare action: p 0.001 → 0.01 gives χ² 0.081 against 2KL 0.013. Read the SE empirically |
| k₀ = 1/16 ("k ≈ 1 in two rounds"); "usable at forecasts ≥ 10–20 Elo" | guessed | Nothing places ε* near 0.032; Q2's recommendation rests on the second |
| Claim miss line "2 SE" | unstated | √(2·4.3² + 3·3.47²) = 8.5 Elo (11.8 at a 0.9 win rate): 17–24 Elo of shortfall per three rounds passes unseen |

## 2. What would make it fail

### 2.1 A null update spends the budget (fatal, repairable)
- **The rule freezes.** Ĝ and its SE both scale with √ε, so Ĝ's z-score is the same at every radius. The rule moves only when Ĝ > 2 SE: at z_G = 1, in 0.16 of rounds at any radius (*rule*).
- **No stop fires.** At zero gain, 0.994 of 12-round chains never see two reverts in a row (*rule*), and the claims compare a zero chain with a zero claim.

About $70 can go with nothing learned and no line crossed (seed §6).

**Repair.** In the noise-dominated regime z_G grows as √(N_t·N_h) [D].
1. Stage 3 already reads Ĝ on held-out games. Read it at two training sizes (2,500 and 5,000 games) and check the √ scaling.
2. Extrapolate z_G to a round. If it is below 2, the user chooses bigger rounds or a stop before any round is bought.
3. Inside the rounds, stop after two unresolved Ĝ in a row.

**Missed, and free.** log π(a|h) + H is exactly the score of the inverse temperature. So on stage 1's own-side decisions, Σ_t y·(log π + H) is the measured first-order gain of sharpening, at a KL of ½Δβ² times the varentropy. It is a measured lower bound on gain per √KL, and it can set ε₀ and the round size in place of [C].

### 2.2 The step rule
The stance's Monte Carlo checks k = ¼ and k = 4 only. From the optimum k = 1, the ×4 cap sends the radius to k ≥ 4 (zero gain) whenever R reads about one SE high (*rule*):
- 0.04–0.36 of rounds when the first-order gain at the optimum is 5–10 Elo;
- 0.09–0.14 at 20 Elo with z_G 3–5.

From k₀ = 1/16, over 12 stationary rounds, the rule keeps 0.39–0.83 of the best gain, and at z_G = 1 no more than the fixed step's 0.44. Q2's case therefore needs F₁ ≥ 20 Elo and z_G ≥ 3, both unknown. Cap the multiplier at ×2 or pool ρ̂ over rounds.

"Certified" overstates: a step truly losing 1 SE stays 0.84 of the time; gains are certified only at claims, with the blind band above.

### 2.3 Planted checks (fatal, repairable)
- **The float-tolerance identity holds only in expectation.** "Ĝ's planted part equals δΣ(π_P − π₀) to float tolerance" is true on average only. The sample Σ_t(r_t − 1)·δΣ_u(1[a_u∈X] − π₀(X|h_u)) carries cross terms with zero mean and nonzero variance, so correct code fails. Make it a z-test.
- **The memory plant is likely unlearnable.** It pays on the parity of the previous side-turn's length. Parity is a standard failure case for gradient learning ([Shalev-Shwartz et al. 2017](https://arxiv.org/abs/1703.07950)), and the count spans 13–24 of the side's decisions against a 16-decision truncation window. A miss would name no path, and its 0-slot arm is not costed. Use a bit visible once and hidden later, shown learnable in CI first.
- **The end_turn plant can pass with wrong code.**
  - If added per side, it bypasses the mover sign s that orients the real result.
  - Ĝ drops every baseline, so a value read one decision late (v̂ after the action) is invisible to it, even though it biases training.

**Add:**
- **Plant through s.** The plant enters z in side 1's frame.
- **Orthogonality.** On stage-1 logs, for every term c of w (v̂, U, S, b) and X ∈ {end_turn, attack, recruit}, the mean of c·(1[a_t∈X] − π(X|h_t)) is 0.
- **Replay identity.** The trainer's forward of π_k on held-out games reproduces the logged log π_k, so r ≡ 1.
  - It catches mismatches in the memory carry, the legal set or the rejection replay. A legal-set mismatch biases Ĝ, since E[r − 1] = 0 needs one normalizer.
  - Its residual KL is the numeric floor ε₀ = 0.002 must clear, still unmeasured.

### 2.4 The optimizer bench reads first order only
Matched KL does not match the second-order loss, which depends on direction (two thirds of the first-order gain in the stance's own record): an arm can win on Ĝ and lose in play. Cross-play the winner against AdamW-hygiene before adopting it (5,000 games, $0.44 [D]). The bench's power is not derived, so a null names no cause.

### 2.5 Smaller
- **Belief loss.** It keeps the imitation weight on the shared trunk. That is the shape of the record's 94–99% value-gradient failure: a coherent dense gradient accumulates while the policy gradient's mean is small. Add a belief-weight-0 arm to the bench.
- **Refused recruits.** The bounced sampled action is not in the record (`tools/game_record.py` keeps only the rejection). Dropping its score term biases the kept term by −b·∇π(refused set). Log refused decisions.
- **Two objectives.** Training maximizes Σ w·log π_θ while the forecast reads Σ w·(r − 1). The ratio surrogate unifies them at no cost.
- **Reference.** `own2` is the reference since 2026-10-21 (`configs/reference_player.json`), so claims go against it, not `own1`.
- **Uncosted.** A revert's retraining (about $2.2) and per-round staging.

### 2.6 Novelty
- **Claim 1** is the classical trust-region ratio test (Nocedal & Wright, ch. 4), noisy form STORM ([Chen, Menickelly & Scheinberg 2018](https://arxiv.org/abs/1504.04231)), model TRPO's L − C·KL. New: reading the ratio from cross-play.
- **Claim 3** is a score-function control variate ([Ranganath, Gerrish & Blei 2014](https://arxiv.org/abs/1401.0118): "we choose h to be the score function").

## 3. Strongest part
Ĝ beside R: a held-out, critic-free forecast of the head-to-head gain, from which every baseline and luck term drops exactly, and the actual gain from cross-play. Together they separate direction from step, and Ĝ's radius-invariant z-score is the cheapest test of whether outcome learning has signal at a given round size.

Second, the Fisher-noise argument: actions are sampled from the model, so the gradient noise is a scaled true Fisher and Kunstner's objection does not apply.
