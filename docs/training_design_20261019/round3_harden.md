# Round 3, hardening: corrected matches as the instrument, a cross-fitted critic, data past the stop

Stance: keep the current design's frame (privileged identity-trained critic, optimal stopping for end_turn, fusion-free fogged student, coin calibration) and repair it. What this file does not change stands as written in `current_design.md`. Every comparison is one decision followed by play that re-decides after every roll; the network picks every action.

## 0. Defects found

1. **The arcsine audit is the wrong instrument for a round's gain.** Its per-game variance is π²Mσ_a² ≈ 47 (M = 12, σ_a² = 0.4); a match of π_{k+1} against `own1` corrected by the frozen critic has ρ_C·Var(z) ≈ 0.65. So 40,000 audit games ($4) read about 12 Elo and 3,200 corrected match games ($0.7) about 5: some 30 times the information per dollar, and the match is exact too. A score-function estimator weights each coin's residual (order σ_a) by a unit-variance score, so its noise grows with the coin count while each coin's signal is about 0.01 ([Mohamed et al. 2020](https://www.jmlr.org/papers/v21/19-346.html)); no forecast control variate removes σ_a², the critic's own error after the coin.
2. **The semi-gradient phase did not cross-fit its corrections.** A critic fitted on the same game makes L_k correlate with that game's z, and the target's mean leaves V.
3. **States past π's stop are almost never visited.** With independent coins (30 per 225 decisions, ε = 0.2), each further act past the stop has probability about 0.026: depth 3 is visited 1,500 times less often than depth 1, and the stop rule's gain lies there.
4. **The Longstaff–Schwartz fit has no planted-signal check.**
5. **The averaged student was calibrated member by member.** Averaging calibrated predictors pushes the slope above 1, and no third split exists that both students and both critics missed.
6. **The record already sets open numbers:** numerics agreement, the value of an extra act, the power of the −2.5 known answer.

## 1. The algorithm

### 1.1 Generation

As current §2.1 (argmax on both sides, a schedule S drawn at game start apart from the dice, weak coins keeping a0 with probability 0.8, a quarter of games coin-free), with two changes.

**Run-length stop coins.** When a stop coin deviates to "continue" where π ends the turn, its schedule entry holds K, uniform on {1, 2, 3, 4}: the side plays a*, π's best non-end act re-decided after every roll, for K acts (fewer if none is legal or L leaves the window), then returns to π. The critic reads the forced acts remaining as part of S, so the identity holds (I12), and depth d is visited at rate (5 − d)/4 per deviating stop coin. An extra act past the −1.5 stop was worth about zero on average (§1.3), so the cost is small.

**Targeted coins from round 2.** A deviating action coin plays the previous round's calibrated student's best alternative with probability ½, otherwise one of the prior's top three uniformly. Propensities are a fixed function of h through a frozen student, so classical randomization inference holds (adaptive propensities would need adaptive weighting, [Hadad et al. 2021](https://arxiv.org/abs/1911.02768)). Where the next switch rule is likely to act, the deviation probability rises from 0.067 to at least 0.1.

### 1.2 The critic

Inputs, outputs and identity as current §2.2. The memory is written from the trunk at each decision before the action (`wesnoth_ai/memory.py`), so every child of a decision or fight carries its parent's memories and needs no extra forward.

**Phase 1: cross-fitted corrected targets.** Games split into folds F1, F2. Critic j regresses C(x_t) on

    y_t = z − Σ_{k>t} L̂_k(C′),   C′ the other fold's critic (pass 0: the seven linear columns).

For any C′ independent of the game, E[y_t | x_t] = V_S(x_t), since each L̂_k has mean zero given its pre-event state (one draw of a sampled tail suffices). Hence E[(C − y)²] = E[(C − V_S)²] + E[(y − V_S)²]: the population minimiser is the L2 projection of V_S on the critic class whatever C′ is, and C′ sets only the target noise Σ_k E[Var_o (V_S − C′)(x_k^o)]. A refresh lowers the noise; it cannot move the fixed point or diverge, so the convergence concern of global-in-time losses ([Hu & Laurière](https://arxiv.org/abs/2303.10257)) does not apply to this phase. Two refreshes, each 1,600 forwards per game.

**Phase 2: the joint residual (optional).** The double-sampled Σ_t R̂¹_t R̂²_t of current §2.2 adds Σ_k E[Var_o e] to the loss: the error differences across chance and coin siblings, which the look-ahead's advantages need. It replaces the second refresh only if, on a 10,000-game pilot, it lowers held-out ρ_C by at least 0.02 and lowers no type's κ.

### 1.3 End_turn

As current §2.3 (forced paths, payoff u_t = C(x_t∘end) luck-corrected per fight, a fogged Longstaff–Schwartz head D scored as played on held-out paths, no offset in the decode). Additions:

- **Cached features:** the head reads four token outputs per path state (about 2 KB), so the trunk runs once for all passes.
- **Privileged twin:** the same fit on the true state; the fogged-to-twin gain ratio is the share fog leaves.
- **Dual bound:** the path's luck terms M_t are a martingale, so any stopping rule's value is at most E[max_t (u_t − M_t)] (Rogers 2002; [Haugh & Kogan 2004](https://pubsonline.informs.org/doi/10.1287/opre.1030.0070)). Read on the same paths, it bounds any rule's gain over π, privileged rules included.

**Sizing from the record** (`docs/endturn_offset_sweep_prereg_20260919.md`, `terrain`, each offset against offset 0, about 20 turns a game). Acts per side-turn 5.90, 9.25, 11.03, 12.85 at offsets 0, −1.5, −2.5, −4; z = 0, 0.578, 0.602, 0.508. Per extra act: +0.0086 ± 0.0004 from 0 to −1.5, +0.0007 ± 0.0011 from −1.5 to −2.5, −0.0026 ± 0.0011 from −2.5 to −4. The mean crosses zero near L = 2: c* ∈ [1.5, 3.0] is worth 0 to +10 Elo over −1.5, and the learned rule gains over c* only through heterogeneity. A fogged spread of D about 0.003 over about 30 boundary decisions per side-game gives 30 · E[max(D, 0)] ≈ 30 · 0.4 · 0.003 = 0.036, about +12 Elo (band +5 to +25).

**The −2.5 known answer is informative.** A corrected 3,200-game match has SE 0.014 in z; with the forecast's SE about 0.01, two combined SE are 0.035 over about 36 extra acts per side-game (`terrain` counts), so a mean critic bias above 0.001 per over-continued act fails it. It cannot see error correlated with the rule's choices; the stop-only match can.

### 1.4 Attacks, moves, recruits

As current §2.4 (top-4 look-ahead at an empty schedule, folds, κ gate on a lower bound above zero, posterior ≥ 0.8, at most 10 switches per side-game). Additions:

**The deployed average, calibrated cleanly.** With X_j student j's centred coin contrast and T the truth, Cov(T, X̄) = ½[Cov(T, X_1) + Cov(T, X_2)]. Each term is estimated on the only coins its student and critic never saw (F2b for student 1, F1b for student 2), and Var(X̄) needs no results:

    β_avg = [Ĉov_{F2b}(T, X_1) + Ĉov_{F1b}(T, X_2)] / (2 Var̂(X̄)).

The folds are random, so each covariance estimates the population one; no symmetry is assumed (I14).

**Privileged twin student:** the same head on the true state; R²_fog/R²_priv separates the fog share of the advantage from the head's capacity.

### 1.5 Measurement

A round's gain and each factor's are read by **corrected matches**: 3,200 games against `own1` under the reference procedure, each game's win and loss indicators corrected by b̂·Σ_k L_k(C), with C frozen before the match and b̂ cross-fitted over its halves (the `tools/match_luck_readout.py` convention), the decisive score (W − L)/(W + L) by the delta method. This is unbiased over all games for any fixed C and b, and consistent for the decisive estimand; the mean correction's t statistic checks the logged laws for free. C and the procedure are fixed before the games, since a value tuned on evaluation games can fake precision ([AIVAT pathologies, 2026](https://arxiv.org/abs/2605.14261)). Strength claims stay PURE at 2 SE.

Ĝ_plug (current §2.5) is read on the match's own states with results sealed until written, so forecast and match are paired. Stop-only and actions-only matches give each factor; their interaction is π_1 − stop-only − actions-only (SE about 8.6 Elo). A mixture match, a fair coin at each disagreement, reads the curvature J(½) − ½J(1) (SE about 5.6 Elo). A miss reads: corrected apart from PURE, the laws or bookkeeping; match apart from Ĝ_plug, critic bias or tail miscalibration; Ĝ_plug apart from Ĝ_old, covariate shift.

### 1.6 Rounds

Round k+1 generates 15,000 coin games of π_k: half self-play, a quarter against `own1`, a quarter against a uniform draw of π_1 … π_{k−1} (`own1` in round 2). Responding to the opponents' history makes fictitious play's time averages converge in two-player zero-sum games (Robinson 1951). Naive policy iteration has no such guarantee, and its convergent variant needs a minimax look-ahead deep enough that α^{H−1}(1 + 2(1 + α^m)/(1 − α)) < 1 ([Winnicki & Srikant](https://arxiv.org/abs/2303.09716)); one step does not meet it, so the mix is a hedge.

Incumbent: the next round starts from π_{k+1} when its corrected gain over π_k (the two matches against `own1`, SE about 7 Elo) gives P(gain > 0) ≥ 0.9 and the corrected and PURE readings agree; otherwise from π_k. Distillation as current §2.6.

### 1.7 Signal per dollar

- **Critic:** each game's 160 chance events are experiments with known laws. Phase 1 lowers target noise from Var(z) = 1 to ρ_C(C′) (1.5 times result regression's effective games after one refresh), and the identity adds about 2.5 χ² per game on chance-sibling differences that result regression never sees.
- **Coins:** sixty weak coins in a game carry about 60 × 0.35 s² of contrast variance; a pair of forks costs about a game for 0.5 s².
- **Student:** labels are the critic's look-ahead at any state, with no result: about 2 million for $1, against about 19,000 games per decision context for outcome policy gradient at a 2-point effect.

### 1.8 Invariants

I1–I7 and I9 stand; I8 stays for P0. New, on an enumerable duel: **I10**, the enumerated mean of y_t equals V_S(x_t) for a constant, a random network and V_S as C′, while a C′ fitted on the same sampled games fails; **I11**, the corrected estimator's mean is exact for any fixed C and its variance is E[R_0²](C); **I12**, R_t(V_S) = 0 with one run-length coin. **I13**: on enumerable paths the dual bound is at least every adapted rule's value. **I14**: on a synthetic population, β_avg equals the slope of truth on the average.

## 2. Regimen

Prices as current §3, plus: a critic-data game $2.1e-4 (generation, memory recompute, two refreshes of 1,600 forwards); a corrected 3,200-game match $0.7 (eval numerics, recompute, about 1,150 forwards a game).

| Stage | Cost | Predicted readout | Kill line |
|---|---|---|---|
| **S0 instruments** | $3.5 | (a) Agreement within generation (a1) and against eval numerics (a2), 2,000 games: ≥ 98% each; the record reads 98.2% (27,171 `parity3` decisions replayed, `critic_step1_20261009/prior_gaps.json`), and 2% unmodelled flips add about 3e-4 of Var(z) (8 a game, each ≤ s²/4). (b) Match A, `own1` at offset 0 against −1.5: E[z] in [−0.60, −0.25] (−0.58 under `terrain`); Match B, `own1` playing its second-ranked joint action wherever the top one is an attack: [−0.7, −0.2]; 3,200 games each. (c) **P0**: arcsine coins between the decodes of A (stop coins) and of B (attack coins), linear corrections, 25,000 games each, each reproduces its match within 2 combined SE (SE 0.07). (d) Weak coins with run-length stop coins on side A against `own1`, 3 and 9 deviations per side-game, 20,000 games each: cost 0.01–0.04 and 0.03–0.12, linear within 2 SE. | (a) below 95%: batch-invariant kernels (+$7). (c) outside: logging, propensity, weight or side sign is broken; nothing downstream runs. Density: the largest of 3, 6, 9 costing ≤ 0.06. |
| **S1 critic** | $16 (+$10.5 conditional) | **P1** as current (15,000 games). Then 50,000 `own1` games. ρ_C: pass 0 at 0.70–0.85, pass 1 at 0.58–0.75, pass 2 within [−0.04, 0] of pass 1, the final value at least 0.05 below the same games' seven-column ratio (0.83 on record); points at 10,000, 20,000 and 50,000 games. κ: attack 0.6–1.0, recruit and stop 0.4–1.0, move 0.0–0.6, flat in remaining coins. C(x_0; S) − C(x_0; ∅) equals S0(d)'s cost within 2 SE. **Extension:** if the curve's power-law fit predicts ρ_C(100k) ≤ ρ_C(50k) − 0.03, buy 50,000 more games. | P1 outside: that path is broken. ρ_C not 0.05 below the linear ratio, or every κ lower bound ≤ 0: fallback F. |
| **S2 operators** | $6 | **P2** as current. **P3** (new; CPU, on S1's recorded paths): (i) a planted payoff of 0.05 per act past π's stop when the side's gold is even sends the rule to the path's end on ≥ 95% of those turns, ≥ 97% agreement with the unplanted fit elsewhere; (ii) at 0.003 the switched share matches the one-pass prediction from the unplanted head within 2 SE; (iii) a planted mean-zero term at each fight leaves ≥ 99% agreement; (iv) with the bit hidden (a per-game coin only the critic sees), the fogged rule moves equally in both arms and the twin splits. **Stop:** offset 0's forecast matches Match A, −2.5's a fresh corrected `own1`(−2.5) match (record −0.035 ± 0.035), each within 2 combined SE; c* ∈ [1.5, 3.0]; LS stable within 8 passes; rule over c* +5 to +25 Elo; dual bound minus achieved 0.01–0.06 per side-game; fog-to-twin gain ratio 0.3–0.8. **Actions**, per open type: student R² 0.2–0.6, the twin's above it; β_avg 0.5–1.0; held-out gain over the nested model above 2 SE. | P2 or P3 outside: stop. A stop known answer missed or κ_stop's lower bound ≤ 0: the stop factor leaves round 1. Rule over c* under 2 SE: round 1 stops at c*. β_avg < 0.3: F on coin labels. |
| **S3 round 1** | $5 | Ĝ_old and Ĝ_plug per factor written first. Corrected matches against `own1`, SE about 5 Elo each: π_1, stop-only, actions-only, `own1` at c*; π_1 against `parity3`; the mixture match. Sizing prior: stop-only +5 to +35, actions-only +5 to +30, interaction within ±10, curvature within 2 SE of 0. | Adopt by the incumbent rule; claim at PURE 2 SE. Match outside 2 combined SE of Ĝ_plug: diagnose by §1.5. Stop-only not 2 SE above c*: the rule is a constant. Gain over `parity3` under half that over `own1`: exploitation. |
| **S4 rounds 2 and 3** | $10 each | P1(i) on the warm refit (3,000 games); 15,000 coin games with targeted coins and the opponent mix; refit; S2–S3 with matches against `own1` and every earlier round. Predicted gain 0.5 times the previous round's corrected gain (band 0.2–0.8; not derived). | P1(i) outside: refit path broken. Forecast and match both under one match SE: S5. |
| **S5 distillation** | $5 | As current. | As current. |

The planned path (S0–S3, both S4 rounds, S5) costs about $55.5: three rounds where the audit paid for two. The S1 extension adds $10.5 and kernels, if S0(a2) fails, $7: at most $73 of the $85.

**The $1,000 case** is argued only after two rounds land within 2 SE of their forecasts. With a per-round decay ratio r the rounds total G_1/(1 − r), so beyond-human strength needs critic refits that restart the gains: 200,000-game refits (κ's SE halved), 25 to 40 rounds with distillation every three, a league check each round. Its prediction is the measured r after a refit, with its error. No measured link from `own1` to human ratings exists.

## 3. What is new, and the nearest work

Searched 2026-10-19; finding nothing does not prove absence.

1. **The identity-trained privileged critic for a fogged player:** schedule in the state, run-length coins, and a cross-fitted phase whose fixed point does not depend on the correcting critic. Nearest: Off-policy DAE ([Pan & Schölkopf 2024](https://arxiv.org/abs/2402.12874)), [AIVAT](https://arxiv.org/abs/1612.06915), [MIVAT](https://www.ijcai.org/Abstract/09/326), the hindsight baselines of counterfactual credit assignment ([Mesnard et al. 2021](https://arxiv.org/abs/2011.09464)), cross-fitting in double machine learning ([Chernozhukov et al. 2018](https://arxiv.org/abs/1608.00060)), deep BSDE solvers.
2. **Turn stopping as optimal stopping on forced paths** with run-length coins, a fogged LS rule, a privileged twin and a pathwise dual bound whose martingale is the critic's luck terms. Nearest: Longstaff–Schwartz, Andersen–Broadie's primal–dual method, [deep optimal stopping](https://jmlr.org/papers/v20/18-232.html); I found no game-turn application.
3. **Corrected matches with an identity-trained privileged critic, per outcome, as a policy-iteration step's instrument.** Nearest: AIVAT, MIVAT; an application of modest novelty.
4. **Calibration of a cross-fitted pair by pooled covariances.** Probably known in DML practice; I found no statement of it.

## 4. The weakest point

**Decision differences.** S1 carries 140 to 320 χ² on decision differences against about 125,000 on chance differences; from round 2, targeting raises the coins at switch points 1.5 to 2 times, and the extension doubles S1. Move switches rest on generalisation from chance siblings and state levels; if κ_move does not open, round 1 changes stops, attacks and recruits, with a prior of +10 to +45 Elo. ρ_C still has no first-principles value.

**For the user:** κ, β and fallback F regress outcomes on coins. Here they only calibrate and gate, since the critic's look-ahead and the path payoffs move the policy; whether that falls under the killed "outcome policy gradient" is the user's call.

## 5. Decision record

- Rejected: the arcsine audit as the round's instrument (defect 1). It stays in P0, where a known answer certifies the coin logs; turn coins go with it.
- Rejected: corrections computed by a critic fitted on the same games, because their mean is no longer V.
- Rejected: within-round adaptive coin propensities. Fixed targeting already puts half the deviation mass on the proposed switch, so adaptation can at most double it, and it would need adaptive inference.
- Deferred: intents and swing layers as current §6, now triggered by the factor interaction or the mixture curvature exceeding 2 SE.
