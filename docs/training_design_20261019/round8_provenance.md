# Round 8, provenance stance: decision rules that hold whatever the shape of the near-tie advantage

Tags: **[M]** measured, **[D]** derived, **[U]** unknown, **[C]** a choice with derived consequences. Scripts: `analysis/round8_provenance_{attack_type, interim_shapes, knob_checks, rules, threshold, label_variance, spread_estimator, plan}.py`, outputs beside them.

## Summary

The core of `current_design.md` stays: near-tie coins with logged propensities, labels exactly unbiased through the chance laws, a closed-form head on frozen `own1`, and adoption only by PURE match.

Four of its rules (the interim stop, the sign rule, E's share, the reading of a null) rest on round 5's Gaussian model of the near-tie advantage, which nothing measures. Each now reads a model-free statistic or has a property proved for every shape:

1. **One pair type has a negative mean.** By the performance-difference lemma, the decode-knob matches show that an attack-promoting switch at gaps below 0.35 nats averages at most −0.0024 z (2 SE, three fits).
2. **The interim stop goes.** In that regime, changing only the residual's shape in the design's own model gives false stops up to 0.50.
3. **The switch threshold is tested.** The sign rule f > 0 is the best threshold only for a Gaussian advantage; two-point shapes lose 0.41–0.72 of the first-order gain. The student switches where f > τ̃, with τ̃ = 0 unless a held-out threshold beats 0 by 2 SE: at most 0.1 Elo lost for Gaussian or Laplace shapes, up to 13.7 gained elsewhere.
4. **Shape-free readings.** E keeps at least half the first-order gain; a null reads an unbiased spread estimate and Scarf's cap on any in-span rule, and exhausts the band only below the adoption line; label variance is capped at 5.33 exactly; S4 predicts a one-sided bound.
5. **S0's known answer holds out of sample:** −76 ± 17 Elo by a knob-only fit against −78 ± 16; leave-one-out RMS 0.83.

The plan costs $56.18 of $85.

## 1. What the records settle

### 1.1 The attack type's mean

Each vector of `training/metrics/decode_knobs_20261012` is `own1` (64 slots) under an offset decode, playing `own1` at its own decode. A positive attack offset o switches exactly the decisions where a non-attack leads and the best attack lies within o.

By the performance-difference lemma ([Kakade & Langford 2002](https://dl.acm.org/doi/10.5555/645531.656005)), the knobbed side's mean result moves by the sum of those switches' advantages under `own1`'s play. The 2,400 unselected stage-1 games give the move; the prior-gaps record bounds the switch count by F(0.35) = 16.5 and F(1.0) = 45.1 per `own1` side-game. The mean per switch is at most move/F(o), at the upper 2-SE move:

| Fit | Gaps < 0.35 | Gaps < 1.0 |
|---|---|---|
| Decision counts as covariates (round 6) | −0.0028 | −0.0029 |
| Knob-only, linear | −0.0026 | −0.0027 |
| Knob-only, quadratic | −0.0024 | −0.0030 |

Point estimates are −0.0046 to −0.0078. Demoting attacks reads −0.12 ± 0.08 z per nat; ending turns earlier costs 0.0033 ± 0.0017 per decision removed; every knob around today's decode loses or is flat. The base sits where typed means are at most 0: there round 7's √N ceiling fails, and the residual's shape decides how gain grows with labels.

### 1.2 What the shape decides, in the design's own model

Round 5's model, behind every floor: per pair type a = c + r, c the unpenalised typed mean, r the in-span residual (variance v); the ridge predicts f = c + κ(r + e), e ~ N(0, η_k), κ = v/(v + η_k). Only r's shape varies (Gaussian, Laplace, two-point with positive share 0.02–0.95). Over cases where student 8 is adoptable at 80% power (J8 ≥ 5.1 Elo):

| Rule | Stated | Gaussian | Laplace | Worst two-point |
|---|---|---|---|---|
| Interim false stop, c ≥ 0 | ≤ 0.041 | 0.028 | 0.030 | 0.028 |
| Interim false stop, c < 0 | ≤ 0.15 | 0.194 | 0.136 | 0.497 (J3 0.0, J8 5.4) |
| Sign rule ÷ best threshold | within 0.6 Elo | 1.000 | 0.958 | 0.283 (5.6 against 19.7 Elo) |
| E's share of first-order gain | 0.80–0.87 | ≥ 0.505 | ≥ 0.505 | ≥ 0.500 |

The interim's worst cases sit at c −0.0038 to −0.0078, inside §1.1's range: when a minority of decisions carries the gain, student 3 cannot find them and student 8 can. The sign rule fails because linear shrinkage is the posterior mean only for Gaussian r; otherwise E[a | f] crosses 0 away from f = 0, on either side. `own1`'s shape is unmeasured; probing finds move information in game networks in bilinear form ([Jenner et al. 2024](https://proceedings.nips.cc/paper_files/paper/2024/file/37d9f19150fce07bced2a81fc87d47a6-Paper-Conference.pdf)), not the shape of small value gaps.

### 1.3 The known answer's window

A knob-only fit (no response noise in its regressors) predicts S0(b)'s attack +1 on one seat at −0.2178 ± 0.0487 z (−76 ± 17 Elo), against round 6's −0.2247 ± 0.0459. Leave-one-vector-out residuals have RMS 0.83 (max 1.74): the SE holds out of sample. The window is the union: [−111, −41] Elo; [−0.640, −0.236] z through the label code.

### 1.4 What the records cannot settle, and what holds before any record

The records do not settle v or `own1`'s census (parity3's share gives 76.2 ± 1.8 per side-game, sampling only), nor the second-order term: knob interactions read +28 ± 19 Elo at one nat of attack and end_turn offset, too coarse for the design's scale. Three things hold without records:

- **Label variance has an exact ceiling.** With capped games dropped, z² = 1, so the label without variates has E[T² | H] = 1/(P(1 − P)) = 5.33 exactly and labels within a game are exactly uncorrelated (`label_variance`). Every SE has a worst case (8 batches at 5.33 equal 5.50 at 3.67).
- **The spread has a shape-free estimator.** With ŵ_j the least-squares fit of batch j alone (from its logged Gram matrix), the mean over batch pairs j < k of ŵ_jᵀSŵ_k (S the within-type covariance of x) is unbiased for v = Var(wᵀx) under any law, per type with that type's S (`spread_estimator`; nearest: [Kong & Valiant 2018](https://arxiv.org/abs/1805.01626)). After 8 batches a null bounds v ≤ 2.5e-6 at 2 SE.
- **Scarf's cap.** Any in-span rule's first-order gain is at most (m + √(m² + v))/2 per decision, attained by a two-point law ([Gallego & Moon 1993](https://doi.org/10.1057/jors.1993.141)): over the band, 21 Elo at m = 0, 7.3 at −0.002, 2.7 at −0.006.

## 2. The algorithm

**What is learned.** A switch head on frozen `own1` predicts, at a near-tie, the runner-up's advantage over the top choice:

    f = c_uv + b_uv·G + ŵᵀx,   x = φ(a1) − φ(a0)

φ concatenates the actor token, q(actor) ⊙ k(target hex) and the four factored log-probabilities (d = 772); typed terms are unpenalised and antisymmetric. The network picks every action; the head reorders two candidates at one decision, and play re-decides after every roll.

**Labels.** π_0 is `own1` at `raw:t0+eo-1.5` with 64 slots, playing both sides in `run_elo_batch` at 120 jobs.
- **The coin.** Where the post-offset top-two gap G < w = 0.35 [C], a coin keyed by seed and decision index plays a1 with P_i = 0.25 where π_k plays a0, 0.75 where it plays a1 [C].
- **The label:**

      T_i = s_i ω_i R_g,   ω_i = (D_i − P_i)/(P_i(1 − P_i)),   R_g = z − b(cell) − βᵀS_g

  z is side A's result, s_i the mover's sign; b (the setup cell's mean) and β (on the seven luck sums S_g) are fitted on earlier batches and frozen before each batch ([Hadad et al. 2021](https://arxiv.org/abs/1911.02768) §2.1).
- **Bias.** E[T_i | H_i] = Q̃(H_i, a1) − Q̃(H_i, a0) under the coin policy:
  - b and the earlier luck are fixed by H_i;
  - each later luck term has mean zero given its pre-fight state, which contains D_i;
  - E[(D − P)z | H] = P(1 − P)(E[z | 1, H] − E[z | 0, H]).

  Under fog this is the information-state advantage.
- **Variance.** Var T ≈ Var(z)·r/(p(1 − p)) = 3.67 (4.48 in batch 1).
- **Capped games** are dropped, exact iff capping does not depend on the coins.

**Update.** The head is a closed-form ridge from per-game statistics: ŵ = (G + λI)⁻¹ Σ_g R_g A_g, with A_g = Σ_{i∈g} s_i ω_i x_i. λ̂_k minimises the progressive squared loss, since batch k + 1 runs with student k logging f.

**Switch rule (new): a tested threshold.**
- **Welfare curve.** For each progressive pair (student j − 1, batch j), W_j(τ) = Σ_{i: f_i > τ} T_i estimates the rule {f > τ}'s first-order gain without bias for every τ, the propensities being known: empirical welfare maximisation's (EWM) setting, regret O(n^−1/2) uniformly ([Kitagawa & Tetenov 2018](https://doi.org/10.3982/ECTA13288)).
- **Pre-test.** τ̂ maximises W summed over student k's latest four pairs; τ̃_k = τ̂ iff W(τ̂) − W(0) exceeds 2 SE of their difference, else 0. Only batches up to k enter, so the forecast stays out of sample.
- **Why.** The plug-in cut at 0 is optimal when the regression is the posterior mean; EWM is right for any shape but pays selection noise; the pre-test keeps the plug-in unless the data reject it ([Mbakop & Tabord-Meehan 2021](https://doi.org/10.3982/ECTA16437): penalised welfare selection).
- **Cost.** One float logged per band decision.

**Fallback E** keeps the top half, by f, of τ̃'s switches. Gaussian estimation noise gives a monotone likelihood ratio, so E[a | f] is nondecreasing for any distribution of a and E keeps at least half the first-order sum. Its curvature saving is [U] (a quarter if interactions are homogeneous).

**Forecast, match, adoption.**
Ĝ_pf sums batches 5–8's labels at students 4–7's switches (each at its τ̃_k), per side-game, × 347.4, sealed before the match; its SE and miss line (about 4.6 and 9.9 Elo) come from the logs. The match: the incumbent with the head on one seat, 37,800 independent decisive games on fresh seeds, SE 1.79, $0.72; adopt iff the lower 2-SE bound is above 0 (false adoption 0.023; 0.046 with E); against `own1` it is the PURE claim. A miss with the forecast 2 SE above 0 triggers E; it is curvature or the opponent's coins, Differences-in-Q's truncation ([Farias et al. 2022](https://arxiv.org/abs/2206.02371)).

**Signal per dollar.** One result prices about 152 labels with no playout or critic: about 41 information units per game-equivalent against 0.82 for a fork pair. Every batch trains (a held-out batch forecasts at SE 9.2 Elo for $1.91, a match at SE 1.79 for $0.72).

**Invariants.** I-coin: coins match P, also within capped games, independently of the dice. I2: an enumerable duel through the production generator gives E[T] = Q(a1) − Q(a0) exactly, and a post-coin variate fails. I3: a swap flips x, T and the typed terms. I6: the statistics fit equals the per-decision fit. I-log: logged f, switches, W_j and forecast sums equal recomputation. I-φ: features reproduce the logits. I-V: once a round, 5% family-wise.

**Rounds and reset** as in `current_design.md` §3.6: round 2 by default after an adopted round 1 [C]; S4's distilled N plays at o_k(1 − ĉ).

## 3. Regimen

Prices: $1.912e-5 a pool game [M], $1.17e-4 a two-network game [M, upper bound]; a batch costs $1.91 and 4.55 h. Stage costs are from `round8_provenance_plan.out`, which reproduces `round8_simplify_power.out` §2; this stance adds no compute.

| Stage | Cost | Readout and prediction | Decision |
|---|---|---|---|
| **S0** | $0.14 (+$0.04 of batch 1) | **(b)** Attack +1 on one seat, 6,400 games: [−111, −41] Elo, [−0.640, −0.236] z (wrong side: +0.44). **Smoke** (batch 1's first 2,000 games): I-coin, I-log, I-φ, plant (SE 0.039), census (76.2 ± 1.8 predicted), overhead | **Outside a window:** fix the match path or orientation. **Smoke fails:** debug. **Else:** restate costs; S1 |
| **S1** | $15.30 | **Batch 1:** plant per side (SE 0.0055); ridge plant at uᵀ(G + λI)⁻¹Gu. **Batches 1–2:** threshold plants. **Report:** (non-attack, attack runner-up) mean label, ≤ −0.0024 if §1.1 transfers (SE ≈ 0.0015). **Batch 8:** Ĝ_pf sealed; v̂, c, b, q̂, τ̃_8. v is not predicted | **Plant fails:** debug. **Price × (1 + overhead) > $3.11e-5:** two rounds (to $4.81e-5) |
| **S2** | $0.72 (+$0.72 E) | π_1 against `own1`; Ĝ_pf and its miss line | **Lower bound > 0:** adopt (the claim); S3. **Miss, forecast positive:** E. **Both null:** S3-null, reading play gain < 3.6 Elo (2 SE), Ĝ_pf, and Scarf's cap summed over types at each type's upper v̂ and measured mean |
| **S3** | $16.74 | Coins on π_1; π_2 against π_1 (adoption) and `own1` (claim); its own Ĝ_pf | As S2; S4 after the optimizer discussion |
| **S4** | $3.60 | ĉ, ρ̂, off-band disagreements n_out, held out first; N against π_k, 12,800 two-network games (SE 3.07). **Prediction:** J ≤ −(1 − ρ̂)Ĝ_k + Σ n_type μ̂_type, off-band flips being nonpositive in mean (argmax beats sampling by +412 ± 97; §1.1's families are negative or flat); the shortfall ÷ n_out prices one | **ĉ not above 0 by 2 SE:** distillation broken. **Estimate ≥ 0:** N is the base. **Otherwise:** keep π_k |
| **S5** | $17.52 | A round on N; two-network claim against `own1` | As S2 |
| **S3-null** | when chosen | The readings above | **(i)** w 0.7 (125 band decisions); **(ii)** 8 more batches, only if Scarf's cap exceeds 3.6 Elo (else the span is exhausted); **(iii)** E from the logs; **(iv)** critic column (~$7); **(v)** one-seat coins; **(vi)** stop |

**Plants, one per code path:**
- **Label and fit.** z′ = z + Σ_j s_j^rec(D_j^rec − P_j)(u·x_j), s^rec and D^rec read from the record by separate code. Prediction: slope 1 per mover side; a wrong side sign reads −1.005, a coin one decision off 0.016.
- **Threshold** (offline, batches 1–2). A step a = A·1(u′·x > q) − b in a separate copy of z; its oracle τ* is computed from the known a before τ̃ is read, sized to beat 0 by 10 SE. Prediction: τ̃ ≠ 0, planted welfare within 2 SE of τ*'s. Negative control: the linear plant moves τ̃ off 0 with P 0.04–0.10.
- **Match and orientation:** S0(b).
- **Distillation:** the baked end_turn shift; ĉ must be above 0 by 2 SE.

**Budget** [D]: $56.18, leaving $28.82 (15.1 more batches); a null round 1 costs $16.16; on a 1.41× slower host, $77.12, three rounds fitting at up to 12% overhead. **Branch L** (ruling (a) no): unchanged, about $4.35.

**Toward human level:** coin rounds and resets, each certified by match; the $1,000 case is sized after two adopted rounds from measured v̂, J/Ĝ_pf, ρ̂ and τ̃'s moves. Nothing links `own1` to human ratings, and nothing rests on it.

## 4. What is new, and the nearest work

Searched 2026-10-20; finding nothing does not prove absence.

1. **A pre-tested welfare threshold on a plug-in index,** with known propensities and forward chaining: optimal under correct specification, robust otherwise. Nearest: EWM ([Kitagawa & Tetenov 2018](https://doi.org/10.3982/ECTA13288)) and penalised welfare maximisation with hold-out selection ([Mbakop & Tabord-Meehan 2021](https://doi.org/10.3982/ECTA16437)), abstracts read; [Athey & Wager 2021](https://doi.org/10.3982/ECTA15732), recalled. New here: one threshold on an index fitted elsewhere, tested against the plug-in cut, in self-play.
2. **A pair type's mean advantage read from decode-offset matches** by the performance-difference lemma and a switch-count bound: elementary, but it measures the regime before any coin game. Likewise the cross-batch spread estimator with Scarf's cap gives a shape-free meaning to a null.
3. **Dropping a futility stop where gains can come late.** Trials know futility rules mislead under delayed effects and optimise the rule for an assumed delay ([Wang & George 2023](https://doi.org/10.1177/17407745231181908), abstract read); here the shape is unmeasured and money does not bind, so the stop goes.
4. **Carried:** the coin label's estimand ([Bottou et al. 2013](https://jmlr.org/papers/v14/bottou13a.html)), a micro-randomised trial at near-ties ([Boruvka et al. 2018](https://doi.org/10.1080/01621459.2017.1305274)) made exact by the simulator's laws ([AIVAT](https://arxiv.org/abs/1612.06915)).

## 5. Weakest point

- **The spread v is unmeasured,** and a null round 1 costs $16.16. This proposal makes the rules right for any shape; it does not make the round worth running.
- **The threshold analysis is single-cell** and selects at the deployed student's own f; the design picks one τ̃ for all pair types on earlier students' f, a transfer the threshold plant tests on batches 1–2 only.
- **§1.1 bounds the knob's whole switch set,** attacks ranked third to eighth included; the coin's runner-up attacks are a subset the record does not bound.

## 6. Rulings and decision record

**Rulings:** (a), (c), (d), (e) unchanged; (b) the luck columns only; **new (f):** may the threshold be chosen on held-out labels, the plug-in kept unless rejected at 2 SE?

**Rejected:** the interim stop (false stops 0.24–0.50 at c −0.0038 to −0.0078); λ chosen for welfare (0.547 of the oracle at worst); an untested welfare threshold (−0.6 to −1.1 Elo where the plug-in is right); the √N ceiling (the attack type breaks its premise); γV and the S0(a) replay (cost, not guarantees); exhaustion read from Gaussian floors (Scarf's cap replaces it).

## Provenance

| Number | Value | Tag | Source or derivation | Used for |
|---|---|---|---|---|
| Pool price, steady | $1.912e-5 a game, 21,966 games/h | M | `round7_provenance_price_tail.out` (arm ref_p120x6); the arm log names `own1.pt` sha 70925e77…, 64 slots, 120 jobs (read this round) | Every cost |
| Host spread; two-network price; box setup | 1.41×; $1.17e-4 (median of 18, upper bound); $0.018 | M | `round6_provenance_prices.out` | Slow host; S4, S5; S0 |
| Distillation | $2.1, upper bound | M | `docs/memory_own_trajectories_prereg_20261011.md` | S4 |
| Budget | $56.18, $28.82 left; S0 0.14, S1 15.30, S2 0.72, E 2.17, S3 16.74, S4 3.60, S5 17.52; null round 1 $16.16; three rounds iff price × (1 + overhead) ≤ $3.113e-5, two ≤ $4.808e-5; 1.41× host $77.12 | D | `plan.out` (reproduces `round8_simplify_power.out` §2) | Regimen |
| Var(z), own1 self-play | 1.0004 | M | `round5_simplify_variance.json` | σ² |
| Label residual | 0.688 (cell + columns; own1-only 0.651, 0.581–0.721); 0.839 columns alone | M | `round6_provenance_cells.out` | σ² 3.67, 4.48 |
| Label second moment, no variates | E[T²] = 1/(P(1 − P)) = 5.333 exactly; within-game cross moments 0 exactly; design effect with variates 1.002 (toy) | D | `label_variance.out` (algebra from z² = 1; toy check) | Worst-case SEs (8 batches at 5.33 = 5.50 at 3.67) |
| Spread estimator | unbiased for any law (simulated biases −0.066 to +0.051, ±0.05); SE at v 0 after 8 batches 1.27e-6, so a null bounds v ≤ 2.53e-6 | D | `spread_estimator.out` §1–2 | Null reading |
| Scarf's cap over the band | 21.0, 11.6, 7.3, 4.0, 2.7 Elo at m 0, −0.001, −0.002, −0.004, −0.006 (v 2.53e-6) | D | `spread_estimator.out` §2; bound checked on 20,000 random laws (max ratio 1.000) | Exhaustion rule |
| Ceiling premise certifiable | P 0.013–0.039 for slopes −0.004 to +0.004 | D | `plan.out` | √N ceiling rejected |
| Decisions per own1 side-game | 224.6 | M | `round5_simplify_critique_decision_units.out` | Census units |
| Band census | 76.2 ± 1.8 per side-game (sampling only; parity3, 100 side-games) | M | `attack_type.out` §2 | Smoke prediction; transfer [U] |
| Attack switch-count bound | F(0.35) 16.5, F(1.0) 45.1 per own1 side-game | M | `attack_type.out` §1 (prior-gaps record; agrees with `round7_invent_critique_checks.out` §1) | §1.1 |
| Attack-offset move (stage 1) | −0.2247 ± 0.0459 per nat (decision-count fit); −0.2178 ± 0.0487 (knob-only); at 0.35, −0.076 ± 0.017 linear, −0.128 ± 0.044 quadratic | M | `round6_provenance_critique_attack_stage1.out` B; `knob_checks.out` A, C | §1.1, S0 |
| Mean per attack-promoting switch | ≤ −0.0024 to −0.0028 (gaps < 0.35), ≤ −0.0027 to −0.0030 (< 1.0) at 2 SE; points −0.0046 to −0.0078 | D | move ÷ F(o), performance-difference lemma | Regime; batch-1 report |
| Attack demotion; ending earlier | −0.12 ± 0.08 per nat; −0.0033 ± 0.0017 per decision removed | M | `knob_checks.out` A; `round5_provenance_knob_switches.out` | Regime; S4's sign |
| Argmax against sampling | +412 ± 97 Elo | M | seed §3 | S4's sign |
| Leave-one-out | RMS 0.83, max 1.74 over 24 vectors | M | `knob_checks.out` B | S0's SE is honest |
| S0 windows | [−111, −41] Elo; [−0.640, −0.236] z | D | union of both fits ± 2√(SE_fit² + SE_read²), SE_read 0.0125 z (Elo) or 0.025 (label, `round7_synthesis_power.out` §1) | S0 gate |
| Knob interactions | +28 ± 19, +8 ± 12, +19 ± 32 Elo at one unit each | M | `knob_checks.out` D | Second order stays [U] |
| Interim false stop, J8 ≥ 5.1 | ≤ 0.030 at c ≥ 0 (all shapes); up to 0.497 at c < 0 | D | `interim_shapes.out` (the design's model, shape varied) | Interim removed |
| Sign rule ÷ oracle threshold | 1.000 Gaussian, 0.958 Laplace, 0.283–0.589 two-point | D | `rules.out` | Tested threshold |
| λ chosen for welfare ÷ oracle | ≥ 0.547 | D | `rules.out` | Rejected |
| E's first-order share | ≥ 0.500, every shape (Gaussian 0.505–0.826) | D | `rules.out`; monotone likelihood ratio of a log-concave location family | E |
| Tested threshold | Gaussian, Laplace: within 0.1 Elo, leaves 0 in 0.04–0.10; two-point: 5.5 → 19.2, 4.9 → 10.8, 7.8 → 13.0, 6.1 → 8.6 Elo; untested: −0.6 to −1.1 in Gaussian cases | D | `threshold.out` (4 batches; 7 beside it) | Switch rule; the threshold plant's control |
| σ² | 3.67; 4.48 in batch 1 | D | 1.0004 × r ÷ 0.1875 | Plants, SEs |
| Elo per unit z | 347.4 | D | 800 ÷ ln 10 | All |
| Match SE; false adoption | 1.79 (37,800 games), 3.07 (12,800); 0.023, 0.046 | D | 347.4 ÷ √N; Φ(−2); two candidates | Adoption |
| Forecast SE; miss line | about 4.6 and 9.9 Elo at μ 0, recomputed from logs | D | `round6_synthesis_power.out` §4 | S2 |
| Held-out batch | SE about 9.2 Elo for $1.91 | D | 347.4 × √(3.67 × 38 ÷ 200,000) | Every batch trains |
| Plant SEs | 0.0387 (2,000 games), 0.0055 (100,000) per side | D | `round8_simplify_power.out` §4 | Smoke, batch 1 |
| Attack-type label SE | about 0.0015 in batch 1 at 10 pairs per side-game | D | √(4.48 ÷ (200,000 × 10)); the pair count is [U] until the smoke | Report |
| Information per game-equivalent | 41 against 0.82 | D | 152.2 × 0.1875 ÷ 0.688; round 5 | Signal per dollar |
| Caps; numerics flips | 0 of 1,600; 1.76% at gaps < 0.014 | M | `round6_simplify_walk_cost.out`; `round5_provenance_numerics_ties.out` | Cap rule; logged propensities |
| Residual offset; offset 0 | o(1 − ĉ), −0.27 at ĉ 0.82; −9.9 ± 4.7 Elo | D | `round7_synthesis_power.out` §4; `round7_simplify_power.out` §6 | S4, ruling (d) |
| Tempo closure | 1.00 (0.82–1.23), 1.06 (0.88–1.30) | M | `round6_provenance_closure.out` | ĉ indicative only |
| Band at w 0.7 | 125 per side-game | D | 0.558 × 224.6 | S3-null (i) |
| Spread v; typed c, b; switched share q | — | U | S1's progressive fits; nothing before S1 relies on them | Round gain |
| Shape of the in-span advantage | — | U | No record; every rule here holds for any shape | Nothing |
| own1 census, runner-up mix, attack-pair count | — | U | The smoke | Restated costs |
| Logging overhead | — | U | The smoke against S0(b), same host | Rounds that fit |
| Second-order and excursion loss | — | U | S2 against Ĝ_pf; E's match | Miss reading |
| Transfer of τ̃ across students and types | — | U | Threshold plant (batches 1–2); τ̃'s moves logged each round | Weakest point |
| ĉ, ρ̂, n_out, price of an off-band flip; pool two-network price | — | U | S4, before and at its match | Reset |
| Compounding, exploitability, human level | — | U | Each round's claim; none; none | Nothing |
| w 0.35, p 0.25, 8 × 100,000 games, d 772, four-pair window, 2-SE pre-test, matches of 37,800 and 12,800 games | — | C | Consequences derived above | Design |
