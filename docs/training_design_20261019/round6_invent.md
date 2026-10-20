# Round 6, invent stance: dice-priced innovations

Tags as in `current_design.md`: [M] measured, [D] derived, [U] unknown, [C] choice. Scripts are in `analysis/`.

## Summary

1. At argmax, a coin game's result is the setup's mean plus the innovations of its chance events and coins (numerics flips aside). The core enumerates every chance law, so each chance innovation can be priced, centred exactly and removed from every coin label without bias.
2. The invention is a price head that values every outcome of a fight or a trait draw from the parent decision's own forward. It is fitted by regressing the result on exactly centred outcome features. The dice are randomized with known laws, so the fit is unconfounded: about 110 fights and 46 recruits' trait draws a game are free experiments, with no deviation and no extra forward.
3. It changes one number of the current design, the kept variance r (0.732). A round's cost to reach a floor scales with r, the forecast's SE with √r, and the match's SE with the square root of its own kept ratio.
4. A $0.6 reading on the 18,800 recorded games measures r before anything is built on it. The record does not predict r. Hand-made prices explain at most 0.152 of the variance, and conditioning on one parent scalar removes 0.043 ± 0.004 more than none.
5. The same prices give each attack candidate its exact-law expected priced outcome: one coin-calibrated student column with an internal known answer.

## 1. What a coin game's result is made of

At argmax both players are deterministic. With the setup fixed, a coin game has four random inputs:
- fight outcomes, whose law the core enumerates (`GameCore.attack_outcomes`);
- trait draws, uniform over the pool's untaken entries by `rng.draw() % count` (`rust/wesnoth_core/src/units.rs`, `roll_traits_with`), so the exact law includes the modulo;
- the coins, with logged p_i;
- numerics flips.

With ξ_k = E[z | after event k] − E[z | before k], Doob's decomposition gives

    z − E[z | setup] = Σ_k ξ_k + Σ_i s_i (D_i − p_i) Δ_i + ν

Here ν holds the numerics flips' effects. The terms are martingale differences, hence uncorrelated, so perfect pricing would leave label noise Σ_i p_i(1 − p_i)Δ_i² + Var(ν). Today the setup explains 0.103 of Var(z), and the label keeps r = 0.732 [M]. So about 0.73 of Var(z) is chance the hand prices miss, less any setup effect the cross-fitted baseline misses.

## 2. Dice-priced innovations (DPI)

**Definition.** Event k has parent features u_k, fixed before the event, and outcome features ψ(o):

    L_k(g) = g(u_k, ψ(o_k)) − Σ_o p_k(o) g(u_k, ψ(o)),      g(u, ψ) = s · ψᵀ Θ u

- **u** (25 numbers [C]): a constant and 8-dimensional projections of `own1`'s frozen token outputs for the attacker, the defender and the global token at the attack decision. For a recruit, u is the constant and the recruit slot's 8. The decision's own forward computes them.
- **ψ** (13 numbers [C]): per fighter, dead, change in HP fraction, advanced, slowed, poisoned and change in XP fraction, plus whether the game ends. For a trait draw, the 11 pool traits' indicators [M: `wesnoth_src/data/core/macros/traits.cfg`].
- **s** is the mover's sign.
- **Size.** Θ has about 420 entries [C].

A fight changes only its two fighters, which ψ encodes, and u carries the board through the tokens' attention, so one forward prices every outcome. ψ is the rule state the fight produces; no weight is set by hand.

**Exactness.** Fix Θ outside the batch being labelled. Then E[L_k | before k] = 0 by the enumerated law, so E[(D_i − p_i)L_k] = 0 for every k: before the coin by independence, after it by the tower property. Hence

    T_i = s_i (D_i − p_i) / (p_i (1 − p_i)) · R_g,      R_g = z − b(setup) − Σ_k L_k(Θ̂)

has E[T_i | H_i] = Δ_i exactly. Terms before the coin act as a baseline and terms after it as control variates. All of a game's labels share R_g. Θ̂'s quality enters only the variance.

**The fit is a natural experiment.** Θ̂ regresses z − b on X̃_g = Σ_k s_k(ψ(o_k) − E_{p_k}ψ) ⊗ u_k by least squares. Each centred feature is orthogonal to every other event's innovation and to every coin. So OLS converges to each ξ_k's projection on the features, and the players' choices cannot confound the prices. Minimizing the corrected result's variance is MIVAT's objective ([White & Bowling 2009](https://www.ijcai.org/Abstract/09/326)), here per event and conditioned on the parent. Θ̂ is closed-form from per-batch accumulators.

**Cost.** Enumeration takes 0.06–0.11 ms per fight [M, seed], and a game has 109.7 fights [M], so 7–12 ms of CPU a game [D] and no forward. The child-forward corrector costs 2.0–3.9× the forwards and pays only below 0.19–0.37 [D, `round5_synthesis_outcomes.out`]. DPI pays whenever r_DPI < 0.732.

**What r buys** (derived by `round6_invent_dpi_power.py` from the round-5 synthesis model, inputs unchanged; no row predicts r):

| r | Information per $ | S1 batches for round 5's floors | Forecast SE at sd 0.003 | Match SE, 12,800 games (r_match scaled alike) | Floor sd, μ 0 / μ −0.0018, match also corrected |
|---|---|---|---|---|---|
| 0.732 | ×1.00 | 8 ($19.8) | 4.7 Elo | 2.57 | 0.0020 / 0.0038 |
| 0.5 | ×1.46 | 6 ($14.8) | 3.9 | 2.12 | 0.0016 / 0.0034 |
| 0.4 | ×1.83 | 5 ($12.4) | 3.5 | 1.90 | 0.0014 / 0.0031 |
| 0.3 | ×2.44 | 4 ($9.9) | 3.0 | 1.64 | 0.0013 / 0.0029 |
| 0.2 | ×3.66 | 3 ($7.4) | 2.5 | 1.34 | 0.0010 / 0.0026 |

DPI cuts cost and the forecast's SE. The floors fall slowly, because gain needs spread v in the frozen span and no variance reduction creates spread.

**The record on r** (`round6_invent_conditioning.py`, on the luck walk's per-game sums [M]). One global material price keeps 0.896. The same price conditioned on the board's margin (`value_50`) keeps 0.853, a paired difference of −0.0428 ± 0.0043. All seven columns keep 0.848. Conditioning helps and the hand forms saturate; learned conditioning is another class, which S0(d) measures.

**Retroactive relabelling.** The student's statistics are linear in Θ: Σ_i x_i T_i = Σ_g y_g(z_g − b_g) − CΘ, with y_g = Σ_{i∈g} s_i(D_i − p_i)x_i/(p_i(1 − p_i)) and C = Σ_g y_g X̃_gᵀ per batch.
- **Retroactive.** A better price re-denoises every past batch, cross-fitted (a batch's labels use a Θ̂ fitted without it), with no new games.
- **Rule I4′.** Everything a progressive reading of batch k+1 uses is a function of batches 1..k. A refit price would otherwise break the current design's I4 silently.

## 3. The lottery column

**Definition.** ℓ(a) = Σ_o p_a(o) g(u_a, ψ(o)) for an attack candidate a (unit, hex, target, weapon).
- The law is the core's, at a fork with the unit on the hex.
- Θ̂ comes from S0(d), frozen for the round.
- x gains ℓ(a1) − ℓ(a0), split into attack against attack and attack against other; the column is 0 for non-attacks.
- The coins calibrate it, and labels stay unbiased (the feature is fixed before the coin).

**Known answer.** A weapon-only pair reaches one set of post-fight states under two laws, so Δ = Σ_o (p_{a1} − p_{a0})(o) Q(s∘o) exactly. True prices would give that column coefficient 1. A diagnostic only; S0(a)'s census counts these pairs [U].

**Cost.** 0.13–0.18 ms per candidate [D: fork 0.02 + step 0.05 + 0.06–0.11]. With at most two candidates per band decision, that is at most 55 ms a game [D: 2 × 152 × 0.18].

**Beyond the band.** Round 2 adds coins where the calibrated column would switch outside the band, only if S1's logged count × calibrated gain clears 2 match SEs at its lower bound.

## 4. The algorithm

**Learned:**
1. Θ̂, by closed-form ridge, cross-fitted by batch and refit each round (it prices outcomes under π̃_k).
2. The current design's ridge student plus the lottery columns, on DPI labels.
3. After a band is exhausted, the network, by own-trajectory distillation.

**Update.** The current design's calibrated switch rule, progressive forecast and adoption match. The match is corrected by pairs plus DPI fitted on the round's coin games, since any fixed g is a valid control variate.

**Why more per dollar.** Information per label is p(1 − p)/E[R²]. Against p = 0.5 (5–10% lower floors [D, round 5]), the child-forward corrector (2.0–3.9× the forwards) and forks (47× less information), DPI cuts E[R²] at cost multiple 1, with headroom up to 0.73 of Var(z).

## 5. Regimen

Prices are the current design's: a generation game $2.47e-5, a match game $1.49e-4, replay at 2,440 decisions a second [M: BACKLOG.md, via `round5_simplify_critique_extra.py`].

| Stage | Cost | Readout | Decision per outcome |
|---|---|---|---|
| **S0 (a)–(c)** | ≈ $0.1 | Unchanged. S0(a) also logs the fighters' and global token outputs, whose principal axes fix u. S0(c) also times DPI logging. | Unchanged. |
| **S0(d) prices on the record** | ≈ $0.6 [D: 18,800 games × 609 decisions = 11.5 M forwards at 2,440/s, $0.55; the CPU walk took 4 min 14 s on 31 jobs, M] | Leave-one-match-out fit. r_DPI in `own1` self-play with the setup baseline (against 0.732), in side-swapped pairs (against 0.700), alone (against 0.848); calibration slope; P-DPI; accrual by turn. No prediction of r; the fit penalty (d/N ≈ 420/18,000) inflates r by about 0.02 [D]. | **r_DPI below 0.732 by 2 paired SE** (computed in the run; hand forms here gave 0.0013–0.0043 [M]): DPI enters S1 from batch 1, the power script reruns at the measured r, S1 takes the fewest batches that reach round 5's floors, and the rest go to round 2. **Otherwise** DPI is not built; $0.6 is spent and the accrual curve still settles §8's horizon question. **P-DPI outside its band:** fix the fit code first. |
| **S1 coin round 1** | B × $2.47 × (1 + overhead [U, S0(c)]); B 8 at r 0.732, 5 at 0.4, 4 at 0.3 [D] | The current design's readouts, plus Θ̂ refit under I4′, P-DPI in batch 1, control-variate invariance (DPI, columns, none), the negative control, the lottery coefficients, and out-of-band switch counts | As the current design; early stop at the measured r |
| **S2 adoption** | $2.0 | The current match, corrected by pairs + DPI: SE 347.4·√(r_match/12,800) [D] | As the current design |
| **S3 round 2** | B₂ × $2.47 + $3.9 | Prices refit on π₁'s games; out-of-band coins if S1's logs clear their line | As the current design |
| **S4 reset; S5** | $6.5; as S3 | As the current design, prices refit | As the current design |

**Budget [D].** Round 1 costs $0.7 + B × $2.47 + $2.0: $22.5 at r 0.732, $15.1 at 0.4 and $12.6 at 0.3. The full plan costs $76.6 − 3(8 − B) × $2.47: about $77 if DPI fails, about $55 at r 0.4.

**Beyond `own1`.** The $1,000 case is argued only after two adopted rounds within 2 SE of their forecasts. Each round then costs about r/0.732 of today's. Its candidates each have a derived admission line:
- **A nonlinear price head** on stored per-event features (≈ 17 KB a game [D]), admitted if it lowers r by 2 SE on held-out batches.
- **Prices from a full-state critic's tokens**, at 1.24× the forwards [D: 110/449], admitted if r_priv × 1.24 < r_DPI.
- **Luck-stripped value targets** z − Σ_{k>t} L_k, deferred to their own case (critics trained on results are a killed line).

Nothing measured links `own1` to human ratings.

## 6. Planted checks and invariants

**P-DPI**, offline in S0(d) and again in S1 batch 1 through the generation path:
- **The plant.** z′ = z + δ Σ_k s_k(1[defender dies] − P_k(defender dies))(e·u_k), with e a random unit direction and δ set blind to 0.01 of Var(z).
- **Recovery.** The defender-dead block must recover δe at slope 1 ± 0.15 at 2 SE in S0(d), and ± 0.054 in batch 1 [D: √(r/(0.01N))].
- **Label means** (S1 only, where there are coins) must not move.
- **Sign.** A sign-flipped copy must read −1.

**Negative control.** Prices from the tokens of the decision after the fight see the outcome, so they are not centred and fit z better; the invariance test must reject them.

**Invariants:**
- **I-D1.** Outcomes and trait draws are calibrated to their laws (χ² of the probability integral transform).
- **I-D2.** The cross-fitted luck sum is uncorrelated with every (D_i − p_i) and with pre-coin quantities.
- **I-D3.** z's slope on it is 1 on held-out batches.
- **I-D4.** The accumulators reproduce a per-game rebuild on 1,000 games.
- **I-D5.** I4′ holds.

The current design's I1–I9, P-coin and P-distill stand.

## 7. What is new, and the nearest work

Searched 2026-10-20; finding nothing does not prove absence.

**Nearest work:**
- **MIVAT** ([White & Bowling 2009](https://www.ijcai.org/Abstract/09/326)): a linear value minimizing an advantage-sum estimator's variance, in closed form, from hand features, for evaluation. Read in the text.
- **AIVAT** ([Burch et al. 2018](https://arxiv.org/abs/1612.06915)): a heuristic value at every chance outcome, for evaluation.
- **VR-MCCFR** ([Schmid et al. 2019](https://arxiv.org/abs/1809.03057)) and **zero-variance baselines** ([Davis et al. 2020](https://arxiv.org/abs/1907.09633)): baselines in CFR training, with a value per history.
- **Backgammon rollout variance reduction** ([Montgomery 2000](https://bkgm.com/articles/GOL/Feb00/var.htm)): the bot's evaluations (not re-read).
- **Counterfactual credit assignment** ([Mesnard et al. 2021](https://arxiv.org/abs/2011.09464)): hindsight baselines constrained to carry no action information, so only approximately unbiased. Read at abstract level.
- **RELAX** ([Grathwohl et al. 2018](https://arxiv.org/abs/1711.00123)): a learned control variate.
- **In-experiment covariates** ([arXiv 2410.09027](https://arxiv.org/abs/2410.09027)): post-treatment covariates are valid if their means match across arms; mediators are excluded.
- **Stochastic MuZero** ([Antonoglou et al. 2022](https://openreview.net/forum?id=X6D9bAHhBQ1)): learned chance codes, no enumerated laws.

**New as far as found:**
1. A learned hindsight baseline for per-decision randomized labels that is exactly unbiased by construction. Chance innovations are post-treatment but not mediators, since their mean is zero given any past.
2. A factored price: one parent forward's learned representation, crossed with each outcome's rule state, values every outcome. The work above evaluates a value, or hand features, at each child.
3. Dice as free experiments that identify prices without confounding, reused as a calibrated decision feature with an internal known answer.
4. Retroactive relabelling.

Each part has ancestors; I found the combination nowhere.

## 8. Weakest point, and decision record

**Weakest point.** r_DPI is unknown, and hand prices saturate at 0.152 explained. If chance acts mainly through cascades that no fight-local price expresses, even from tokens that see the board, DPI adds little; finding out costs $0.6 and no generation code. Even at a low r, DPI creates no spread: a null gets cheaper, but the floor moves only from 0.0020 to 0.0014 at r 0.4 [D]. The lottery column helps only if prices learned on chance siblings transfer to decision siblings, which the coins measure and nothing predicts.

**For the user.**
- ψ is offered as rules data, not a hand evaluation; if not accepted, DPI falls under ruling (b).
- Labels drop capped games, per the ruling that a cap is not a draw. That conditions on a post-coin event: `own1` self-play capped 0 of 800 [M], and each batch reports its count.

**Rejected:**
- **Branching continuations.** At most 12.5% more information under uniform accrual [D, `round6_invent_rejections.py`], at the price of a tree-shaped data path.
- **Horizon labels with a rectifier.** They gain only if variance is front-loaded and the horizon value calibrated: 10% at share 0.8, h 0.5 [D, same script]. S0(d)'s accrual curve settles it.
- **A rule-scalar price arm:** it is hand-chosen.

**Deferred:** the nonlinear head, privileged prices and luck-stripped targets (§5).

## Provenance

| Number | Value | Tag, source |
|---|---|---|
| r, setup + columns, `own1` self-play | 0.732 | M, `round5_provenance_match_variance.out` |
| Pairs + columns kept | 0.700 ± 0.009 | M, `round5_provenance_critique_pairs.out` |
| Setup alone kept | 0.897 (explains 0.103) | M, current design ruling (b) |
| Columns alone kept, pooled | 0.848 (reproduced 0.8484) | M, `round6_invent_conditioning.out` |
| Hand prices' explained share | at most 0.152 (1 − 0.848) | M, same |
| Headroom of a perfect pricing | up to 0.73 of Var(z) | D, r = 0.732 with all variance beyond the setup chance at argmax (seed §3) |
| Var(z), `own1` self-play | 1.0004 | M, `round5_simplify_variance.json` |
| Material 0.896, `value_50` 0.853, difference −0.0428 ± 0.0043; joint − `value_50` −0.0047 ± 0.0013; `own1` self-play 0.904 / 0.832 / 0.832 | as stated | M, `round6_invent_conditioning.py` (leave-one-match-out, 2,000 game bootstraps) |
| Paired SEs of kept-ratio differences on 18,800 games | 0.0013–0.0043 | M, same |
| Fights per game; decisions per game | 109.7; 449.1 | M, `round5_synthesis_outcomes.out` |
| Recruits (trait draws) per game | 45.8 | M, `match_luck_20261019/readout.txt` (36,607 / 800, `gen_own1_selfplay`) |
| Capped games, `own1` self-play | 0 of 800 | M, same |
| Enumeration time per fight | 0.06–0.11 ms | M, seed §2 |
| Fork, step | 0.02, 0.05 ms | M, seed §2 |
| DPI CPU per game | 7–12 ms | D, 109.7 × 0.06–0.11 ms |
| Lottery column CPU per game | ≤ 55 ms | D, 2 candidates × 152 band decisions × 0.18 ms |
| Pool traits | 11 | M, `traits.cfg` (strong, dextrous, quick, intelligent, resilient, healthy, fearless, weak, slow, dim, aged) |
| Trait law | uniform over untaken pool entries, draw modulo count | M, `rust/wesnoth_core/src/units.rs` `roll_traits_with` |
| dim u, dim ψ, Θ size | 25, 13 (+11 traits), ≈ 420 | C |
| Fit penalty in S0(d) | about +0.02 on r | D, OLS out-of-sample factor 1 + d/N, 420/18,000 × r ≈ 0.75 |
| Floors, batches, forecast SEs, match SEs by r | table §2 | D, `round6_invent_dpi_power.py` / `.out` (round-5 synthesis model, inputs unchanged) |
| Match SE line in the round-5 power model | uses 0.747 (2.65 Elo), not 0.700 (2.57) | M, `round5_synthesis_power.py` `MATCH_SE_Z`; floors there are about 3% conservative |
| Child-forward corrector's line | r_L < 0.19–0.37 at cost 2.0–3.9× | D, `round5_synthesis_outcomes.out` |
| S0(d) forwards | 11.5 M, $0.55 | D, 18,800 × 449.1 × 27.27/20.1 turns (M, `readout.json` pooled_facts; `readout.txt`), at 2,440/s (M, BACKLOG.md via `round5_simplify_critique_extra.py`), $0.42/h |
| Luck walk wall | 19,466 games in 4 min 14 s on 31 jobs; 0.32 s per game single-thread | M, `match_luck_20261019/walls.txt`, `rows.jsonl` `secs` |
| P-DPI bands | ± 0.15 (S0(d)), ± 0.054 (S1 batch 1) at 2 SE | D, √(r/(0.01 N)) with r ≤ 1, N 18,800; r 0.732, N 100,000 |
| Privileged prices' cost multiple | 1.24 | D, 110 / 449 forwards added |
| Per-event storage | ≈ 17 KB a game | D, 110 × 38 floats × 4 B |
| Branching gain bound | ≤ 1.125 | D, `round6_invent_rejections.out` |
| Horizon-label factor | 1 when variance accrues with cost; 0.900 at a 0.8, h 0.5 | D, same |
| Round costs, budget | §5 | D, from the rows above and the current design's prices ($2.47 a batch, $2.0 adoption, $3.9 for S3's matches, $6.5 S4) |
| Information multiplier | 0.732 / r | D, label information p(1 − p)/E[R²] at unchanged cost |
| r_DPI, r_DPI in pairs | — | U, S0(d); S1's size and S2's SE wait for it |
| DPI logging overhead | — | U, S0(c) |
| Lottery coefficients; weapon-only pair count | — | U, S1; S0(a) census; diagnostic only |
| Out-of-band switch count and gain | — | U, S1 logs; round 2's extra band waits for it |
| v, μ, second-order loss, distillation closure | — | U, as the current design |
| own1 against strong humans | — | U; nothing rests on it |
