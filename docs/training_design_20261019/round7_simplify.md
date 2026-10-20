# Round 7, simplify stance: one label, one match form, one plant per path

Tags: **[M]** measured, **[D]** derived, **[U]** unknown, **[C]** choice. Scripts are in `analysis/`; this stance's is `round7_simplify_power.py` (`.out` beside it, cited as §n).

## Summary

The core stays: near-tie coins with logged propensities, exactly unbiased labels, a closed-form linear head on the frozen network, a sealed model-free forecast, adoption only by match. Eleven mechanisms go or merge; the largest:
- **Matches are independent games** in the existing `run_elo_batch`: 37,800 games keep SE 1.79 Elo for $0.146 more a match. Pair scheduling, the pair rules, I8 and the mirror null go.
- **Every coefficient is frozen before its batch is generated;** labels are computed once. No cross-fitting; I4 holds by construction; cost 0.15 of a batch.
- **DPI and its replay are deferred** until the budget binds. Inside $85 they decide only whether a fourth round fits, in practice only if r_DPI ≤ 0.516.
- **The plant is read as a difference** (SE 0.005 per side, no prior measurement); **the offset is baked into the distillation target**, whose known shift is that path's plant; S4 runs one match.

The plan costs $71.61 (against $73.58 plus $1.61 reserve) and fits three rounds at overhead up to 22% (against 16%).

## 1. What goes, and what each removal costs

| # | Current | Here | Cost |
|---|---|---|---|
| 1 | S0(a) replay; DPI (token logging at fights, Θ fit, admission, I-D1, I-D2) | Deferred to a fourth round or the $1,000 case; batch 1's logs give census, d_eff, E[R²] | Saves $0.42. Above r_DPI 0.602 nothing changes inside $85; at 0.602 a fourth round needs overhead ≤ 0.001; at ≤ 0.516, DPI from round 1 funds a fourth round at overhead ≤ 0.168, from round 3 only at ≤ 0.001 (≤ 0.078 at r ≤ 0.430) [D §4] |
| 2 | Seed-shared side-swapped pairs, pair scheduling, pair drops, I8, mirror null | Independent games, sides alternated as `slot_for` already does | +$0.146 a match, $1.16 over the plan, less $0.08 [D §1]; the pair-correlation unknown goes |
| 3 | Cross-fitted b, β, γ; batch 1 by halves | Coefficients of batch k fitted on batches < k; batch 1: b = 0, β from the record | Batch 1 keeps 0.839, not 0.709: 7.78 effective batches, not 7.93; floors unchanged to 0.0001; ≈ $0.38 once [D §2] |
| 4 | γ·V_i | Dropped; V_i logged as a report | Unknown (no record holds per-decision values). One corrected result per game serves all labels |
| 5 | I2, the exact duel | I-V on every real batch | No exact-Q test; its targets have other checks: sign (plant), propensity (I1), coin not played (I9), uncentred variate (I-V) |
| 6 | Plant δ set from Var(u·x); read on T′ | Read on T′ − T | None: δ and game noise cancel |
| 7 | S0(c) smoke stage | First 2,000 games of batch 1 gate the rest | None |
| 8 | E3, E, two-candidate rule, $1.61 reserve | Interim stops iff J3 < 0; S2 adopts iff lower bound > 0; E computable from logs as a user option | The automatic rescue of a curvature-type miss, worth an unmeasured amount; false adoption 0.046 → 0.023 |
| 9 | Round 2's forecast scaled by J/Ĝ_pf | Each round's own Ĝ_pf | None derivable; the ratio is reported |
| 10 | P-distill run ($1.09); S4 at offsets 0 and −1.5 | Offset baked into the target; one match at offset 0 | Saves $2.59. A closure c < 1 costs −7.3 ± 3.8 Elo at c 0.82 [D §6], priced from the measured c. The raw-target offset-0 arm faced Match A's −34 to −55 Elo [D, knob records] |
| 11 | Kind indicators in φ | Removed (collinear with the pair constants), d = 772 | None |

Everything else in the current design is kept; branch L and the S3-null options stay unbuilt.

## 2. The algorithm

### 2.1 Generation

π_0 is `own1` at `raw:t0+eo-1.5` (64 slots), both sides, in `run_elo_batch` at 120 jobs. At a decision whose top two joint log-priors after the offset differ by G < 0.35 [C], a coin keyed by seed and decision index plays the base runner-up a1 with probability P_i: p = 0.25 where π_k plays a0, 1 − p where it plays a1 [C]. One code path thus serves every round in base orientation. Play continues under π_k, re-deciding after every roll. Logged: per band decision P_i, D_i, pair type, mover, G, V_i and (from batch 5) the student's switch bit; per game z and A_g = Σ_{i∈g} s_i ω_i x_i; per batch the Gram matrix.

### 2.2 The label

    T_i = s_i ω_i R_g,   ω_i = (D_i − P_i)/(P_i(1 − P_i)),   R_g = z − b(cell) − βᵀS_g

z is side A's result, s_i the mover's sign, b(cell) the full setup cell's mean, S_g the game's seven luck sums from `tools/match_luck.py`. b and β for batch k are fitted on batches 1..k−1; batch 1 uses b = 0 and β from the 18,800 walked games [C].

**Claim.** E[T_i | H_i] = Q̃(H_i, a1) − Q̃(H_i, a0), under the coin policy π̃. **Proof.** The coefficients are constants; b and the luck before i are functions of H_i; each later luck term has mean zero given the state before its fight, by the core's enumerated law, and that state contains D_i. So E[(D_i − P_i)(b + βᵀS) | H_i] = 0, and E[(D_i − P_i)z | H_i] = P_i(1 − P_i)(E[z | D_i = 1, H_i] − E[z | D_i = 0, H_i]). ∎ Under fog this is the information-state advantage: no strategy fusion ([Baisero & Amato](https://arxiv.org/abs/2105.11674)).

**Variance.** σ² = Var(z)·r/(p(1 − p)): 3.67 from batch 2 (r 0.688 plus ≤ 0.011), 4.48 in batch 1 (r 0.839) [D]. **Caps** are dropped (ruling of 2026-08-17); `own1` self-play capped 0 of 1,600 [M]; a batch with caps runs the coin χ² inside them.

### 2.3 The student

x_i = φ(a1) − φ(a0), φ = [actor token output, q(actor) ⊙ k(target hex), four factored log-probabilities], d = 772 [C]; the target logit is q·k/√d (`wesnoth_ai/model.py`). Unpenalised terms are built antisymmetric: per unordered kind pair {u, v}, a constant column (+1 for (u, v), −1 for (v, u)) and a slope column G·1[{u, v}], six constants and ten slopes. The fit is ŵ_k = (G_{1..k} + λI)⁻¹ Σ_g R_g A_g. λ̂_k minimises the progressive loss Σ_{j<k}[ŵ_j(λ)ᵀG_{j+1}ŵ_j(λ) − 2ŵ_j(λ)ᵀr_{j+1}] from statistics alone; the first student read is student 3, so no special case. Switch where the prediction is above 0, the posterior-mean rule under held-out λ (`round6_simplify_rule.py`).

### 2.4 Forecast, matches, adoption

- **Forecast.** In batches 5–8, student k−1 runs in log-only mode. Ĝ_pf sums those labels at its switches, signed as its choice against π_k's, per side-game, × 347.4; sealed before the match; SE 4.6 Elo at μ 0 [D, `round6_synthesis_power.out` §4].
- **Matches.** `own1` plus the head on one seat, sides alternated, 37,800 decisive games on fresh seeds, Elo and SE from `tools/elo_collect.py` (1.79) [D].
- **Adoption** iff the lower 2-SE bound is above 0 (false adoption 0.023 at zero gain); against `own1` it is the PURE claim.
- **Interim.** After batch 3, student 3 plays the incumbent; stop iff its estimate is below 0 (P 0.50 at a null). With the stop counted, a null bounds the spread at sd 0.0016 (μ 0) or 0.0036 (μ −0.0018) at d_eff 776, 0.0012 / 0.0028 at 200 [D §2].
- **Miss.** J below Ĝ_pf − 2√(SE_f² + SE_m²), ≈ 9.9 Elo at μ 0. Labels price the gradient at the coin point; the match integrates to deterministic switches against an uncoined opponent; the gap is curvature or interaction with the opponent's coins (current §3.5), inseparable in one round, and [Qian 2025](https://arxiv.org/html/2502.13500v2) §2.2 bounds neither (read). A miss is reported as "first-order labels do not transfer at q̂ switches".

### 2.5 Rounds and reset

Round k puts the coins of §2.1 on the base band under π_k; the student is refitted on all rounds' labels; a null round exhausts the band. The reset distils π_k by the own-trajectory recipe, the target being `tools/own_targets.py`'s top-16 raw distribution with the switched pair's logits exchanged and end_turn lowered by 1.5 through `shift_end_turn_prior` (`tools/raw_player.py`) before `top_target` renormalises. The distilled network plays at offset 0.

### 2.6 Why this extracts more per dollar

One result informs about 152 independently randomised decisions: 39.0 information units per game-equivalent against 0.82 for fork pairs [D, round 5]. The luck columns save 1.225 times the games (residual 0.843 → 0.688 [M]) for 2.4–6.6% of generation's CPU. Every batch trains (a match is 17 times a held-out batch's information per dollar [D, round 5]). Net of the pairs' $1.16, the plan is $3.6 cheaper than round 6's with its reserve; the frozen coefficients cost 0.15 of a batch in information; about half the new code paths go.

### 2.7 Invariants and plants

**Invariants:** I1 D matches P (χ²), independent of dice and pre-coin quantities; I3 a swap flips x, T and the pair terms; I6 the statistics fit equals a per-decision fit on 1,000 games; I7 logged switches equal recomputation with a planted ŵ; I9 at D_i = 1 the record's command is a1; I-φ features reproduce the logits; I-F labels read only earlier batches' coefficients; **I-V** (every batch) per pair type, mean s_i ω_i (b + βᵀS_g) is 0 within 2 SE.

**Label and fit path (batch 1).** z′ − z = Σ_j s_j^rec (D_j^rec − P_j) g_j, g_j = u·x_j for a random unit u, with s^rec and D^rec read from the game record by code separate from the label code. T is linear in z, so readings use T′ − T on the same games:
- per mover side, the slope of T′ − T on g: predicted 1, SE 0.0051 at 100,000 games [D §5, simulated; closed form √(3/n) = 0.0055]; a wrong side sign reads −1.005, a coin read one decision off 0.016;
- (ŵ′ − ŵ)ᵀu, predicted uᵀ(G + λ̂I)⁻¹Gu, SE from the logs.

On the planted increment, I-V's statistic weighted by g is this slope, so the check can detect a coin-correlated term.

**Match path (S0).** Attack offset +1 on one seat through the per-seat hook that carries the head, 6,400 games: −78 ± 16 Elo [M via D, `round6_provenance_critique_attack_stage1.out`], window ±33.2 [D §1]. A sign error reads ≈ +78, a hook on both seats ≈ 0.

**Distillation path (S4).** The baked shift is known per position. ĉ, the slope of the student's end_turn log-probability shift on the target's over held-out positions, has an indicative band 0.82–1.30 (tempo closure, `round6_provenance_closure.out`); not above 0 by 2 SE means distillation is broken.

## 3. Regimen

Prices [M, `round6_provenance_prices.out`]: $2.51e-5 a pool game ($2.51 a batch, $0.95 a match), $1.17e-4 an eval-driver game.

| Stage | Cost [D] | Readout, prediction | Decision |
|---|---|---|---|
| **S0** | $0.18 | Attack known answer, −78 ± 33 | Outside: fix the match path before any match (batches may start; labels do not use it) |
| **S1** round 1 | $20.08 + $0.95 | Smoke: invariants, price, overhead. Batch 1: plant per side, I-V, census, d_eff, E[R²] with V; the power script reruns. Batch 3: J3. Batch 8: Ĝ_pf sealed; v̂, c, b, q̂ (no record predicts v) | Smoke or plant fails: debug. Price × (1 + overhead) > $3.07e-5: drop round 3 (two fit to $4.78e-5). J3 < 0: stop; S3-null |
| **S2** | $0.95 | π_1 vs `own1`; Ĝ_pf ± miss line | Lower bound > 0: adopt (the claim); S3. Otherwise S3-null, as a bound (Ĝ_pf null) or a transfer miss |
| **S3** round 2 | $22.93 | As S1 under π_1; adoption vs π_1, claim vs `own1` | Adopt by the rule; then S4 |
| **S4** reset | $3.60 | Distillation $2.1 [M]; ĉ, ρ̂. Sealed prediction −(1 − ρ̂)Ĝ_k + Σ_type n_type μ̂_type − 347.4 × 0.0774 × 1.5 max(0, 1 − ĉ) [D, knob slope, `round5_provenance_critique_knobs.out`]; out-of-band disagreements unpriced [U]. One match vs π_k at offset 0, 12,800 eval games, SE 3.07 | ĉ not > 0 by 2 SE: debug. Estimate ≥ 0: new base, offset gone (a wrong acceptance costs the next round's gain, never a claim). Else keep π_k; S3-null (i) |
| **S5** round 3 | $22.93 | As S3 on the new band | As S2 |
| **S3-null** | when chosen | A bound, or a transfer miss named at S2 | User: (i) w 0.7, same code, 125 band decisions per side-game, $22; (ii) E from logs, $0.95, after an S2 miss; (iii) critic column, ≈ $7; (iv) stop |

**Budget** [D §3]: $71.61, $13.39 left. Three rounds fit iff price × (1 + overhead) ≤ $3.07e-5 (22% at the measured price; the 1.41× host spread does not fit, so the smoke picks the host). A null round 1 costs $8.66 stopped, $22.16 run through, $15.41 expected. Branch L is unchanged (current §4). The $1,000 case opens after two adopted rounds within their miss lines, DPI's $0.42 reading first.

## 4. What is new, and the nearest work

Searched 2026-10-20; finding nothing does not prove absence. The core is the current design's; I found its combination nowhere. Nearest:
- **RESLOPE** ([Daumé et al., ICLR 2018](https://mlanthology.org/iclr/2018/iii2018iclr-residual)): episodic loss reduced to contextual bandits through a learned per-step decomposition, with a "multiple deviation mode" in its code ([hal3/reslope](https://github.com/hal3/reslope); abstract and README read). Its credit is a learned regressor; here each label is unbiased by its coin alone.
- **LOLS** ([Chang et al. 2015](https://arxiv.org/abs/1502.02206)): one-step deviations scored by roll-outs, one per roll-out, against about 19 coin deviations per side-game here.
- Micro-randomised trials ([Boruvka et al. 2018](https://doi.org/10.1080/01621459.2017.1305274)), the score-function identity ([Williams 1992](https://link.springer.com/article/10.1007/BF00992696)), [AIVAT](https://arxiv.org/abs/1612.06915).

New in this stance, as engineering: (1) labels frozen by forward chaining (nearest: cross-fitting, [Chernozhukov et al. 2018](https://arxiv.org/abs/1608.00060), not re-read; prequential validation, Dawid 1984), 0.15 of a batch for no folds; (2) a plant read on T′ − T, its SE fixed by the coins alone; (3) the decode offset as the distillation run's own plant.

## 5. Weakest point

About $22 tests one hypothesis, a near-tie spread of sd ≥ 0.0012–0.0036 in the frozen span [D], which nothing on disk measures; and with E, E3, the critic column and DPI out of the automatic path, a null or a transfer miss yields only a bound or a named failure, the next step waiting on the user and on code not yet written. Two removals carry costs I cannot price: the value-head baseline and the exact duel's coverage. The first reset rests on a closure the record measures only as tempo, on another target; at c 0.82 it would cost about 7 Elo, which S4 prices but cannot avoid.

## 6. Rulings for the user

- **(a)** Coin labels pooled by one linear head: unchanged; if no, branch L.
- **(b)** The seven luck columns as label control variates only; DPI is no longer asked for.
- **(c)** Adoption and claim by the standing PURE procedure scaled to 37,800 decisive games in `run_elo_batch` at 120 jobs (SE 1.79, $0.95).
- **(d)** The offset is baked into the first reset's target and leaves play iff the distilled network at offset 0 is non-inferior to π_k.

## Provenance

| Number | Value | Tag | Source or derivation | Used for |
|---|---|---|---|---|
| Pool price, one-network game | $2.51e-5 | M | `round6_provenance_prices.out` (ref_p120x6, generation_speed_r5 speed.json) | All coin and match costs |
| Eval-driver price | $1.17e-4 a decisive game (median of 18) | M | same | S4's two-network match |
| Box setup | $0.018 (median) | M | same | S0 |
| Pairs keep | 0.847 ± 0.009 | M | `round6_simplify_pairs_raw.out` | Sizing independent matches at equal SE |
| Independent match | 37,800 games, SE 1.79 Elo, $0.95; +$0.146 a match, +$1.16 over 8 matches | D | 32,000/0.847; 347.4/√N | Matches, budget |
| Elo per unit z | 347.4 | D | 800/ln 10 | All SEs |
| Var(z), own1 self-play | 1.0004 | M | `round5_simplify_variance.json` | σ² |
| Label residual, cell known | 0.688 | M | `round6_provenance_cells.out` | σ² = 3.67 |
| Label residual, columns only (record-fitted) | 0.839 | M | same | Batch 1 under frozen coefficients |
| Finite term of a cell baseline | 0.696 × 1,512/N | D | same | r by batch: 0.839, 0.699, 0.693 … 0.690 |
| Effective batches at 8 | 7.78 (frozen) vs 7.93 (round 6) | D | `round7_simplify_power.out` §2 | Cost of item 3 |
| Floors, interim bounds | Floors 0.0011 / 0.0027 (d 200), 0.0016 / 0.0034 (d 776); bounds 0.0012 / 0.0028, 0.0016 / 0.0036 | D | same §2 (round 6 Gaussian model) | What a null bounds |
| Interim P(stop) at a null | 0.50 | D | `round6_synthesis_power.out` §3 | S1 |
| Forecast SE; miss line | 4.6 Elo; ≈ 9.9 Elo at μ = 0 | D | same §4 (match SE 1.79, unchanged) | S2 |
| False adoption | 0.023 a round | D | Φ(−2) | S2 |
| I-V threshold | 2 SE of the per-type mean of s ω (b + βᵀS) | D at run time | Per-game sums from the logs | Variate check |
| Band per side-game | 76.1 | D | 0.339 × 224.6, `round5_simplify_band.py`, `round5_simplify_critique_decision_units.out`; batch 1 re-measures | Sizes, σ² per side |
| Coins against forks | 39.0 against 0.82 units per game-equivalent | D | round 5, current §2 | §2.6 |
| Cell plus columns against cell alone | 0.688 against 0.843 | M | `round6_provenance_cells.out`, `round6_simplify_pairs_raw.out` | Columns earn their place |
| Walk CPU | 0.321 s a game; 2.4–6.6% of generation | M; D | `round6_simplify_walk_cost.out` | same |
| Caps, own1 self-play | 0 of 1,600 | M | same | Cap rule |
| Attack known answer | −78 ± 16 Elo; window ±33.2 | M via D | `round6_provenance_critique_attack_stage1.out`; `round7_simplify_power.out` §1 | S0 |
| Plant slope SE | 0.0051 per side, 0.0044 pooled at 100,000 games; closed form 0.0055, 0.0045 | D | same §5 (200 simulated replications) | Plant band |
| Plant failure readings | wrong side sign −1.005; coin one decision off 0.016 | D | same §5 | What the plant detects |
| Knob slopes around −1.5 | above −0.0774 ± 0.0406 z per unit, below +0.0012 ± 0.0245; stage-1 games −0.1059 ± 0.0505, +0.0569 ± 0.0315 | M | `round5_provenance_critique_knobs.out` | S4 prediction |
| Closure price | c 0.82: −7.3 ± 3.8 Elo (stage 1: −9.9 ± 4.7); c 1.30: +0.2 ± 3.8 | D | `round7_simplify_power.out` §6 | Item 10; S4 prediction |
| Tempo closure | 1.00 (0.82–1.23), 1.06 (0.88–1.30) | M | `round6_provenance_closure.out` | ĉ's indicative band |
| Match A, offset 0 against −1.5 | −34 ± 17 to −55 ± 26 Elo | D from M | `round5_provenance_knob_switches.out`, `round5_provenance_critique_knobs.out` | Why the raw-target offset-0 arm fails |
| Distillation recipe | $2.1 | M | `docs/memory_own_trajectories_prereg_20261011.md` | S4 |
| Budget | $71.61; $13.39 left; three-round line $3.07e-5 (overhead ≤ 0.222); two-round $4.78e-5; null $8.66 / $22.16 / $15.41 expected | D | `round7_simplify_power.out` §3; two-round: (85 − 8.53)/1.6 M | Regimen |
| DPI's batches by r; fourth-round lines | 7 at ≤ 0.602, 6 at ≤ 0.516; overhead lines in §4 | D | same §4 | Cost of item 1 |
| Numerics flips | 1.76% of decisions, at gaps < 0.014 | M | `round5_provenance_numerics_ties.out` | Logged P; antisymmetric terms |
| Band at w 0.7 | 125 per side-game | D | 0.558 × 224.6 | S3-null (i) |
| κ_T reading | about $7 | D | round 5 synthesis | S3-null (iii) |
| Near-tie spread v; c, b per type; q | — | U | S1's progressive fits; nothing before S1 relies on them | The round's gain |
| d_eff; census; runner-up mix | — | U | Batch 1 (Gram matrix, logs); the power script reruns; floors are stated for d 200 and 776 | Floor case |
| E[R²] with V | — | U | Batch 1, as a report; nothing relies on it | A later admission, if asked |
| Logging overhead; host speed | — | U | The smoke | Whether round 3 fits |
| Second-order and excursion loss | — | U | S2: match against Ĝ_pf | Whether a miss is declared |
| ρ̂; ĉ; out-of-band disagreement cost | — | U | S4, before its match; the last unpriced, reported as spillover | S4 prediction |
| r_DPI | — | U | Deferred; nothing in the plan relies on it | $1,000 case |
| Compounding, exploitability, human level | — | U | Each round's claim; none; none | Nothing |
| p 0.25, w 0.35, 8 × 100,000 games, d 772, interim after batch 3, 37,800-game matches, 12,800-game S4 match, batch-1 coefficients | — | C | Consequences derived above | Design |
