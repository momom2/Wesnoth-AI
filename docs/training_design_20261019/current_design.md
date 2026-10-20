# Current design (after round 8)

Supersedes the round-7 design. The core is unchanged: near-tie coins with logged propensities, labels exactly unbiased through the simulator's chance laws, a closed-form linear head on the frozen network, adoption only by match. What changed:
- **Rules that rested on a Gaussian model of the near-tie advantage now hold for any shape:** no interim stop; a switch threshold tested on held-out labels; a null read by an unbiased spread estimator; E within each pair type.
- **Stage order.** S0 confirms `own2` as a candidate base and ends with a coin smoke; round 3 is adopted against the pre-reset incumbent.
- **Checks added:** a dice positive control on every batch, the head's swap checked inside the match, plant lines at 3 SE.
- **Removed:** γV and the census replay, the √N ceiling.
- **Branch S,** a rating-conditioned base certified by one match, behind two new rulings.
- **Ruling (a) restated:** the head's fit is a natural policy-gradient step (§1).

Reasons: `round8_synthesis.md`. Tags: **[M]** measured (record or script named), **[D]** derived (inputs named), **[U]** unknown (measured before anything relies on it, or nothing relies on it), **[C]** a choice whose consequences are derived. Scripts are in `analysis/`.

## 1. The idea

At argmax both players are deterministic, so with the setup fixed, all of a result's variance is chance (seed §3). Where the network's top two choices nearly tie, a logged coin plays the runner-up, and play continues under the network, re-deciding after every roll. The coin is independent of the past, so the result identifies that decision's effect without a critic, and the enumerated laws remove part of the dice's share without bias. A linear head on the frozen token outputs pools the labels across positions; the player switches where the head, at a threshold tested on held-out labels, says the runner-up is better. A sealed forecast predicts the gain; a PURE match against `own1` certifies it.

**In the literature's terms.** With P(1 − P) = 0.1875 at every band decision, the ridge fit equals (F̂ + 0.1875λI)⁻¹ Σ_g R_g ψ_g: ψ_g is the game's summed score of a logistic coin policy at θ = 0, F̂ its expected Fisher matrix. That is one episodic natural-actor-critic step ([Peters & Schaal 2008](https://doi.org/10.1016/j.neucom.2007.11.026); form via [Su et al. 2017](https://arxiv.org/abs/1707.00130) Eq. 7), with the regression's constant replaced by the cell and the luck columns; switching is the greedy step ([Kakade 2001](https://papers.nips.cc/paper/2073-a-natural-policy-gradient)). The seed lists outcome-driven policy gradient as killed, so ruling (a) names it (§7).

## 2. Measurements first

**Settled by existing records**

| Quantity | Value | Tag, source | Decision it feeds |
|---|---|---|---|
| Pool price, steady | $1.912e-5 a one-network game | [M] `round7_provenance_price_tail.py` | Every cost: batch $1.91 (4.55 h), match $0.72 |
| Two-network price; host spread | $1.17e-4 a decisive game (eval median, upper bound); 1.41× | [M] `round6_provenance_prices.py` | S0(b), S4, S5, Branch S |
| Budget | Coin path $59.23 of $85; with Branch S's core $64.04. Three rounds fit iff price × (1 + overhead) ≤ $2.986e-5 (room 0.56); on a 1.41× host $79.58 (room 0.08). A null round 1 costs $17.66 with all of S0. | [D] `round8_synthesis_power.py` §1 | Stage order; host choice |
| Label residual | 0.688: own1's column share minus the cell share pooled over 23 matches (own1 alone 0.651, 0.581–0.721). σ² 3.67; 4.48 in batch 1 | [D from M] `round6_provenance_cells.py` | Floors, plant SEs |
| Variate-free label | E[T²] = 1/(P(1 − P)) = 5.333 exactly (z² = 1 with caps dropped); labels within a game uncorrelated | [D] `round8_provenance_label_variance.py` | Worst-case SE of every floor: 8 batches at 5.33 equal 5.50 at 3.67 |
| Interim stop | Interim and adoption are independent games, so a stop only lowers power; P(stop) ½ at a null. False stops reach 0.497 in shapes where a minority of decisions carries the gain (design's model, shape varied); at the μ −0.0018 floor, power 0.80 → 0.63 | [D] `round8_provenance_interim_shapes.py`; `round8_simplify_power.py` §1 | Interim removed |
| Futility lines, if money binds | Line 0: $4.42 saved for 0.17 of power; −0.5 SE: $2.45 for 0.08; below −1.4 SE both lost (Gaussian model) | [D] `round8_simplify_critique_checks.py` §1 | The fallback if the user restores a stop |
| Sign rule against the best threshold | 1.000 Gaussian, 0.958 Laplace, 0.283–0.589 two-point | [D] `round8_provenance_rules.py` | Tested threshold |
| Tested threshold | Within 0.1 Elo of f > 0 for Gaussian and Laplace shapes (moves off 0 with P 0.04–0.10); two-point gains 5.5 → 19.2, 4.9 → 10.8, 7.8 → 13.0, 6.1 → 8.6 Elo. Choosing τ on students 4–7 for student 8 loses ≤ 0.4 Elo. | [D] `round8_provenance_threshold.py`; `round8_provenance_critique_checks.py` B | Switch rule |
| E's share | ≥ 0.500 of the first-order sum within a cell, any shape; pooled types can keep less | [D] `round8_provenance_rules.py`; provenance critique §2.4 | E within each type |
| Spread estimator | Cross-batch ŵ_jᵀSŵ_k, unbiased for v under four laws (simulated); after 8 batches SE 1.27e-6 at v = 0, so a null bounds v ≤ 2.53e-6 (sd 0.0016) | [D] `round8_provenance_spread_estimator.py` | Reading of a null; S3-null choice |
| Coin plant | SE 0.0387 per side on the smoke, 0.0055 on batch 1; four readings at 3 SE raise a false alarm with P 0.011; a 2% misread sits 3.7 SE away on batch 1 | [D] `round8_simplify_power.py` §4; `round8_simplify_critique_checks.py` §2–3 | Plant lines |
| Dice control | z's slope on side A's hp luck: +0.00416 ± 0.00009 over 23 player pairs (Q 24.9 on 22 df, between-pair sd ≤ 0.00049); window at 3 sd [+0.00268, +0.00565] a batch, [+0.00245, +0.00587] on the smoke; a flipped result reads −0.00416; batch-to-batch line 0.00013–0.00017 | [M] `round8_invent_chance_ledger.py` §B, §E; [D] `round8_synthesis_power.py` §2 | Orientation and join gate on every batch |
| Attack known answer | Offset +1 on one seat: window [−111, −41] Elo; through the label code [−0.640, −0.236] z (union of two fits; leave-one-out RMS 0.83) | [D from M] `round8_provenance_knob_checks.py` | S0(a) gate |
| Attack-type mean | Per attack-promoting switch at gaps < 0.35: ≤ −0.0003 at 2 SE with a hinge at 0.35 (point −0.0096); smooth fits ≤ −0.0024 to −0.0028 | [D] `round8_provenance_critique_checks.py` A; `round8_provenance_attack_type.py` | Report only |
| Stop decisions in the band | At w 0.35, 0.140 of end-on-top decisions (0.113–0.168); 0.310 at w 0.7 (`parity3`) | [M] `round8_challenge_stop_census.py` | Record correction; S3-null (i) |
| `own2` against `own1` | +16.5 ± 12.3 Elo at 64 slots (predicted +15) | [M] `docs/memory_own_round2_prereg_20261011.md` "Measured" | S0(b) |
| Rating axis | Out-of-sample log loss 0.605 against 0.693. In-sample ratings leak the result: E[s \| won] − E[s \| lost] +116 Elo, +70 out of fold. At s = +253 one player supplies 0.605 of the sides. | [M] `round8_challenge_skill.py` §3; `round8_challenge_critique_checks.py` §1 | Branch S's inputs |
| Branch S settings | Highest centre with ≥ 10 players and none above 0.2 of the sides within ±52 Elo: s_top = +200 (42 players, largest 0.146); s_low = +40 (the rated 25th percentile; 56 players, largest 0.116). M1 detects a transfer f ≥ 0.022 of the 160-Elo gap | [D] `round8_synthesis_skill_coverage.py`; `round8_synthesis_power.py` §4 | Branch S |
| Tempo | Own side-turns 2–4: `own1` 7.56 actions, lower-rated regulars 8.32, top quartile 10.07 (faction-matched). Acting more through the stop logit is flat at `own1`: offset −2.5 reads −12.2 ± 12.3 Elo | [M] `round8_challenge_critique_checks.py` §2; `round5_provenance_stop_acts.py` | Logged in Branch S's matches; no gate |
| Learned luck pricing | Setup-conditioned columns: null after overfit (−0.0006 ± 0.0011). CES at the reset: plant κ/SE 1.5 at the distillation's 2,227 game-equivalents | [M] `round8_invent_chance_ledger.py` §A; [D] `round8_invent_critique_checks.py` §1 | Deferred to the $1,000 case |
| Reset, round 3 | S5 adopted against π_k: prediction J_S4 + Ĝ_pf(S5), SE 5.53, miss line 12.7 Elo | [D] `round8_synthesis_power.py` §3 | S5 |
| Second-order loss, a priori | The Differences-in-Q bias bound is in the total-variation distance between the two actions' transitions and in mixing constants ([Farias et al. 2022](https://arxiv.org/abs/2206.02371), Theorem 1, read): not measurable for this game | Literature | A miss is detected, not sized in advance |
| Carried from rounds 5–7 | 76.1 band decisions per side-game (76.2 ± 1.8, sampling only); numerics flips 1.76% at gaps < 0.014; caps 0 of 1,600; forecast SE 4.6, miss line ≈ 9.9 Elo; residual offset o(1 − ĉ); tempo closure 1.00–1.06; offset 0 at ĉ 0.82 costs −9.9 ± 4.7 Elo | [M/D] `round7_synthesis.md`, `round6_synthesis.md` | Unchanged |

**Still to measure**

| Quantity | Cheapest way | Cost [D] | Decision it feeds |
|---|---|---|---|
| Match path and label orientation | S0(a): attack +1 on one seat, 6,400 games, read both ways | $0.14 | Gate before any coin game |
| Whether `own2` is the better base | S0(b): `own2` against `own1`, 12,800 two-network games, SE 3.07 | $1.50 | π_0 |
| Base's census, runner-up mix, spectrum; logging overhead (the server's feature gather) | S0(c): the first 2,000 games of batch 1 | $0.04 | Power rerun and restated costs before S1 is proposed; three rounds fit iff price × (1 + overhead) ≤ $2.986e-5 |
| Spread v per type, typed c and b, switched share q, τ̃ | S1's batches; v̂ by the cross-batch estimator | inside S1 | The round's gain; what a null means |
| Second-order and excursion loss | S2 against Ĝ_pf; E on a miss | $0.72 each | Fallback; round 2's reading |
| Closure ĉ, reproduced share ρ̂, J_S4; pool two-network price | S4, held out before its match | $3.60 | Reset; S5's prediction |
| Whether the network uses s; transfer f; M2 | Branch S: permutation gap, M1, M2 | $4.81 | Base choice |
| κ_T of a privileged critic | Only if chosen after a null | ~$7 | Critic column |
| Compounding, exploitability, human level | Each round's claim; none; none | — | Nothing rests on them |

## 3. The algorithm

### 3.1 Base and generation

π_0 is `own1`, or `own2` if S0(b) certifies it, at `raw:t0+eo-1.5` with 64 memory slots, playing both sides in `run_elo_batch` at 120 jobs.
- **Coins.** At a decision whose top two joint log-priors after the offset differ by G < w = 0.35 [C], a coin keyed by seed and decision index plays the base runner-up a1, with P_i = 0.25 where π_k plays a0 and 0.75 where it plays a1 [C]. End_turn is an ordinary candidate.
- **Logged per band decision:** P_i, D_i, pair type, mover, G, u·x_i and u′·x_i (plants), and from batch 2 the log-only student's f_i. Per game: z, the seven luck sums S_g, A_g. Per batch: the Gram matrix per pair type, and x for a fixed 1,000-game subsample (0.23 GB [D, `round8_synthesis_power.py` §5]).
- **Volume.** About 76 band decisions and 19 deviations per side-game [D; the smoke re-measures].

### 3.2 The label

    T_i = s_i ω_i R_g,   ω_i = (D_i − P_i)/(P_i(1 − P_i)),   R_g = z − b(cell) − βᵀS_g

- z is side A's result, s_i the mover's sign, b(cell) the setup cell's mean, S_g the walk's seven luck sums (`tools/match_luck.py`). b and β for batch k are fitted on batches 1 to k − 1 and frozen before batch k is generated; batch 1 uses b = 0 and β from the 18,800 walked games ([Hadad et al. 2021](https://arxiv.org/abs/1911.02768) §2.1).
- **Claim:** E[T_i | H_i] = Q̃(H_i, a1) − Q̃(H_i, a0) under the coin policy: the coefficients are constants, b and the earlier luck are fixed by H_i, each later luck term has mean zero given its pre-fight state (which contains D_i), and E[(D − P)z | H] = P(1 − P)(E[z | 1, H] − E[z | 0, H]). Under fog it is the information-state advantage ([Baisero & Amato](https://arxiv.org/abs/2105.11674)).
- **Caps** are dropped (ruling 2026-08-17), exact iff capping does not depend on the coins (I-coin).

### 3.3 The student and the switch rule

- **Head.** f = c_uv + b_uv·G + ŵᵀx, x = φ(a1) − φ(a0), φ = [actor token output, q(actor) ⊙ k(target hex), four factored log-probabilities], d = 772 [C]; per unordered kind pair an unpenalised antisymmetric constant and a margin slope. ŵ_k = (G_{1..k} + λI)⁻¹ Σ_g R_g A_g; λ̂_k minimises the progressive loss.
- **Welfare curve.** For each progressive pair (student j − 1, batch j), W_j(τ) = Σ_{i: f_i > τ} T_i estimates the rule {f > τ}'s first-order gain without bias for every τ, the propensities being logged ([Kitagawa & Tetenov 2018](https://doi.org/10.3982/ECTA13288)).
- **Tested threshold.** τ̂_k maximises W summed over the latest four progressive pairs (batches up to k); τ̃_k = τ̂_k iff W(τ̂) − W(0) exceeds 2 SE of the difference, else 0. The cut at 0 is the posterior-mean rule only for a Gaussian residual; the pre-test keeps it unless the labels reject it.
- **Switch** where f > τ̃.

### 3.4 Matches, forecast, adoption, E

- **Matches.** The incumbent with the head on one seat, sides alternated, 37,800 independent decisive games on fresh seeds, SE 1.79 Elo, $0.72 [D]; two-network matches 12,800 games, SE 3.07, $1.50 [D].
- **Forecast.** Ĝ_pf sums batches 5–8's labels where students 4–7 switch at their τ̃, per side-game, × 347.4, sealed before the match; SE about 4.6 Elo, miss line ≈ 9.9, both recomputed from the logs.
- **Adoption** iff the match's lower 2-SE bound is above 0 (false adoption 0.023). Against `own1` it is the PURE claim.
- **E.** When the adoption match falls below its line while Ĝ_pf is above 0 by 2 SE, E plays one match on its own sealed forecast: within each pair type, it switches only above the half-quantile of f among the student's switches, each quantile fixed on the previous batch's stored subsample (student 8's on batch 8's). It keeps at least half the first-order sum within a type in the single-noise model [D]; its sealed forecast and match protect the claim regardless. Same adoption rule (0.046 for two candidates).

### 3.5 What a null and a miss mean

- **Null** (no adoption, Ĝ_pf not above 0): the band is exhausted for this span. v̂ per type with its upper bound states, for any shape, how much linear near-tie spread the frozen span holds (v ≤ 2.53e-6 at 8 batches if v̂ is near 0); the match bounds the play gain at J + 3.6 Elo. v̂ above 0 by 2 SE means spread exists that gave no first-order gain.
- **Miss** (Ĝ_pf above 0 by 2 SE, match below): the labels give the gradient at the coin point, both sides coined; the match integrates it against an uncoined opponent. With the plants, the dice control and I-match passed, a miss is curvature or the opponent's coins. No measurable bound exists in advance (§2), so a miss leads to E, not to a retuned p.

### 3.6 Rounds and reset

- **Round k** re-coins the base band under π_k and refits on all rounds' labels; its own Ĝ_pf and match read π̃_k, so stale labels can bias the student, never a claim. **Round 2 runs by default after an adopted round 1** [C]: the reset trains weights and waits for the optimizer discussion (ruling 2026-10-15). A null round exhausts the band.
- **Reset (S4).** π_k is distilled by the own-trajectory recipe onto `tools/own_targets.py`'s top-16 target, the switched pair's logits exchanged and end_turn lowered by π_k's offset o_k (`shift_end_turn_prior`). ĉ, ρ̂ and the off-band disagreements are read on held-out positions first. N plays at o_k(1 − ĉ) [D]. N against π_k (two networks) reads J_S4 against −(1 − ρ̂)Ĝ_k + Σ_type n_type μ̂_type, off-band disagreements unpriced [U]; the shortfall divided by their count prices one.
- **N becomes the next band's generator iff ĉ is above 0 by 2 SE.** No value threshold applies at S4, because S5's player is adopted only against π_k (§4). S5 runs unless J_S4's upper 2-SE bound is below minus the largest per-round gain measured so far; then S3-null.
- **Not covered.** Confident errors are never sampled; only the reset moves the band.

### 3.7 Invariants and plants

| Invariant | Check |
|---|---|
| I-coin | Coins match P (χ²) over all games and within capped ones, independent of dice and pre-coin quantities |
| I2 | An enumerable duel through the production generator, coin on each seat: E[T] = Q(a1) − Q(a0) exactly under the cell and columns; a post-coin variate must fail |
| I3 | A swap flips x, T and the typed terms |
| I4 | Each batch's labels read only earlier batches' coefficients (by construction) |
| I6 | The statistics fit equals a per-decision fit on the stored subsample |
| I-log | Logged f, switch sets, W_j and forecast sums equal recomputation from stored x; every D_i = 1 has a1 in the record |
| I-φ | The feature blocks reproduce the network's logits to bf16 tolerance |
| I-V | Once a round: per type, the mean of s_i ω_i (b + βᵀS_g) is 0 at a 5% family-wise line (3.4–3.8 SE) |
| I-match (new) | On the match's head seat, f and the switch logged at every band decision, x on a 1,000-game subsample: every switch played a1, f recomputes, the switch rate is reported beside the forecast's |

| Plant or control (lines at 3 SE) | Prediction | Failure readings |
|---|---|---|
| Label and fit: z′ = z + Σ_j s_j^rec(D_j^rec − P_j)(u·x_j), s^rec and D^rec parsed by separate code | Slope of T′ − T on u·x is 1 per mover side, on the smoke (SE 0.0387) and batch 1 (SE 0.0055); (ŵ′ − ŵ)ᵀu = uᵀ(G + λI)⁻¹Gu | Wrong side sign −1.005; coin one decision off 0.016 |
| Threshold: a = A·1(u′·x > q) − b in a separate copy of z, batches 1–2's stored subsamples, sized to beat τ = 0 by 10 SE; τ* computed before τ̃ is read | τ̃ ≠ 0; planted welfare within 2 SE of τ*'s | τ̃ = 0, or welfare short of τ*'s |
| Dice control: separate code fits z (as the labels read it) on side A's S_hp (`tools/match_luck.py`), smoke and every batch | Batch 1 in [+0.00268, +0.00565]; later batches within 3 SE of the earlier mean | Flipped result −0.00416; a share f of mis-joined results reads (1 − f) × 0.00416, caught at f ≥ 0.36 by the window and at a change of 0.03–0.04 batch to batch. The coin plant sees neither: z cancels in T′ − T |
| Match path and orientation, S0(a) | §2's windows | Sign error about +78 Elo or +0.449 z; hook on both seats about 0 |
| Distillation, S4: the baked end_turn shift | ĉ above 0 by 2 SE; 0.82–1.30 indicative (a tempo closure on another target) | ĉ near 0 |

**Report, not a gate:** the typed constant of (non-attack top, attack runner-up), expected negative (§2).

## 4. Regimen

| Stage | Cost [D] | Readout and prediction | Decision per outcome |
|---|---|---|---|
| **S0(a)** | $0.14 | Attack known answer: [−111, −41] Elo and [−0.640, −0.236] z | Outside either window: the match path or the label orientation is broken; fix before any coin game |
| **S0(b)** | $1.50 | `own2` against `own1`, SE 3.07. Record: +16.5 ± 12.3, so a reading in 16.5 ± 25.4 agrees with it | Lower bound > 0: π_0 = `own2` (also a PURE claim). Otherwise π_0 = `own1` |
| **S0(c)** smoke | $0.04 (in batch 1) | I-coin, I-log, I-φ, coin plant per side, dice control; census, runner-up mix, spectrum, overhead | A gate fails: debug. Else the power script reruns on the measured census and overhead, and S1 is proposed on restated floors and costs. Price × (1 + overhead) > $2.986e-5: two rounds (≤ $4.666e-5) or another host |
| **S1** round 1 | $15.30 | Batch 1: coin plant per side, ridge plant. Batches 1–2: threshold plant. Every batch: dice control. Batch 8: v̂, c, b, q̂, τ̃_8; Ĝ_pf and E's forecast sealed. v is not predicted: nothing on disk measures it | A plant or control fails: debug, the batch's labels quarantined |
| **S2** adoption | $0.72 (+ E $0.72) | π_1 against π_0; Ĝ_pf ± its miss line; I-match. Against `own1` the match is the claim; if π_0 is `own2`, the claim is a separate two-network match (+$1.50) | Lower bound > 0: adopt; S3. Miss with a positive forecast: E. Both null: S3-null with v̂ per type and the play bound |
| **S3** round 2 | $16.74 | Coins on π_1; its own Ĝ_pf; π_2 against π_1 (adoption) and `own1` (claim); round 1's J/Ĝ_pf reported | As S2; then S4, after the optimizer discussion |
| **S4** reset | $3.60 | ĉ, ρ̂, n_out held out; J_S4 against its one-sided prediction | ĉ not above 0 by 2 SE: distillation broken. Else N generates the next band (§3.6) |
| **S5** round 3 | $18.29 (+ E $1.50) | Round on N; adoption against π_k, two networks, predicted J_S4 + Ĝ_pf, miss line 12.7; claim against `own1` two-network | As S2 |
| **S3-null** | when chosen | v̂, Ĝ_pf, typed c, the play bound | The user chooses. v̂ near 0: **(i)** w 0.7 (125 band decisions per side-game; 0.310 of end-on-top decisions), **(ii)** critic column κ_T (~$7), **(iii)** Branch S's base, **(iv)** stop. v̂ above 0 with a miss: E from the logs, then one-seat coins |

**Budget** [D, `round8_synthesis_power.py` §1]: coin path $59.23 (E reserve $2.94 included), $25.77 left; $62.23 if `own2` becomes the base (two-network claims in S2 and S3); three rounds fit at up to 56% logging overhead on the measured host, 8% on a 1.41× slower one. A null round 1 costs $17.66 with all of S0. A batch takes 4.55 h, a round about 36 h.

**Branch S** (rulings (f), (g); it trains weights, so it waits for the optimizer discussion). One factor, a rating input. N is π_0 plus two zero-initialised inputs on the global token, a rating scalar s and a rated/unrated/own indicator, so N equals π_0 at every input (I-zero). Data: both sides of the non-holdout corpus games; rated sides (≥ 30 games) carry their out-of-fold rating jittered by its posterior sd, others are marked unrated, built-in AI sides give no targets; π_0's own game-sides carry their memoryless targets under "own". 4 M positions [C], sequence trainer.

| Stage | Cost [D] | Readout and prediction | Decision |
|---|---|---|---|
| **B1** | $2.59 | Plant (0.5 M positions): in 20% of games, at s = +500, one faction's commonest recruit relabelled to another; closure above 0 by 4 SE (0.82–1.30 indicative). Then the run; holdout loss with true s against s permuted among sides | Plant fails: the path is broken. Permutation gap not above 0 by 2 SE: s is unused, route closed at $2.59 |
| **B2** M1 | $0.72 | N(+200) against N(+40), pool, SE 1.79; f̂ = M1/160. No magnitude predicted; the sign is the hypothesis | Not above 0 by 2 SE: no transfer at this size; route closed at $3.32 |
| **B3** M2 | $1.50 | N(+200) against `own1`, one pre-registered arm, SE 3.07: the PURE claim. Decisions per side-turn logged | Lower bound > 0: a candidate base |
| **B4** | $1.50 (+ $3.60) | Only if M2 is null while M1 > 0: the arm at 0 slots against `own1` at 0 slots | Above 0 by 2 SE: the memory is the cost; the own-trajectory remedy and its match follow |

Order: if the discussion and the rulings come before S1, Branch S runs first, since its result picks the base and its exits cost less than a null round 1. Otherwise it runs when allowed, and one two-network match ($1.50) picks the next base between it and the incumbent coin player.

**Branch L, if ruling (a) is no.** Unchanged: five PURE matches of 3,800 independent games against `own1`, about $4.35; each arm minus its placebo has SE 8.0 Elo.

**The $1,000 case** is argued only after two adopted rounds within their miss lines, sized from measured v̂, J/Ĝ_pf, ρ̂ and τ̃'s moves; one replay then reads γV, DPI, antithetic siblings and a repaired CES as cost cuts. Nothing measured links `own1` to human ratings, and nothing rests on it.

## 5. What is new, and the nearest work

Searched 2026-10-20; finding nothing does not prove absence.
1. **Near-tie coins in deterministic self-play, made exact by the simulator's laws:** a natural policy-gradient step ([Peters & Schaal 2008](https://doi.org/10.1016/j.neucom.2007.11.026); [Bottou et al. 2013](https://jmlr.org/papers/v14/bottou13a.html) App. D.1) on a micro-randomised trial ([Boruvka et al. 2018](https://doi.org/10.1080/01621459.2017.1305274)), with exact chance-law control variates ([AIVAT](https://arxiv.org/abs/1612.06915)), a closed-form head on frozen features, certified by match. The assembly is new as far as found; the parts are not.
2. **A pre-tested threshold with known propensities.** Proxy-score evaluation covers its parts: BLP and GATES ([Chernozhukov et al.](https://www.nber.org/system/files/working_papers/w24678/revisions/w24678.rev0.pdf); E is the top group), [Imai & Li](https://arxiv.org/abs/1905.05389), RATE ([Yadlowsky et al.](https://arxiv.org/abs/2111.07966)), EWM. Post-selection caveats ([Leeb & Pötscher 2005](https://doi.org/10.1017/S0266466605050036), recalled) do not reach the claim, read on later batches and a match.
3. **A shape-free reading of a null:** the cross-split inner product ([Kong & Valiant 2018](https://arxiv.org/abs/1805.01626)) per pair type.
4. **The dice as a positive control** inside the same games: standard, in a new place.
5. **Branch S:** skill-conditioned imitation is covered ([Maia-2](https://arxiv.org/abs/2409.20553), KataGo's human networks); only the coverage-chosen setting and the match-certified base are this design's.

## 6. Weakest points

1. **v is unmeasured, and no record bounds it.** A null round 1 costs $17.66; it now buys a shape-free bound on the span.
2. **Ruling (a)** under its real name: without it, only Branch L and Branch S remain.
3. **From labels to play.** 19–38 switches against an uncoined opponent; a miss is detected, not sized.
4. **The band moves only through the reset,** whose closure on an exchanged-pair target is unmeasured.
5. **The optimizer ruling** holds S4 and Branch S, and S1–S3 too if ruling (e) treats the head as training.
6. **Branch S tops out near the covered human range;** past it, only the coin rounds move.

## 7. Rulings for the user

- **(a), restated.** The head's fit is one natural policy-gradient step for a near-tie coin policy: results regressed on randomised coins, exact control variates, one linear head on frozen features, a threshold tested on held-out labels, every step adopted only by match. The seed lists outcome-driven policy gradient as killed. Is this allowed? If not, branches L and S.
- **(b), narrowed.** The seven luck columns as control variates in labels only, never in play or matches. If not, the setup cell alone, at 1.225 times the coin games for the same floors.
- **(c).** Adoption and the claim by one PURE match of 37,800 independent decisive games in `run_elo_batch` at 120 jobs (SE 1.79, $0.72); two-network matches of 12,800 games (SE 3.07) at S0(b), S4, S5 and in Branch S. Claims stay against `own1` even if `own2` becomes the base.
- **(d).** The offset is baked into the reset's target, and the network plays at the residual o(1 − ĉ). Does that count as "gone"? If not, offset 0 costs about −9.9 ± 4.7 Elo at ĉ 0.82.
- **(e).** Does a closed-form head fitted on frozen features fall under the 2026-10-15 ruling? S4 does.
- **(f), new.** Is rating-conditioned training on both sides of the corpus within "imitation's remaining gains, parked" (2026-10-04)? It uses information that more epochs do not.
- **(g), new.** Two zero-initialised inputs on the global token (rating, and a rated/unrated/own indicator): an architecture change that starts as π_0.

## 8. Decision record

**Rejected or deferred this round:**
- **The interim match** (with E3): a stop only lowers power, fires at a null with P ½, and false-stops up to 0.50 in plausible shapes; money does not bind. If the user values dollars over power, a futility line at −0.5 SE, not 0.
- **γV and the census replay:** cost, not guarantees; no record holds V on `own1`'s games; the value head ranks turns at 0.22.
- **The √N ceiling:** its certification passes with P 0.013–0.039.
- **The fixed sign rule, λ chosen for welfare (0.547 of the oracle at worst), an untested welfare threshold (−0.6 to −1.1 Elo where the plug-in is right):** replaced by the tested threshold.
- **Scarf's cap as an exhaustion rule:** it needs a linear conditional mean, and at m = 0 it caps at 21 Elo, closing nothing.
- **The census as a pass/fail gate:** the transfer from `parity3` is unmeasured.
- **"N kept iff J_S4 ≥ 0":** it contradicted its own prediction and kept a 2.8-Elo deficit with P 0.18.
- **CES at the reset:** κ/SE 1.5 at the distillation's scale; a frozen-token ridge overfits by 0.36; an unfunded $13–15 replay; a plant sharing the label's join; post-strike inputs break exactness unseen. Deferred with its repairs (κ ≥ 0.0014, a separate parse, pre-strike inputs under I2 and I-V, a funded replay, the head's overhead measured).
- **Setup-conditioned luck prices:** null.
- **Guidance past the top rating, its three-arm selection, directed coin bands:** extrapolation is argued against by [UniMaia](https://arxiv.org/abs/2605.27767) (as cited by the challenge), [Transcendence](https://arxiv.org/abs/2406.11741) and [Decision Transformer](https://arxiv.org/abs/2106.01345) §5.2; the raw contrast lets implausible candidates win (1.5 nats at γ 2); the head needs spread, not a mean.
- **s_top = +253 and in-sample ratings:** one player supplies 0.605 of the sides; the input leaks the result.
- **Sentinel coins, prefix-branched coin games, span changes chosen on a pilot,** and the challenge's rejected cores: reasons in the round-8 files.

**Record corrections:** the (end, best act) pair covers 0.140 of end-on-top decisions at w 0.35, not the "continue" edits in general; the near-tie readings do not lean negative (stop-act readings −0.0028, +0.0043 on `parity3`, −0.0001, −0.0018); a 2% coin misread sits 3.7 SE away on batch 1, not 4.6; 0.688 is derived; the core's nearest work is the episodic natural actor-critic, not RESLOPE or LOLS.

**Standing from rounds 5–7** (reasons in their syntheses): corrected and mixture matches; α* and the g, A, C identification; constant intercepts per ordered type; the calibration slope; products of coin scores; the exchange score as a play rule; DPI and antithetic siblings before the $1,000 case; seed-shared pairs and the mirror null; cross-fitting; known-switch calibration; intents as the first route; D2; the separate P-distill run; the d_eff 200 case; the stopping machinery.

**Standing rules:** single-decision comparisons only; logged propensities; fresh-seed adoption; PURE claims; nothing distilled before a match certifies it.
