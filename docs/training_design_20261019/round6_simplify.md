# Round 6, simplify stance: one label, one head, one kind of match

The current design's guarantees rest on a coin with a logged propensity, luck terms of mean zero from the simulator's exact laws, a closed-form ridge head and a fresh match before adoption. This proposal keeps those and the guarantees (unbiased labels, no strategy fusion, a sealed model-free forecast, adoption only on a match, a plant per training path, no unknown relied on unmeasured), and removes or merges ten other mechanisms: about twenty code paths become ten. The largest cost is a 10% wider match SE. Scripts: `analysis/round6_simplify_*`. Tags: [M] measured, [D] derived, [U] unknown, [C] choice.

## 1. What goes, and what it costs

| Current mechanism | Here | Cost |
|---|---|---|
| **S0 stage**: replay of 800 games twice; flips; turn-over counts; census, E[R²] and d_eff before S1 | A 2,000-game smoke at the head of batch 1; the census, E[R²] and d_eff come from batch 1's own logs (d_eff from its Gram matrix) | None of S0's outputs fed a decision before batch 1. The flips carried none (current §4). One box session less. |
| **S0(b)**: the value head at every fight child, as a luck corrector | Deferred to the $1,000 case | Labels: it pays only if r_L < 0.19–0.37 (synthesis). Matches: it would have to beat the columns (r_L < 0.54 [D]), which this design already drops from matches as worth only about $2.6. It is a new code path. The value head ranks turns worse than the HP margin, 0.22 against 0.40 [M, seed §3]. |
| **Luck columns in matches**, the correction-term check, I8, separate PURE matches | A PURE side-swapped paired match. One reading is both the adoption test and the claim. | Pairs alone keep 0.847, against 0.700 with columns [M]. The SE rises from 2.57 to 2.83 Elo at 12,800 games. Restoring it costs about $2.6 over the plan [D]. The floors move by at most 0.0001 [D]. Ruling (b) leaves the adoption path. |
| **The per-decision luck split** (c_i before i, β·S_{>i} after) | One corrected result per game from the existing walk (`tools/match_luck.py`), plus the value head at i | None: the same estimator with a shared β (§2.2) |
| **The calibration slope β̂** | Ridge sign rule, λ chosen by progressive loss from sufficient statistics | None found. In simulation it is never behind and is within 0.6 Elo of oracle tuning (§2.3). |
| **Ĝ_8**, the model-bound forecast | Ĝ_pf only | In the model, the final student sits about half a forecast SE above Ĝ_pf [D] |
| **Mixture match, α*, rematch** | One fallback: the same student thresholded to half its switches, with a model-free forecast | It keeps at least 0.80 of the first-order gain against the mixture's 0.50 [D]. CPI's bound and G2's identification are lost. |
| **P-coin (a) and (b); the negative control on batches** | One plant through the game record, read two ways; I-φ; the negative control on the exact duel | The parity plant's independence from x is replaced by I-φ, which is exact |
| **Critic column and κ_T after a null; branch L's detail** | Deferred | After a null, the only designed continuation is w = 0.7 |
| **p and w retuning; S4's two offset matches** | Fixed; offset 0 is matched first | p = 0.5 would lower the floors by 5–10% (synthesis) |

## 2. The algorithm

### 2.1 Data

π_0 is `own1` at `raw:t0+eo-1.5` with 64 memory slots, playing both sides.

**Coins.** At a decision whose top two joint log-priors, after the offset, differ by G < w = 0.35 [C], a coin keyed by game seed and decision index plays the base runner-up a1 with probability P_i. P_i is p = 0.25 where the acting policy plays a0 and 1 − p where it plays a1 [C]. Play then re-decides after every roll. (End, best act) is an ordinary pair type.

**Logs.** Per band decision: P_i, D_i, the pair, the mover, the value reading V_i from the forward that set the pair, the progressive prediction f_i, the plant's g_i = u·x_i, the pair type. Per game: setup, z, A_g = Σ s_i ω_i x_i and B_g = Σ ω_i V_i x_i, with ω_i = (D_i − P_i)/(P_i(1 − P_i)) (2.5 GB per 800,000 games in fp16). Per batch: the Gram matrix. Afterwards: D^rec from the game record (checked against the logged D_i), S_g from the walk.

**Every coin game that starts enters the labels**, a capped one at z = 0, since dropping it would select on a post-coin event (`own1` self-play capped 0 of 1,600 [M, `round6_simplify_walk_cost.out`]). A crash stops the round.

### 2.2 The label

    T_i = s_i · ω_i · R_i,     R_i = z − b(e) − βᵀS − γ s_i V_i

Here e is the setup and S the game's seven luck sums. b, β and γ are cross-fitted over two folds of games, within the batches the reading student is fitted on (I4).

**Claim.** E[T_i | H_i] = Q̃(H_i, a1) − Q̃(H_i, a0) = Δ_i under the coin policy π̃, for any such coefficients.

**Proof.**
- b(e), βᵀS_{<i} and γ s_i V_i are functions of H_i, so they are uncorrelated with D_i − P_i.
- Each later luck term has mean zero given the position before its fight, and that position contains D_i.
- E[(D_i − P_i)z | H_i] = P_i(1 − P_i)(E[z | D_i = 1, H_i] − E[z | D_i = 0, H_i]). ∎

Conditioned on the mover's observation, the label estimates the information-state advantage, so there is no strategy fusion ([Baisero & Amato](https://arxiv.org/abs/2105.11674)).

**Consequences.**
- **One number per game.** Apart from γV_i, all labels of a game share z − b − βᵀS. The walk computes S in 0.321 core-seconds a game [M], 2.4–6.6% of generation's CPU [D, `round6_simplify_walk_cost.out`].
- **The columns pay in labels.** Setup alone keeps 0.882 in `own1` self-play, setup plus columns 0.732 [M]: 17% of the games, $3.36 a round, on any box above 8.9 cores [D]. σ² = 3.91 [D].
- **The value head.** If V_i does not help, its cross-fitted γ is near 0 and costs only estimation noise, so it needs no admission rule.

### 2.3 The student

**Features.** φ(a) = [actor token output, q(actor) ⊙ k(target hex), kind indicators, four factored log-probabilities], so d = 776 [C], and x_i = φ(a1) − φ(a0). The actor logit is linear in the actor token, and the target logit is q·k/√d (`wesnoth_ai/model.py:383, 401`), so x spans the network's own contrast.

**Fit.** Ridge with unpenalised intercepts per pair type, in closed form from the per-batch G_b and r_b = Σ_g [A_g(z_g − b − βᵀS_g) − γB_g].

**λ.** Student k takes λ̂_k = argmin_λ Σ_{j<k} [ŵ_j(λ)ᵀG_{j+1}ŵ_j(λ) − 2ŵ_j(λ)ᵀr_{j+1}], the progressive validation loss computed from statistics alone. Student 1 uses the two halves of batch 1.

**Rule.** Switch where μ̂_type + ŵᵀx_i > 0.

**No calibration slope.** Under a Gaussian prior, ridge at λ = σ²/τ² returns the posterior mean of Δ, and switching where it is positive maximises the expected first-order gain. A held-out λ estimates that shrinkage; a BLP slope on top re-estimates it from the same labels, and at small spread it crosses 0, where the calibrated rule turns unstable. Simulated (`round6_simplify_rule.py`: the synthesis's model made multivariate, d = 776, two spectra, 8 batches of 15.2 M labels, 300 replications), the sign rule is never behind the calibrated one, is up to 1.1 Elo ahead where P(β̂ ≤ 0) is 0.18–0.68 (sd ≤ 0.001), and is within 0.6 Elo of oracle tuning.

### 2.4 Forecast and adoption

**Forecast.** Ĝ_pf is sealed before the match. It sums batches 5–8's labels, per side-game, at the decisions where student k−1's choice differs from the acting policy's, each signed as the student's choice against the acting one. Its SE is 4.7 Elo at μ = 0 and 2.9–3.0 at μ = −0.0018 [D, `round6_simplify_power.out`]. A miss is a match below Ĝ_pf by more than 2√(SE_f² + SE_m²), about 11 Elo.

**Adoption.** 6,400 side-swapped pairs (same map and factions, seats and factions exchanged) on fresh seeds; Elo on the decisive games (`tools/elo_collect.py`), SE from the pair means, 2.83 Elo [D] (elo_collect's independent-games 3.07 is the conservative fallback). Adopt iff the lower 2-SE bound is above 0; false adoption at zero gain is 0.023 per candidate. Against `own1` the same reading is the PURE claim. Fishtest pairs color-reversed games the same way; its pentanomial README reports the independent-games test taking about 20% longer ([vdbergh/pentanomial](https://github.com/vdbergh/pentanomial)), consistent with 1/0.847.

**Fallback, on a miss whose lower bound is ≤ 0.** E is the same student with a threshold that halves its switches. Its first-order forecast is the sum of the progressive labels at f_i > t, which is model-free from the logged f_i. It gets one match under the same rule (two candidates: 0.046).
- **First order.** E keeps 0.80–0.87 of the first-order gain [D, `round6_simplify_rule.out`].
- **Second order.** If the losses come from pairwise interactions spread evenly, halving the switches quarters them. That is an assumption, and the match tests it.

### 2.5 Rounds and reset

**Round k** generates under the adopted π_k. Coins sit on the base band, oriented by π_k's choice. The student is refitted on all rounds' labels in the base orientation, and the progressive student logs beside the acting one: two weight vectors, one code path.

**Reset.** When the forecast's upper bound falls under 2 match SEs (5.7 Elo), the band is exhausted. π_k is then distilled by the `own1` recipe, and the distilled network's near-ties form the next band. Confident errors are never sampled; only distillation moves the band.

### 2.6 Signal per dollar

Coins carry 39.0 information units per game-equivalent against 0.82 for fork pairs [D, current §2], because one result informs about 150 independently randomised decisions. A match carries 17 times a held-out forecast batch's information per dollar [D, synthesis], so every batch trains. The luck columns earn their place in labels (17% of the games for 2.4–6.6% of the CPU), not in matches (about $2.6 over the plan, for a second reading and a correction fault mode).

### 2.7 Invariants and plants (tests from this spec)

**Invariants.**
- **I1** D matches the logged P (χ²), uncorrelated with pre-coin quantities.
- **I2** On an enumerable duel, E[T] = Q(a1) − Q(a0) exactly under each control variate; a post-coin variate fails (the negative control).
- **I3** Swapping a pair flips x and T, not the switch.
- **I4** No game a student reads enters its fit or coefficients.
- **I5** Every started game is labelled; crashes are 0.
- **I6** The statistics-based fit and progressive loss equal per-decision versions on 1,000 games.
- **I7** Logged switches and forecast sums equal offline recomputation, also with a planted ŵ.
- **I8** Pairs swap seats and factions exactly; the pair SE matches a hand computation.
- **I-φ** The feature blocks reproduce the network's own logits (actor head on the actor block, the q⊙k sum over √384, the log-probs) to bf16 tolerance; a wrong unit, hex, side or forward fails.

**Plants.**
- **P-coin (batch 1).** z ← z + 0.01·Σ_i s_i(D_i^rec − P_i)g_i, with g_i = u·x_i for a random unit vector u, and D^rec taken from the game record. It passes through z, the walk and the cross-fits. It is read as:
  - the slope of T on g, predicted 1 ± 2 SE, the SE computed blind from the logged g;
  - ŵ·u/0.01, predicted 1 ± 0.10 [D, current §3.6].

  The plant is then subtracted.
- **Every batch.** The mean labels per pair type, and along ŵ, with and without each variate group, agree within 2 paired SE.
- **P-distill (S4).** `own1` at −2.5 is distilled into `own1`. Known answer: 9.63 decisions per side-turn at −1.5 [M].

## 3. Regimen

Prices [D]: $2.47e-5 a game, $2.47 a 100,000-game batch, $1.90 a 12,800-game match.

| Stage | Cost | Readout and prediction | Decision |
|---|---|---|---|
| **S1 round 1** | $19.8 plus overhead [U] | **Smoke** ($0.05): I1, I5–I7 and I-φ pass; the price per game. **Batch 1:** the plant; the census (76 band decisions per side-game if `own1`'s margins match `parity3`'s [D]); E[R²] with and without V (no prediction: no record holds per-decision values); d_eff; coin cost by margin. **After batch 4:** stop if students 1–3's upper bound is under 5.7 Elo. P(stop) is 0.17 at a μ = 0 null, 1.00 at μ < 0 with sd ≤ 0.003, and 0.04 at sd 0.002 [D]. **After batch 8:** Ĝ_pf, v̂, μ̂. | **Smoke fails:** debug, before any batch. **Plant outside its band:** stop. **Early stop:** S3-null. **Otherwise:** S2. |
| **S2 adoption** | $1.90 (+$1.90 for the fallback) | π_1 against `own1`, paired and PURE, SE 2.83. Prediction: Ĝ_pf ± 2 combined SE. | **Lower bound > 0:** adopt; this is the claim. **Within the forecast's band:** S3. **Below it:** second-order loss; if the lower bound is ≤ 0, fallback E; if E is null too, S3-null. **Null forecast and null match:** no spread above sd 0.0015–0.0039 [D]; S3-null. |
| **S3 round 2** | $23.6 | As S1, under π_1; adoption against π_1; the claim against `own1`. Prediction: its Ĝ_pf. | Adopt by the rule. A forecast upper bound under 5.7 Elo: S4. |
| **S4 reset** | $4.6–6.5 | P-distill ($0.6). π_k distilled ($2.1 [M]). Then the distilled network at offset 0 against π_k, non-inferior if the estimate is ≥ 0 and the lower bound is above −5.7 Elo. | **P-distill outside:** keep π_k, widen to w = 0.7. **Offset 0 non-inferior:** the new base, with the offset gone (ruling d). **Otherwise:** the distilled network at −1.5 against π_k; if non-inferior it is the new base, else keep π_k and widen. |
| **S5 round 3** | $23.6 | As S3, on the new band | As S3 |
| **S3-null** | none until chosen | Round 1 bounded the spread | The user chooses between w = 0.7 under the same code (125 band decisions per side-game [D], $21.7) and a stop |

**Budget** [D]:
- At most $77.3. The $7.7 left covers logging overhead and the walk up to 13% of generation; beyond that, round 3 is cut.
- A null round 1 ends at $9.9 or $21.7.

**The $1,000 case** is argued only after two adopted rounds that fell within their forecast bands. It is sized from the measured v̂, second-order losses and per-round gains. Nothing links `own1` to human ratings, and nothing rests on such a link.

## 4. What is new, and the nearest work

Searched 2026-10-20; finding nothing does not prove absence. This stance subtracts; the core's novelty is the current design's.

**New here, as observations:**
- With independent coins and luck terms that are martingale differences, one corrected result per game serves all ~150 labels, so the AIVAT correction ([Burch et al.](https://arxiv.org/abs/1612.06915)) runs once per game, in an existing tool.
- λ is tuned by a progressive loss built from per-batch sufficient statistics.

**Nearest work:** micro-randomised trials ([Boruvka et al.](https://arxiv.org/abs/1601.00237)); Fishtest's SPSA and paired games ([pentanomial](https://github.com/vdbergh/pentanomial)); SPIBB, which keeps the baseline where data is lacking ([Laroche et al. 2019](https://arxiv.org/abs/1712.06924), not re-read); safe policy iteration, which maximises a lower bound on the improvement ([Pirotta et al. 2013](https://proceedings.mlr.press/v28/pirotta13.html), per-state variant not verified); the posterior-mean rule is textbook.

## 5. Weakest point

**One hypothesis.** About $22 tests a single hypothesis: that the frozen token features express a near-tie spread of sd ≥ 0.0015–0.0039 [D]. Nothing on disk measures it. On a null, the only designed continuation, a wider band under the same features, is likely to fail for the same reason. That is the price of removing the critic column and branch L from the design.

**The fallback has no theorem.** CPI bounds mixtures, not thresholds, so E's second-order behaviour is tested only by its match.

## 6. Provenance table

| Number | Value | Tag | Source or derivation | Used for |
|---|---|---|---|---|
| Pairs alone, kept share | 0.847 ± 0.009 | M | `analysis/round6_simplify_pairs_raw.py/.out`: swap covariance of setup cells on the raw centred result, `match_luck_20261019/rows.jsonl`, 18,800 decisive games | Match SE; dropping columns from matches |
| Pairs plus columns | 0.700 ± 0.009 | M | `round5_provenance_critique_pairs.out` | What matches give up |
| Paired match SE | 2.83 Elo at 12,800 | D | 347.4·√(0.847/12,800) | Adoption, floors |
| Independent-games SE | 3.07 Elo | D | 347.4/√12,800 | Conservative fallback |
| Elo per unit of z | 347.4 | D | 800/ln 10 at an even match | All SEs |
| Fishtest: independent SPRT ~20% longer than pentanomial | — | literature | vdbergh/pentanomial README (read) | Consistency check of 0.847 |
| Label kept share, setup plus columns | 0.732 (0.730, 0.734) | M | `round5_provenance_match_variance.out` (gen_own1_selfplay, own64_selfplay) | σ² |
| Setup alone, `own1` self-play | 0.882 | M | same; `round6_simplify_walk_cost.out` | Ruling (b) "no": floors ×√(0.882/0.732) = 1.10 [D] |
| Var(z) | 1.0004 | M | `round5_simplify_variance.json` | σ² |
| σ² | 3.91 | D | 1.0004 × 0.732/0.1875 | Label SEs |
| Walk time | 0.321 s mean (median 0.176) per game, one core | M | `round6_simplify_walk_cost.py` on rows.jsonl `secs`, 19,466 games, 31 jobs on an EPYC 7D12 slice (box.txt) | Columns in labels |
| Walk as a share of generation CPU | 0.024–0.066 | D | 0.321/(cores × 3,600/17,000), cores 64 to 23 (docs/box_specs.md) | same |
| Columns' saving in labels | 0.170 of games, $3.36 a round; pays above 8.9 cores | D | 1 − 0.732/0.882; same script | same |
| Columns' worth in matches | ≈ $2.6 over eight matches | D | (1 − 0.700/0.847) × 8 × $1.90 | Dropping them from matches |
| Learned corrector, match line | r_L < 0.54 to beat the columns | D | 0.700/1.3, m_match 1.3 from current §4 | Deferring S0(b) |
| Value head against HP margin, turn ranking | 0.22 against 0.40 | M | seed §3 | same |
| Capped games | `own1` self-play 0 of 1,600; all records 666 of 19,466, at z = 0 | M | `round6_simplify_walk_cost.out` | I5, pair drops |
| Band per side-game | 76.1 | D | 0.339 × 224.6 (`round5_simplify_band.json`, `round5_simplify_critique_decision_units.out`); batch 1 re-measures | Sizes |
| Labels per batch | 15.22 M | D | 2 × 76.1 × 100,000 | Simulation, SEs |
| Sign rule against calibrated rule against oracle | Sign ≥ calibrated in all 20 cases, ahead by up to 1.1 Elo; within 0.6 of oracle | D | `round6_simplify_rule.py/.out` (stated Gaussian model, 300 replications) | Removing β̂ |
| P(β̂ ≤ 0) | 0.18–0.68 at sd ≤ 0.001 | D | same | same |
| Threshold fallback E | 0.80–0.87 of the sign rule's gain at half its switches (sd ≥ 0.002) | D | same | Fallback |
| Mixture's share | 0.50 | D | first-order gain linear in α (Kakade & Langford 2002) | Fallback comparison |
| Floors (8 batches, p 0.25) | sd 0.0015, 0.0031 (d_eff 200; μ 0, −0.0018); 0.0020, 0.0039 (d_eff 776) | D | `round6_simplify_power.out` | Meaning of a null |
| 80%-power line | 8.0 Elo | D | 2.84 × 2.83 | Floors |
| Early-stop P | 0.17, 0.12, 0.04, 0.00 at μ 0, sd 0–0.003; 1.00 at μ −0.0018, sd ≤ 0.003 | D | `round6_simplify_power.out` | S1 |
| Forecast SE | 4.7 (μ 0); 2.9–3.0 (μ −0.0018) Elo | D | same | S2 band |
| Final student against Ĝ_pf | 16.3 against 14.0 Elo at sd 0.003 | D | `round5_synthesis_power.out` §2 | Removing Ĝ_8 |
| Prices | $2.47e-5 a game; $2.47 a batch; $1.90 a match; $0.05 the smoke | D | $0.42/17,000; 12,800 × $0.42 × 17/60/800 | Budget |
| Budget | ≤ $77.3; $7.7 left; overhead line 0.130 | D | `round6_simplify_power.out` | Regimen |
| Distillation recipe | $2.1 | M | `docs/memory_own_trajectories_prereg_20261011.md` | S4 |
| P-distill | $0.6; known answer 9.63 against 8.58 | D; M | current §4; `round5_provenance_stop_acts.out` | S4 |
| Plant, ridge reading | 1 ± 0.10 | D | current §3.6 | S1 |
| Plant, slope reading | SE from the logged g, before unblinding | D at run time | — | S1 |
| Numerics flips | 1.76% at gaps under 0.014 nats | M | `round5_provenance_numerics_ties.out` | Logged P, antisymmetry |
| Winner's curse | +35.7 → +1.7 | M | `docs/decode_knobs_prereg_20261012.md` | Fresh seeds |
| Band at w = 0.7 | 125 per side-game | D | 0.558 × 224.6 | S3-null |
| Coins against forks | 39.0 against 0.82 | D | current §2 | Instrument |
| Near-tie spread v; μ per pair type | — | U | S1's progressive fits; nothing before S1 relies on them | The round's gain |
| d_eff(λ) | — | U | batch 1's Gram matrix | Floor case (200 against 776) |
| E[R²] with V; `own1`'s census | — | U | batch 1; nothing before it relies on them | σ², band count |
| Logging overhead | — | U | the smoke | S1's dollars; whether round 3 fits |
| Second-order loss | — | U | S2: match against Ĝ_pf, and E's match | Round 2's rule |
| Distillation closure on a policy target | — | U | S4 | Reset |
| Compounding, exploitability, human level | — | U | per-round claims; none; none | Nothing rests on them |
| p 0.25, w 0.35, 8 × 100,000 games, d 776, δ 0.01, 12,800 games, E halves the switches | — | C | consequences derived above | Design |

## 7. Rulings and decision record

**Rulings.**
- **(a)** Coin labels, unchanged. Without them this design has no label.
- **(b)** The luck columns are used in labels only.
- **(c)** Adoption by PURE paired matches at SE 2.83, with the same reading as the claim.
- **(d)** Offset 0 first at each reset.

**Rejected:** the rows of section 1, for the reasons given there, and dropping capped games from labels, which selects on a post-coin event.
