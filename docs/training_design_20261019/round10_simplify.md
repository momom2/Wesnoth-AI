# Round 10, simplify stance: one head, one hook, one plant, and an $85 plan that trains no weights

Tags: **[M]** measured (record or script named), **[D]** derived (inputs named), **[U]** unknown, **[C]** a choice whose consequences are derived. Scripts are in `analysis/`; `r10 §n` is `round10_simplify_checks.out` section n, `r9s §n` is `round9_synthesis_checks.out` section n.

## Summary

The core stays: near-tie coins with logged propensities, labels exact through the chance laws, a closed-form linear head on the frozen network, adoption only by match, a fresh PURE claim after each adoption. Every guarantee of the current design is kept: false adoption 0.023 per step, per-round claim checks at false-miss 0.042–0.053, a planted check on every code path, and no costly stage resting on an unmeasured number.

- **Removed:** T, C and H1; `own2`+R as a base; the typed block; the R² report; the in-run spread estimator. The reset, Branch S and Branch L leave the $85 plan.
- **Merged:**
  - R becomes the head with a unit block and zero token weights, so one fit, one hook and one plant serve both.
  - I-R folds into I-match.
  - The smoke moves onto L's box.
- **New use of a measured finding:** R is the known answer for the production code. The processor must reproduce R from batch 1's records, and L plays R with S2's own hook.

The path through a third round costs $79.57 at zero dump overhead, against $86.87 [D, r10 §4].

## 1. What the records move

| Measurement | Decision |
|---|---|
| S0(b): `own2` +22.7 ± 4.4 over `own1` [M] | π_0 = `own2`. Round 1's generation is the certified batch-1 path with only the checkpoint changed |
| Batch 1's recruit spread; R at +21.3 ± 3.1 Elo first order [M, r9s §1] | R is the known answer of the fit code (R0) and of the hook in play (L) |
| R's welfare by ridge: +22.9 at 500–2,000 labels, +21.3 at 5,000, +19.3 at 50,000 [M, r10 §1] | One ridge for every column. A token block that drives λ up costs the recruit part at most 3.6 Elo on 60,000 games, and less on 800,000 |
| R's welfare curve: best τ beats τ = 0 by +0.8 ± 1.6 (τ −0.01); τ +0.005 loses 1.7 ± 1.2 [M, r10 §2] | The tested threshold stays at 0 on the one measured structure, as derived where the plug-in is right. It is kept as a guard on unmeasured shapes |
| Typed cells: type V z +0.39, cross-fitted type-sign welfare +8.3 ± 15.9 Elo; type × gap z +0.16 [M, `round9_provenance_cells.out` §2] | The typed block goes |
| The coin code takes one `coin_p` per player [M, `exp/near-tie-coins:tools/raw_player.py` lines 145–176, 244–249] | A per-decision propensity (0.75 where the player switches) is new code. `own2`+R as π_0 would have needed it in round 1. Here it first appears in round 2, after a head has passed S2 |
| Prices [M, `round9_simplify_labels.out`] | Three rounds by one code fit iff o ≤ 0.040 (§5) |

## 2. The algorithm

**The player.** π_k is `own2` at `raw:t0+eo-1.5` with 64 memory slots, plus at most one linear head.
- **Band.** At a decision whose top two joint log-priors, after the end_turn offset, differ by G < 0.35 [C], call the argmax a0 and the runner-up a1.
- **Switch.** π_k plays a1 iff f_k = w_kᵀ(φ(a1) − φ(a0)) > τ_k.

The network proposes every action. The head chooses between the network's top two at one decision, and play re-decides after every roll. The head reads only the mover's own tokens and candidates, so the acting player uses only its own observation.

**Features.** φ(a) = [actor token output, q(actor) ⊙ k(target hex), four factored log-probabilities], 772 values. A 43-wide faction|unit one-hot is added on recruit candidates iff ruling (h) allows.
- **R is this head with the token weights at zero.** The synthesis fitted R by `fit_pred` on the one-hot difference e(a1) − e(a0) (`round9_synthesis_checks.py`), which is this head's fit restricted to the unit block.

**Generation.** π_k self-plays on the pool, both sides coined.
- **Coin.** Keyed by (seed, side, decision index), it deviates from π_k's choice with probability ¼. This is built for k = 0 and new for k ≥ 1.
- **Logs.** Per band decision: D, P, G, both candidates, their unit identities and the pair type. Per game: z and the seven luck sums S_g.
- **Dump.** The inference server writes x = φ(a1) − φ(a0) per band decision (24 GB per 100,000-game batch [D]).

**Label.** In the network's orientation (D = 1 iff a1 was played; P = ¼ where π_k plays a0, ¾ where it plays a1):

    T_i = s_i ω_i R_g,   ω_i = (D_i − P_i)/(P_i(1 − P_i)),   R_g = z − b(cell) − βᵀS_g

- **Baselines.** b and β are fitted once on `own1`'s 60,000 coin games and frozen.
- **Exactness.** E[T_i | H_i] = Q̃(H_i, a1) − Q̃(H_i, a0) under the coin policy, for any coefficients fixed before the game. Two facts give it: the coin is independent of H_i, and each later luck term has mean zero given its pre-fight state. Under fog it is the information-state advantage.
- **Noise.** E[T²] is 4.01; the within-game correlation ratio is 0.999 [M].

**Fit.** An offline processor makes one pass per match j:
- G_j = Σ x xᵀ;
- b_j = Σ_g R_g Σ_{i∈g} s_i ω_i x_i;
- x kept for a fixed 1,000-game subsample.

Then:

    ŵ_k(λ) = (Σ_j G_j + λI)⁻¹ Σ_j b_j
    λ̂ minimises the progressive loss Σ_j [−2 ŵ_{j−1}ᵀ b_j + ŵ_{j−1}ᵀ G_j ŵ_{j−1}]

- **The progressive loss** has expectation Σ_i (ŵᵀx_i − Δ_i)² − Σ_i Δ_i². It is an exact estimate of the prediction error up to a constant, from per-batch statistics. It also decides whether an earlier round's statistics are pooled.
- **Bias and variance.** The bias is ridge shrinkage only; the variance is σ²(G + λI)⁻¹G(G + λI)⁻¹.
- **Name.** With P(1 − P) constant, the fit is one episodic natural actor-critic step for a logistic coin policy ([Peters & Schaal 2008](https://doi.org/10.1016/j.neucom.2007.11.026)). Ruling (a) covers it.

**Threshold.** The welfare curve W_j(τ) = Σ_{f_i>τ} T_i, of student j − 1 on batch j, is unbiased for the first-order gain of {f > τ} at every τ ([Kitagawa & Tetenov 2018](https://doi.org/10.3982/ECTA13288)). τ̃ maximises W over the latest four pairs if that beats τ = 0 by 2 SE of the difference, and is 0 otherwise.

E's restriction, used only after a miss, switches only above the previous batch's half-quantile of f within each pair type. Both rules are one per-type threshold vector applied by one function.

**Forecast, adoption, claim.**
- **Forecast.** Ĝ_pf = 347.4 × the labels per side-game of batches 5–8 where students 4–7 switch at their τ̃. It is sealed with SE 4.88 and a miss line of 10.4 [D].
- **Adoption.** The new head plays one seat, chosen by the built seat coin, against π_k: 37,800 decisive games (SE 1.79), read on raw z. Adopt iff the lower 2-SE bound is above 0 (false adoption 0.023).
- **Claim.** π_{k+1} then plays `own1` (12,800 two-network games, SE 3.07). The check compares D_k = claim_k − claim_{k−1} with the adoption reading at 2√(SE_D² + 1.79²): 9.4 Elo, or 11.2 against S0(b). False miss is 0.042–0.053 at any step size [D, r9s §5].
- **A claim miss stops the chain.** The next round would compound on a gain that does not reach `own1`.

**Rounds.** Round k coins π_k's band. The head is refit on round k's batches, plus earlier rounds' statistics iff the progressive loss says so. Stale labels can bias a student, never a claim.

**Signal per dollar.**
- One result prices about 155 near-tie decisions with no playout or critic: 38.7 information units per game-equivalent against 0.82 for a fork pair [D, `round9_simplify_budget.out`].
- Exploration costs +2.0 ± 4.7 Elo, against −412 ± 97 for sampling at temperature 1.
- The update is a ridge solve: no optimizer, no learning rate, every statistic additive and recomputable.
- A match carries 78× the labels' information per game about one fixed rule, so labels fit and matches certify.

**Invariants.**

| Check | What it sees |
|---|---|
| I-coin, I-log (built, passed on `own1`) | Coins match the logged P and are independent of the past; every band decision played as the coin chose. A wrong P in round 2's new code fails I-coin's χ² directly |
| I2, I3 (tests from the spec) | An enumerable duel through the production generator gives E[T] = Q(a1) − Q(a0) exactly, and a post-coin variate fails it. A swap flips x and T |
| I-φ | The dump's log-probability block reproduces the gap the coin record logged, checked by separate code. This catches a mis-joined x |
| I-proc | The statistics recompute from the subsample |
| I-V per type | The mean of sω(b + βᵀS) is 0 at the family-wise 5% line |
| I-match (absorbs I-R) | On a head's seat, f and the switch are logged at every band decision. Every f > τ played a1 and no other did. f recomputes from stored x, or from unit identities in L. L's fire count is above 0 on every faction |
| I-side | The seat coin's (seat, side) χ²; the first-mover effect is +9.3 ± 1.4 Elo [M] |

## 3. What goes, what merges, and what each costs

| Change | Cost | Gain |
|---|---|---|
| **T and C removed; R never enters a player except as the head's unit block** | No reading of R across bases. R's value on `own2` is banked only through S2's head. If S2 and E both miss, recruit structure goes unbanked. A unit-block-only head refit on `own2` can then be played with L's code for $0.98, at the user's call | $2.11. Round 1 runs the built coin path. The new propensity code waits until a head is certified |
| **H1 removed** | No head is read in play before batch 8. Students never play in generation, so H1's actions (E's restriction, unit columns) can all be applied at S2 from the logs. Where H1 would fire and S2 would miss, H1 then restricted-S2 and S2 then E both cost two matches. H1 can also lose gain: a false fire (P 0.023 [D, r9s §3]) or a one-batch student unlike the eight-batch one imposes E's restriction where the full head would pass | $0.98 in every world where H1 does not fire; one match type and its decision table |
| **R's own fit, plant and hook merged into the head's** | None: the synthesis's R is the ridge on the one-hot difference | L tests S2's hook on real signal; R0 tests the processor against independent code |
| **Typed block removed** (16 unpenalised columns, the Λ mask) | A type-level mean on `own2` that the token span does not carry goes uncollected. On `own1` the typed cells hold no spread (table in §1) | No antisymmetric construction and no mask. Even kept, penalising the constants would move them by a factor of 0.857 to 0.988 at λ 5,000 [D, r10 §3] |
| **One ridge, not one per block** | At most 3.6 Elo of R's part on 60,000 games [M, r10 §1] | One hyperparameter |
| **R² report and recruit readout as gates removed** | None. Unit columns are in φ iff (h), whatever R² reads; the recruit readout stays an offline diagnostic | — |
| **v̂ out of the run** | A null is read from Ĝ_pf's upper bound and per-type welfare curves. A pooled v̂ remains computable from the stored G_j and b_j | One estimator fewer in the run |
| **Smoke on L's box** | o is measured on box A's host and re-read from every batch's walls | One setup ($0.21); the user approves S1 knowing L and o |
| **Reset, Branch S and Branch L out of the $85 plan** | The band does not move within $85 | No training run and no rulings (d), (f), (g) inside the plan. A third refit fits at o ≤ 0.040 |

**Kept, with reasons:**
- **E.** It costs $0.98 and is spent only on a miss. It can rescue a $25 round when second-order loss is the cause, and it is one mask on the threshold function.
- **The per-round claims.** They give attribution, and a miss now has an action.
- **τ̃.** It guards against unmeasured shapes, and it stays at 0 on R.
- **I-V per type.** All the measured signal sits in one type.
- **The rest of the machinery:** the seat coin (built), frozen b and β, the dump and the offline processor.

## 4. Regimen

| Stage | Cost [D, r10 §4] | Readout and prediction | Decision |
|---|---|---|---|
| **R0**, offline | $0 | **CI:** I2, I3, I-match and I-side on a planted table; I-φ on a CPU forward; the coin plant with per-decision P on synthetic games. **Known answer:** the processor, fed batch 1's records with x = the unit block, reproduces the synthesis's λ̂ = 5,000, R's 43 values and F = +21.3 ± 3.1 to float tolerance. **Merged plant** on the same records, z′ = z + Σ s(D − P)h(x) with h = a·1(uᵀx > q) − c at 10 SE: the fit change equals the processor's predicted (G + λI)⁻¹Σ x h(x) within 3 SE, τ̃ leaves 0, and the planted welfare is within 2 SE of τ*'s | Any red: fix before a box |
| **Box A: L, then the smoke** | $1.28; L's part $0.98(1 + o_h) | **L:** `own1` with R, through S2's hook, on one seat against `own1`, uncoined, 37,800 games. Sealed at F ± 7.2. P(lower bound > 0) is 0.92 at transfer ratio 0.3 and 1.00 at 0.5 [D, r9s §2]. Reported beside it: the switch rate against 2.21 (expected a few percent lower, since after a played switch the rate is 0.56× [M]); I-match; the hook's overhead o_h. **Smoke:** `own2`, 2,000 dumped and 1,000 undumped coin games. Gates: I-coin, I-log, the coin plant, the dice control in [+0.00245, +0.00587], I-φ, I-proc. Readings: census, E[T²], o, processor time | **L's lower bound > 0:** S1 is proposed. Whether L lands inside F ± 7.2 is reported; the ratio is never used to scale a forecast. **Lower bound ≤ 0** (ratio < 0.34): re-check I-match and the label path. If both are clean, no round is bought, and the user chooses between the reset, Branch S and Branch L. **A smoke gate fails:** debug. **o:** the user learns how many rounds fit before approving S1 |
| **S1:** batches 1–8 | $22.79(1 + o), 39.5 h | **Every match:** coin plant, dice control with the batch-to-batch line, I-coin, I-log, I-V. **Batches 1–2:** the merged plant on the full φ. **Batch 8:** λ̂ and τ̃; Ĝ_pf and E's forecast sealed; per-type welfare curves; the learning curve (student j − 1's welfare on batch j, j = 2–8) | A check fails: the batch is quarantined |
| **S2 + claim** | $0.98 + $1.13 | The head against π_0, at Ĝ_pf ± 10.4; I-match. The claim's D_1 against A_1, line 11.2 | **Adopt** iff the lower bound > 0, then claim. **Claim inside its line:** round 2 is proposed, with the learning curve beside it. **Outside:** the chain stops and the miss is reported. **Ĝ_pf above 0 by 2 SE, match below its line:** E. **Both null:** this head collects nothing above Ĝ_pf's upper bound; the user chooses (band 0.7, the reset, Branch S, stop) |
| **E**, after a miss only | $0.98 | The restriction against π_0, read against its own sealed forecast | Adopt iff the lower bound > 0. Otherwise the round adopts nothing |
| **Round 2** | $25.12(+o) | The same code on π_1. Its first match gates the new propensity code (I-coin, coin plant, I-log); pooling is decided by the progressive loss | As S2 |
| **Third slot** | $25.12, or the reset | Bought iff round 2 adopted, its claim check passed and o ≤ 0.040. The alternative is the reset, if the optimizer discussion has concluded | The user's choice |

**Planted or known-answer check per code path:**
- **Coins and labels:** the coin plant on every match (0.994 and 0.995 ± 0.008 on `own1`), the dice control (0.00436), and I-log.
- **Dump join:** I-φ.
- **Processor and fit:** R0's reproduction of R, and I-proc.
- **Fit, threshold and forecast:** the merged plant, at R0 and on S1's batches 1–2.
- **Hook in play:** L's sealed prediction on real signal, and I-match on every head seat.
- **Round 2's propensity:** I-coin and the coin plant.

## 5. Budget

From $82.34 [D; this assumes the $5 window counted], at $0.577 an hour, with E reserves counted [D, r10 §4]:

| Through | Cost at o = 0 |
|---|---|
| Box A | $1.28 |
| Round 1 | $27.38 |
| Round 2 | $53.47 |
| A third refit | $79.57 |

- **Two rounds** fit iff o ≤ 0.632.
- **Three rounds** fit iff o ≤ 0.040, or o ≤ 0.079 if the window did not count. At o = 0, they fit iff a host's all-in cost per coin game is at most $2.964e-5 (1.040× the measured host).
- **A null round 1** after box A costs $25.26.

Round 9's path costs $86.87 whole, and $56.57 through round 2 [D, r9s §6].

## 6. From `own1` to beyond the strongest humans

**The loop.**
1. Coin rounds on a frozen network collect the value of its near-tie band, one certified head at a time.
2. The reset distils the adopted head into the network, on `tools/own_targets.py`'s top-16 target with the switched pairs exchanged (round 8's S4, its plant ĉ > 0 by 2 SE). This moves the band.
3. Coin rounds run on the new network.

Every step is adopted by match and claimed against `own1`. Self-play is the generator throughout.

**What waits.** The reset is the only training run. It waits for the optimizer discussion (ruling of 2026-10-15). The $1,000 case is argued only from at least two adopted rounds with passing claim checks and their measured learning curves.

**What nothing rests on.** Nothing measured links `own1` to human ratings, so no stage depends on that link.

## 7. What is new, and the nearest work

Searched 2026-10-21; not finding something does not prove it absent.

1. **A finding from the labels used as the known answer of its own production path.** R is refit by the production processor, which must agree with independent code. This is differential testing ([McKeeman 1998](https://dblp.org/pid/40/4856), *Digital Technical Journal* 10(1); bibliographic record only). R is then played by the production hook at a sealed out-of-sample prediction against an uncoined opponent. This is the offline–online check of [Gilotte et al. 2018](https://arxiv.org/abs/1801.07030), here with an exact estimator. The planted plant stays; L adds a check on real signal with no game modification. I found no self-play learner whose first in-play check is its own measured finding.
2. **The player: a frozen imitation network plus one closed-form least-squares head over its near-tie band, refit each round on exact per-decision advantage labels from logged coins.** The nearest is LS-DQN ([Levine et al. 2017](https://arxiv.org/abs/1705.07461)), which refits a deep network's last layer by regularised batch least squares, but on bootstrapped TD targets and over all actions. The others are the episodic natural actor-critic, and tie-breaker designs ([Owen & Varian 2020](https://arxiv.org/abs/1808.07563)), which randomise only near a cutoff.
3. **Fresh per-round claims instead of selection-adjusted inference.** [Andrews, Kitagawa & McCloskey 2024](https://www.nber.org/papers/w25456) (QJE 139(1); abstract read) give conditional and hybrid inference on a selected winner. Here no selected reading is ever used as an estimate, so no correction is needed. Rejected: their hybrid intervals, because the plan has no use for them.

**Contradicting evidence on the token span.** [Brunner et al. 2020](https://arxiv.org/abs/1908.04211) find that input tokens largely keep their identity in a transformer, but that identity fades with depth. The 8-layer span may therefore not carry unit identity linearly, which is why ruling (h) matters.

## 8. Weakest point

**Within $85 the band never moves, and the only measured spread sits in recruit pairs:** 4.29 of 77.6 band decisions per side-game on `own1`. Unit-keyed cells outside recruits read +0.90 SD [M].

- **If `own2`'s band outside recruits holds nothing a linear head collects,** round 1 adopts at most a recruit head.
- **Round 2 re-prices the same band.** Its purchase rests on round 1's adoption and claim, not on a forecast of round 2's own gain: the value of re-pricing is the first measurement of policy iteration on this band, and nothing measured predicts it. The learning curve shown to the user before round 2 says whether more labels of the same kind help, not whether re-pricing does.
- **If ruling (h) is no and the span does not carry unit identity,** R's value reaches no player within $85.

The current design shares all three. This one reaches the end of round 2 for $3.10 less ($53.47 against $56.57).

## 9. Rulings for the user

- **(a), unchanged:** one natural policy-gradient step on a near-tie coin policy, adopted only by match.
- **(b), unchanged:** the luck columns and the frozen cell baseline are used in labels only.
- **(c):** per-round claims; a claim miss stops the chain.
- **(e):** does a closed-form head on frozen features fall under the 2026-10-15 ruling? L plays R as a measurement.
- **(h), narrowed:** may the head carry 43 faction|unit columns? R itself never enters a player.
- **Budget:** did the $5 window come out of the $85? It decides whether a third refit fits at o ≤ 0.040 or at o ≤ 0.079.
- **Not asked:** (d), (f) and (g) belong to the reset and Branch S, which sit outside this plan.

## Provenance

| Number | Value | Tag | Source or derivation | Used for |
|---|---|---|---|---|
| `own2` against `own1` | +22.7 ± 4.4 (SE 4.35), 6,400 decisive | M | `training/metrics/own2_base_20261020/reading.json` | π_0; claim_0 |
| Attack known answer | −41.3 ± 4.4 Elo; −0.242 ± 0.023 z | M | `near_tie_coins_20261020/readout_a_offline.txt` | Transferred windows are reports |
| Coin path on `own1` | plant 0.994 / 0.995 ± 0.008; dice slope 0.00436 ± 0.00005; I-log 0 of 9,328,402 mismatched | M | `readout_b_offline.txt` | Gates kept as built |
| Band census | 77.6 ± 0.13 per side-game | M | `readout_b_offline.txt` | Dump size; information per game |
| Pooled label mean; coin play's cost | +0.0003 ± 0.0007; +2.0 ± 4.7 Elo | M; D | `readout_b_offline.txt`; `round9_simplify_budget.out` | Free exploration |
| R | F +21.3 ± 3.1 Elo at ridge 5,000; 2.19 switches per side-game (2.21 for the all-data rule); 43 values, 262 cells | M | r9s §1 | L's prediction; R0's known answer |
| F by ridge | +22.9 (500–2,000), +21.3 (5,000), +20.1 (10,000), +20.2 (20,000), +19.3 (50,000) | M | r10 §1 | One ridge |
| R's welfare curve | W(−0.01) − W(0) +0.8 ± 1.6; W(+0.005) − W(0) −1.7 ± 1.2; W(+0.02) − W(0) −7.3 ± 2.2 | M | r10 §2 | τ̃ kept; stays 0 on R |
| Typed cells | type V z +0.39, welfare +8.3 ± 15.9; type × gap z +0.16, +16.2 ± 13.3 | M | `round9_provenance_cells.out` §2 | Typed block removed |
| Typed shrinkage | factor 0.857 (smallest type, 30,000 labels a batch) to 0.988 (median) at λ 5,000 | D | r10 §3, counts from `round9_provenance_cells.out` §4 × 200,000 | No unpenalised mask |
| Recruit spread | V_rr 1.59e-3 ± 3.7e-4; 4.29 per side-game | M | `round9_provenance_recruit_cells.out` | v > 0 for one family |
| Outside recruits | unit cells +0.90 SD | M | `round9_provenance_unit_cells.out` | Weakest point |
| Label noise | E[T²] 4.01; residual 0.699 ± 0.003, 0.692 on `own2`; within-game ratio 0.999 | M | `round9_simplify_labels.out`, `round9_simplify_residual.out` | Every SE; frozen b, β |
| Forecast | SE 4.88 at switched share ½; miss line 10.4 | D | `round9_simplify_budget.out` | S2 |
| Match SEs | 1.79 (37,800), 3.07 (12,800) | D | 347.4/√N | L, S2, E, claims |
| False adoption | 0.023 | D | Φ(−2) | S2, E |
| L's lines | miss half-width 7.2; null line 3.57; ratio bound 0.34; P(lower bound > 0) 0.92 at ratio 0.3 | D | r9s §2 | L |
| Claim checks | lines 9.4 / 11.2; false miss 0.042–0.053; cumulative prediction biased +5.85, miss rate 0.126 | D | r9s §5 | Per-round claims |
| H1's false fire | 0.023 at SE 2.53 | D | r9s §3 | Cost of removing H1 |
| R's clustering | 0.583 of switches follow an earlier switch of the same side; same-turn rate after a played switch 0.56× | M | `round9_simplify_critique_clusters.out` | L's switch-rate report |
| First-mover effect | +9.3 ± 1.4 Elo | M | `round9_challenge_critique_checks.out` A | I-side |
| Coins against forks; match against labels | 38.7 against 0.82 per game-equivalent (47×); 78× | D | `round9_simplify_budget.out` | Signal per dollar |
| Coin propensity in code | one `coin_p` per player | M (code read) | `exp/near-tie-coins:tools/raw_player.py` lines 145–176, 244–249 | Round 2 needs new code; T removed |
| Prices | coin game all-in 20,254 an hour ($2.849e-5); pool 22,209; two-network 6,521; $0.577 an hour; setup 22 min (7 measured + 15 staging [C]) | M | `round9_simplify_labels.out` | Every cost |
| Path costs | box A $1.28; a round $25.12; E $0.98; through round 1 $27.38, round 2 $53.47, a third refit $79.57; null round 1 $25.26 | D | r10 §4 | Regimen |
| Rounds that fit | two iff o ≤ 0.632; three iff o ≤ 0.040 ($82.34) or 0.079 ($85); ≤ $2.964e-5 a coin game at o = 0 | D | r10 §4 | Third slot; host choice |
| Round 9's path | $86.87 whole; $56.57 through round 2 | D | r9s §6 | Comparison |
| Budget left | $82.34 | D, on an assumption | $85 − $2.66 (seed §7); the window counted | Third slot |
| Dump; round wall | 24 GB per batch; 39.5 h | D | 2 × 77.6 × 772 × 2 B × 100,000; `round9_simplify_budget.out` | Box disk; schedule |
| Dice windows | smoke [+0.00245, +0.00587]; batch [+0.00267, +0.00565] | D from M | `round8_synthesis_power.out` §2 | Smoke and batch gates |
| Dump overhead o; processor time; hook overhead o_h | — | U | o at box A's smoke and from every batch's walls; o_h at L. Nothing before them relies on them | Rounds that fit; S2's cost |
| L's transfer at R's step | — | U | L | Whether S1 is bought |
| `own2`'s spread, τ̃, q, Ĝ_pf | — | U | S1. Nothing before S1 relies on them | The round's gain |
| Second-order loss at the head's step | — | U | S2 against Ĝ_pf; E | Adoption (certified by match either way) |
| Value of re-pricing in round 2 | — | U | Round 2 | Round 2's gain. Its purchase rests on round 1's adoption and claim (weakest point) |
| Whether the $5 window counted | — | U | The user | Third slot |
| Compounding past the band, exploitability, human level | — | U | Claims; none; none | Nothing |
| w 0.35, P ¼, 8 × 100,000 games, match sizes, plant at 10 SE | — | C | Consequences derived above and in round 9 | Design |
