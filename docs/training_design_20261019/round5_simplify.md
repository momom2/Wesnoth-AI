# Round 5, simplify stance: one coin, one regression, one match

The current design's guarantees rest on three facts:
- A coin with a logged propensity makes one decision's effect identifiable from the result.
- The simulator's exact chance laws give control variates of mean zero.
- A fresh match certifies any change.

The rest of it (section 1) exists to extract decision information from a critic. The record says no critic ranks alternatives better than material (`critic_step1_20261009/readout.md`), so this proposal removes it.

What remains:
- argmax self-play, with coins at the decode's near-ties;
- a per-decision label, exactly unbiased whatever its control variates;
- a closed-form ridge head on the network's frozen token outputs;
- a switch to the runner-up where the head is positive;
- a forecast on held-out coins;
- a critic-free corrected match.

That is four new code paths and two planted checks, a third if the critic re-enters. Section 6 traces every number.

## 1. Kept, merged, removed

| Current mechanism | Here | Cost of the change |
|---|---|---|
| Privileged critic: multistep loss, folds, corrector, P1, I5, I6 | Removed; re-enters only as a control variate (2.5), where it cannot bias | Dense look-ahead targets: if a critic ranks single decisions (κ > 0, never measured), round 1 learns more slowly than it could |
| Decision windows | Removed | None measured: their χ² rested on a guessed spread |
| Consequence heads, gating, P-cons | Removed | An inductive-bias bet for trunk training, which no round here does |
| Factored attack arm, P-law | Removed | Exact fight laws in the student (it also needed ruling b) |
| LS stop on forced paths, stop window, run lengths, twin, P3 | Merged: (end, best act) is one more near-tie pair | Stops corrected only near the offset's boundary, one act per round |
| Tapered band coins, weak coins, λ, c0, coin-cost arms, coin-free quarter | One flat coin; the labels' mean by margin bin reads the cost | The taper carries 6.67 units of information per nat of shift, the flat coin 4.67 (`round5_simplify_band.json`), a cost only if deviation cost grows with margin (unmeasured); margins ≥ 0.35 are never sampled |
| κ gate, β_avg, posterior ≥ 0.8, at most 10 switches | One ridge fit and its held-out slope | More switches, larger second-order error, read by the match |
| Forecast ladder, factor and mixture matches | One forecast | A miss does not localize horizon against selection |
| Critic-corrected matches | Critic-free correction, measured 0.747 | A critic might cut it further (unmeasured) |
| Distillation every round, stacked heads | Distillation only to reset an exhausted band | None in round 1 |

## 2. The algorithm

### 2.1 Data

π is `own1` at argmax with its offset. At a decision whose top two joint log-priors, after the offset, differ by G < w = 0.35 (the band), a coin with its own generator, keyed by game seed and decision index, plays the runner-up with probability p = 0.25. The log records p_i, the pair (a0_i, a1_i), D_i = 1[a1 played] and the side. End_turn is an action like any other, so near the boundary the pair is (end, best act). After a coin, π re-decides after every roll; both sides draw coins. `own1` has about 62 band decisions per side-game, 124 labels a game (derived: `parity3`'s band share 0.339 times `own1`'s 182.7 decisions per side-game; S0 measures it).

### 2.2 The label

    T_i = s_i · (D_i − p_i) / (p_i (1 − p_i)) · R_i,     R_i = z − c_i − β·S_{>i}

- z is side A's result, and s_i the mover's sign.
- c_i is any function of information fixed before the coin, with coefficients fitted outside the batch it labels: the setup baseline, the luck before i, `own1`'s value head at i.
- S_{>i} holds the seven luck columns' terms after i (`tools/match_luck.py`).

**Claim.** E[T_i | H_i] = Q̃(H_i, a1) − Q̃(H_i, a0) =: Δ_i, the runner-up's advantage under the coin policy π̃, for every c and β.

**Proof.** D_i is independent of the history H_i, so E[(D_i − p_i)c_i | H_i] = 0. Each later luck term has mean zero given its past, which includes D_i. Finally, E[(D_i − p_i)z | H_i] = p_i(1 − p_i)(E[z | D_i = 1, H_i] − E[z | D_i = 0, H_i]). ∎

Consequences:
- **No strategy fusion.** Conditioning on the mover's observation h_i gives the information-state advantage E[Δ_i | h_i] ([Baisero & Amato](https://arxiv.org/abs/2105.11674)), so the fogged student is safe.
- **Other coins** enter R as mean-zero noise.
- **Variance.** Var(T_i | H_i) ≈ E[R_i²]/(p(1 − p)). The setup baseline and seven columns leave E[R²] = 0.732 in `own1` self-play, so σ_ε² = 3.91. The choice of c and β changes only this variance.

### 2.3 The student

A ridge regression of T_i on x_i = φ(a1_i) − φ(a0_i), with d = 776. φ(a) concatenates the frozen network's actor token output, its target hex token output, kind indicators and the four factored log-probabilities.

It is a second actor head trained on outcomes (the actor head is itself linear in the actor token, `wesnoth_ai/model.py`), fitted in closed form from per-game sufficient statistics (a Gram matrix and two vectors a game), so no per-decision features are stored. Swapping the pair flips its sign, so the 1.76% of argmax flips between numerics, which the record ties to gaps under about 0.014 nats, cannot change a switch. Excess risk: η = σ_ε² d_eff / N ([Hsu, Kakade & Zhang](https://arxiv.org/abs/1106.2363)).

### 2.4 Switch, forecast, adoption

**Switch.** π′ plays a1 at band decisions where ŵ·x > 0, and π elsewhere.

**Gain, to first order.** By the performance-difference lemma under π̃ ([Kakade & Langford 2002](https://dl.acm.org/doi/10.5555/645531.656005)), π scores −pΔ at each band decision, π′ scores (1 − p)Δ where it switches and −pΔ elsewhere, so J(π′) − J(π) ≈ E_π̃[Σ_switched Δ_i]: the coins' cost cancels at first order.

**Forecast.** Ĝ = β̂·Σ_switched f per side-game, where β̂ is the slope of T on f over held-out coins ([Chernozhukov et al.](https://arxiv.org/abs/1712.04802)). Its SE follows from the held-out predictions and the training half's label noise, before any held-out result is read.

**Adoption.** A corrected match against the incumbent: 12,800 decisive games, SE 2.65 Elo, $1.90. Adopted iff its lower 2-SE bound is above 0 and the corrected and PURE readings agree within 2 combined SE. False adoption at zero gain is 0.023.

### 2.5 Rounds, reset, critic re-entry

**Rounds.** Round k puts coins on π_k at the same base band, with the pair oriented by π_k's choice and the ridge centred on the previous fit.

**Reset.** A band is exhausted when a round's forecast upper bound falls below two match SEs. π_k is then distilled into the network by own-trajectory imitation with the human corpus (the `own1` recipe, about $2.1 measured), the −1.5 offset leaves the decode, and the new network's near-ties form the next band. If that band is exhausted too, w widens to 0.7 (about 102 decisions per side-game, derived).

**Critic re-entry**, when a round's forecast half certifies nothing: a result-regression critic is trained on that round's games (true state, cross-fitted by batch, $1–4 at the record's 285 positions a second) and enters c_i only, so it cannot bias; the labels are refitted and the forecast half re-read once, with no new games. Its plant: a terminal bonus that is a known function of the board at turn 10, recovered at slope 1 within 2 SE at later states.

### 2.6 Signal per dollar

**Against forks.** Per game-equivalent, coins carry 124 × 0.1875 / 0.732 = 31.7 units of information; a fork pair carries 1/(2 × 0.608) = 0.82, 39 times less.

**Against the critic route:** equal if the critic carries no decision information, as the record shows; they differ by κ, never measured.

**What $19.8 buys.** Eight batches of 100,000 games: four train, four read the forecast. At d_eff = 776 and zero mean advantage (`round5_simplify_power.out`):

| Spread of near-tie advantages (sd) | Gain | Detected? |
|---|---|---|
| 0.001 | +1 Elo | no |
| 0.0022 | +5.3 Elo | the floor |
| 0.003 | +9.2 Elo | yes (β 0.13, 3.8 SE) |
| 0.005 | +23 Elo | yes |
| 0.010 | +68 Elo | yes |

At d_eff = 200 the floor is 0.0016. The forecast's SE is 2.4 Elo. Halving the label variance is worth as much as doubling the data, so a better control variate is the cheap lever.

### 2.7 Invariants (tests from this spec)

I1, coin frequency equals the logged p (χ²), independent of the dice. I2, on an enumerable duel E[T] equals Q(a1) − Q(a0) exactly under three control variates, and not under one that reads D_i. I3, swapping the pair flips x and the label, not the switch. I4, no forecast-half game enters the frozen student. I5, labels regressed on pre-coin quantities read 0. I6, the sufficient-statistics fit equals a per-decision fit on 1,000 games. I7, logged switches equal sign(ŵ·x) recomputed offline. I13 (provenance draft), corrected-match mean preservation on 23 recorded matches and a planted handicap.

## 3. Regimen

Prices: a generation game $2.47e-5, a decisive match game $1.49e-4, a PURE match $0.12.

| Stage | Cost | Readout and prediction | Decision per outcome | Plant |
|---|---|---|---|---|
| **S0** instruments | $0.6 | `own1`'s 800 recorded self-play games replayed under generation numerics, logits and value head logged: agreement (`parity3` 98.2%); flips at G ≥ 0.35, predicted 0; band census, predicted 62 per side-game if `own1`'s margins match `parity3`'s; E[R²] with and without the value head, no prediction. 2,000 coin games timed; I1, and I7 with a planted ŵ. | A flip at G ≥ 0.35 means the generation and eval bands differ: fix numerics first ($7). Measured census, E[R²] and timing replace the derived inputs; `round5_simplify_power.py` reruns before S1. | — |
| **S1** round 1 data | ≤ $19.8 | **Batch 1, P-coin**: two planted columns booked into a separate result stream at every band decision. (a) δ1 × (even(played hex) from the game record, minus its logged expectation): right answer δ1 = 0.01 exactly, ± 0.0016 (2 SE, blind). (b) δ2 × (1[a1 played] − p)(u·x), u a random unit direction: δ2 = 0.002 ± 0.0011. Mean label by margin bin (the coin cost; SE 0.0006 a batch over all bins). **Batches 1–4** train, each read progressively by the student of the batches before; then the student is frozen. **Batches 5–8** read β̂ and Ĝ (SE 2.4 Elo). No prediction for β. | P-coin outside: label path broken, stop. After batch 4: if batches 2–4's readings, through the stated model, put batch 8's gain below 5.3 Elo at its upper bound, skip batches 5–8 ($9.9 saved): critic re-entry. After batch 8: β̂'s lower bound ≤ 0 certifies the spread below the floor: critic re-entry, then report. Otherwise S2. | P-coin |
| **S2** match | $2.1 | π_1 against `own1`, 12,800 corrected games; PURE against `own1` and `parity3`. Prediction: Ĝ, within 2 combined SE (3.6). | Adoption rule. Below the forecast by 2 combined SE (second-order error): switch only where f exceeds one SD of f (61% of the gain with 32% of the switches, Gaussian model), re-read the forecast half once, one rematch ($1.9). Gain over `parity3` below that over `own1` by 2 combined SE: exploitation, to the user. | — |
| **S3** round 2 | ≤ $21.9 | Coins on π_1 under the S1 protocol; a direct match against π_1 and PURE against `own1`. Prediction: its Ĝ. | Continue while the forecast's lower bound is above 0; otherwise S4. | P-coin if the code changed |
| **S4** band reset | ≈ $4.6 | **P-distill** (about $0.6): 2,000 games of `own1` at −2.5 distilled into `own1`; known answer: at −1.5 the student plays the teacher's decisions per side-turn (9.63 against 8.58 in the record) and agrees with it at S0's replay agreement, within 2 SE; no rate prediction. Then π_k is distilled and matched against π_k (12,800 games). | P-distill outside: distillation broken. Distilled network more than 2 SE below π_k: keep π_k and widen to w = 0.7. | P-distill |
| **S5** round 3 | ≤ $21.9 | The new band, under the S1 protocol | As S3 | P-coin |

**Budget.** S0 to S5 about $71; the reserve (numerics $7, one rematch $1.9, critic re-entry ≤ $4) brings it to $85, at the edge. A null round 1 costs $10.5 to $24.4, and each outcome names its next step.

**The $1,000 case** is argued only after two adopted rounds whose matches fell within 2 combined SE of their forecasts. It is scaled from the measured gain per round and per batch, with w = 0.7 and p = 0.5 (33% more information per label). Nothing measured links `own1` to human ratings.

**Rulings for the user.**
- **(a)** The labels regress outcomes on coins. This is ruling (a) of rounds 1–4 at full strength: the design has no other label source.
- **(b)** The seven luck columns are hand-written material evaluations, used only as control variates in labels and matches: they cannot bias either, and the player never sees them. Without them the setup baseline keeps 0.897 instead of 0.747, 17% less information; the value head, the learned alternative, is measured in S0.

## 4. What is new, and the nearest work

Searched 2026-10-20; finding nothing does not prove absence.

**New, as far as found.** Near-tie randomization of a deterministic self-play policy, analysed as a factorial experiment. Each of about 124 band decisions a game gets a label that is exactly unbiased, from logged propensities and the simulator's chance laws. These labels are policy iteration's only data: no critic, fork or search. Improvement is confined to the randomized binary choices, through a closed-form second actor head, reset by distillation.

**Nearest work:** micro-randomized trials, hundreds of randomized decisions per unit with known propensities ([Boruvka et al. 2018](https://arxiv.org/abs/1601.00237); distal outcomes, [Qian 2025](https://arxiv.org/abs/2502.13500)); tie-breaker designs, whose efficiency rises with the window's width at a cost in short-term value ([Owen & Varian](https://arxiv.org/abs/1808.07563); [Li & Owen](https://arxiv.org/abs/2202.12511)); AggreVaTe, one explored action per trajectory ([Ross & Bagnell 2014](https://arxiv.org/abs/1406.5979)); doubly robust policy learning ([Athey & Wager 2021](https://arxiv.org/abs/1702.02896)); baselines in sampled game training ([Schmid et al. 2019](https://arxiv.org/abs/1809.03057)); AIVAT, evaluation only ([Burch et al.](https://arxiv.org/abs/1612.06915)); LSPI ([Lagoudakis & Parr 2003](https://jmlr.org/papers/v4/lagoudakis03a.html)); the episodic natural actor-critic (round-1 critique).

**Not new:** corrected matches, ridge, progressive validation ([Blum, Kalai & Langford 1999](https://ttic.edu/blum/Papers/kfold.pdf)), the PDL.

## 5. Weakest point

The design pays only if the observation-predictable spread of advantages between the top two near-tie choices is at least about 0.002 in result units. Nothing on disk measures it. The record's nearest number is a mean, not a spread: −0.0018 per extra act in the stop window (`round5_provenance_stop_acts.out`). At sd 0.001, a $20 round gains about +1 Elo and certifies only an upper bound.

Two structural limits:
- Decisions the network is confident about are never sampled, so its confident errors stay until a band widens.
- The sign rule switches about 31 decisions per side-game, half the band. There the first-order forecast can miss by an amount nothing here predicts.

## 6. Provenance table

| Number | Value | Used for | Tag | Source or derivation |
|---|---|---|---|---|
| Variance kept, setup + 7 columns | 0.747 pooled (0.733–0.758); 0.730, 0.734 in `own1` self-play | label noise, match SE | measured | `analysis/round5_provenance_match_variance.py/.out` (leave-one-match-out) |
| Same, setup only / columns only | 0.897 / 0.847 | ruling (b) | measured | same |
| Mean preservation | Σt² 17.7 on 23 df | I13 | measured | same |
| Var(z), `own1` self-play | 1.0004 | label noise | measured | `analysis/round5_simplify_variance.py/.json` |
| Band share, w = 0.175, 0.35, 0.7 | 0.190, 0.339, 0.558 of decisions (`parity3`) | band count | measured | `analysis/round5_simplify_band.py/.json` on `prior_gaps.games.jsonl` (gaps after the offset, `tools/prior_gaps.py` at a72071a) |
| `parity3` band, w = 0.35 | 91.4 (sd 72.9) per side-game; tops: 65.8 moves, 10.3 recruits, 9.6 attacks, 5.7 end | census prediction | measured | same |
| `own1` decisions per side-game | 182.7 = (8.42 + 1) × 19.4 | band count | measured | `analysis/round5_provenance_stop_acts.out` (gen_own1_selfplay), plus one end_turn per side-turn |
| `own1` band count | 62.0 per side-game, 124 labels a game | all sizing | derived | 0.339 × 182.7, assuming `parity3`'s margins per decision; S0 measures it |
| Label noise σ_ε² | 3.91 | sizing, SEs | derived | 1.0004 × 0.732 / 0.1875 |
| Flat against taper | 4.67 against 6.67 information per nat of shift | coin merge | derived | `analysis/round5_simplify_band.py/.json` on `parity3`'s measured margins (w = 0.35, p = 0.25, taper matched on flips) |
| Argmax flips | 1.76%; near-ties under 0.0136 nats; partial r 0.53 ± 0.10 | antisymmetric student, S0 flip criterion | measured | `analysis/round5_provenance_numerics_ties.py/.out` |
| Replay agreement | 98.2% (`parity3`) | S0 comparison | measured | `critic_step1_20261009/prior_gaps.json` |
| Playout noise | 0.608 (temperature 0.5) | coins against forks only | measured | `analysis/round5_provenance_playout_noise.py/.out` |
| Belief-world rollouts | 683 of 7,728 carry a result | why not used | measured | `analysis/round5_simplify_forks.py/.json` (replaces this stance's earlier, misdescribed version) |
| Coins against forks | 31.7 against 0.82 (39×) | instrument choice | derived | 124 × 0.1875 / 0.732; 1/(2 × 0.608) |
| Elo per z | 347.4 | SEs | derived | 800 / ln 10 at an even match |
| Match SE, cost | 2.65 Elo, $1.90 at 12,800; PURE $0.12 | adoption | derived | 347.4 √(0.747/n); $0.42 × 17/60/800 per game |
| Game price, batch | $2.47e-5; $2.47 per 100,000 | all costs | derived | $0.42/17,000; the statistics' overhead is unknown, timed in S0 |
| Floors at $19.8 | sd 0.0022 (d_eff 776), 0.0016 (200); gains 1.1 / 4.3 / 9.2 / 23 / 68 Elo at sd 0.001 / 0.002 / 0.003 / 0.005 / 0.010 | stage meaning, budget | derived | `analysis/round5_simplify_power.py/.out` (stated Gaussian model, random-design ridge) |
| Forecast SE | 2.4 Elo (4 + 4 batches) | S2 prediction | derived | same |
| Plant SEs | 0.00079 (even-hex, contrast variance 0.5 assumed, read blind from the logs); 0.00056 (unit direction) | P-coin bands | derived | same |
| Mean label SE | 0.00056 per batch | coin cost readout | derived | same |
| Threshold rule | 61% of the gain with 32% of the switches | S2 remedy | derived | Gaussian model at zero mean: φ(1)/φ(0); (1 − Φ(1))/0.5 |
| False adoption | 0.023 per match | adoption | derived | one-sided 2 SE |
| PDL first-order gain | Σ_switched Δ | forecast | derived | section 2.4 |
| Per-decision feature storage | about 150 GB at 800,000 games | rejecting stored-feature cross-fitting | derived | 776 × 2 bytes × 124 band decisions × 800,000 |
| Band at w = 0.7 | about 102 per side-game | widening | derived | 0.558 × 182.7 |
| Stop window, mean per act | −0.0018 (−0.0061, +0.0017) | weakest point; a sensitivity case in the script | measured | `analysis/round5_provenance_stop_acts.out` |
| Teacher's decisions per side-turn | 9.63 against 8.58 | P-distill known answer | measured | same; read again on the P-distill games |
| Critic training rate | 285 positions a second | re-entry cost | measured | `training/metrics/critic_step1_20261009/T100.summary.json` |
| Distillation cost | about $2.1, two matches included | S4 | measured | `docs/memory_own_trajectories_prereg_20261011.md`, "Measured" |
| Winner's curse | +35.7 → +1.7 | fresh forecast half | measured | provenance draft; `docs/decode_knobs_prereg_20261012.md` |
| Near-tie spread v | — | whether any round gains | unknown | S1's forecast half; nothing before it relies on it |
| d_eff | — | floors | unknown | the ridge fit's trace in S1; floors shown at 200 and 776 |
| Value head's variance cut | — | label noise | unknown | S0 |
| Second-order error | — | forecast against match | unknown | S2 |
| Compounding over rounds | — | rounds 2–3 | unknown | each round's match; a continuation rule only |
| `own1` against strong humans | — | end goal | unknown | nothing rests on it |
| p = 0.25, w = 0.35, batch 100,000, 4 + 4 batches, d = 776, δ1 = 0.01, δ2 = 0.002, distillation tolerance 2 SE | — | design | choices | their consequences are derived above |
| Budget | $85; $0.42 an hour | everything | given | seed §1 |

## 7. Decision record

Rejected this round:
- **A critic as student target or gate in round 1.** Its benefit needs κ > 0, never measured, and its cost is the larger half of the current design.
- **Separate stop machinery.** End against best act is a near-tie pair.
- **The taper.** It rests on an unmeasured cost model, and the labels read the cost by margin.
- **Coin-free and one-sided games.** The labels' mean reads the coin cost.
- **Two-fold cross-fitting with stored per-decision features**, about 150 GB at 800,000 games. Sequential halves on sufficient statistics replace it.
