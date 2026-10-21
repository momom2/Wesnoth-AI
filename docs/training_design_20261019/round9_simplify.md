# Round 9, simplify stance: frozen baselines, one feature dump, one candidate, and a $1 transfer check

Tags: **[M]** measured (record and script named), **[D]** derived (inputs named), **[U]** unknown, **[C]** a choice whose consequences are derived. Scripts are `analysis/round9_simplify_<name>.py`, each with its `.out`.

## Summary

The core stays: near-tie coins with logged propensities, labels exact through the chance laws, a closed-form linear head on the frozen network, a sealed forecast, adoption only by match. Section 7 showed that the core works on real games, moved the base to `own2`, and priced a round at $23.98 where the current design had $15.30.

Batch 1's records hold the first measured near-tie spread. Among near-ties between two recruits of different units, a rule fitted on four of the seven matches gains **+22.0 ± 4.7 Elo** at first order on the other three [M, `recruit_spread` §7]. The rule is 43 per-unit values and a constant, which is the head's own form.

- **Removed:** the server's statistics (moved offline), per-batch baseline refits, E, per-type checks, per-round claims. Branch S and the reset leave the budget.
- **Merged:** the two feature plants.
- **Added: C1,** a $0.98 match that tests the step from labels to play on that signal before a $24 round.

Three rounds fit in the $82.34 left if the dump's overhead is at most 0.117 on the measured host [D, `budget`].

## 1. What the records move

| Measurement | Decision |
|---|---|
| `own2` +22.7 ± 4.4 over `own1` [M] | π_0 = `own2`. Batch 1's `own1` games price `own1`'s future and log no features. They fit the baselines and C1's rule |
| Mean label +0.0003 ± 0.0007. A per-kind-pair sign rule's in-sample 24.3 Elo has P 0.32 under a null whose selection alone gives 20.8 [M, `typed_rule`] | No stage keyed by kind pair |
| Recruit-versus-recruit near-ties (different units, 4.29 per side-game): cell means spread with sd 0.040 (≥ 0.029 at 2 SE). Cross-fitted rule +23.1 ± 3.4 Elo; a rule frozen on b00–b03 reads +22.0 ± 4.7 on b04–b06, at 2.28 switches per side-game [M, `recruit_spread`, on the extraction of the provenance stance's `round9_provenance_cells.py`] | v > 0 is measured for one family, in the head's form. C1 runs before round 1; round 1 reads the head against this rule |
| Label slope on the margin +0.0157 ± 0.0070 per nat [M, `labels` §3] | Within 0.35 nats the margin does not rank the pair. Round 1 keeps 0.35 ([0.35, 0.7) is unmeasured) |
| E[T²] 4.01 with frozen baselines; within-game ratio 0.999 [M, `labels`] | Forecast SE 4.88 Elo; miss line 10.4 |
| A match carries 78× the labels' information per game on one rule's value [D] | Only a match adopts |
| Coin games all-in 20,254 an hour at $0.577 [M] | The host's cost per game and the dump's overhead decide two or three rounds |
| S0(a) passed at the edge of a window from other networks, at half the predicted effect [M] | Gates are agreement between code paths and a sign at 3 SE. A transferred window gates only where the between-network spread was measured (the dice control: Q 24.9 on 22 df) |

## 2. The algorithm

**What is learned.** A linear second actor head on `own2`'s frozen token outputs, f(x) ≈ Q̃(H, a1) − Q̃(H, a0), for the base's top two joint actions a0, a1 within G < 0.35 nats (after the end_turn offset). The player plays a1 where f > τ̃ and re-decides after every roll.

**Labels.** In π_k's self-play, a coin keyed by seed, side and decision index plays the pair's non-chosen action with probability ¼ [C; built; I-coin and I-log passed on 9.3 M decisions]. With z side A's result and s_i the mover's sign:

    T_i = s_i ω_i R_g,   ω_i = (D_i − P_i)/(P_i(1 − P_i)),   R_g = z − b(cell) − βᵀS_g

(b, β) are fitted once on `own1`'s 60,000 coin games and frozen:
- β by least squares on the walk's seven luck sums;
- b as the setup cell's mean of z − βᵀS, shrunk toward an additive fit of map × side plus faction pair.

**Claim.** E[T_i | H_i] = Q̃(H_i, a1) − Q̃(H_i, a0) under the coin policy, for any coefficients fixed before the game (`tools/coin_labels.py`, proof step 2): later luck terms have mean zero given their pre-fight states, and E[(D − P)z | H] = P(1 − P)(E[z | D = 1, H] − E[z | D = 0, H]). A fixed baseline sets only the variance ([Williams 1992](https://link.springer.com/article/10.1007/BF00992696)). The frozen coefficients leave 0.699 ± 0.003 of Var(z) on held-out coin games and 0.692 on `own2`'s uncoined games [M, `residual`].

**Head.**
- **Dump:** per band decision, x = φ(a1) − φ(a0), where φ = [actor token output, q(actor) ⊙ k(target hex), four factored log-probabilities]. 772 bf16 values, 2.40 GB a 10,000-game match.
- **Statistics,** offline per match: G_j = Σ x xᵀ, and b_j = Σ_g R_g A_g with A_g = Σ_{i∈g} s_i ω_i x_i. x is kept for a fixed 1,000-game subsample.
- **Typed columns:** sixteen, unpenalised. A constant per unordered pair of different kinds, and a margin slope per kind pair.
- **Fit:**

      ŵ_k(λ) = (Σ_{j≤k} G_j + λΛ)⁻¹ Σ_{j≤k} b_j,   Λ = 1 on network columns, 0 on typed ones

  λ̂_k minimises the progressive loss Σ_{j=2..k} [−2ŵ_{j−1}ᵀb_j + ŵ_{j−1}ᵀG_jŵ_{j−1}], exact from per-batch statistics.
- **What the fit is:** one episodic natural actor-critic step for a logistic coin policy at θ = 0, since P(1 − P) = 0.1875 everywhere ([Peters & Schaal 2008](https://doi.org/10.1016/j.neucom.2007.11.026)). Bias: ridge shrinkage only. Variance: σ²(G + λΛ)⁻¹G(G + λΛ)⁻¹.
- **Its form** is f = h(a1) − h(a0), h(a) = wᵀφ(a). The recruit signal has it: 43 unit values and a constant keep +22.0 Elo on held-out games, the 262-cell table +21.6.

**Switch rule.** Student j − 1's f on batch j gives W_j(τ) = Σ_{f_i > τ} T_i, unbiased for the first-order gain of {f > τ} ([Kitagawa & Tetenov 2018](https://doi.org/10.3982/ECTA13288)). τ̃_k maximises W over the latest four pairs if that beats τ = 0 by 2 SE of the difference; otherwise τ̃_k = 0.

**Forecast, adoption, claim.**
- **Forecast:** Ĝ_pf = 347.4 × W per side-game over batches 5–8 at students 4–7's τ̃, sealed before the match. SE 4.88 Elo.
- **Adoption:** π_k with the head on one seat against π_k: 37,800 decisive pool games on fresh seeds, SE 1.79. Adopted iff the lower 2-SE bound is above 0; false adoption 0.023.
- **Claim:** one PURE two-network match of the last adopted player against `own1`, 12,800 games, SE 3.07, predicted at 22.7 plus the adoption readings.

**Signal per dollar.**
- **Coins against forks:** one result prices 155 labels with no playout or critic: 38.7 information units per game-equivalent, a fork pair 0.82 (47×) [D].
- **Exploration:** free (+2.0 ± 4.7 Elo; sampling at temperature 1 costs −412 ± 97).
- **Matches:** a match evaluates one rule and cannot fit 788 coefficients.

**Rounds.** Round k coins π_k's band (roles exchanged where π_k switches) on the same frozen features, and refits on all rounds' labels. Stale labels can bias a student, never a claim.

**Invariants.**
- **Passed on `own1`:** I-coin, I-log and the dice control.
- **New:**
  - **I-φ:** the dump reproduces the logits.
  - **I-proc:** the statistics recompute from the subsample.
  - **I-match:** on a switching seat every switch played a1, and the rule recomputes from that match's dump.
  - **I-V, pooled:** the mean of sω(b + βᵀS) is 0 within 3 SE.
- **Tests from the spec:**
  - **I2:** an enumerable duel through the production generator; a post-coin variate must fail it.
  - **I3:** a swap flips x, T and the typed columns.

## 3. What goes, what merges, what is added

| Change | Cost | Gain |
|---|---|---|
| In-server statistics, logged plant products and f, seven student reloads → the server writes x, an offline processor does the rest | 24 GB a batch written, then deleted; overhead o [U, R1] | One write in the server; every statistic recomputable |
| Per-batch refits of b, β (I4) → fitted once on `own1` | 0.699 against the refit's 0.688: at most 1.6% more games | No leakage path left to test |
| E → removed | After a miss, that round adopts nothing | False adoption 0.046 → 0.023; C1 measures what E was meant to salvage |
| Two feature plants → one: z′ = z + Σ s(D − P)h(x), h = a·1(u·x > q) − c, sized at 10 SE | Off-by-one reading depends on adjacent decisions' correlation [U, smoke]; the built random-sign plant still reads it as 0 | One copy, one parse |
| I-V per type → pooled | A one-type defect shows only once it moves the pooled mean by 0.0021 | One test in place of 15 |
| v̂ → an offline reading; one PURE claim at the end, not per round | Intermediate players have no PURE number | $1.13 a round |
| Branch S → its own proposal; S4 and S5 → after the optimizer discussion | No compounding past the base's band in this plan | Two rulings and the distillation pipeline |
| **Added: C1** | $0.98; a 44-number rule file; its switch hook is the adoption match's | Labels-to-play transfer measured first |

**Kept:**
- **The tested threshold:** ≤ 0.1 Elo where the sign rule is right; it recovers most of the 11.0–14.5 Elo the sign rule leaves when a minority carries the gain [D, `rules`].
- **The typed block:** a mask.
- **The luck walk:** built, feeds the dice control, residual 1 → 0.82.

**Rejected:**
- **A fixed top-half rule:** it leaves up to 22 Elo [D, `rules`].
- **A pilot to bound v:** SE(v̂) ∝ 1/N, so a pilot cannot kill the round.

## 4. Regimen

| Stage | Cost [D] | Readout and prediction | Decision |
|---|---|---|---|
| **R0** build | $0 | CI: I2, I3, the plant identities on synthetic games, I-φ on a CPU forward, the switch hook on a synthetic band | Red: fix before any box |
| **C1**, first box | $0.98 | `own1` with the recruit rule frozen on b00–b03, on one seat, against `own1`: 37,800 pool games on fresh seeds. Prediction **+22.0 ± 4.7** (b04–b06's labels); miss line 10.1. I-match; switch rate against 2.28 per side-game | **Within the line:** labels transfer at first order; S1 proposed. **Below the line, above 0 by 2 SE:** κ̂ = reading / 22.0 reported; S1 proposed with adoption predicted at κ̂Ĝ_pf. **≤ 3.6** (P 0.0001 if the chain holds): the chain is broken on a signal at z 4.7; no round until found (hook, coin-policy pricing, the opponent's coins). **Above the line:** reported; S1 proposed |
| **R1** smoke, `own2`, same box | $0.29 | 2,000 dumped, 1,000 undumped coin games. At 3 SE: I-coin, I-log, label plant, dice control in [+0.00245, +0.00587], I-φ, merged plant. Readings: census, E[T²], o | A check fails: debug. o sets the rounds |
| **S1** round 1 | $22.79(1 + o) + $0.21; 39.5 h | Every match: batch 1's checks. Batches 1–2: the planted τ̃ leaves 0. Batch 8: λ̂, τ̃, head; Ĝ_pf sealed; v̂. **Recruit readout:** the head's held-out welfare on recruit pairs against the 44-value rule refitted on `own2`'s labels, same batches; predicted within 2 SE or above if the tokens carry unit identity linearly [U] | A check fails: the batch is quarantined. Recruit shortfall beyond 2 SE: I-φ and the join re-run on those decisions before S2; if both pass, it measures what the frozen features miss |
| **S2** adoption | $0.98 | π_1 against `own2`, read against Ĝ_pf ± 10.4; I-match | **Lower bound > 0:** adopt; S3. **Ĝ_pf > 0 by 2 SE, match below its line:** a miss, read against C1's κ̂; the user rules. **Both null:** if the head missed the recruit signal, the frozen features are the limit (the reset, after the optimizer discussion); if not, v̂ bounds the rest, and the user picks w 0.7, a privileged critic column, or stop |
| **S3** rounds 2, 3 | $22.79(1 + o) + $1.19 each | As S1–S2, on π_k | As S2 |
| **Claim** | $1.13 | Last adopted player against `own1`, SE 3.07; predicted 22.7 + Σ readings; miss line 11.2 / 11.8 / 12.3 after 1 / 2 / 3 rounds | Below the line: intransitivity, or a pool or two-network defect; reported |

**Budget** [D, `budget`], from $82.34 left:
- **Two rounds,** with C1, R1 and the claim, cost $50.38 (margin $31.96).
- **Three rounds** cost $74.36. They fit iff o ≤ 0.117 on the measured host, or, at o = 0, iff the host's all-in cost per coin game is at most $3.155e-5 (1.107× the measured). Throughput is host-specific, so an hourly price alone settles nothing.
- **A null round 1** costs $25.26.
- **Branch L,** if ruling (a) is no: unchanged.

**Beyond this plan:** coin rounds alternating with a reset that distils π_k and moves the band, after the optimizer discussion; the $1,000 case is argued only from two adopted rounds' measured gains. Nothing links `own1` to human ratings, and nothing rests on it.

**Checks per code path:**
- **Labels:** the random-sign coin plant on every match (passed, 0.994 and 0.995 ± 0.008).
- **Dump, fit, threshold:** I-φ and the merged plant.
- **Hooks:** S0(a) for logits (passed); I-match and C1 for switches, C1 being the plan's first known answer of measured size on real signal.
- **Head:** the recruit readout.

## 5. What is new, and the nearest work

Searched 2026-10-21; finding nothing does not prove absence.
- **Tie-breaker designs** ([Owen & Varian](https://arxiv.org/abs/1808.07563)) randomise near an assignment cutoff. The coin band is one, on the network's margin, with 155 decisions sharing one outcome.
- **The other parts:** [micro-randomised trials](https://doi.org/10.1080/01621459.2017.1305274), the episodic natural actor-critic, [AIVAT](https://arxiv.org/abs/1612.06915) and empirical welfare maximisation.
- **C1 is standard practice in a new place:** an offline estimate checked online before it is relied on. [Gilotte et al. 2018](https://arxiv.org/abs/1801.07030) benchmark offline uplift estimators against 39 online A/B tests. The weak correlation between offline and online comparisons is called an open problem ([Recommender Systems: A Primer](https://arxiv.org/pdf/2302.02579)). That contradicting evidence is why C1 runs first.
- **No probe of action-value gaps at near-ties** turned up; [Jenner et al. 2024](https://arxiv.org/abs/2406.00877) decode look-ahead, not value gaps.
- **This stance adds no method,** only measurements: the frozen variate's transfer, and a signal of known size in the head's form.

## 6. Weakest point

**Only one family of near-ties has a measured spread.** Recruit pairs are 4.29 of the 77.6 band decisions per side-game.
- **The other 94%:** whether they hold spread the frozen features carry is unmeasured; the full round is its cheapest measurement.
- **The signal is `own1`'s.** `own2` recruited closer to the measured unit values for 16 units and further for 12 (`round9_provenance_own2_recruits.py`), so its training did not absorb it; its own near-ties stay unread until S1.
- **C1 measures transfer at 2.3 switches per side-game;** the head may switch 19–38 times, where second-order loss can grow. Without E, a miss at that scale adopts nothing.

## 7. Rulings for the user

- **(a), unchanged:** one natural policy-gradient step on a near-tie coin policy, adopted only by match; if no, Branch L.
- **(b):** the luck columns and the frozen cell baseline, fitted on `own1`'s coin games, used in labels only.
- **(c):** adoption by pool match, and one PURE claim at the end with its predicted value.
- **(e):** does a closed-form head on frozen features fall under the 2026-10-15 ruling?
- **(h), new:** C1's rule is 44 numbers fitted on labels and keyed by faction and unit names. It is a measuring device, never part of a player. Is it acceptable?
- **Host:** chosen by measured cost per coin game.
- (d), (f) and (g) wait for the reset and Branch S.

## Provenance

Script names abbreviate `analysis/round9_simplify_<name>.py`.

| Number | Value | Tag | Source or derivation | Used for |
|---|---|---|---|---|
| `own2` against `own1` | +22.7 ± 4.4 Elo, 6,400 games | M | `training/metrics/own2_base_20261020/reading.json` | π_0; claim prediction |
| Attack known answer | −41.3 ± 4.4 Elo (window [−111, −41]); label −0.242 ± 0.023 z | M | `near_tie_coins_20261020/readout_a_offline.txt`; seed §7 | Gate style |
| Band census | 77.6 ± 0.13 per side-game | M | `readout_b_offline.txt` | Dump size, information |
| Pooled mean label | +0.0003 ± 0.0007 | M | `readout_b_offline.txt` | Coin play's cost |
| Per-kind-pair sign rule | in sample 24.3 Elo; null mean 20.8, 95th pct 41.9, P 0.32 | M/D | `typed_rule` (normal null per type) | No kind-pair stage |
| Recruit cell spread | V_rr 1.59e-3 ± 0.37e-3 (z 4.3); sd 0.040, ≥ 0.029 at 2 SE; mover's faction alone z 1.3 | M | `recruit_spread` §1–2: U-statistic over 20 game folds on the label sums extracted by `round9_provenance_cells.py` (batch 1's β, b = 0) | v > 0 for one family |
| Recruit rule welfare | cross-fitted +23.1 ± 3.4; leave-one-match-out +20.9 ± 3.4 (without luck columns +13.6 ± 3.6); positive in 5 of 6 factions | M | `recruit_spread` §4–6 | §1; S1 readout |
| C1's forecast | frozen on b00–b03, read on b04–b06 (28,000 games): additive +22.0 ± 4.7 at 2.28 switches per side-game; table +21.6 ± 4.8; reverse split +16.7 ± 4.5 | M | `recruit_spread` §7 | C1 prediction |
| C1's lines | miss 10.1; null at ≤ 3.6; P(null) 0.0001 at κ = 1 | D | 2√(4.7² + 1.79²); 2 × 1.79; Φ(−(22.0 − 3.6)/5.03) | C1 decisions |
| `own2`'s recruit rates | 16 units toward `own1`'s measured values, 12 away, beyond 2 SE | M | `round9_provenance_own2_recruits.out` (provenance stance; not re-run) | Weakest point |
| Margin slope | +0.0157 ± 0.0070 per nat, z 2.23 | M | `labels` §3 (weighted LS on 7 bins) | w kept at 0.35 |
| Coin play's cost | +2.0 ± 4.7 Elo | D | 77.6 × ¼ × (+0.0003 ± 0.0007) × 347.4; `budget` | §1, §2 |
| Residual share, frozen | 0.699 ± 0.003 out of sample; β only 0.824 | M | `residual` (two folds of 30,000) | Baselines; σ² |
| Transfer of frozen coefficients | `own2`_64 vs `own2`_0 0.692; `own2` vs `own1` 0.704 (SE ≈ 0.04) | M | `residual` (fit on 60,000) | Baselines on `own2` |
| Refit's asymptote | 0.688 | D from M | `round6_provenance_cells.out` | Cost of freezing |
| E[T²]; within-game ratio | 4.01 ± 0.007 frozen (4.56 as batch 1 ran); 0.999 | M | `labels` (b00 and b06 held out, 1.56 M labels) | Forecast SE, v̂ |
| Forecast SE; miss line | 4.88 Elo at switched share ½ (3.41 at ¼); 10.4 | D from M | per-side-game variance 158.1 (`labels`); 2√(4.88² + 1.79²) | S1, S2 |
| Match against label information | 78× per game | D from M | 158.1 / 2 | Only a match adopts |
| Coins against forks | 38.7 against 0.82, 47× | D | 2 × 77.6 / 4.01; 1/(2 × 0.608), `round5_provenance_playout_noise.out` | Signal per dollar |
| Prices | coin all-in 20,254 games/h ($2.849e-5); pool 22,209/h; two-network 6,521/h; $0.577/h | M | `labels` §4 on `matches.txt`, `walls.txt`, `own2_base_20261020/match.walls` | Every cost |
| Budget left | $82.34 | D | $85 − $2.0 − $0.66 (seed §7); assumes the window came out of the $85 | Rounds that fit |
| Costs | C1 $0.98; first box $1.28; round $23.98 at o 0 (batches $22.79, adoption $0.98, setup $0.21); claim $1.13 | D | `budget` (setup 7 min [M] plus 15 min staging [C]) | Regimen |
| Rounds that fit | three iff o ≤ 0.117, or ≤ $3.155e-5 a coin game (1.107×); two iff o ≤ 0.701, or ≤ 1.634× | D | `budget` | R1; host |
| Dump size | 24.0 GB a batch, 2.40 a match, subsample 0.24 | D | 2 × 77.6 × 772 × 2 bytes × games | Offline processing |
| Overhead o; processing time | — | U | R1, against 1,000 undumped games on the same box; nothing before R1 relies on them | Rounds that fit |
| Fixed rules against the oracle | sign worst 0.244 (11.0 Elo left); top half up to 22.4 Elo left | D | `rules` (round 8's model at E[T²] 4.01, band 77.6) | Keep τ̃ |
| Tested threshold | ≤ 0.1 Elo cost where sign is right | D | `round8_provenance_threshold.out` | Keep τ̃ |
| v̂ after 8 batches | SE 1.39e-6, bound 2.78e-6; SE ∝ 1/N | D | `round8_provenance_spread_estimator.out` × 4.01/3.67; Var(e_jᵀSe_k) = dσ⁴/(n_j n_k) | No pilot |
| Match SEs | 1.79 (37,800), 3.07 (12,800); false adoption 0.023 | D | 347.4/√N; Φ(−2) | C1, S2, claim |
| Claim prediction | SE 4.70 / 5.03 / 5.34; miss lines 11.2 / 11.8 / 12.3 | D | √(4.35² + k × 1.79²); 2√(· + 3.07²) | Claim |
| Label plant on `own1` | 0.994 / 0.995 ± 0.008 | M | `readout_b_offline.txt` | R1 lines |
| Dice windows | smoke [+0.00245, +0.00587]; batch [+0.00267, +0.00565] | D from M | `round8_synthesis_power.out` §2 | R1, S1 |
| Transfer κ at 2.3 switches | — | U | C1 | S1's adoption prediction |
| v outside recruits; τ̃; Ĝ_pf's mean; whether tokens carry unit identity linearly | — | U | S1; nothing before S1 relies on them, and every S2 outcome has a decision | Round gain |
| Miss size at 19–38 switches | — | U | S2 against Ĝ_pf and C1's κ̂; no a-priori bound (Farias et al. 2022, read in round 8) | Reading a miss |
| w 0.35, p ¼, 8 × 100,000 games, d 772 + 16, match and smoke sizes, plant at 10 SE, C1's split | — | C | Consequences derived above | Design |
