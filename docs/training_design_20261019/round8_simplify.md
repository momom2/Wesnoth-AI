# Round 8, simplify stance: one residual per game, no futility stop, the plant before the batches

Tags: **[M]** measured, **[D]** derived, **[U]** unknown, **[C]** a choice whose consequences are derived. This stance's script is `analysis/round8_simplify_power.py` (`.out` beside it), cited as §n.

## Summary

The core is `current_design.md`'s: near-tie coins with logged propensities, labels exactly unbiased through the chance laws, a closed-form ridge head on frozen `own1`, a sealed forecast, adoption only by match. Three mechanisms go and one check moves earlier:

1. **The interim match goes,** with its stop rule and E's interim branch. With the budget no longer binding, the stop only trades money for power, and it loses that trade where the record leans: at the μ −0.0018 floor it cuts adoption power from 0.80 to 0.63 (§1). A null round 1 costs $16.16 instead of an expected $11.74; a round that is not null saves $0.72.
2. **γV goes, and with it the S0(a) replay.** The label residual is one number per game. Every stated floor is already computed without V. The census comes from a 2,000-game coin smoke on S0's box.
3. **The √N ceiling and its margin-bin certification go.** No decision rested on them.
4. **The coin plant is also read on the smoke** (SE 0.039 per side), so gross label-path errors stop the run before the other 798,000 coin games are paid for; batch 1 reads it again at SE 0.0055 (§4).

The plan costs $56.18 of $85 (now $57.67). Three rounds fit at up to 63% logging overhead, 12% on a 1.41× slower host (now 60% and 9%) (§2).

## 1. What goes, and what each removal costs

| # | Current | Here | Cost | Gain |
|---|---|---|---|---|
| 1 | Interim match after batch 3, stop iff J3 < 0, E3 | Removed; E only at adoption | A null round 1: $16.16 always, against $6.60 or $16.88 (expected $11.74), **+$4.42** [D §2] | Power at the floor 0.80 against 0.63 (μ −0.0018), 0.80 against 0.77 (μ 0); null bound sd 0.0034, not 0.0036 [D §1]; $0.72 a non-null round; one rule, one branch |
| 2 | S0(a): `own1`'s recorded games replayed through the prior-gaps forwards | Census, runner-up mix, spectrum from the S0 coin smoke | None on any stated number; the census is read under the coin policy the labels sample | No second computation of band membership that could disagree with the generator's |
| 3 | γV: per-decision V_i in R_i, its admission, the V arms of I2 and I-V | R_g = z − b(cell) − βᵀS_g | Unknown variance reduction; stated floors (r 0.688, no V) unchanged. The slack buys the floor V would give at a kept share 0.614 [D §3] | Removes "V read after the coin", which would bias every label; the fit needs two objects per game |
| 4 | √N ceiling in round 2's preregistration, with per-bin certification | Removed | Round 2 states only its Ĝ_pf and miss line | One computation and one rule that decided nothing |
| 5 | Report on (non-attack top, attack runner-up) labels | Removed | None: it gated nothing; S0's label reading gates orientation | — |
| 6 | I1, I5; I7, I9 | Merged into I-coin and I-log | None | Two named checks |

**Removal 1 is new.** No round proposed it. The interim replaced round 5's model-based stop when money bound the plan ($9.81 left, three rounds not fitting on a 1.41× host, `round6_synthesis_power.out` §6); round 7's steady price moved that bottleneck. Group-sequential practice states the same trade: a futility stop costs power, which must be budgeted ([Li, Herrmann & Rauch 2020](https://doi.org/10.1186/s12874-020-01141-5), abstract read).

**Removal 3 uses round 7's own rule.** Round 7 kept γV as one cheap coefficient, but deferred DPI and the antithetic siblings because they change cost, not guarantees, while the budget does not bind. γV is in that class, and the $1,000 case's one replay can read all three. No record holds V on `own1`'s games (`tools/game_record.py` keeps commands; the only value-head table is the seed checkpoint's on human games), so V's share is [U], and nothing relies on it.

## 2. The algorithm

**Generation (unchanged).** π_0 is `own1` at `raw:t0+eo-1.5`, 64 slots, both sides, `run_elo_batch` at 120 jobs. Where the top two joint log-priors after the offset differ by G < w = 0.35 [C], a coin keyed by seed and decision index plays the base runner-up a1, with P_i = p = 0.25 where π_k plays a0 and 1 − p where it plays a1 [C]. End_turn is an ordinary candidate. Play continues under π_k, re-deciding after every roll, so each comparison is one decision followed by adaptive play. About 76 band decisions and 19 deviations per side-game [D; the smoke re-measures]. Logged per band decision: P_i, D_i, pair type, mover, G, u·x_i (plant), from batch 5 the log-only student's switch bit; per game z, S_g, A_g; per batch the Gram matrix.

**The label.**

    T_i = s_i ω_i R_g,   ω_i = (D_i − P_i)/(P_i(1 − P_i)),   R_g = z − b(cell) − βᵀS_g

z is side A's result, s_i the mover's sign, b(cell) the full setup cell's mean, S_g the seven luck sums (`tools/match_luck.py`). b and β for batch k are fitted on batches 1 to k − 1 and frozen before batch k is generated; batch 1 uses b = 0 and β from the 18,800 walked games (adaptive AIPW's plug-in, [Hadad et al. 2021](https://arxiv.org/abs/1911.02768) §2.1).

*Claim:* E[T_i | H_i] = Q̃(H_i, a1) − Q̃(H_i, a0) under the coin policy π̃. *Proof:* the coefficients are constants; b and the luck before i are functions of H_i; each later luck term has mean zero given the state before its fight, which contains D_i; so E[(D_i − P_i)(b + βᵀS_g) | H_i] = 0, and E[(D_i − P_i)z | H_i] = P_i(1 − P_i)(E[z | D_i = 1, H_i] − E[z | D_i = 0, H_i]). ∎ Under fog this is the information-state advantage, with no strategy fusion ([Baisero & Amato](https://arxiv.org/abs/2105.11674)). Variance σ² = Var(z)·r/(p(1 − p)) = 3.67, 4.48 in batch 1 [D]. Capped games are dropped (ruling 2026-08-17), exact iff capping does not depend on the coins (I-coin); `own1` self-play capped 0 of 1,600 [M].

**The student (unchanged).** x_i = φ(a1) − φ(a0), φ = [actor token output, q(actor) ⊙ k(target hex), four factored log-probabilities], d = 772 [C]; per unordered kind pair an unpenalised antisymmetric constant (six) and margin slope (ten). ŵ_k = (G_{1..k} + λI)⁻¹ Σ_g R_g A_g with A_g = Σ_{i∈g} s_i ω_i x_i: one residual and one vector per game. λ̂_k minimises the progressive loss from statistics. Switch where ŵᵀx > 0, within 0.6 Elo of oracle tuning in the stated model [D, `round6_simplify_rule.out`].

**Signal per dollar.** Every band decision is a label whichever way its coin falls, and one result prices about 152 of them at the generator's price, with no playout and no critic: about 41 information units per game-equivalent (152.2 × 0.1875/0.688) against 0.82 for a fork pair [D, round 5's form]. Cell and columns remove 0.312 of Var(z) without bias [M]. Every batch trains: a match carries 17 times a held-out batch's information per dollar [D, `round5_synthesis.md`].

**Matches, forecast, adoption.**
- Matches: the incumbent with the head on one seat, 37,800 independent decisive games on fresh seeds, SE 1.79 Elo, $0.72 [D].
- Forecast Ĝ_pf: batches 5–8's labels at log-only students 4–7's switches, per side-game × 347.4, sealed before the match; SE 4.6 at μ 0, miss line ≈ 9.9 Elo [D, `round6_synthesis_power.out` §4].
- Adoption iff the lower 2-SE bound is above 0 (false adoption 0.023); against `own1` it is the PURE claim.
- E: if not adopted while Ĝ_pf > 0 by 2 SE, the student switching only above its half-quantile of f plays one match on its own sealed forecast, adopted by the same rule (false adoption 0.046 for two candidates). It keeps 0.80–0.87 of the first-order gain; a quadratic curvature loss falls to a quarter [D, round 6].
- A miss: labels give the gradient at the coin point with both sides coined; the match integrates it against an uncoined opponent, a first-order truncation like Differences-in-Q ([Farias et al. 2022](https://arxiv.org/abs/2206.02371)). With the plants passed, a miss is curvature or the opponent's coins, which one round cannot separate.

**Rounds and reset (unchanged).** Round k coins the base band under π_k and refits on all labels; its own Ĝ_pf and match read π̃_k, so stale labels can bias the student, never a claim. Round 2 runs by default after an adopted round 1, since the reset trains weights (ruling 2026-10-15). A round not adopted with a null forecast exhausts the band. S4 distils π_k by the own-trajectory recipe onto `tools/own_targets.py`'s top-16 target, the switched pair's logits exchanged and end_turn lowered by o_k (`shift_end_turn_prior`); ĉ and ρ̂ are read on held-out positions first, and N plays at o_k(1 − ĉ), −0.27 at ĉ 0.82 [D].

**Invariants.** I-coin: coins match P (χ²) over all games and within capped ones, independent of dice and pre-coin quantities. I2: on an enumerable duel through the production generator, coin on each seat, E[T] = Q(a1) − Q(a0) exactly under the cell and column variates; a post-coin variate must fail. I3: a swap flips x, T and typed terms. I4: by construction. I6: the statistics fit equals a per-decision fit on 1,000 games. I-log: logged switches and forecast sums equal recomputation (also with a planted ŵ), and every D_i = 1 has a1 in the record. I-φ: features reproduce the logits. I-V, once a round at a 5% family-wise line (3.4–3.8 SE); an alarm holds the match.

**Plants, one per code path.**
- *Label and fit:* z′ = z + Σ_j s_j^rec (D_j^rec − P_j)(u·x_j), with s^rec and D^rec read from the record by separate code; the slope of T′ − T on u·x is 1 per mover side. On the smoke (SE 0.036–0.039) a wrong side sign sits 56 SE away, a coin one decision off 27 SE, a 20% misread 5.5 SE; on batch 1 (SE 0.0055) a 2% misread sits 4.6 SE away [D §4]. (ŵ′ − ŵ)ᵀu is predicted at uᵀ(G + λI)⁻¹Gu.
- *Match path and orientation (S0):* attack offset +1 on one seat through the head's hook, the seat a logged coin at P ½, 6,400 games: −78 ± 16 Elo (window ±33) by `elo_collect`; 2Δ = −0.449 z (window [−0.640, −0.259]) by the label code [M via D, `round7_synthesis_power.out` §1–3]. A wrong-side result reads +0.449, half flipped 0.
- *Distillation (S4):* the baked end_turn shift; ĉ not above 0 by 2 SE means broken.

## 3. Regimen

Prices $1.912e-5 a pool game, $1.17e-4 a two-network game [M]; a batch is $1.91 and 4.55 h.

| Stage | Cost [D §2] | Readout and prediction | Decision per outcome |
|---|---|---|---|
| **S0**, one box | $0.14 + $0.04 of batch 1 | Attack known answer both ways. If ruling (a) is yes, the smoke (batch 1's first 2,000 games): I-coin, I-log, I-φ, gross plant, own1's census, runner-up mix, spectrum, overhead against the attack games on the same host | Known answer outside its window: match path or orientation broken, fix first. Smoke or plant fails: debug. Else the power script reruns on the measured census and overhead, and S1 is proposed on the restated floors |
| **S1** | $15.30 | Batch 1: fine plant per side. Batch 8: v̂, c, b, q̂; Ĝ_pf sealed. Nothing predicts v | Plant fails: debug. Price × (1 + overhead) > $3.11e-5: two rounds (fit to $4.81e-5) |
| **S2** | $0.72 (+ $0.72 E) | π_1 vs `own1`, against Ĝ_pf ± miss line | Lower bound > 0: adopt (the claim), S3. Not adopted, Ĝ_pf > 0 by 2 SE: E. Both null: S3-null; no spread above sd 0.0016 (μ 0) or 0.0034 (μ −0.0018) in the frozen span [D §1] |
| **S3** | $16.74 | Coins on π_1; π_2 vs π_1 (adoption), vs `own1` (claim); its own Ĝ_pf, round 1's J/Ĝ_pf beside it | As S2; a null exhausts the band; S4 after the optimizer discussion |
| **S4** | $3.60 | ĉ, ρ̂ held out; N at o_k(1 − ĉ) vs π_k, 12,800 two-network games (SE 3.07); prediction −(1 − ρ̂)Ĝ_k + Σ_type n_type μ̂_type, out-of-band unpriced [U] | ĉ not > 0 by 2 SE: broken. Estimate ≥ 0: N is the base. Else keep π_k; S3-null |
| **S5** | $17.52 | As S3 on N; two-network claim vs `own1` | As S2 |
| **S3-null** | when chosen | A bound or a named miss | User: w 0.7 (125 band decisions per side-game); E from logs; critic column (~$7); intents after rulings; stop |

**Budget** [D §2]: $56.18 with a $2.17 E reserve (one match a round), $28.82 left; $77.12 on a 1.41× host, three rounds at up to 12% overhead. The slack could buy 5.0 more batches a round (μ 0 floor 0.0016 → 0.0014, no new code) [D §3]; that is the user's budget choice, not this plan.

**Branch L**, if ruling (a) is no: unchanged, five 3,800-game PURE matches (~$4.35, SE 8.0 per arm against placebo); S0 is then the attack known answer alone.

**Beyond the yardstick:** cycles of a coin round and a reset, each certified by match. The $1,000 case comes only after two adopted rounds within their miss lines, sized from measured v̂, J/Ĝ_pf and ρ̂, and opens with one replay that reads γV, DPI and the antithetic siblings as cost cuts. Nothing measured links `own1` to human ratings, and nothing rests on it.

## 4. What is new, and the nearest work

Searched 2026-10-20; finding nothing does not prove absence. The core is the current design's: coin labels at decode near-ties in deterministic self-play, exact through enumerated chance laws, pooled by a closed-form second actor head, certified by match. Nearest:
- [Bottou et al. 2013](https://jmlr.org/papers/v14/bottou13a.html), App. D.1 ([ar5iv text](https://ar5iv.labs.arxiv.org/html/1209.2355) read): the derivative of a deployed system's expected outcome at its own randomised operating point, from logged data: the coin label's estimand. Their randomisation is of continuous auction parameters per query, with learned predictors for variance (§5.2), not of discrete near-ties sharing one game result, with exact chance-law variates.
- Micro-randomised trials ([Boruvka et al. 2018](https://doi.org/10.1080/01621459.2017.1305274)); [AIVAT](https://arxiv.org/abs/1612.06915); RESLOPE ([Daumé et al. 2018](https://mlanthology.org/iclr/2018/iii2018iclr-residual)) and LOLS ([Chang et al. 2015](https://arxiv.org/abs/1502.02206)), whose credit comes from a learned regressor or one roll-out per deviation. A search for near-tie randomisation with logged propensities in game self-play found nothing closer.

This stance adds no method. Its removals rest on standard results: a futility stop costs power ([Li, Herrmann & Rauch 2020](https://doi.org/10.1186/s12874-020-01141-5)); a state-dependent baseline changes only variance ([Williams 1992](https://link.springer.com/article/10.1007/BF00992696); [Greensmith, Bartlett & Baxter 2004](https://jmlr.org/papers/v5/greensmith04a.html), abstracts read).

## 5. Weakest point

The money is spent before the answer is known: the spread v is unmeasured, no record bounds it, and without the interim a null round 1 always costs $16.16 for a bound. I trade $4.42 of expected spend at a null for 0.17 of adoption power at the μ −0.0018 floor, the regime the record leans toward. The trade is right only while money does not bind; if the user values the dollars more than that power, the interim returns as it was. The γV deferral could also forgo more than the slack can buy, if V keeps less than 0.614 of the residual; only the $1,000 replay reads that share.

## 6. Rulings for the user

- (a), (c), (d), (e): unchanged (`current_design.md` §7).
- (b), narrowed: the seven luck columns as label-only control variates; V is no longer asked for.

## Provenance

| Number | Value | Tag | Source or derivation | Used for |
|---|---|---|---|---|
| Plan slack when the interim entered | $9.81 left; three rounds did not fit on a 1.41× host | D | `round6_synthesis_power.out` §6 | Why removal 1 is new |
| Pool price, steady | $1.912e-5 a game; batch $1.91, 4.55 h | M | `round7_provenance_price_tail.out` (ref_p120x6, completions 120–480) | Every cost |
| Host spread | 1.41× | M | `round6_provenance_prices.out` (via BACKLOG.md) | Slow-host budget |
| Two-network price | $1.17e-4 a decisive game (median of 18) | M, upper bound | same | S4, S5 claim |
| Box setup | $0.018 | M | same | S0 |
| Distillation | $2.1 | M, upper bound | `docs/memory_own_trajectories_prereg_20261011.md` | S4 |
| Var(z), own1 self-play | 1.0004 | M | `round5_simplify_variance.json` | σ² |
| Label residual | 0.688 (cell + columns); 0.839 (columns alone, record-fitted); cell alone 0.843 | M | `round6_provenance_cells.out`; `round6_simplify_pairs_raw.out` | σ² 3.67 / 4.48; floors |
| σ² | 3.67; batch 1 4.48 | D | 1.0004 × r / 0.1875 | Floors, plant |
| Effective batches, frozen coefficients | 2.80 at 3, 7.78 at 8 | D | `round7_simplify_power.py` label_r, reused in §1 | Floors, interim |
| Floors (no interim) | sd 0.0016 (μ 0), 0.0034 (μ −0.0018), d 772 | D | §1 (round 5's Gaussian model) | What a null bounds |
| Interim cost in power | 0.80 → 0.77 at μ 0, 0.80 → 0.63 at μ −0.0018, at the floors; P(stop) 0.50 at a null; with-interim bound 0.0036 | D | §1 | Removal 1 |
| Interim false stop, shape-free | ≤ 0.044 (the current design's 0.041 used 3/8; frozen batches give 2.80/7.78) | D | §1 | Record correction, moot after removal |
| Budget | $56.18, $28.82 left; E reserve $2.17; S0 $0.14, S1 $15.30, S2 $0.72, S3 $16.74, S4 $3.60, S5 $17.52 | D | §2 | Regimen |
| Round price lines | three rounds iff price × (1 + overhead) ≤ $3.113e-5 (room 0.628; 0.122 on a 1.41× host); two rounds iff ≤ $4.81e-5 | D | §2 | S1 decision |
| Null round 1 | $16.16; with interim $6.60 / $16.88, expected $11.74 | D | §2 | Removal 1 |
| Slack against γV | 5.0 batches a round; floor 0.0016 → 0.0014; V equivalent iff kept share ≤ 0.614; 1.1 / 2.7 / 8.0 batches a round at shares 0.875 / 0.75 / 0.5 | D | §3 | Removal 3 |
| V's share of the residual | — | U | No record holds V on own1's games (`tools/game_record.py` keeps commands). The $1,000 replay reads it. Nothing in the plan relies on it. | — |
| Plant SE | 0.0387 per side at 2,000 games (simulated 0.036), 0.0055 at 100,000; failure readings −1.005, 0.016, 1 − f for misread share f | D | §4; `round7_simplify_power.out` §5 | Smoke and batch-1 gates |
| Attack known answer | −0.2247 ± 0.0459 z per unit on unselected stage-1 games → −78 ± 16 Elo, window ±33; label 2Δ −0.449 ± 0.092, window [−0.640, −0.259] | M via D | `round6_provenance_critique_attack_stage1.out` B; `round7_synthesis_power.out` §1–3 | S0 gate |
| Elo per unit z | 347.4 | D | 800 / ln 10 | All |
| Match SE | 1.79 at 37,800; 3.07 at 12,800; false adoption 0.023, 0.046 with E | D | 347.4/√N; Φ(−2); 1 − (1 − 0.023)² | Adoption |
| Forecast SE, miss line | 4.6, ≈ 9.9 Elo at μ 0 | D | `round6_synthesis_power.out` §4 | S2 |
| E's first-order share | 0.80–0.87 | D | `round6_simplify_rule.out` | E |
| Sign rule against oracle | within 0.6 Elo | D | `round6_simplify_rule.out` | Rule |
| Band per side-game | 76.1 (19 deviations at p 0.25) | D | 0.339 × 224.6 from 100 parity3 side-games, `round5_simplify_band.py`, `round5_simplify_critique_decision_units.out`; the smoke re-measures for own1 | Volume, floors (restated at S0) |
| Band at w 0.7 | 125 per side-game | D | 0.558 × 224.6 | S3-null |
| Information per game-equivalent | ≈ 41 (coins) against 0.82 (fork pair) | D | 152.2 × 0.1875 / 0.688; 1/(2 × 0.608), `round5_simplify.md` | §2, signal per dollar |
| Match against forecast batch | 17× the information per dollar | D | `round5_synthesis.md` | Every batch trains |
| Caps | 0 of 1,600 own1 self-play games | M | `round6_simplify_walk_cost.out` | Cap rule |
| Walk cost | 0.321 s a game; ≤ 1.8% slowdown | M; D | `round6_simplify_walk_cost.out`; `round7_provenance_price_tail.out` | Overhead |
| Numerics flips | 1.76% of decisions, gaps < 0.014 | M | `round5_provenance_numerics_ties.out` | Logged propensities |
| Residual offset | o(1 − ĉ): −0.27 at ĉ 0.82; offset 0 costs −9.9 ± 4.7 Elo at ĉ 0.82 | D | `round7_synthesis_power.out` §4; `round7_simplify_power.out` §6 (stage 1) | S4, ruling (d) |
| Tempo closure | 1.00 (0.82–1.23), 1.06 (0.88–1.30) | M | `round6_provenance_closure.out` | ĉ indicative only |
| I-V line | 3.4–3.8 SE family-wise; resolves 0.0013 pooled | D | `round7_simplify_critique_checks.out` §2 | I-V |
| S4 acceptance error | P 0.18 at a 2.8-Elo deficit | D | Φ(−2.8/3.07) | S4 |
| Branch L | about $4.35; SE 8.0 per arm against its placebo | D | `round7_synthesis.md` | If (a) is no |
| Spread v; typed c, b; switched share q | — | U | S1's progressive fits; nothing before S1 relies on them | Round gain |
| own1 census, runner-up mix, spectrum | — | U | The S0 smoke, before S1 is proposed | Restated floors |
| Logging overhead (feature gather) | — | U | The S0 smoke against the attack games on the same host | Rounds that fit |
| Second-order and excursion loss | — | U | S2 against Ĝ_pf; E's match | Miss reading |
| ĉ, ρ̂, spillover; two-network pool price | — | U | S4, before its match; timed at S4 | Reset |
| κ_T; r_DPI; antithetic capture | — | U | Only if chosen; the $1,000 replay | Nothing in the plan |
| Compounding, exploitability, human level | — | U | Each round's claim; none; none | Nothing |
| w 0.35, p 0.25, 8 × 100,000 games, d 772, 37,800- and 12,800-game matches, smoke 2,000 games | — | C | Consequences derived above | Design |
