# Round 5, provenance stance: the current design with every number measured, derived or removed

The frame of `current_design.md` stands: an exact luck-and-coin identity, a privileged training-only critic, coins as randomized single-decision deviations with logged laws, a fogged student, optimal stopping for end_turn, adoption by match. This memo audits every number it rests on (section 7), measures what the records settle, and removes or restructures what rested on a guess. No stage relies on a number not measured before it; where a readout cannot be predicted, each outcome maps to a decision.

## 1. What the records settle now

Scripts in `analysis/`, numpy and the standard library, each under 40 s. I read every script the paused attempt left; its outputs follow from its code except where marked.

1. **A corrected match needs no critic.** Seven luck columns plus an additive setup baseline (map × side, faction pair), fitted leave-one-match-out over 23 argmax matches, keep **0.747 (0.733–0.758)** of a result's variance, 0.70–0.82 per match from `parity3` to `own2`; the mean is kept (Σt² = 17.7 on 23 df; `round5_provenance_match_variance.py`). **Side-swapped pairs** need no fitted baseline: the full setup cell carries 0.151 ± 0.005 of the column-corrected variance, so pairs keep at least **0.696** (`round5_provenance_paired_setups.py`).
2. **The constant offset has taken the mean.** Match C is on disk: `own1` at −2.5 against −1.5, z = −0.035 ± 0.035, **−0.0018 (−0.0061 to +0.0017)** per added act (`round5_provenance_stop_acts.py`). The 24 decode-knob vectors (6,400 games) agree (`round5_provenance_knob_switches.py`, the other knobs as covariates, the 40 extra decisions per side-game per unit z that winning brings removed): the offset moves **19.6 ± 1.0 decisions per side-game per unit**; a decision dropped by ending earlier is worth **+0.0033 ± 0.0017**, one added by ending later **+0.0002 ± 0.0010**; **Match A** (offset 0 against −1.5) is predicted at **z = −0.097 ± 0.048** (−34 ± 17 Elo). The draft's reading of the knob fit had its sign wrong: the slope, −0.017 ± 0.009 win rate per unit toward 0, says going below −1.5 gains; the matches read flat over [−3.0, −1.5].
3. **A learned stop gains only on heterogeneity τ.** In a stated normal model, its gain in the [1.5, 2.5) window reaches 2 SE of a 12,800-game corrected match only if √q·τ ≥ 0.0038 (3,200 games: 0.0059), q the share the observation predicts (`round5_provenance_stop_bound.py`). τ is unknown; no stop gain is forecast before S2.
4. **More games of plain result regression buy prediction, not ranking.** The record's true-state critics reach held-out cross-entropy 0.4915, 0.4712, 0.4595 nats at 1,784, 3,475, 6,832 games, each before half an epoch; a doubling gives 0.57 of the previous one's gain, so 50,000 games add about 0.013 nats (`round5_provenance_critic_curve.py`). Their ranking stayed flat (T100 − T25: +0.020 ± 0.047, `critic_step1_20261009/readout.md`). S1's decision information must come from what this design adds, the luck-corrected objective and coin siblings; neither is measured.
5. **Numerics.** 1.76% of replayed decisions flip, at top-two gaps under 0.014 nats (partial r 0.53 ± 0.10; `round5_provenance_numerics_ties.py`): propensities must be logged at generation.
6. **Playout noise** at temperature 0.5, human mid-game positions: 0.608 (0.564–0.654) (`round5_provenance_playout_noise.py`); argmax noise is not on disk.
7. **Winner's curse:** kv15 read +35.7 ± 12.3 when selected, +1.7 ± 12.3 fresh.
8. **Rejected:** `round5_simplify_forks.json`. The belief-world rollouts are four-player-turn horizons and the script kept only the 683 of 7,728 that ended by a leader's death.

## 2. The algorithm

### 2.1 What is learned, from which labels, by which update

- **Generation.** π, the argmax decode, plays both sides. At band decisions (top-two margin G < w) a coin with step propensity p(G) plays a1, drawn from the other top-four candidates, instead of a0; π then re-decides after every roll. Stop coins flip end against continue for L ∈ [−1.5, 4.5]; a continuation plays π's best non-end action for K ~ U{1..4} acts, each re-decided after its roll. A quarter of games are coin-free. Propensity, pair and draw are logged. The schedule's cost needs no model: by the performance-difference lemma J(π_coin) − J(π) = E_{π_coin}[Σ_t p_t·A^π(s_t, a1)], linear in p to first order, so S0 measures it at one trial schedule and scales the steps to the cap; λ and c0 are not needed.
- **Critic C** (training only: full state, both memories, the schedule; two folds; no gradient to the actor): round 4's multistep loss ℓ_k = (y_{k+1} − C(x_k) − L_k(C))², y_s = z − Σ_{j≥s} L_j(C′), on each fight's enumerated law, each trait's and coin's logged law and the result. Its expectation is E[e²] + E[Var_{p_k} e] + E[η²], E[η²] free of C′. Pass 0's C′, the seven columns, leaves 0.847 of the result's variance: the label noise at a game's start.
- **Calibration κ_type:** the weighted centred least-squares slope of the result on (D_i − p_i)·X_i, X_i the critic's predicted a1-minus-a0 contrast, on the other fold's coins ([Boruvka et al. 2018](https://doi.org/10.1080/01621459.2017.1305274); the critic as the proxy of [Chernozhukov et al.](https://arxiv.org/abs/1712.04802)'s best linear predictor). Its estimand is the excursion effect from π_coin, which depends on the randomization elsewhere ([Qian 2025](https://arxiv.org/abs/2502.13500); secondary sources, original unread): a bias of the forecast, never of the match.
- **Student A_θ:** a two-layer head on `own1`'s frozen tokens regressing the critic's one-step look-ahead Â_C over the top four, hence E[Â_C | h] without strategy fusion ([Baisero & Amato](https://arxiv.org/abs/2105.11674)); calibrated to β on unseen coins.
- **Stop head D** (fogged): Longstaff–Schwartz on forced paths, payoffs C(x∘end), scored as played by the other fold's critic.
- **Switch:** on types whose calibration lower bound is above 0, play b where α + β·A_θ(h, b) is largest and positive with posterior ≥ 0.8, at most 10 per side-game.

### 2.2 Estimators

- **Corrected match:** G = z − β̂·S over paired setups, β̂ from other matches. Each fight's luck is centred given its past, so E[G] = E[z] for any β̂. SE(Elo) = 347.4·√(r/n) (347.4 = 800/ln 10, Elo per unit z at an even match), r = 0.747 until a paired match reads its own.
- **SE(κ̂) = σ_T/√(Σp_i(1−p_i)X_i²)**, cross terms exactly zero for decisive games (z² = 1, the D_i − p_i martingale differences). p is logged and X is the critic's output, so the SE is known before truth is read: blinded recalculation from an internal pilot ([Wittes & Brittain 1990](https://pmc.ncbi.nlm.nih.gov/articles/PMC3845535); [Liao et al. 2016](https://arxiv.org/abs/1504.00238) for trials).
- **Floor:** κ²Var(X) ≤ Var(A) (Cauchy–Schwarz on the BLP), so κ̂_lower·sd(X) bounds the advantage spread from below.
- **κ_min = 2·SE_match/Γ_C**, Γ_C = E[Σ_t max(0, max_b X_t(b))] the critic's claimed gain per side-game, read without truth. Below κ_min no calibrated switch set reaches a resolvable gain.
- **Forecast** Ĝ = Σ_switches(α̂ + β̂·A_θ) plus the stop's cross-critic score, SE by the delta method, sealed before the match: the expected gain if calibration holds at the switches (performance-difference lemma).
- **Adoption** ([Thomas et al. 2015](https://proceedings.mlr.press/v37/thomas15.html), a match as the bound): a candidate fixed beforehand, fresh seeds; adopted iff the corrected head-to-head's lower 2-SE bound is above 0 and corrected and PURE agree within 2 combined SE (false adoption 0.023).

### 2.3 Why more signal per dollar

Chance gives about 110 fights, 46 trait rolls and 2.4 advancements a game (`readout.txt`), exact targets at no playout cost. A coin game ($2.5e-5) and a fork pair ($2.4e-5) cost the same; about κ, coins carry Σp(1−p)X²/σ_T², a pair X²/(2σ_f²). Coins win iff Σp(1−p) ≥ σ_T²/(2σ_f²), 0.61 at the measured noise, about one deviation a game; any accepted schedule carries tens, so the choice holds unless argmax σ_f² is below about 0.012 (S0(e) checks). One decision cannot be certified here (an advantage of 0.01 needs (2/0.01)² × 2 × 0.61 ≈ 49,000 fork pairs), so coins only calibrate, one slope per type.

### 2.4 Out of round 1, and what brings each back

- Decision windows: κ fails while the critic beats the columns on chance.
- Consequence heads or a larger critic: the critic does not beat the columns.
- The factored attack arm: κ_attack open, student R² against Â_C low, and ruling (b).
- Ĝ at horizon 2, the mixture match, the privileged twin: a match misses its forecast.
- Fallback F (ridge on frozen features, fitted on coin labels, SE blind like κ's): the critic route closes and ruling (a) allows it.

### 2.5 Invariants

Round 4's I1–I11; I12 exact. New: **I13** the corrected-match code reproduces Σt² ≈ χ²₂₃ on the 23 matches and reads a planted ±0.1 handicap within 2 SE; **I14** permuting results leaves every computed n unchanged; **I15** a pair swaps seats and factions exactly, and a capped game drops its pair.

## 3. The regimen

Prices at $0.42/h, derived from the seed's throughputs: generation game $2.5e-5, eval-numerics match game $1.49e-4, playout $1.2e-5, training position $7.8e-8, critic forward $3.6e-8, critic-data game about $3.1e-4 (generation, memory recompute, two refreshes of about 1,600 forwards and multistep children; the 1,600 assume about 8 outcomes per fight, which S0 counts).

| Stage | Cost | Readout and prediction | Decision per outcome | Plant |
|---|---|---|---|---|
| **S0 instruments** | $2.5 | (a) Replay of `own1`'s 1,600 recorded games: agreement (none predicted; `parity3` 98.2%), band census by type, runner-ups, outcomes per fight. (b) Generation self-agreement and against eval, 2,000 games each. (c) Coin cost at a trial schedule, 20,000 games, SE 0.0061 z per side-game. (d) Match A, 3,200 corrected games, SE 5.3 Elo, **predicted −0.097 ± 0.048 z**. (e) 1,000 argmax fork pairs: σ_f². (f) Critic forward time. | (b) generation worse than self-agreement by 2 SE: generate at eval numerics (+$7.4 at 60,000 games). (c) scales the steps to the cap. (d) outside 2 combined SE: the decision-count model is wrong; the stop bound's N comes from the replay. (e) σ_f² < 0.012: forks calibrate. (a), (f) size S1 in dollars. | **P-coin**: 0.05·even(played hex) per coin decision, slope 1 ± 0.1; n = (σ_T² + m·a²/4)/(0.05²·(a²/2)·Σp(1−p)), for example 8,900 games if the schedule gives Σp(1−p) = 31 |
| **S1a** | $2 | **P1**: mean-zero lottery (scale 0.05, asymmetric logged law) at about 160 events a game, booked into z, 5,000 games; held-out residual's slope on the lottery sum 0 ± 0.045 (SE 0.022 = 1/(√5,000·√(160·0.05²))); dropped corrections 1, sign-flipped 2. | Outside: the critic path is broken; nothing downstream runs. | P1 |
| **S1b pilot** | $5 | 10,000 coin games; critics at 2,500 and 5,000 games per fold. In order: ρ̂ against the columns, paired; blind SE(κ̂), Γ_C, κ_min per type; then κ̂. **No prediction**: nothing measured predicts this critic (point 4). | (i) a κ̂ lower bound above 0: size S1c to SE(κ̂) ≤ κ_min/2.84, cap $20; skip S1c if 5,000 does not beat 2,500 by 2 paired SE. (ii) every κ upper bound under κ_min, ρ̂ beats the columns: windows ($6.5), re-read. (iii) ρ̂ not below the columns: consequence heads or a larger critic ($5). (iv) still failing: the route closes at about $15, the failing quadrant named; F if ruling (a) allows. | — |
| **S1c main** | ≤ $20 | κ̂ per type at the derived SE; floor κ̂_lower·sd(X). | Types with lower bound above 0 open. | P1, 2,000 games, if code changed |
| **S2 operators** | $6 | Plants first, bands 2 SE of each stream's variance on a 500-game pilot: **P2** village tax at a hidden rate, the student's planted part 0.0044 per village-turn whatever the bit; **P3** payoffs −0.01, −0.01, +0.05 per act past the stop, LS reaches depth 3, greedy 0. Known answers: the stop machinery values offsets 0 and −2.5 at Match A and Match C within 2 combined SE. Then student, β̂, sealed Ĝ per factor. | A plant outside: that path is broken. A known answer missed: stop payoffs biased; the stop leaves round 1. β̂ lower bound ≤ 0: the type stays closed. | P2, P3 |
| **S3 round 1** | $6 | π_1 against `own1`, corrected, n = 0.747·(347.4·2.84/Ĝ)², cap 12,800 (SE 2.65); Ĝ < 7.5 Elo is declared unresolvable beforehand. Stop-only and actions-only at the same n (diagnostics); PURE against `own1` and `parity3`. Prediction: Ĝ ± 2 combined SE. | Adoption rule. Outside the forecast: the factor whose diagnostic misses failed; Ĝ at horizon 2 at its switches. π_1's Elo over `parity3`, less `own1`'s measured +88.6 ± 12.7, below its gain over `own1` by 2 combined SE: exploitation of `own1`. | — |
| **S4 rounds 2–3** | $9.4 each | 15,000 coin games of π_k, warm refit, S1b's readouts, S2, S3 against π_k. | Continue iff the last lower bound is above 0 and the new Ĝ exceeds 7.5 Elo. | P1, P2, P3 |
| **S5 distillation** | $3 | **P5**: `own1` at −2.5 into a fresh head; agreement within 2 SE of S0's self-agreement, level within 2 SE in a 3,200-game corrected match. | Outside: distillation broken. | P5 |

Costs: S0 = replay $0.02 + agreement $0.37 + coin cost $0.50 + Match A $0.48 + forks $0.02 + P-coin $0.22–0.70; S1a = 5,000 critic-data games $1.55 + training; S1b = 10,000, $3.1 + four critics; S3 = three 12,800-game matches $1.9 each + PURE $0.24; S4 = 15,000 games $4.65 + P1 $0.6 + refits + a match.

**Budget:** S0–S3 about $42 at S1c's cap; $63 with two more rounds and S5; $22 left for one remedy branch and the eval-numerics fallback. The window price ($3.5e-5, round 4) is untraced and re-derived only if that branch opens. **The $1,000 case** is argued only from two certified rounds that fell within 2 SE of their forecasts; any extrapolation is a model fitted to them, re-measured every five rounds, stopped below its lower bound. Nothing measured links `own1` to human ratings, and nothing rests on it.

**Rulings for you:** (a) do κ, β, P-coin and F, regressions of results on randomized coins, fall under the killed outcome policy gradient? (b) the factored arm, if it returns; (c) adopt the learned stop rule on non-inferiority (lower bound above −2 SE, estimate ≥ 0), since it replaces a hand-set offset you want gone.

## 4. What is new, and the nearest work

Searched 2026-10-20; absence of a find does not prove absence.

1. **Coins as a micro-randomized trial on a game's decisions,** the result as distal outcome, calibrating a privileged critic's predicted contrasts by a BLP slope sized blind from logged propensities. Nearest: micro-randomized trials ([Liao et al. 2016](https://arxiv.org/abs/1504.00238); [Boruvka et al. 2018](https://doi.org/10.1080/01621459.2017.1305274); [Qian 2025](https://arxiv.org/abs/2502.13500)), generic ML inference ([Chernozhukov et al.](https://arxiv.org/abs/1712.04802)), Bellman calibration with doubly robust targets ([van der Laan & Kallus 2025](https://arxiv.org/abs/2512.23694)), which calibrates levels on observational data, not contrasts under designed randomization. I found no use in game self-play.
2. **A regimen whose sizes, predictions and kill lines are formulas of earlier measurements.** Nearest: internal pilots with blinded recalculation, HCPI, AlphaGo Zero's gate, Fishtest's SPRT and pentanomial pairs; the combination with a critic's blind information measure appears new.
3. **Decode records as an instrument** for marginal decision values on both sides of the stop (performance-difference lemma): weakly new.
4. Round 4's algorithmic novelty stands: the identity as an exact system for a fogged player's privileged critic, and turn stopping as optimal stopping on forced paths. Not claimed: corrected matches (AIVAT, [Burch et al.](https://arxiv.org/abs/1612.06915); CUPED), paired setups, the double estimator.

## 5. Weakest point

Every gain path passes through κ: whether a critic trained on chance events and coin games ranks decision siblings. The strongest evidence on record is against it: the record's critics improved in prediction with games and not in ranking, and pass 0's labels remove only 0.153 of the result's noise at a game's start. What coins add to ranking is unmeasured. The pilot measures it for about $7 and every outcome names a next step, but the route can close at about $15 with no gain, knowing only which of chance learning and decision transfer failed. Second: the stop, round 4's "likeliest gain", has a measured mean of zero at the margin; its gain rests on τ, unmeasured.

## 6. Decision record

Rejected: predicting ρ_C, κ, R², β, τ or a round's decay (ranges with no source); the critic-corrected match as yardstick (critic-free correction measured, pairs remove the fitted baseline); S0's Match C (on disk); "the stop is the likeliest gain" (mean measured at zero); separate weak and band coins (one logged step propensity, scaled by the performance-difference lemma); kernel work as a numerics fix (eval-numerics generation is the derived +$7.4 fallback; logged propensities keep the identity exact either way); `round5_simplify_forks.json` as playout noise. Corrected: the draft's sign on the knob fit's offset slope; "coins 40× forks", replaced by the break-even inequality.

## 7. Provenance table

| Number | Value | Used for | Tag | Source or derivation |
|---|---|---|---|---|
| Events per game | 110 fights, 46 trait rolls, 2.4 advancements | critic data, P1 | measured | `match_luck_20261019/readout.txt`, gen_own1_selfplay counts / 800 |
| Columns' kept variance | 0.847 (leave-one-match-out) | label noise, match SE | measured | `round5_provenance_match_variance.out` |
| Setup baseline + columns | 0.747 (0.733–0.758); per match 0.70–0.82 | every match size | measured | same |
| Pairs + columns | ≥ 0.696 | yardstick design | measured (lower side) | `round5_provenance_paired_setups.out` |
| Mean preservation | Σt² = 17.7 on 23 df | I13 | measured | `round5_provenance_match_variance.out` |
| Setup shares | 0.157 ± 0.006 full cell; 0.097 factions; 0.010 map × side | pairing | measured | `round5_provenance_setup_cells.out` |
| Elo per unit z | 347.4 | all SEs | derived | 800/ln 10 at p = ½ |
| Match SE | 12.3 (800 PURE); 5.3 (3,200 corrected); 2.65 (12,800) | sizes | derived | 347.4·√(r/n) |
| Prices | $2.5e-5 per game; $1.49e-4 per eval game; $1.2e-5 per playout; $7.8e-8 per position; $3.6e-8 per forward | costs | derived | $0.42/h over seed §2 throughputs |
| Outcomes per fight | about 8 | critic forward cost | unknown | round 4 invent, untraced; S0(a) census before S1 is sized |
| Critic-data game | about $3.1e-4 | S1, S4 costs | derived (one unknown input) | generation + recompute + 2 × 1,600 forwards + children |
| Replay agreement | 98.2% (`parity3`) | numerics | measured | `critic_step1_20261009/prior_gaps.json`; `own1`'s in S0(a) |
| Near-tie flips | 1.76%; ε 0.014 nats; partial r 0.53 ± 0.10 | logged propensities | measured | `round5_provenance_numerics_ties.out` |
| Band census | 309, 230, 138 per nat per side-game; 91 decisions at w = 0.35, 66 of them moves | schedule | measured (`parity3`) | `round5_simplify_band.json`; `own1`'s in S0(a) |
| λ, c0 | — | old taper | unknown, removed | S0(c) cost at one schedule, linear scaling |
| Per added act, Match C | −0.0018 (−0.0061, +0.0017) | stop prior removed | measured | `round5_provenance_stop_acts.out` |
| Per decision removed / added | +0.0033 ± 0.0017 / +0.0002 ± 0.0010 | stop prior removed | measured | `round5_provenance_knob_switches.out` |
| Decisions per unit of offset | 19.6 ± 1.0 per side-game | stop bound's N | measured | same |
| Winning's extra decisions | 40 per side-game per unit z | count correction | measured | same (1,600 self-play games) |
| Match A | predicted −0.097 ± 0.048 z | S0(d) check, S2 known answer | derived | piecewise fit stated in the script |
| Match C | −0.035 ± 0.035 z | S2 known answer | measured | `round5_provenance_stop_acts.out` |
| Attack offset | −0.126 ± 0.027 z per unit | context | measured | `round5_provenance_knob_switches.out` |
| Knob fit, offset slope | −0.017 ± 0.009 win rate per unit toward 0 | sign correction | measured | `round5_challenge_records.txt` |
| Stop visibility | √q·τ ≥ 0.0059 (3,200); 0.0038 (12,800) | the stop's meaning | derived | `round5_provenance_stop_bound.out` (normal model) |
| τ, q | — | stop gain | unknown | nothing relies on them; S2 cross-critic score, S3 stop-only match |
| Critic learning curve | 0.4915, 0.4712, 0.4595 nats; r = 0.57; +0.013 at 50,000 games | S1 expectations, scaling test | measured; extrapolation derived | `round5_provenance_critic_curve.out` |
| Critic ranking with data | +0.020 ± 0.047 (T100 − T25) | weakest point | measured | `critic_step1_20261009/readout.md` |
| ρ_C, κ, β, student R², Γ_C | — | gates | unknown | S1b, S1c, S2; gates on measured bounds |
| SE(κ̂) | σ_T/√(Σp(1−p)X²) | S1c sizing | derived | martingale differences, z² = 1 |
| κ_min | 2·SE_match/Γ_C | S1b decision | derived | Γ_C read blind |
| Decision-spread floor | κ̂_lower·sd(X) | sizing any fork branch | derived | Cauchy–Schwarz on the BLP |
| Playout noise | 0.608 (0.564–0.654) at temperature 0.5 | break-even | measured | `round5_provenance_playout_noise.out`; argmax in S0(e) |
| Coins-vs-forks break-even | Σp(1−p) ≥ σ_T²/(2σ_f²) = 0.61 | instrument choice | derived | information per equal-price unit |
| One-decision certification | 49,000 fork pairs | instrument choice | derived | (2/0.01)² × 2 × 0.61 |
| P1 SE | 0.022 | P1 band | derived | 1/(√5,000·√(160·0.05²)) |
| P-coin n | formula in S0's row (8,900 games if Σp(1−p) = 31, an illustration) | P-coin | derived after S0(c) | schedule fixed by S0(c) |
| Coin-cost SE | 0.0061 z per side-game | S0(c) | derived | √(0.747/20,000) |
| Power multiplier | 2.84 | sizing | derived | 2 + 0.84 |
| False adoption | 0.023 per round | adoption | derived | one-sided z = 2 |
| Unresolvable forecast | 7.5 Elo | S3, S4 | derived | 2.84 × 2.65 |
| Eval-numerics fallback | +$7.4 | budget reserve | derived | 60,000 × ($1.49e-4 − $2.5e-5) |
| Window price | $3.5e-5 | failure branch only | unknown | round 4, untraced |
| Remedy costs | windows $6.5; heads or larger critic $5 | budget reserve ($22 covers both) | unknown | round 4; re-derived from S0's prices before a branch opens |
| `own1` over `parity3` | +88.6 ± 12.7 Elo | exploitation check | measured | CLAUDE.md lineage (docs/own_confirm_prereg_20261011.md) |
| Winner's curse | +35.7 → +1.7 Elo | fresh-match rule | measured | `docs/decode_knobs_prereg_20261012.md` |
| Shift cap 0.06; posterior 0.8; ≤ 10 switches; quarter coin-free; plant scales; pilot sizes; S1c cap $20 | — | design | choices | nothing derived rests on them; the forecast includes them |
| Removed ranges | s, χ² budgets, ρ_C 0.55–0.75, κ ranges, Match A [−0.60, −0.25], band 4–9×, round prior 0.2–0.8, stop and action priors, R² and β ranges, P5 97%, "0.65 for 5 Elo", "coins 40×" | — | unknown, removed | replaced as stated in the rows above |
| `own1` against strong humans | — | end goal | unknown | nothing rests on it; needs rated human games |
| Budget | $85; $0.42/h | everything | given | seed §1 |
