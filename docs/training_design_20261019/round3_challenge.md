# Round 3 challenge: play each candidate out, then read

Stance: the current design learns a critic and then trusts its look-ahead about decisions. This proposal never trusts a critic about decisions. For each candidate at a decision, it forks the true state and lets the game play out for k side-turns, the policy re-deciding after every roll. A critic only cancels the dice inside that window and reads the state at its end. A fogged student learns from these contrasts. Matches, not per-switch statistics, certify each new policy. The critic's errors on positions become label noise, which more forks average away, instead of operator bias, which nothing removes.

## 1. Why the current design's operator is the wrong bet

- **Its decisions rest on the critic's generalisation.** By its own accounting, S1 carries about 125,000 χ² on chance contrasts and 140 to 320 on decision contrasts. The student regresses the critic's one-step look-ahead Â_C; coins only calibrate it, through slopes with standard errors of 0.11 to 0.17. If Â_C is wrong on moves, nothing in the loop learns the right answer.
- **The record says static reads miss what play reveals.** `obs8`'s value head ranked candidate turns at 0.27, 0.42 and 0.53 read 1, 3 and 7 side-turns later through the policy's playouts, and at 0.22 read statically; the linear grader scored 0.27, the HP margin 0.40 (`docs/turn_value_prereg_20260925.md`). The material look-ahead was neutral; true-state rollouts selected better turns (+0.049 ± 0.017) where fitted critics did not. Caveat: whole turns with their own dice, continued at temperature 0.5.
- **The literature agrees, in a dice game.** Rollouts truncated after 7 steps and read by the net cut TD-Gammon 2.1's equity loss per move from 0.0120 to 0.0018, 6.6 times, and a 10-hidden-unit net's by 4.8 times. The base net also graded the test positions ([Tesauro & Galperin 1996](https://arxiv.org/html/2501.05407v1)).
- **Its certification is costly and caps the gain.** The arcsine audit's per-game variance is π²Mσ_a² ≈ 47 (M = 12, σ_a² = 0.4), a match's at most 1: per unit of variance, matches cost a tenth (two matches) to a fortieth (round 1). The switch rule (posterior ≥ 0.8, at most 10 per side-game) leaves most decisions untouched, and a $15 critic refit puts each round at $17.

## 2. The algorithm

### 2.1 Fork labels

A decision point is a state x of a game of π, with mover history h and candidates B: π's choice a0 and the next three non-end joint actions by prior. For each b in B, fork the core (true state, both 64-slot memories, rejection sets) and apply b. Both sides then play π at argmax with fresh dice until k side-turn boundaries pass or the game ends. The label is

    Y_k(b) = C(x_τ) − Σ_{chance events j in the window} ℓ_j,    ℓ_j = f(x_j, o_j) − Σ_o p_j(o)·f(x_j, o)

where x_τ is the window's last state (a finished game reads its result), p_j the core's exact law of event j (fights, b's own included; trait rolls; advancements), and f a luck head.

**Exactness.** E[ℓ_j | x_j] = 0 for any f, so E[Y_k(b) | x] = Q_k(x, b) := E[C(x_τ) | x, b, π] whatever f is. With e = V^π − C, and V^π a martingale along the closed loop,

    Y_k(b) = V^π(x_b) + Σ_j [L_j(V^π) − ℓ_j] − e(x_τ)

A contrast D_k(b) = Y_k(b) − Y_k(a0) is biased by E[e(x_τ^{a0})] − E[e(x_τ^b)], the critic's error at the window's end, which shrinks as the decision's consequences turn into what the critic prices well (hit points, kills, gold). Its noise is f's error on each event's outcomes plus the spread of e(x_τ). For a set S of decisions, the **materialisation factor** m(k) = E Σ_S D_k / E Σ_S D_∞ is measured against matches in S2.

**Luck head.** f reads the state before an event and one token per outcome, and prices all outcomes in one forward (an afterstate value, as in Stochastic MuZero, fitted to the critic's per-outcome values). A fight then costs one forward instead of the critic's 10 or so. This is an assumption until S1 measures it.

**Cost.** A side-turn holds about 11 decisions and 2.75 fights (87,720 attacks in 800 `own1` games of 20.1 turns; `training/metrics/match_luck_20261019/readout.txt`). That is about 14 forwards, $8×10⁻⁷. A decision point at k = 3, with four forks, costs $1×10⁻⁵.

**Information.** The chance variance per side-turn is v ≈ Var(z)/40 ≈ 0.025. A fork carries σ_k² ≈ ρ_f·v·k, where ρ_f is the share of that variance f leaves. A contrast of true spread s_k carries s_k²/(2σ_k²) and costs in proportion to k, so information per dollar falls as 1/k². At k = 3, ρ_f = 0.7 and s_k = 0.008 to 0.012, a 400,000-point round gives the student about 700 to 1,700 χ² on decision contrasts. The current design's S1 gives 140 to 320, and gives them to a slope.

### 2.2 The critic

C reads the true state, both memories, the rejection sets and the matchup tag, and starts from `own1`'s trunk. It is fitted semi-gradient on luck-corrected returns with exact enumeration (the current design's U_0), and f is distilled from it. Its quality sets the forks' noise, not their direction: at ρ_f = 0.85, the linear columns' level, a round needs 1.2 times the forks it needs at 0.7.

From round 2, C also trains on fork states: the state right after b gets the target Y_k(b), an exactly luck-corrected n-step target on the off-policy states a look-ahead prices. Its readout is **horizon contraction**: m(1) rises, k* can fall, and forks get cheaper as 1/k².

### 2.3 Student, stop, decode

- **Student.** A two-layer head on `own1`'s frozen token outputs (memory included) regresses the within-point centred D_k by squared loss, with folds split by game. Its minimiser is E[A_k | h], the information-state advantage, so no strategy fusion enters ([Baisero & Amato](https://arxiv.org/abs/2105.11674)).
- **Stop.** In the current design's window L ∈ [−1.5, 4.5] the variants are: end now; act once, then π decides; act by π's best non-end choice for 2, or 5, more decisions, re-deciding after every roll, then π decides. A head predicts D_m(h) = Q_k(variant m) − Q_k(end); the rule ends iff max_m D̂_m ≤ 0. Outside the window every offset from 0 to −4.5 makes the same choice, so the −1.5 offset leaves the decode in round 1. The long variants price several more acts, which a one-step rule misses when a turn is not monotone.
- **Decode.** π_{k+1}(h) = argmax_b [ℓ_k(h, b) + λ_type·Â_θ(h, b)], where ℓ_k is π_k's joint log-probability. This is the mode of π_k·exp(λÂ_θ), [piKL](https://arxiv.org/abs/2112.07544)'s form, with no cap.
- **Choosing λ.** Every candidate at a held-out point has a label, so S(λ), the mean of D_k(π_λ(h)), is unbiased for the one-step k-horizon gain. Its maximiser λ* is the prior; matches at λ*/2, λ* and 2λ* read the real curve.

### 2.4 Forecast and truth

- **Ĝ_old** sums S(λ*) over a side-game.
- **Ĝ_plug(k) is measured.** In games of π_{k+1} against `own1`, paired forks at every switch compare π_{k+1}'s choice with π_k's, each continued by π_k. By the performance-difference lemma their sum is the round's gain at horizon k. 6,000 games give about 9 Elo of standard error for $1.2.
- **Truth** is the luck-corrected match z − Σ_j ℓ_j, unbiased in the AIVAT pattern ([Burch et al.](https://arxiv.org/abs/1612.06915)). Its per-game variance is about ρ_f, so 3,200 games give about 5 Elo.
- **Prediction:** the match lands at m̂·Ĝ_plug(k*), with m̂ from S2 (stops from P0f, actions from P-PDL).
- **Reading a miss:** match below the prediction with the λ curve peaking below λ*, the switches interact; with the curve flat, the horizon is myopic. Ĝ_plug below Ĝ_old: covariate shift.

### 2.5 Rounds and invariants

**Rounds.** Round k+1 plays 25,000 games of π_k (a quarter against `own1`), forks 300,000 points at k* continued by π_k, and stacks a new head with its λ on the frozen trunk. C and f refit warm on fork states every second round; distillation is own-trajectory training, as in the current design.

**Invariants (tests from this spec):** each law sums to 1 and Σ_o p(o)(f(o) − E_p f) = 0; on an enumerable duel with f = C = V, Y_∞ = Q with zero variance; a fork given the record's dice reproduces it; no privileged token reaches a head (static check, plus P2); no game's labels reach both the training and the λ fold; permuted labels null S(λ) and Ĝ; λ = 0 reproduces π_k; π_k's forecast against itself is 0.

## 3. Regimen

Prices are the current design's: a game $2.5×10⁻⁵ (generation numerics), a forward about $5×10⁻⁸, a 3,200-game match $0.40 (eval numerics).

| Stage | Cost | Predicted readout | Kill line |
|---|---|---|---|
| **S0 instruments** | $3 | (a) Numerics agreement ≥ 98%, as in the current design. (b) 1,000 forks given the record's dice reproduce it. (c) Match A, `own1` at offset 0 against −1.5, 3,200 games: E[z] in [−0.60, −0.25]; disagreements per side-game (M_A) logged. (d) **P-luck**, f = `own1`'s value head, 3,000 games: Σℓ per game has mean 0 within 2 SE, variance ratio 0.80–0.92 (record: 0.82–0.86). | (a) under 95%: batch-invariant kernels first. (b) or (d) missed: fork copy, law logging or enumeration broken; nothing downstream runs. |
| **S1 critic, luck head** | $6 | 20,000 `own1` games. **P1** (a fair coin at turn 3–8 grants 30, 60 or 90 gold): f's term at the coin against luck-corrected fork truth at 1,000 positions, slope within 2 SE of 1. Held-out ρ_f 0.45–0.75; f within 0.03 of the critic's enumeration. | P1 outside: luck path broken, stop. ρ_f ≥ 0.85: the value head serves as f, with a fifth more forks. No other kill: C sets noise, not direction. |
| **S2 horizon** | $6 | **P0f**: 2,000 games of offset 0 (side A) against −1.5; paired forks (end against continue, continued by −1.5) at every disagreement, read at k = 1, 2, 3, 4, 6 and at the end. The end read reproduces Match A within 2 combined SE (fork SE about 0.09 at M_A = 25); m_stop(2) in [0.5, 1.0], m_stop(4) in [0.7, 1.1]. **−2.5**: the same at −2.5's disagreements, within 2 combined SE of the record's −0.035 ± 0.035. **P-PDL**: 4,000 `own1`-against-`parity3` games, paired forks at `own1`'s disagreements continued by `parity3`, read to k = 6: m_act(k) against +88.6 ± 12.7 Elo (combined SE about 25 Elo). σ_k² within 25% of ρ_f·v·k. k* is the smallest k with m ≥ 0.75 on both. | P0f outside: the fork or performance-difference path is broken, stop. m(6) < 0.5 on both: consequences do not materialise at affordable horizons; hand over to the current design's critic look-ahead, losing only S2. |
| **S3 round 1** | $9 | **P2**, the village tax with a hidden rate (current design): the fogged head's planted part reads 0.0044 per village-turn whatever the bit, a privileged copy 0.010 and 0.002; π_1 on the taxed game holds villages within 2 SE of its fork forecast. Then 400,000 points (a quarter in the stop window) from 30,000 `own1` games at k*. Match predicted at m̂·Ĝ_plug within 2 combined SE; prior +25 to +95 Elo, central +50. Matches: π_1 at λ*/2, λ*, 2λ*; stop-only; actions-only; against `parity3`. | P2 outside: student or decode path broken. Adopt at 2 SE in a fresh match. A miss, read as in 2.4: halve λ, or double k*. Gain over `parity3` under half that over `own1`: exploitation. |
| **S4 rounds 2–5** | $8 each | As 2.5, four matches each. Round 2 reruns P1 on the warm refit. By round 3, m(1) on a 500-game P0f slice has risen ≥ 0.1. Prior: 0.4–0.8 of the previous round's gain; from round 3, the measured ratio. | P1 outside: refit path broken. Two rounds under one match SE (about 5 Elo): S5. |
| **S5 distillation** | $4 | **P5**: `own1` at −2.5 distilled into a fresh head, held-out agreement ≥ 97%, level within 2 SE; then the stacked decode into one head, level within 2 SE. | P5 outside: distillation path broken. |

**Budget.** S0–S3, three S4 rounds and S5 cost about $52; a fifth round adds $8 and leaves one rerun inside $85. If generation and eval numerics disagree, batch-invariant kernels (+$7) cost one round.

**Why this gives more strength per dollar.** The student learns from about 5 times the current design's decision χ², on an estimand whose bias is calibrated against matches; every decision may switch; a round costs $8, not $17; certification buys the cheapest unbiased readout of a policy's total; a weak critic costs forks, not correctness, and the fallback is the current design itself. Sizing priors: +50 Elo in round 1 and about +110 after four rounds for $52, against about +40 and +20 over two rounds for $54 at the midpoints of the current design's priors. Round 1's match against m̂·Ĝ_plug is their first test.

**The $1,000 case**, argued after two rounds land within their predictions: 2 million points a round (about 7 times the χ²), a full student with a budgeted critic refit, 30 or more rounds, a league check. Nothing measured links `own1` to human ratings.

## 4. What is new, and the nearest work

Searched 2026-10-19; absence is not proven.

**The core is old:** truncated rollout improvement over a strong base with a learned evaluator at the cut ([Tesauro & Galperin 1996](https://proceedings.neurips.cc/paper/1996/hash/996009f2374006606f4c0b0fda878af1-Abstract.html); DPI-Critic, [Gabillon et al., ICML 2011](https://icml.cc/imls/conferences/2011/papers/541_icmlpaper.pdf); CBMPI, [Scherrer et al., JMLR 2015](https://jmlr.org/papers/v16/scherrer15a.html), whose rollout length trades classifier against value error; AlphaGo's rollout and value mix; Bertsekas's truncated rollout as a better start for a Newton step, [lecture summary](https://www.digitalfutures.kth.se/event/distinguished-lecture-dimitri-bertsekas-arizona-state-university/), text not checked).

**New, as far as I found:**
1. **Exact luck cancellation inside training forks**, by an amortised afterstate head at one forward per fight, unbiased for any head. Nearest: luck-adjusted backgammon rollouts ([Montgomery 2000](https://bkgm.com/articles/GOL/Feb00/var.htm)) and AIVAT, which use a fixed evaluator and serve evaluation.
2. **The horizon's bias calibrated against matches.** The performance-difference lemma, in fork form, is applied at the disagreements of policy pairs whose match is on record. Nearest: [STEVE](https://arxiv.org/abs/1807.01675) blends horizons by model uncertainty, and CBMPI tunes its rollout length by score; neither has an unbiased known answer.
3. **Horizon contraction**: fork-state targets train the critic, read as m(1) rising.
4. **The round's gain measured by paired forks at the new policy's own switches.**
5. **Fusion-free fogged regression on privileged fork labels**, with stop variants that price several more acts.

Deferred: choosing which points to fork, with known inclusion probabilities ([Zrnic & Candès 2024](https://proceedings.mlr.press/v235/zrnic24a.html)).

## 5. The weakest point

**Information.** A fork label's noise has a standard deviation near 0.23 at k = 3, against advantages near 0.01. A round's 700 to 1,700 χ² can teach a regularised head a coarse ranking, not a fine one. Two things can make this worse:
- **Myopia.** If a decision's value takes more than about four side-turns to appear in what the critic prices, labels favour short-term material. m(k) then stays low, and information per dollar falls as 1/k². S2 measures m(k) for stops and for `own1`'s disagreements with `parity3`. The student's own switches may be of another kind, which only the match against m̂·Ĝ_plug shows.
- **Fog.** Rollouts from the mover's view, hidden units removed, selected no better turns. Hidden units placed from a belief recovered part of the gain (`docs/belief_worlds_prereg_20261012.md`). E[A | h] may keep little of the privileged contrast.

Compounding under zero-sum play has no theory here either.

**Rulings for the user:**
- Do the 2- and 5-decision stop variants, each act re-decided after every roll, satisfy "never whole turns"?
- Is reading λ's three-point curve by matches acceptable decode tuning, like the offset sweep?

## 6. Decision record

- **Rejected: full-horizon forks as labels.** Seven times the variance per pair at seven times the cost: about 50 times less information per dollar than k = 3. They serve known answers only.
- **Rejected: enumerating the window's dice.** About 6^2.75 ≈ 140 leaves per side-turn.
- **Rejected: Rhee–Glynn randomised-horizon labels.** Their far increments carry the critic's whole residual, divided by a small probability.
- **Rejected: common random numbers.** Measured dead under any keying (`docs/selfplay_redesign_20260904.md`).
- **Rejected: per-switch gates and caps, and the arcsine audit as the main readout.** Matches certify the total for less (section 1).
- **Rejected: Longstaff–Schwartz on the critic's end-of-turn value.** That payoff is the position read statically, the critic's weakest quantity.
