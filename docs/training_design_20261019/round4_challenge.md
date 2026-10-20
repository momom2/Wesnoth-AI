# Round 4 challenge: forks teach what a decision causes, results teach what a board is worth

Stance: the current design prices decisions by differencing a learned value of positions, on 140 to 1,040 χ² of decision information, most of it biased by the critic. This proposal splits the job:
- **Consequences.** A fogged network learns what each candidate causes, the expected board after the opponent's reply, from realised fork boards, which need no critic.
- **Valuation.** A low-dimensional linear valuation of boards is fitted on results.
- **Trust.** One luck-corrected factorial self-play experiment per round measures every decision group's effect at once.

No coins, no critic in the loop, about $6.5 a round.

## 1. Why change the core

- **The current design.** Its decision values rest on the critic generalising from chance to decisions, decided at S1 ($21) with no prediction of ρ_C. A critic-data game costs ten times a plain one. Band coins move about 50 decisions per side-game. A round costs $11 and measures one contrast.
- **The record: reads after play beat static reads.** `obs8`'s head ranked candidate turns at 0.27, 0.42 and 0.53 read 1, 3 and 7 side-turns later. The static HP margin after the turn (0.40) beat the fitted graders (0.27) and the static head (0.22) (`docs/turn_value_prereg_20260925.md`). The failed material look-ahead (−8.7 Elo) read material before any play.
- **For:**
  - KataGo's ownership and score targets, which predict the final board: removing them cost 190 Elo at 2.5G queries, or 1.65 times the training time ([Wu 2019](https://arxiv.org/html/1902.10565v5), Table 2).
  - Direct Future Prediction regresses future measurements per action and acts on their weighted sum. It beat A3C by 2.5 to 6 times on the hard Doom scenarios ([Dosovitskiy & Koltun 2017](https://arxiv.org/abs/1611.01779)).
- **Against:** KataGo's targets were auxiliary to a value head; DFP's weights were given and its game single-agent; expectation models are exact only for values linear in the features ([Wan et al. 2019](https://arxiv.org/abs/1904.01191)); value-equivalence says a model should predict only what matters to value ([Grimm et al. 2020](https://arxiv.org/abs/2011.03506)), and here φ is small and g weighs it.

## 2. The algorithm

### 2.1 Consequence labels

φ(x) is the board's dynamic rule quantities, as sufficient statistics:
- per side and cell (unit type, level): units alive, HP sum and XP sum;
- each leader separately;
- villages and gold per side;
- terminal flags.

Per-token heads add each present unit's state and each village's owner. H1 is the mover's next side-turn start, after the opponent's reply; H2 is the one after.

**A window point** is a decision of a game of π_k. Its candidates B are the top 4 non-end joint actions by prior, plus end in the stop window. For each b, the point forks the core (true state, both memories, rejection sets) and applies b. Both sides then play π_k at argmax with fresh dice, re-deciding after every roll, and the point records Y_H(b) = φ(x_H^b) at H1 and H2. So E[Y_H(b) | x] = E_π[φ(x_H) | x, b], with no critic in the label. Stop points fork "end" at each depth of the current design's forced path.

**Cost.** About 38 decisions per fork to H2, 154 forwards a point: $8.6e-6 in round 1, about twice that once π_k carries a consequence head. The current window costs $3.5e-5. Each decision of each generated game is also a free label for its played action.

**Choice of points.** Points are sampled in proportion to the model's predicted spread among candidates, floored at a quarter of the mean. Selecting on h changes the regression's design, not its target.

### 2.2 The consequence model

ψ_f(h, b) is initialised from `own1`. It reads the mover's observation, memory and candidate tokens. Its heads are per token (unit state, village owner) and global (all cells, hidden units included; gold; terminal flags). Weighted squared loss on Y_H gives

    ψ_f(h, b) = E[φ(x_H) | h, b]

This is the information-state expectation ([Baisero & Amato](https://arxiv.org/abs/2105.11674)). Both sides continue with the fogged π, so no strategy fusion enters; hidden units enter only as labels. No gradient from results reaches ψ. An arm pretrains ψ on the realised futures of the human corpus (about 6.5M decisions, labelled on CPU).

### 2.3 The valuation

g_H(φ) = w_H·φ, with coefficients w_H(cell, side, turn, time of day) produced by a small shared network, about 10⁴ parameters. It is fitted by ridge on boards at H along self-play games:

    min Σ (y − g_H(φ(x_H)))²,    y = z − Σ_{fights k after H} L_k(g_0)

Its minimiser is E[y | φ(x_H)]. Caps count as losses for both sides. g_0, the same valuation at H = 0, is the luck column. A fight changes only two units' cells, so L_k(g_0) is CPU arithmetic.

**Composition.** By linearity, V(h, b) := g_H(ψ_f(h, b)) = E[g_H(φ(x_H)) | h, b] exactly. The one approximation is E[z | x_H] ≈ E[z | φ(x_H)]: the value at H that the board does not carry. Its difference between candidates is what this design gets wrong.

**Embedding cumulants** (an option S−1 decides) append `own1`'s frozen pooled trunk embedding at x_H, mover's view, to φ. On the trunks probed on 2026-08-19, a linear probe came within 0.007–0.025 AUC of the value head (`docs/archive/credit_assignment_design_20260817.md`, A3). So this adds a value read after play and keeps g linear.

**Not the killed critics.** Those had up to 15.6M parameters and an effective sample near the game count. g has about 10⁴ shared coefficients over boards after play and 50,000 games. It is a learned material valuation, the level those critics failed to beat; decision differences come from ψ. If P-val fails, an arm fits g by the current design's identity.

### 2.4 Information

- **In value units, at H1.** Noise per pair is 2·v·k ≈ 2 × 0.025 × 1.5 = 0.075, with no luck correction. At s = 0.010–0.015 that is 1.3–3.0 × 10⁻³ χ² per pair. With about 2 effective pairs per point, 300,000 points give 800–1,800 χ² for $2.6. The current 100,000 windows give 460–1,040 for $3.5, and those are biased by C. This is about 2.3 times per dollar, unbiased. At H2 it is about equal per dollar.
- **Per entity, the larger gain.** The acting unit's survival to H1 differs across candidates by 0.1–0.3, with label variance at most 0.25. That gives 0.02–0.18 χ² per pair, 15 to 60 times the value-unit figure.
- **So** the local consequence map learns from dense labels, and a low-dimensional map, pinned by results, prices it. A flexible critic must learn both through one scalar.

### 2.5 Decode and stop

    A_f(h, b) = g_H(ψ_f(h, b)) − g_H(ψ_f(h, a0))
    π_{k+1}(h) = argmax_{b ∈ B} [ℓ_k(h, b) + λ_{G(h)}·A_f(h, b)]

- G(h) is attack, move or recruit, crossed with turns 1–4, 5–10 and 11+.
- λ_G ≥ 0 comes from the factorial, and λ = 0 reproduces π_k bit-exactly.
- Every action is the network's; every comparison is one decision followed by π's play, re-deciding after every roll.
- Heads stack and are distilled every second round.

**Stop.** Longstaff–Schwartz on forced paths, as in the current design, with the payoff g_H(ψ_f(h_d, end)): ending at depth d, read after the reply, not statically. The three stop groups have four levels: the −1.5 offset, and LS shifted by −δ, 0 and +δ.

### 2.6 The factorial

Each side of each game draws, independently per group, a level from {0, ½, 1, 2}·λ̄_G, from a generator separate from the dice. λ̄_G is set so that level 1 switches about 3 decisions per side-game.

- **Response and SE.** The response y = z − Σ L_k(g_0) is unbiased for any frozen corrector ([AIVAT](https://arxiv.org/abs/1612.06915)). It is regressed on per-side level indicators, side A minus side B. A contrast has variance 4σ²/n: at σ² ≈ 0.80 and n = 40,000, SE 0.0089 in z (3.1 Elo at 347 Elo per unit z), for all 36 contrasts at once, $2.0.
- **Selection.** Levels are selected on one half and estimated on the other (4.4 Elo). A level is adopted if its held-out lower 2-SE bound is above 0. The 66 pairwise interactions are read at about twice that SE.
- **Forecast and check.** The forecast is the adopted main effects plus interactions. It is written before a direct corrected match against π_k (6,400 games, SE 3.9 Elo) and a PURE match against `own1`.
- **Calibration.** Each group's measured-to-predicted slope is read, not gated; the prediction is Σ A_f over its switches.

### 2.7 Invariants (tests from this spec)

The current design's I1 and I15; no result gradient reaches ψ; g(Eφ) = E g(φ) on synthetic mixtures; cell statistics round-trip with the encoder; on an enumerable duel ψ's target equals the enumerated expectation; permuted levels give null contrasts; λ = 0 reproduces π_k; no privileged token reaches ψ_f.

## 3. Regimen

Prices: a game $2.5e-5, or $5e-5 with one consequence head; a window point $8.6e-6, or $1.7e-5 from round 2; PURE 800-decisive with the head $0.21.

| Stage | Cost | Predicted readout | Kill line |
|---|---|---|---|
| **S−1 records** | $0.3 | 1,600 `own1`-era argmax records (`gen_own1_selfplay`, `confirm_kv15`) are rebuilt, with boards at H0, H1 and H2 and g_H cross-fitted. (a) g_0 as the luck column leaves 0.78–0.84 of Var(z) (`value_50`: 0.829 on these games). (b) Adding `own1`'s value head, read at H2, to g_H2 raises the held-out R² of y by at most 0.03 (not derived). | (a) at 0.86 or above: collapse the cells. (b) 0.03–0.10: embedding cumulants. (b) above 0.10: the board misses what a learned value sees; stop before renting. |
| **S0 instruments** | $2 | Numerics at least 98% (2,000 games each); I15 on 1,000 forks. **P-fact**: 40,000 `own1` games with 12 null groups at logged levels, plus +12 gold at a side's turn 3 as a planted factor, its effect measured by a 6,400-game direct match. The null contrasts' χ²₃₆ falls in [21.3, 54.4], and the gold factor within 2 combined SE. | Numerics under 95%: kernels (+$7). P-fact outside: the factorial is broken; nothing runs. |
| **S1 consequences** | $5.3 | 20,000 `own1` games and 300,000 points; the human arm; a $0.2 dice-keying probe (section 6). **P-cons**: B is a fair per-game bit that nothing reveals before turn 6. At turn 6's start, units on even hexes lose 4B HP. On 60,000 turn-5 points, the fogged contrast (same unit, even against odd hex) reads 2.0 ± 0.3 HP for either B, a copy given B 4.0 and 0.0; its SE falls as 1/√n from 15,000 points, within 30%. Held-out corrected r between predicted and realised contrasts: the acting unit's survival at H1, 0.6–0.85; value units, 0.35–0.6, against the record's 0.40 and 0.52–0.53 (not derived). | P-cons outside: the label or fit path is broken. Value-unit r under 0.3: consequences do not carry decisions; hand over to the current S1 (about $8 spent). |
| **S2 valuation, stop** | $2 | g_H is fitted on 50,000 games. **P-val**: in 20,000 games, a fair coin per side grants 30 gold at that side's turn 5. g, fitted on games without windfalls, predicts the coin's measured effect on y within 2 combined SE (measured SE about 0.009). **P-stop**: from turn 10, the first two acts past π's stop cost 15 gold each, the third grants 60; on 20,000 forced-path points LS reaches depth 3 on at least 95% of eligible turns, a greedy fit 0. λ̄_G and the groups' predicted gains are written. | P-val outside: broken or confounded; run the identity arm with the current design's P1 (+$5). P-stop outside: the stop path is broken. |
| **S3 round 1** | $3 | Factorial of 40,000 games; direct match; PURE against `own1` and `parity3`. **Priors:** stop groups +5 to +35 Elo (the current prior). Action groups 0 to +35, central 8: N·E[max(X, 0)] with N = 150, X ~ N(−m, r²s²), m/s from 0.5 to 1, s = 0.01, and r from S1. Slopes 0.3–1.0. The match lands within 2 combined SE of the forecast. | No group's lower bound above 0: report r and stop (about $13). Gain over `parity3` under half that over `own1`: exploitation. |
| **S4 rounds 2–7** | $6.5 each | 10,000 games of π_k, 100,000 points, warm refits, a 30,000-game factorial (3.6 Elo per contrast) and confirmation; P5, then distillation, every second round. Prior: 0.3–0.8 of the previous round's action gain (not derived). | The forecast's upper 2-SE bound under 5 Elo in two consecutive rounds: S5. |
| **S5** | $4 | P5 and the final distillation; a larger ψ if held-out r rises by at least 0.05. | P5 outside: distillation broken. |

**Budget.** About $13 for S−1 to S3, plus $39 for rounds 2–7 and $4 for S5: $56 for seven rounds. The current design reaches round 3 for $65. An S1 rerun, the identity arm and the kernels together fit in $85.

**Against the current design.** At equal per-round priors and decay d, seven rounds give (1−d⁷)/(1−d) of round 1's action gain against (1−d³)/(1−d) for three: 13% more at d = 0.5, 62% more at d = 0.8, for 86% of the price. The larger difference is the cost of failure: the central bet is decided at $8, not $21. That unbiased dense labels raise r is the mechanism's claim, not a derivation.

**The $1,000 case**, argued only after two rounds land within their forecasts: 10 times the points, embedding cumulants, a full-trunk ψ, 30 or more rounds, a league check, forecast by the measured decay. Nothing measured links `own1` to human ratings.

## 4. What is new, and the nearest work

Searched 2026-10-19; finding nothing does not prove absence.

1. **A candidate-conditional expected-board model, trained on exact-simulator forks at decision siblings, as the channel of decision values.** Labels are realised under the policy's own two-sided continuation, critic-free, regressed by a fogged network. Nearest: DFP (on-policy, single agent, given weights); KataGo's ownership head (final board, auxiliary, not action-conditioned); action-conditional predictions ([Oh et al. 2015](https://arxiv.org/abs/1507.08750); [Zheng et al. 2021](https://arxiv.org/abs/2102.04897)); successor features ([Barreto et al.](https://arxiv.org/abs/1606.05312)); GNU Backgammon's rollouts.
2. **A linear valuation fitted on luck-corrected results at the horizon, composed exactly with the expectation model.** Nearest: Wan et al.'s linear condition; the reward regression of successor features; DFP's goals, evolved in [Ellefsen et al.](https://arxiv.org/abs/1904.02435).
3. **Luck-corrected factorial screening of decode-trust groups in self-play, with split selection.** This is weakly new. Nearest: Stockfish's SPSA ([chessprogramming](https://chessprogramming.org/Stockfish%27s_Tuning_Method)) and AIVAT.
4. **The board-sufficiency test (S−1)**, a pre-box check of the factorisation.

Not claimed: Longstaff–Schwartz, the fusion-free regression, corrected matches.

## 5. The weakest point

- **Value at H that the board does not carry.** Tempo, unexecuted threats and placement for later turns are invisible until they become board changes by H2. Such decisions get A_f ≈ 0, and move groups would read null. Embedding cumulants patch this only as far as `own1`'s weak head sees; S−1 bounds it against that head only.
- **The prior's form.** The per-round prior has the current design's form. The gain is rounds per dollar and cheaper failure; a larger r is hoped for, not derived.
- **Staleness.** Consequences go stale once π switches, and compounding under zero-sum play has no theory.
- **Rulings for the user:**
  - g is a result fit, though low-dimensional and after play.
  - The λ_G are set by outcomes: at most 48 numbers per round, like the offset sweep.
  - Both may fall under the killed "critics trained on results" and "outcome policy gradient". The identity arm replaces the first; nothing here replaces the second.

## 6. Decision record

- **Rejected: coins as the source of decision labels.** Forks give unbiased labels at chosen points without coin drift; the factorial measures policy-level effects.
- **Rejected: a privileged critic in the loop.** It is kept only as the arm for g.
- **Rejected: a nonlinear valuation of expected features.** The Jensen gap makes it inexact; an MLP arm is judged only by fit and the factorial.
- **Rejected: the forecast ladder as a gate.** The factorial's held-out half measures the adopted switches directly.
- **Deferred: per-unit keyed dice across forks.** The 2026-08-17 kill measured a collapsed checkpoint rolling 0.55 fights per turn, within one turn (`docs/archive/credit_assignment_design_20260817.md`, Q8); `own1` rolls 2.75 per side-turn. Keying each strike on (unit birth key, side-turn, swing count, role) meets the strike-order objection. S1's probe measures the fork-pair correlation; above 0.5 it halves the noise per pair.
