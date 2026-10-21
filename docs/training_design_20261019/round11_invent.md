# Round 11, invent stance: the exact-law advantage walk under a KL-set policy gradient

Tags: **[M]** measured (record or script named), **[D]** derived (inputs named), **[U]** unknown (measured before anything relies on it, or nothing relies on it), **[C]** a choice whose consequences are derived. My scripts are in `analysis/`, each with its `.out`: `round11_invent_luck_strata.py` (*strata*) and `round11_invent_budget.py` (*budget*, § for its sections).

## Summary

- **The update.** A single-pass policy gradient over `own2`'s self-play at temperature 1, with both seats learning. Each round's step size is set by the KL it moves the policy on held-out positions, not by a learning rate.
- **New: the exact-law advantage walk.** Every random event of a game, whether a sampled action of either side or an attack's dice, becomes an innovation whose expectation is exact under the logged policy or the combat tables. A ridge of the result on the summed innovations identifies every event's advantage with no value level, which is what the project's critics failed to learn. The label I_t + ε_g is unbiased for any walk fitted before the game, and ε_g is the walk's residual, so the games a round needs scale with its held-out residual share r.
- **Measurements first.** No game at temperature 1 is on disk. Two stages that train nothing measure r, the sampling loss and the gradient noise scale for about $2.
- **The optimizer.** Policy-gradient noise has the shape of the policy's Fisher matrix: Adam and SOAP whiten it, while Muon gives every noise direction a full-size step. AdamW applies round 1. The built bench reads a different regime unless it gains one arm.

## 1. The algorithm

### 1.1 The walk (new)

**What is learned.** The policy with its memory and belief head. The value head is out of the loss: a value gradient once owned 94–99% of the trunk's updates (seed §3). The generator π_k is `own2` at round 1: temperature 1, no offset, 64 slots, both seats.

**Events and columns.** A game's events s are every decision of either side, under law π_k over the enumerated legal set, and every attack's outcome, under its exact distribution. Over 87,720 attacks the enumeration left 0 outside support and 0 not enumerable [M, `match_luck_20261019/readout.txt`]. Traits and advancement choices take I = 0: unbiased, their variance left in the residual.

There are about 70 columns [C], each crossed with four turn bins. The bins are fixed before each event, so the innovations keep their zero mean.
- **Dice:** the luck walk's seven columns.
- **Actions:** kind indicators, log π_k by kind, the exact expected material swing of an attack, recruit cost, village capture.

The innovation of event s is I_s = σ_s wᵀ(x_s − E_{p_s}x), where σ_s signs it to side A. The expectation is a sum over the legal set or over the outcome table.

**Fit.** w is a ridge of z − c(cell) on X_g = Σ_s σ_s(x_s − E x_s), one 70-number vector per game. It is fitted on earlier games and frozen before the games it labels, at CPU-seconds' cost and with no network forward.

**Why it finds advantages.** The Doob decomposition gives z = V_0 + Σ_s I_s(Q), with I_s(Q) = Q(h_s, x_s) − E Q(h_s, ·). Innovations are martingale differences, so for any fixed f:

    E[(z − c − Σ_s I_s(f))²] = E[(V_0 − c)²] + Σ_s E[(I_s(Q) − I_s(f))²]

The minimiser is f = Q + (any function of the history), the advantage of every event. No level enters. The project's critics failed at levels: the value head ranks turns at 0.22, and observation critics do no better than material (seed §3). Within-event contrasts, by contrast, are identified by each event's own randomisation.

**The record so far weighs against large gains.**
- In the near-tie band the coin label rises with the prior's logit gap, +0.0157 ± 0.0070 per nat [M, `round9_simplify_labels.out`], where Boltzmann-rational play makes it fall: the log π columns may be empty there. Off the band nothing is measured [U].
- The state-saturated column value_50 explains 0.147 of the result's variance at argmax, material 0.104 [M, `readout.txt`, 18,800 games]; setup-varying coefficients add +0.0007 ± 0.0002 [M, *strata*, 60,000 games, cross-fitted]. Luck's value depends on the state inside games, not on the setup.

### 1.2 The label, the pass and the step

    Ã_t = I_t + ε_g,   ε_g = z − c(cell) − Σ_{all events s} I_s

**Unbiased for any w fixed before the game.** ε_g + I_t = z − c − Σ_{s≠t} I_s:
- earlier innovations are known at t, so they are zero against the score;
- each later innovation has zero mean given its own history, which contains a_t.

The conditions: the centring law is the sampling law (temperature 1, no offset, masks as played), and w and c come from other games. Columns may read the full state and the opponent's history: they are training-only and fixed before each event, so fog does not break them.

**Credit.** A decision's label is the walk's estimate of its own advantage plus the residual left once every other event's explained innovation is removed, about 453 decisions and 110 attacks a game [M, *budget* §1; `readout.txt` 87,720/800]. The residual reaches decision t through its correlation with t's score across games. Only single sampled actions are compared, each followed by play that re-decides after every roll.

**Pass.** One pass of the sequence trainer over the round's game-sides on the surrogate Σ_t (π_θ/π_k)(a_t) Ã_t. That is the own-trajectory loss (`wesnoth_ai/own_loss.py`) with a signed one-hot target, under the same masks, memory unroll and BPTT windows that trained `own1` and `own2`.

**Step.** θ_α = θ_k + α(θ_pass − θ_k), with the largest α in {1, ½, ¼, …} whose mean per-decision KL(π_k‖π_α) on held-out positions is at most κ. Forwards only.

**Forecast.** On G_h held-out games, Σ_t (π_α/π_k − 1)Ã_t is the first-order gain (performance-difference lemma), at 347.4 Elo per unit mean z [D, 800/ln 10]. It is sealed before the match. Its SE is 2.77 Elo at G_h 10,000 and κ 0.001 [D, *budget* §3], and its relative precision does not depend on κ.

### 1.3 Bias and variance

**Bias.** None from the estimator. What remains: truncated BPTT at window edges (as in own training), drift within the pass (the ratio corrects it to first order), and any generation-trainer arithmetic gap (a log-ratio barrier reads it before the first step).

**Variance.** The coin labels' within-game cross terms are absent (ratio 0.999 [M, `round9_simplify_labels.out`]). So the per-game gradient covariance is about r·F_game, the policy's Fisher summed over a game [D]. The noise scale under a preconditioner P is then G* ≈ r·tr(P F_game)/(gᵀPg) games ([McCandlish et al.](https://arxiv.org/abs/1812.06162)). G* is proportional to r, so the walk's held-out r is its price in games:
- at argmax, the dice walk with the shrunk setup cell leaves r = 0.699 [M];
- at temperature 1, r is unknown [U, M1].

### 1.4 Prior, memory, extension

**The human prior** is the starting point, released only by KL-bounded steps that the labels drive. There is no anchor term [C]. An anchor to `own2` at temperature 1 would penalise the sharpening sampling needs: argmax beat sampling by +412 ± 97 Elo in 40 games on the September seed (seed §8). The holdout imitation loss is a crash barrier, never a target.

**The memory** is trained by BPTT on the player's own trajectories, the recorded cure for its causal confusion. The belief loss stays at the recipe's weight, and the memory barrier is read every round.

**The chance head.** If M1 shows the turn-binned dice block beating the pooled block by more than 2 SE, a chance head follows: a detached head emitting w(h) over outcome features φ(o), so that E_o[w(h)·φ(o)] stays exact at one forward. It is trained by the same game-level loss. Nothing relies on it before then.

### 1.5 Signal per dollar

The walk costs nothing at the margin: its columns come from priors and tables the generator already computes, and as a ridge on a superset of the dice walk's columns it loses at most its degrees of freedom, 70/10,000 = 0.7% of r [D]. A2C or GAE need the value level the project never learned, and GAE's bootstrap biases the gradient. Per-decision forks lose to more games when downstream variance dominates ([Nauman & Cygan 2023](https://proceedings.mlr.press/v202/nauman23a.html)), and paired forks share a median of 0 fights (`docs/selfplay_redesign_20260904.md`). A game generated and trained once costs $7.52e-5 (training 1.89× generation); a 100,000-game round $8.34, nine in $81 [D, *budget* §2, §4].

## 2. The optimizer

### 2.1 The candidates under Fisher-shaped noise

- **AdamW:** its second moment estimates r·diag(F)/B, so it acts as a diagonal square-root natural gradient [D].
- **SOAP:** the same, Kronecker-factored.
- **Natural gradient:** steepest per unit KL, and F here is exact and analytic over the legal actions. But Fisher-vector products through the packed flash trunk and the memory's BPTT are new code. ACKTR reports 2–3× sample efficiency ([abstract](https://arxiv.org/abs/1708.05144)).
- **Muon:** sets every singular value of the momentum to 1. Against it: "Muon collapses to zero" in low-SNR GRPO ([2605.19282](https://arxiv.org/abs/2605.19282), abstract), and Adam beat it fine-tuning AdamW-pretrained Qwen2.5-7B on all four benchmarks ([2502.16982](https://arxiv.org/html/2502.16982v1), Table 7); `own2` was trained with AdamW. For it: one single-seed agentic-RL gain ([2607.16169](https://arxiv.org/pdf/2607.16169), unverified).
- **Schedule-free:** its average is a noise average. No RL evidence found [U].
- **PPO:** epochs refit one batch's noise, and its gains come from code-level details, not clipping ([Engstrom et al.](https://arxiv.org/abs/2005.12729)).

**The [full-batch post](https://www.lesswrong.com/posts/9gqjdRyJxSM6kS7zi/i-trained-gpt-2-with-full-batch-gradient-descent-so-you-don)** (one run per setting, by its author's caveat) supports three choices:
- Without noise, a fixed rate rides the stability edge: consecutive gradient cosine −0.945, the loss rising on a third of steps, the top eigenvalue going from 48 to 113. The KL line search guards against this.
- A reused set generalises worse: fixed-set validation 3.411 at 6.5 epochs against 3.228 on fresh data. Hence one pass per round.
- Large fresh batches save steps, not data. Our cost is data, so G_r sits near G*.

### 2.2 Recommendation

- **AdamW for round 1** [C]: it trained `own2`, the fine-tuning evidence favours keeping it, and its normalisation approximates the right metric.
- **Fresh second moments:** a warm-up, since imitation's moments are not the PG's.
- **Step size:** κ by line search. The learning rate only has to make the chord long enough to search.
- **Batch:** G_r ≥ G* from M2.

### 2.3 What the bench settles

**As built** (16 imitation passes from scratch, 1M positions), it reads high-SNR pretraining efficiency, speed and memory. It does not read fine-tuning from `own2` or low-SNR behaviour.

Its pre-registered criterion [C]: an arm displaces AdamW iff both hold:
- its final 64-slot holdout loss, at a rate that is not a grid edge, beats `adamw_mid` by more than 2 paired SE and more than 2·|`adamw_mid2` − `adamw_mid`|;
- its wall is at most 1.2× AdamW's.

**The missing arm** [C], bandit-converted imitation from `own2`: on corpus positions the trainer samples a ~ π_θ, labelled 1[a = human action] − π_θ(human action) plus Gaussian noise scaled to M2's B_simple. The objective (Σπ(human action) on holdout) and its exact gradient are known. Each optimizer, under KL-set steps, is read by gain per unit KL at equal positions against an AdamW repeat; the first step's cosine with the exact gradient checks the score path. Six arms, $0.84 [D, $0.14 an arm from the bench script].

## 3. The regimen

| Stage | Cost | Readout and prediction | Decision |
|---|---|---|---|
| **R0** (laptop, CI) | $0 | Logging, fit, labels. I-centre: every centred column's mean is 0 within 3 SE (exact by construction). I-duel: in an enumerable duel through the production generator, the label's mean difference between two first actions equals their exact Q difference; a swap flips the sign | Red: fix before any box |
| **M1** (trains nothing) | $1.09 at argmax prices [D, *budget* §2]: 20,000 `own2` games at temperature 1 ($0.52); 6,400 decisive `own2` at temperature 1 against `own1` at its reference decode ($0.57). Game length at temperature 1 [U] scales it | D_0, no prediction. r per block, cross-fitted halves. Offline plant z′ = z + δΣσ_s(1[a_s∈C] − π(C\|h_s)): the C column recovers slope 1 within 3 SE. Census by kind | Sets w, c, r. If the turn-binned dice block wins by more than 2 SE, build the chance head. If the action block adds less than 2 SE, drop it |
| **M2** (gradients, no step) | $0.98: 9.07M positions at 1,477 a second [D] | B_simple per game, Euclidean and Adam metric. Sign-flip null: \|G\|² is 0 within 2 SE. Planted gradient against the exact δΣ_t∇π(C\|h_t): ratio 1 within 3 SE. Sampled against all-action estimator. PG positions a second. Log-ratio barrier | G_r and the rounds that fit (*budget* §4). If G* is above 10⁶ games, no round fits: report |
| **Discussion** | $0 | §7 | Optimizer, bench, anchor |
| **Bench arm** (optional) | $0.84 | Gain per KL by optimizer, against the repeat | Switch only on a 2-SE win |
| **P**: planted round | $1.04: M1's games trained once ($0.98) + 2,000 fresh games ($0.05) [D] | Labels are the plant alone. Sealed from M2's statistics at κ: the change of Σ_tπ(C\|h_t) on held-out positions and C's count per side-game in fresh games, each within 3 SE of first order | **Inside:** pass, step and generation certified at κ. **Below:** halve κ. **Null:** debug |
| **R1** | G_r·$7.52e-5 + $0.26 held-out + $0.57 match | Sealed forecast; `own2`′ against `own2`, both at temperature 1, 6,400 decisive (SE 4.3); the claim against `own1`. Miss line 2√(SE_f² + 4.3²) = 10.3 Elo at SE_f 2.77 [D] | **Inside:** continue at κ. **Below:** halve κ. **Null with a positive forecast:** labels do not transfer; debug the trainer against generation. **Claim lower tail:** stop, the user decides |
| **Rounds 2+** | as R1 | w refitted on the previous round's games | as R1 |

## 4. Provenance

| Number | Value | Tag and source |
|---|---|---|
| Decisions per game, argmax with coins | 453.3 | [M] *budget* §1: server requests over results, b00–b06 |
| Attacks per game, `own1` self-play | 109.7 | [M] `readout.txt`, gen_own1_selfplay, 87,720/800 |
| Enumeration coverage | 0 outside support, 0 not enumerable | [M] same |
| r, dice walk + shrunk cell, argmax | 0.699 | [M] `round9_simplify_residual.out`; *strata* reproduces 0.6988 |
| Dice columns alone, argmax pooled | material 0.896, value_50 0.853, joint 0.848 | [M] `readout.txt` pooled, 18,800 games |
| Setup-varying luck coefficients | +0.0007 ± 0.0002 of variance | [M] *strata* (every penalty at the grid's top) |
| Label slope on prior gap, in band | +0.0157 ± 0.0070 per nat | [M] `round9_simplify_labels.out` |
| Within-game label cross terms | ratio 0.999 | [M] same |
| Generation | 22,209 games an hour, $2.598e-5 a game | [M] *budget* §2, b01–b05 walls at $0.577 an hour |
| Training | 1,476.9 positions a second (imitation loss), $4.92e-5 a game | [M] `trainer_calibration_20261015/status.txt`; [D] *budget* §2 |
| PG-loss training rate | — | [U] M2; prices scale by it |
| Two-network decisive game | $8.848e-5 | [M] `own2_base_20261020/match.walls` |
| Elo per unit mean z | 347.4 | [D] 800/ln 10 |
| Match SE | 4.3 at 6,400 games | [D] 347.4/√N |
| Forecast SE | 2.77 at G_h 10,000, κ 0.001 | [D] *budget* §3, r 0.699, 453 decisions |
| Round price | $8.34 at 100,000 games | [D] *budget* §4 |
| Budget left | $81.4 | [D] `current_design.md` §4 $82.34, less match L's about $0.9 (BACKLOG) |
| Argmax over sampling | +412 ± 97 Elo, 40 games | [M] seed §8; another network, a report only |
| Fit's degrees-of-freedom cost | 0.7% of r | [D] 70 columns / 10,000 games |
| r at temperature 1, D_0, G*, game length at temperature 1 | — | [U] M1, M2, before any training |
| κ | — | [C] set at P from M2 |
| Bench arm | $0.14 | [D] `optimizer_bench_box.sh` header |

## 5. What is new, and the nearest work

Searched on 2026-10-21. Finding nothing does not prove absence.

**Nearest prior work.**
- [Direct Advantage Estimation](https://arxiv.org/abs/2109.06093) regresses the return on summed π-centred advantages.
- [Off-policy DAE](https://ar5iv.labs.arxiv.org/html/2402.12874) adds a "luck" term for the environment's randomness. That term's centring needs the transition law, which it approximates with a CVAE ("holds only approximately"). It ran 15 hours against 2 for the other methods, on MinAtar.

**New here:**
1. The centring is exact for both seats' actions and every roll, through the combat tables and the logged policy.
2. Linear columns turn the fit into a closed-form ridge from per-game sums computed during generation and frozen per round.
3. The label I_t + ε_g is unbiased for any walk, where DAE plugs in Â.
4. ε_g is the walk's own residual, so its held-out loss is the gradient's noise and its price in games.

**Neighbours:** [AIVAT](https://ar5iv.arxiv.org/html/1612.06915) and [MIVAT](https://www.ijcai.org/Abstract/09/326) (evaluation only); [TrajCV](https://ar5iv.org/html/1908.03263) (action-only control variates, TD-learned Q); [MRDR](https://proceedings.mlr.press/v80/farajtabar18a.html) (variance-fitted model, off-policy evaluation); [VR-MCCFR](https://arxiv.org/pdf/1809.03057) (chance and opponent baselines in regret minimisation). The bandit arm is a design, not a method claim.

## 6. The weakest point

The walk may explain little at temperature 1, and then the critical batch decides everything. Both of this round's measurements lean that way:
- the logit gap is empty in the band;
- luck coefficients do not vary across setups.

If the variance sits in chaotic continuations, r stays near the dice walk's level. If G* is also around 10⁶ games, one resolved round costs about $76 and the plan reduces to M1, M2 and a report. Those two stages cost about $2 and decide this first.

## 7. Questions for the optimizer discussion

1. **The update.** Single pass with KL-set steps (proposed); PPO epochs (reuse fits noise); TRPO or K-FAC (2–3× in ACKTR's abstract, new second-order code).
2. **The optimizer.** AdamW (proposed: it trained `own2`); SOAP (untested here); schedule-free (no RL evidence); Muon (a published low-SNR collapse).
3. **The bench.** As built with §2.3's criterion; plus the bandit arm (proposed, $0.84); neither.
4. **M1 and M2** (about $2, no training): approve?
5. **The prior.** No anchor (proposed). A KL anchor to `own2`: piKL reports gains in Diplomacy (recalled), but the anchor fights sharpening. A human-corpus mix.
6. **The yardstick.** `own1` at its pinned reference decode (proposed: D_0 is the debt to repay), or `own1` at temperature 1.
7. **The opponent.** The current self (proposed), or a league against cycling.
8. **The chance head.** On M1's evidence (proposed), or never.
9. **The price.** If M2 puts one round above about $30: the $1,000 case, or stop.
