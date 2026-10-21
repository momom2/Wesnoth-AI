# Round 11, challenge: certified steps

**The stance.** At temperature 1 every decision is a randomized experiment with a logged propensity, so a direction comes from every decision at no extra cost. What nothing on record fixes is the step: no policy-gradient run on a current network was ever measured, and on the one structure measured, a step's own second-order loss was two thirds of its first-order gain. This proposal learns both from game outcomes: the direction from each decision's luck-corrected result, the step from the next round's games, in which the new policy plays its predecessor and which are also training data. Forks, critics, epochs and a fixed step each buy less per dollar (§1.6).

Tags: **[M]** measured (record or script), **[D]** derived (inputs named), **[U]** unknown (measured before anything uses it), **[C]** a choice whose consequences are derived. *forks* = `analysis/round11_challenge_forks.py`, *budget* = `analysis/round11_challenge_budget.py`, each with its `.out`.

## 1. The algorithm

### 1.1 What is learned, from which labels

π_k plays both sides at temperature 1, 64 slots, on the pool. Per decision the player logs log π_k(a_t|h_t) of the sampled action, the entropy H_t of the distribution it sampled from, and the value head's reading v̂_t from the same forward (the head is not trained, so its quality can only decay; α is refitted each round). Decision t of side s in game g carries the weight

    w_t = y_g − α·v̂_t,    y_g = s·(z_g − b(cell) − βᵀS_g − κ·U_g)

- z ∈ {−1, 0, +1}; b(cell) the frozen setup baseline; β the luck walk's coefficients, refitted on temperature-1 games.
- U_g = Σ over the opponent's decisions of (log π(a|h) + H) under the policy that sampled them: the opponent's "surprisal luck", each term of zero mean given its own history.
- α, β, κ are cross-fitted on other games, so no term raises the label's variance in expectation; each enters only if stage 1 measures it lowering E[w²] by 2 SE.

**The update** maximizes Σ_t w_t·log π_θ(a_t|h_t) over the round's fresh games, in one pass, subject to a per-decision KL radius ε_k from π_k, read on held-out positions with (r − 1) − log r, unbiased for KL(π_k‖π_θ) when a ~ π_k ([Schulman 2020](http://joschu.net/blog/kl-approx.html)). Every head, the trunk and the memory train; the value head does not (the AlphaZero-style loop's value gradient owned 94–99% of trunk updates [M, seed §3]).

### 1.2 Credit, bias, variance

- **Credit.** Each decision receives its side's corrected result; averaging over games through shared weights carries credit to the decisions that earned it, since E[w_t∇log π(a_t|h_t)] = E[A(h_t,a_t)∇log π(a_t|h_t)] for a baseline depending only on h_t and luck terms of zero mean given the history they follow [D].
- **Bias.** Zero at π_k [D]. Truncated back-propagation through time (the recipe's 16-decision window) drops the memory write's gradient past the window, as imitation does.
- **Variance.** Per-decision noise E[w²]. Nearest record, human mid-game positions with `obs8`'s temperature-0.5 playouts: a playout's result variance 0.928; luck-corrected 0.854; with the value-head baseline too, 0.725 [M, *forks*, transferred]. At temperature 1, [U]: stage 1.

### 1.3 The step: forecast, certification, radius

Round k:
1. **Generate**, in order: N_c = 10,000 cross-play games π_k against π_{k−1}, seat by coin [C]; N_t self-play games of π_k; N_h = 10,000 held-out self-play games [C].
2. **Read** R_{k−1}, the new seat's mean corrected result in Elo: SE 3.47 at sd(y) ≤ 1 [D, *budget*].
3. **Radius.** ρ = R_{k−1}/Ĝ_{k−1}; ε_k = ε_{k−1}/(2(1 − ρ))², capped ×4 or ÷4 a round [C], moved only when Ĝ_{k−1} > 2 SE.
4. **Revert** if R_{k−1} < −2 SE: retrain from π_{k−1} on its round's games (on disk) at ε/4. A zero step is falsely reverted 0.023 of the time [D].
5. **Train** π_{k+1} on the N_t games and the new seat's cross-play decisions, stopping at held-out KL ε_k.
6. **Seal** Ĝ_k = mean over held-out game-sides of Σ_t w_t(r_t − 1), r_t = π_{k+1}/π_k at the played action: the first-order performance difference ([Kakade & Langford 2002](https://mlanthology.org/icml/2002/kakade2002icml-approximately/)) of one seat switching to π_{k+1} against π_k, the next R's estimand. Since E_{a~π_k}[r − 1|h] = 0, every baseline and luck term drops out and Ĝ is unbiased [D]. SE 2.40 Elo at ε 0.002, 5.36 at 0.01, 10,000 held out [D, *budget*].

**Why ρ = ½.** With first-order gain a√ε and second-order loss Cε (a quadratic in the step), the best radius ε* = (a/2C)² gives ρ = ½, and at ε = kε* a round keeps 2√k − k of the best gain [D, *budget*]:

| k | 1/16 | 1/4 | 1 | 2 | 4 | 9 |
|---|---|---|---|---|---|---|
| Share of the best gain | 0.44 | 0.75 | 1 | 0.83 | 0 | −3 |

Overshooting is the expensive side, so ε_0 = 0.002 nats a decision [C]; from k = 1/16 the rule reaches k ≈ 1 in two rounds [D]. The step matters here: recruit rule R gained +18.0 Elo at first order and lost −12.3 ± 4.5 to its own second-order term at about 2.2 switches a side-game [M, `current_design.md` §2], ρ ≈ 0.32, k ≈ 1.85 by the model. Under noise the rule seldom errs on the expensive side: it grows the radius at a true k = 4 in at most 0.015 of rounds, and shrinks it at k = ¼ in 0.05–0.22, for forecasts of 10–20 Elo [D, *budget* Monte Carlo].

**Claims against `own1`** every third round (6,400 games, SE 4.3–7.2 Elo by win rate [D]) are predicted by the chain Σ R. A miss more than 2 SE below means head-to-head gains do not transfer (intransitivity); the chain stops for the user.

### 1.4 The human prior

Kept as the base measure, released by evidence: with KL to the previous iterate, π_K ∝ `own2`·exp(Σ_k δ_k), each δ_k bounded by its radius; where outcomes say nothing, the prior stands. No fixed anchor by default: the sampling loss sits in the prior's mistake mass, which an anchor of strength λ would keep (π ∝ `own2`·exp(Q/λ)). The risk: self-play mirror descent without an anchor can cycle in a two-player zero-sum game, while magnetic mirror descent (a KL pull toward a reference) converges linearly to the regularized equilibrium ([Sokota et al. 2023](https://arxiv.org/abs/2206.05825)). The claims' transitivity check is the alarm, the magnet the remedy (Q3). The human holdout cross-entropy is reported each round.

### 1.5 The memory

Trained by the same loss through each 16-decision window, on histories the policy produced: the remedy the record validated with another target (own-game training cut the memory's cost from 105 to about 22 Elo, then to nothing inside `own2` [M, CLAUDE.md status]). On-policy, past actions are the agent's own interventions, so history-conditioned imitation's self-delusion does not arise ([Ortega et al. 2021](https://arxiv.org/abs/2110.10819)). The belief loss stays at the recipe's weight, its gradient share reported; the crash barrier stays.

### 1.6 Why more strength per dollar

- **Prices.** Training a game's decisions costs 1.99× generating it [D, *budget*]: games are cheap, trained positions and wrong steps are not.
- **Forks** (exact Monte Carlo baselines, as in [VinePPO](https://arxiv.org/abs/2410.01679)). Over the free value-head baseline a K = 8 fork buys 1.19× (95% 1.13–1.25) on post-fork decisions, the bulk of a tree, and 2.81× (2.58–3.08) on the decisions just before the fork [M, *forks*, transferred], for a server-side memory copy, tree records and tree labels. **Rejected.**
- **A bootstrapped critic.** The record's critics rank turns at 0.22–0.27 against playout truth [M, seed §3]; bootstrapping imports that error into every label. The value head serves only as a baseline, which cannot bias.
- **Epochs.** A second pass costs 0.58 of a fresh game [D: 5.16/(2.60 + 1.09 + 5.16)] and brings no new labels; in the user's GPT-2 post, full batch over a fixed set reached 3.521 validation loss against 3.248 for minibatches [M, external; repetition and noise confounded]. One pass; "two epochs on half the games" can be a bench arm.
- **A fixed step** is off by an unknown factor (§1.3); certification costs 5–16% of a round [D, *budget*].

## 2. The optimizer

The trust region is the outer loop (KL radius, ratio test). The inner optimizer's job: the most held-out first-order gain per unit KL per position.

**Mechanism.** The label w is nearly independent of the score, so the gradient's noise covariance is ≈ E[w²]·F, with F the Fisher [D, stated assumption]. Adam's second moment then estimates a scaled diagonal Fisher, and SOAP's and Shampoo's factors its Kronecker factors ([Morwani et al. 2025](https://arxiv.org/abs/2406.17748) relate Shampoo to Gauss–Newton and the gradient covariance), unlike supervised training, where the empirical Fisher misleads ([Kunstner et al. 2019](https://arxiv.org/abs/1905.12558)). Both apply roughly F^−½, the natural gradient F^−1. Muon orthogonalizes the gradient, equalizing singular directions that are mostly noise when the gradient is: a prior against it, contradicted by one study ([agentic RL](https://huggingface.co/papers/2607.16169): GiGPO 0.290 → 0.546 validation success, single seeds).

**Natural gradient and trust regions.** ACKTR's K-FAC natural gradient reported 2–3× sample efficiency over earlier on-policy actor-critics ([Wu et al. 2017](https://arxiv.org/abs/1708.05144)); TRPO's conjugate gradient needs Fisher-vector products through the memory's recurrence (not built). Surrogates track the true objective poorly at typical sample sizes ([Ilyas et al. 2020](https://arxiv.org/abs/1811.02553)): games, not the surrogate, certify the step.

**Recommendation.** AdamW with the bench's hygiene by default; SOAP the candidate to beat.

**Step size and batch.** The radius by the rule. The learning rate set so one pass ends at the radius, rescaled each round by the share of positions used [C]. Minibatch 512 positions (32 streams × 16) [C]. Below the critical batch, the batch changes steps, not samples ([McCandlish et al. 2018](https://arxiv.org/abs/1812.06162)); in the user's post, fresh data at 50M tokens a step matched 393K a step (3.230 against 3.248) after 19× the tokens [M external; D]. Large batches buy wall-clock, not dollars. This update's noise scale is [U]: stage 3 reads it from per-step gradient norms at two batch sizes.

**What the built bench can and cannot settle.** It settles the optimizer for supervised cross-entropy on human data, not this update: here one result covers 238 decisions, and an optimizer that wins at high signal-to-noise can lose at low. This update needs: each arm trains the same recorded temperature-1 games to the same held-out KL; read Ĝ at matched KL, paired across arms, with positions to the radius; adopt an arm over AdamW-hygiene only by 2 paired SE. Six arms on 5,000 games, each read on 5,000 held out: $3.21 [D, *budget*]. The bench's arms carry over; the loss and the readout are added.

## 3. Regimen

Generation is priced as argmax pool games; every cost scales with the temperature-1 game length [U].

| Stage | Cost [D, *budget*] | Readout and prediction | Decision per outcome |
|---|---|---|---|
| **0. CI, offline** | $0 | On an enumerable two-decision game the production code's expected update equals the analytic gradient, Ĝ the first-order gain, k3 the KL; a seat swap flips R | Red: fix before any box |
| **1. Measure, no training** | $1.09 | 20,000 `own2` temperature-1 self-play games with logs; a 6,400-game claim `own2`@T1 against `own1`@reference (two networks, 120 in flight). Reads the gap to claw back, decisions a game, cap rate, price per game (two-network pool price included), E[w²] under each correction, v̂'s R² by turn. No value predicted; bound E[y²] ≤ 1 [D] | Each correction kept iff it cuts E[w²] by 2 SE; N_h, N_c resized from E[w²]; the cross-play price sets N_c's share |
| **2. Planted rounds** (after the discussion) | $1.51 | Train on 10,000 stage-1 games with z′ = z + δΣ_t(1[a_t ∈ X] − π_0(X\|h_t)), X = end_turn, δ sized for 10 forecast SEs. Ĝ's planted part equals separately computed δ·Σ(π_P(X\|h) − π_0(X\|h)) to float tolerance; 5,000 cross-play games read it within 2 SE; P(X) rises. Memory plant: pay an action only when the side's previous side-turn length (unobserved) has a given parity; 64 slots beat 0 by 2 SE. Revert check: the same games re-scored with the plant's sign flipped must revert | A miss names its path: sign (planted reading negative), log join (identity fails), seat labels (reading near 0), memory credit (64 = 0) |
| **3. Matched-KL bench** | $3.21 | Six arms on stage-1 games to KL ε_0: held-out Ĝ paired; positions to the radius; B_noise. No ranking predicted | Best arm iff it beats AdamW-hygiene by 2 paired SE, else AdamW-hygiene; batch from B_noise |
| **4. Rounds** | $3.54–3.91 at N_t 20,000; $5.31–5.68 at 40,000 | Each round seals Ĝ before reading R; a claim every third round, predicted by Σ R | The ratio rule; revert on R < −2 SE; stop after two reverts in a row (no direction certifies), or when a claim misses its chain by 2 SE (user: magnet λ) |

**Budget.** Stages 1–3 $5.80 [D]. About $70 then buys 11–12 rounds at N_t 40,000 with a claim every third, or about 16 at 20,000 [D, *budget*]; start at 20,000 while the radius is found [C]. The $1,000 case is argued only from measured R per dollar.

## 4. Provenance

| Number | Tag | Source or derivation | What rests on it |
|---|---|---|---|
| $2.598e-5 a pool game | M | `round9_simplify_labels.out` [prices] | All costs |
| $8.848e-5 a two-network game (20 connections, mean batch 5.54) | M | same; `own2_base_20261020/own2_vs_own1.server_0.json` | Upper side of certification and claims |
| $5.196e-5 a cross-play game in the pool | D | Twice the pool price if a forward's cost is flat in its batch (BACKLOG, generation round 2, qualitative); stage 1 measures it | Lower side of certification |
| 1,477 positions a second | M, transferred | BACKLOG; `trainer_calibration_20261015` (imitation windows) | Training cost |
| $1.085e-7 a trained position; 1.99× | D | 0.577/3600/1477; × 475.7 / 2.598e-5 | §1.6, rounds |
| 7,003 positions a second, encoding | M | `memory_own_round2_20261011/candidates.log` | $1.09e-5 a game (upper side) |
| 475.7 decisions a game, 237.9 a side | M, argmax | `round11_provenance_census.out` | Costs, SEs; T = 1 unknown |
| 347.4 Elo per unit z | D | 1/(2 ln10/1600) | Every Elo SE |
| 0.928, 0.608, 0.320; luck-corrected 0.854, 0.534 | M, transferred | *forks* (`bench_truth.npz`, cross-fitted) | §1.2 report; fork rejection |
| Value head explains 0.40 (0.29–0.51) of var(V); label 0.725 | M, transferred | *forks* | Baseline kept if stage 1 confirms |
| Fork K = 8: 1.19× (1.13–1.25) post; 2.81× (2.58–3.08) pre | M, transferred | *forks*, 300 bootstraps | Forks rejected |
| R: +18.0 first order, −12.3 ± 4.5 second | M | `current_design.md` §2 | The case for certifying the step |
| 2√k − k; ρ = ½ at the optimum; rule ε/(2(1−ρ))² | D | Quadratic step model | Radius rule |
| Wrong-way rates 0.000–0.015 (k 4), 0.001–0.22 (k ¼) | D | *budget* Monte Carlo at the SEs below | The rule is usable at forecasts ≥ 10–20 Elo |
| Certification SE 3.47 at 10,000 | D | 347.4/√N, sd(y) ≤ 1 | N_c |
| Forecast SE 2.40 (ε 0.002), 5.36 (ε 0.01) at 10,000 held out | D | 347.4·√(237.9·2ε)/√(2N_h); (r−1) uncorrelated across decisions, E(r−1)² ≈ 2ε | N_h; when the rule moves |
| False revert 0.023 | D | One-sided 2 SE | Revert line |
| ε_0 0.002; ×4 cap; N_c, N_h 10,000; minibatch 512 | C | Consequences above | Stage 4 |
| Stage costs $1.09, $1.51, $3.21; round $3.54–5.68 | D | *budget* [stages], [rounds] | Regimen, budget |
| Budget about $81 | M | `current_design.md` §4 ($82.34) less match L (BACKLOG) | Rounds affordable |
| +412 ± 97 argmax over T = 1 | M, transferred | seed §8 (September seed, 40 games) | Report only |
| `own2` +22.7 ± 4.4 over `own1`; first mover +9.3 ± 1.4 | M | seed §7; `current_design.md` | π_0; the seat coin |
| GPT-2 post: 3.248, 3.521, 3.230; 19× tokens | M, external; D | Post read 2026-10-21; 50M×350 / 393K×2,372 | Epochs; batch |
| ACKTR 2–3×; Muon 0.290 → 0.546; VinePPO ≤ 3.0× | External, abstract level | Linked | Priors only |
| E[w²] at T = 1; game length; cap rate; κ; the T = 1 gap; two-network pool price | U | Stage 1 | N_h, N_c, corrections |
| B_noise; optimizer ranking | U | Stage 3 | Inner optimizer, batch |
| a, C, ε*; second-order loss | U | Rounds' ratios | The radius (measured online) |
| Gain per round, Elo per dollar | U | Rounds | Nothing until measured; then the $1,000 case |

## 5. What is new, and the nearest work

Searched 2026-10-21; not finding something does not prove it absent.
1. **A trust region whose actual-improvement reading comes free from cross-play training games against the predecessor**, targeting ρ = ½ (the quadratic optimum) and reverting on a confident loss. Nearest: a stochastic policy trust region whose ratio adds the reward's sample standard deviation ([Zhao, Li & Wen 2022](https://arxiv.org/abs/1911.11640), abstract-level); TRPO's same-batch line search; PPO's adaptive KL; HCPI ([Thomas et al. 2015](https://proceedings.mlr.press/v37/thomas15.html)); AlphaGo Zero's 55% gate; [PBT](https://arxiv.org/abs/1711.09846).
2. **A matched-KL optimizer criterion for a policy-gradient update.** None found; the natural gradient is defined by this geometry ([Kakade 2001](https://sham.seas.harvard.edu/sites/projects.iq.harvard.edu/files/kakade/files/natural-2002.pdf)).
3. **The surprisal control variate**: AIVAT's known-strategy correction ([Burch et al. 2018](https://arxiv.org/abs/1612.06915), abstract-level) with κ·log π as the action-value proxy, whose expectation is exactly −κH. No prior use found.
4. **Derived, not claimed new:** label noise makes the policy-gradient second moment proportional to the Fisher, so Adam and SOAP statistics estimate it, unlike the supervised empirical Fisher.

## 6. The weakest point

The proposal spends its novelty on the step, not on the signal: information per game is that of plain policy gradient with free baselines. If the natural-gradient signal is small, every round is small however well its size is chosen, and the rule degrades exactly then (at forecasts near 10 Elo it shrinks a too-small radius in up to 0.22 of rounds [D], costing rounds, not losses). And R measures π_{k+1} against π_k only: exploiting the predecessor reads as progress until a claim against `own1` misses its chain prediction, up to three rounds later (about $11–17 [D]).

## 7. Questions for the user (optimizer discussion)

- **Q1. Inner optimizer.** (a) AdamW-hygiene now; (b) the matched-KL bench first ($3.21); (c) the built imitation bench ($2.50), which settles only supervised cross-entropy. Evidence: ACKTR's 2–3×; Muon in one agentic-RL study (single seeds); the Fisher-shaped noise (§2).
- **Q2. Step control.** (a) The certified trust region (+5–16% a round); (b) a fixed KL target set a priori, which nothing measured on this network fixes, and a 4× overshoot wastes the round; (c) a surrogate-only line search, which Ilyas et al. found tracks true gains poorly.
- **Q3. The human prior.** (a) Released, mirror descent only; (b) a magnet toward `own2` at λ from the start: convergence (MMD) but a capped clawback; (c) (a), switching to (b) when a claim misses its chain.
- **Q4. The memory's dense signal.** (a) Keep the belief loss at the recipe's weight; (b) drop it: the policy gradient owns the trunk, the crash barrier loses its signal.
- **Q5. Round size.** 20,000 trained games ($3.5–3.9) or 40,000 ($5.3–5.7). In the noise-dominated regime, gain per dollar at the best radius does not depend on round size [D]; larger rounds amortize claims and take larger steps, which ρ tests.
- **Q6. Stage 1's box** ($1.09, no training): approve it before the discussion ends? It measures the unknowns every option depends on.
