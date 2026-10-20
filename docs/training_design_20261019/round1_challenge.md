# Round 1 challenge: learn the evaluator from the dice

Stance: replace playouts as the source of labels with a privileged critic whose training loss has no noise floor, and improve the policy by exact one-step look-ahead with that critic. Playouts remain only as ground truth for the planted checks.

## 0. The claim

The seed buys one label of noise variance about 1.8 with two playouts, to measure a choice difference of a few hundredths. An argmax self-play game already holds about 150 randomized experiments with known probabilities: 105 fights and 45 trait rolls per game (match a1_eo25 in `training/metrics/match_luck_20261019/readout.txt`: 84,566 attacks and 36,351 recruits in 804 games). At argmax both players are deterministic. So with the true value function, the result minus the luck of every chance event equals the starting value exactly. I make that identity the critic's loss. Its floor is zero, its held-out value is the critic's error with nothing to subtract, and every chance event trains the local value differences that a one-step comparison needs. I estimate this at about three orders of magnitude more information per dollar than a fork label. The trained critic then prices every candidate at every decision for about $1e-6 a decision, with no playouts.

## 1. The algorithm

### 1.1 What is learned

- **Critic C** (training only). Its input is the full board, hidden units included, plus both sides' 64-slot memories. Its output is the probability of three outcomes: A wins, B wins, capped. A side's value is v = P(win) − P(loss) − P(cap). A cap scores as a loss for both sides, so stalling into the cap never pays; the PURE fit reads decisive games only and would otherwise reward it. The critic is a separate network initialized from `own1`'s trunk, and none of its gradient reaches the actor.
- **Actor**: `own1`'s network with its memory, trained only toward the improvement targets of 1.4. Its trunk gets no value loss, so the record's 94-99% value share of the trunk's update cannot recur.

Why both memories: a side writes its memory from the trunk at each decision, before acting (`wesnoth_ai/memory.py`). So (board, memory A, memory B) is the Markov state of argmax self-play. A critic that reads only the board averages over memories and keeps a floor. A critic that reads the history, here through the policies' own sufficient statistic, is unbiased (Baisero and Amato 2022, theorems 4.2 and 5.1).

### 1.2 The zero-floor loss

Index by k the chance events after decision t (fights and trait rolls). Each has an exact outcome law p_k: the core's enumeration for fights, and for trait rolls an enumeration the core must add from the WML trait rule. For any function U of post-chance states that was fixed before the game was generated:

```
L_k(U) = U(x_k^realized) − Σ_o p_k(o) U(x_k^o)          E[L_k(U) | history] = 0
Y_t(U) = z − Σ_{k ≥ t} L_k(U)                            E[Y_t(U) | x_t] = V^π(x_t)
```

So Y_t is unbiased for any U (AIVAT's Lemma 1). The policy is deterministic, and V^π does not change across a deterministic step, so z − V^π(x_t) = Σ_k L_k(V^π) exactly: **Y_t(V^π) = V^π(x_t), with zero variance.** For a critic C, write e = V^π − C and R_t = Y_t(C) − C(x_t). Then R_t = e(x_t) + Σ_k [e(x_k^realized) − E_o e(x_k^o)], a sum of orthogonal martingale differences, and

```
E‖R_t‖² = ‖e(x_t)‖² + Σ_{k ≥ t} E[ Var_{o~p_k} e(x_k^o) ]
```

What follows from this:
- **The floor is 0**, and only V^π reaches it on the visited states and their outcomes. The **zero-floor ratio** is the held-out mean of ‖R_0‖² divided by Var(z). It measures the critic without a benchmark and without playouts. Plain regression on results has the floor Var(z | x) ≈ 0.9, which hides any critic difference smaller than its own noise.
- **The second term is the critic's error on the difference between the outcomes of one fight.** The loss trains exactly the local differences a one-step comparison uses.
- The outcome expectation is computed, not sampled. So a single trajectory gives an unbiased loss and an unbiased gradient, and the double-sampling bias of residual-gradient methods (Baird 1995) does not arise.
- **Floor caveat:** randomness that is not chance adds a floor. The record has `parity3`'s replayed choices agreeing with the recorded ones at 98.2% (docs/selfplay_program_20261008.md), which points to nondeterministic arithmetic. Stage 0 measures it.

**Training:** C(x_t) regresses on Y_t(U), where U is a frozen past critic (semi-gradient).
- **Invariant:** every correction function is fixed before the games it corrects were generated. Generation runs in chunks, and chunk j is corrected by the critic as it stood at the end of chunk j−1. A critic that had trained on a game would explain away that game's result through its corrections.
- Each refresh lowers the targets' noise to the previous critic's error, and the population minimizer stays V^π at every refresh.
- A fight is computed exactly over its outcomes of probability at least 2%, with one conditional draw standing for the tail's mass. That stays unbiased and costs about 5 forwards a fight; the average support size is my estimate, not measured.

### 1.3 Information per dollar

Information here is signal² / (noise variance × cost).
- **The seed's paired fork label:** a choice difference of 0.02, noise 1.8, cost $2.5e-5 (two playouts at the seed's $1.2e-5). That is **9 per dollar**.
- **One chance event:**
  - Its consequence has an RMS of about 0.05; the material-scaled column `value_50` reads 0.049 RMS per fight on the record, and the true consequence is unmeasured.
  - Its noise is the rest of the game's residual, about 0.55 × 0.9 × 0.5 ≈ 0.25 at the predicted ratio.
  - It costs about $4e-7: a 150th of a game's generation, five critic forwards, and its share of training.
  - That is **about 25,000 per dollar.**

The gap is near 3×10³, or 1.5×10³ if the forks use the luck correction too. These are orders of magnitude, not measurements. They also concern chance differences, not choice differences (section 4). The critic also prices all nine candidates at every decision, where a fork label prices two.

### 1.4 The improvement operator

**Candidates** at a decision: `own1`'s top 8 joint actions plus end_turn. Q(x, a) = Σ_o p(o | x, a) C(T(x, a, o)). That is one forward for a move or an end_turn, the enumerated outcomes for a fight, and the trait outcomes for a recruit: about 22 forwards a decision, so 2M decisions cost $1.6. Each candidate is a single decision, valued by the policy's own continuation, which re-decides after every roll.

**Target:** magnetic mirror descent (Sokota et al. 2023), with `own1`'s decoded policy ρ as the magnet. It converges to a regularized equilibrium in two-player zero-sum games, where single-agent updates can cycle. It uses Muesli's clip (Hessel et al. 2021):

```
q(a | x) ∝ [ π_k(a|o) · ρ(a|o)^{αη} · exp(η · clip(Â(x,a)/σ, −c, c)) ]^{1/(1+αη)}
```

- π_0 = ρ includes the −1.5 end_turn shift. The actor trains its raw logits toward q and plays raw argmax. After round 1 the offset lives in the weights, and the stop rule is learned: Q of end_turn against the other candidates.
- η is set offline so that the argmax changes on f = 10% of held-out decisions. This is the small step of conservative policy iteration (Kakade and Langford 2002).
- **Predicted gain, before any game:** G_pred = Σ of Â over the flipped decisions, from the performance-difference lemma at first order. For small G, Elo ≈ 347 · G per game-side (decisive games).

**Fog:** the actor sees its observation o, while q depends on the true state x. Cross-entropy distillation converges to E[q_x | history]. Expanding exp(ηÂ) to second order, E[q_x](a) equals the tilt by E[Â | history] up to π(a) · O(η² Var(Â | history)). Strategy fusion therefore enters only at second order in the step size, and small steps make the privileged teacher fog-correct.

## 2. The regimen

**Prices:**
- a 4090 at $0.42/h;
- 17,000 games/h, so $2.5e-5 a game;
- critic inference at 3,200 forwards/s, so $3.6e-8 a forward. This is the `relset` serve figure; a critic without memory at a larger batch is likely faster, not measured;
- training at 1,500 positions/s;
- a match at $0.12;
- an operator match, privileged, with 22 critic forwards per decision of side A, at about $0.3.

| stage | cost | readout and prediction | kill |
|---|---|---|---|
| 0. Does the critic learn the dice? | ~$7 | Zero-floor ratio at game start **0.55 (0.40-0.70)**; the seven linear columns left 0.848 | ratio ≥ 0.80 |
| 1. Planted signals | ~$6 | Windfall slope 0.9 (0.75-1.05); chest slope 0.75-1.05; η = 0 null reproduces `own1` | windfall slope < 0.6; any chest readout outside |
| 2. Privileged operator | ~$16 | All types ≥ **+60 Elo**, = s · G_pred with s in 0.5-1.0; end_turn known answer within ±0.5 nats | ≤ 2 SE above 0, or s < 0.3 |
| 3. The actor | ~$3 | ≥ **0.6** of the operator's gain; within 2 SE of its own G_pred | < 2 SE above `own1` |
| 4. Rounds | ~$12 each | G_pred × the last round's s | below prediction by > 2 SE halves f |

Details of each stage:

**Stage 0.**
- **Determinism.** Replay 500 `own1` self-play games from their seeds, in shuffled batch slots. Predicted: at least 99.5% of decisions agree, and the implied floor is under 0.01 of Var(z).
- **Data.** 20k games train a bootstrap critic U_0 on raw results. 60k more games train C_1 with U_0's corrections, at 150 positions a game, with refreshes by chunk. 10k fresh games are read out.
- **Readout.** The 0.848 left by the seven linear columns had no setup or trait term; the critic's value at the start absorbs the setup draw.
- **Scaling check.** C trained on 60k games must read a ratio at least 0.03 lower than C trained on 30k. Otherwise capacity, not data, is the limit, and the critic grows before stage 2.

**Stage 1: the planted-signal check of each training code path.**
- **Critic path, "windfall".** At the start of each side's turn 4, a fair coin grants +60 gold.
  - Truth: the coin's consequence at 300 positions, from luck-corrected paired forks (9,600 playouts, $0.12).
  - Predicted: after 30k modified games, the critic's luck at the coin regresses on the truth with slope 0.9 (0.75-1.05). Its RMS error at 10k games is √3 ≈ 1.7 times that at 30k (±0.3), the 1/√n rate.
- **Operator and distillation path, "chest".** On turns 2 to 6, the recruit list offers once per game a zero-cost chest worth +40 gold next turn. The actor's prior on it is pinned at 0.001.
  - Truth: its advantage at 300 positions, by forks.
  - Predicted:
    - the operator's Â regresses on the truth with slope 0.75-1.05;
    - after one distillation, the actor's chest probability is within 0.05 of q's;
    - the match on the modified game against `own1` lands within 2 SE of G_pred.
  - Null: η = 0 must reproduce `own1`, with at least 99% argmax agreement and a match within 2 SE of 0.

**Stage 2.**
- Compute G_pred from the critic, grown to 200k games, then play operator matches against `own1`: one with all decision types, then one per type (fights, moves, recruits, end_turn), which locates the gain.
- Known answer: averaged over positions, the operator moves end_turn by no more than ±0.5 nats from `own1`'s −1.5 decode. That decode was tuned as the best constant, so an average shift beyond it means the critic is biased about stopping.

**Stage 3.**
- Distil 2M operator decisions, then match the actor against `own1`.
- The predicted fraction of 0.6 allows for fog and distillation losses; the record's critic that read the mover's observation lost most of the true-state critic's edge.

**Stage 4.**
- **Each round:** 100k games of π_k; the critic refitted warm with past-only corrections; 2M operator labels; distillation; the matches.
- **Second-order check:** a match that falls below its prediction by more than 2 SE signals interaction between flips, and f is halved.
- **When two rounds in a row gain less than 2 SE:** first a look-ahead of two decisions inside the mover's turn (still single decisions, the second chosen after the first one's roll), then a critic four times larger.

**Budget:**
- **$85:** stages 0-3 cost about $32, which leaves three to four rounds.
- **$1,000:** only after three rounds land within their predictions. It buys about 60 rounds, or fewer with larger critics.
- No human anchor exists to tell when the player passes the strongest humans; the regimen reports Elo over `own1` only.

## 3. What is new, and the nearest prior work

New, as far as my search found (not proof of absence):

1. **The martingale luck residual used as the evaluator's training loss, with a zero floor under a deterministic policy and an exact chance model.** Nearest prior work:
   - Martingale control variates in simulation (Henderson and Glynn 2002), and their adaptive tuning to minimize the estimator's variance (Kim and Henderson 2007): the same objective, for low-dimensional approximators in option pricing.
   - The luck adjustment in JellyFish and GNU Backgammon rollouts: with a good net, 100 games are as accurate as 2,500 (Montgomery 2000); the factor falls to about 3 when the net misjudges the position.
   - AIVAT and MIVAT, which evaluate with a fixed or separately learned value.
   - The learned baselines of VR-MCCFR (Schmid et al. 2019).
   - None of these trains a deep value network on the residual. Searched 2026-10-19, the closest hit was an evaluation-only value-head control variate.
2. **A privileged critic that reads both policies' memories.** It makes Baisero and Amato's history critic finite and makes the zero floor reachable under fog with a memory policy.
3. **The dice as the evaluator's label source,** with the accounting of 1.3.
4. **Fog-correctness to first order of distilling privileged tilts,** with the second-order bound. Nearest: the asymmetric actor-critic (Pinto et al. 2018) and Suphx's oracle guiding (Li et al. 2020), neither of which bounds the fusion.
5. **Magnetic mirror descent with the imitation policy as the magnet,** at the granularity of a single decision with exact chance. The end_turn offset is absorbed by construction. Nearest: Sokota et al. 2023, and the human-regularized piKL (Bakhtin et al. 2023).

Not new:
- One-step expectimax with exact chance nodes (Ballard 1983; TD with exact expectations in 2048, Szubert and Jaśkowski 2014).
- This project's program of 2026-10-08. Its step 1 killed the learned critic, and that kill does not transfer here, for three reasons:
  - those critics trained on at most 6,800 games of raw results, the regime where AlphaGo's value network overfit (Silver et al. 2016);
  - its benchmark compares turns played with their own dice, which the user has since ruled invalid;
  - its holdout loss fell with data (0.492 to 0.460) while the benchmark, at ±0.05-0.1, could not resolve rankings.

## 4. The weakest point

- **The dice teach chance differences:** hit points, deaths, status, a recruit's traits.
  - A fight buys a distribution over exactly those, and recruits and end_turn are close to them.
  - A move's value is positional. The critic learns positions only through the level term ‖e(x_t)‖², whose signal is the slow drift of a game.
  - If the available gain lies mostly in moves, the critic's move advantages will be uncalibrated.
  - Stage 2's per-type matches read this. If it happens, the operator flips only the calibrated types, and moves go to the seed's forks, which this design keeps as a fallback rather than a competitor.
- **The second risk** is that step 1's flat size curve reflects capacity rather than data. Stage 0's scaling check reads it for $7, before anything else is spent.
- **The third risk** is that one reference does not show exploitability. The magnet limits drift but does not measure it.

## References

- Baird 1995, "Residual Algorithms", ICML. http://www.leemon.com/papers/1995b.pdf
- Baisero and Amato 2022, arXiv 2105.11674. https://arxiv.org/abs/2105.11674
- Bakhtin et al. 2023, piKL, ICLR. https://arxiv.org/abs/2210.05492
- Ballard 1983, "The *-minimax search procedure for trees containing chance nodes", Artificial Intelligence 21.
- Burch et al. 2018, AIVAT, arXiv 1612.06915. https://arxiv.org/abs/1612.06915
- Henderson and Glynn 2002, Math. Oper. Res. 27(2). https://pubsonline.informs.org/doi/10.1287/moor.27.2.253.329
- Hessel et al. 2021, Muesli, arXiv 2104.06159. https://arxiv.org/abs/2104.06159
- Kakade and Langford 2002, ICML. https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf
- Kim and Henderson 2007, Math. Oper. Res. 32(3). https://pubsonline.informs.org/doi/10.1287/moor.1070.0251
- Li et al. 2020, Suphx, arXiv 2003.13590. https://arxiv.org/abs/2003.13590
- Montgomery 2000, "Variance Reduction", GammOnLine. https://bkgm.com/articles/GOL/Feb00/var.htm ; GNU Backgammon manual on rollouts: https://www.gnu.org/software/gnubg/manual/html_node/Rollouts-in-GNU-Backgammon.html
- Pinto et al. 2018, asymmetric actor-critic, arXiv 1710.06542. https://arxiv.org/abs/1710.06542
- Schmid et al. 2019, VR-MCCFR, arXiv 1809.03057. https://arxiv.org/abs/1809.03057
- Silver et al. 2016, Nature 529.
- Sokota et al. 2023, MMD, ICLR, arXiv 2206.05825. https://arxiv.org/abs/2206.05825
- Szubert and Jaśkowski 2014, "Temporal difference learning of N-tuple networks for the game 2048", IEEE CIG.
