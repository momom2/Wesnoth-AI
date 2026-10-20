# Round 2 (invent): swing layers, turn-scale policy iteration with a budget of deviations

## Summary

This proposal keeps the current design's critic, coins and plants, and replaces its improvement operator. Each round optimizes the mover's whole turn against the critic:
- one action at a time, re-deciding after every roll;
- end_turn chosen freely;
- at most j deviations from the incumbent's actions.

The values for j = 0 to 3 ("swing layers", after the swing-option recursion over remaining exercise rights) are outputs of one fogged Q head. Their targets carry no dice noise. Each layer's marginal gain is forecast, then measured by coins that pick whole turns. So the depth at which critic errors start being exploited is measured, not guessed.

## 1. Why turn scale

- **The record places the headroom in multi-decision changes.**
  - Under `terrain`, 7 of 60 positions had an alternative turn confirmed at least 0.25 better (docs/turn_gap_ref_prereg_20260921.md).
  - In 6 of those 7 the better turn took more decisions (14.4 against 10.3), and in 5 it attacked more.
  - Caveats: the grading read the true state, and 2 of the 7 carry in-turn luck.
  - The largest lever on record, the end_turn offset (+229), is itself a stopping rule.
- **Single switches reach these changes slowly.** A switch followed by the incumbent's continuation cannot value "act three more times". The current design names this as its weakness 2.
- **The critic is the expensive object.** It costs about $11 per policy, for about one equation per game. A turn-scale operator gets more improvement out of each critic. With exact values and unrestricted choices, policy iteration on a finite-horizon problem makes the last i stages optimal after round i (backward induction). Turns as stages give about 10 per side; single switches give about 225.

## 2. The algorithm

### 2.1 Definitions

The incumbent and its critic:
- π is the incumbent decode.
- C is its privileged critic: it reads the true state, both memories and the matchup tag, as in the current design.

At the mover's information state h (observation plus memory), with true state x:
- **a\*(h)** is π's best non-end action, the joint argmax over non-end actions.
- **K(h)** is {end, a\*} plus π's next three non-end actions by joint prior.
- Choosing a\* or end is free. Choosing any of the other three is a **deviation**.
- **u(x) = C(x∘end)**. A game that ends inside the turn scores its result.

With j deviations left in this turn:

    Q_j(h, end) = E[u(x) | h]
    Q_j(h, a*)  = E[ Σ_o p_{a*}(o|x) V_j(h'_{a*,o}) | h ]
    Q_j(h, b)   = E[ Σ_o p_b(o|x) V_{j−1}(h'_{b,o}) | h ]     (b a deviation, j ≥ 1)
    V_j(h)      = max over allowed b of Q_j(h, b)

Here h'_{b,o} is the mover's next information state: the fork's observation, which shows any unit the move reveals, and the memory written at that decision. Fight outcomes come from the core's enumeration: every outcome of probability at least 2%, plus one conditional draw for the tail. Recruit traits use two independent draws.

Two special cases:
- **V_0 is the Snell envelope of the turn:** optimal stopping over π's own actions, with C as the payoff.
- **The current design's one-step look-ahead** is V_1 with π's own stopping in place of free stopping. That quantity stays as an extra output, so its coin calibration (current design 2.4) runs unchanged.

### 2.2 Properties (exact case, C = V^π)

1. **Monotone.** V_3 ≥ V_2 ≥ V_1 ≥ V_0 ≥ C. Each class contains the previous one, and V_0 contains π's own play, whose value is C because C is a martingale along π.
2. **Improvement and exact forecast.** Let π^(j) play every side-A turn by the layer-j rule against fixed `own1`. The policy-improvement theorem on the turn-level process gives V^{π^(j)} ≥ V^π. The performance-difference lemma at turn granularity is exact:

       J(π^(j)) − J(π) = E_{π^(j)} Σ_{A's turns i} [ V_j(h_i) − C(x_i) ]

   Side B's turns contribute nothing, because `own1` is the same in both games. Layer j's marginal gain is the same sum with V_j − V_{j−1}, and it is a separate factor.
3. **Targets free of dice noise.** The decision's own chance is enumerated. Later in-turn chance is integrated by the bootstrapped V, a regression and so an expectation. Chance after the turn is integrated by C. Only the fog floor and the head's approximation error remain. Compare round 1's turn-closure label: about 0.011 of later-dice variance per candidate, against an advantage of about 0.03 (0.0009 squared).
4. **The stop rule.** "End now versus act once, then stop as π does" is a one-stage look-ahead rule. Such rules are optimal for monotone stopping problems (Chow, Robbins & Siegmund 1971, *Great Expectations*; classical result, not re-read). A turn is not monotone: an action can open attacks that make acting again worth more. So V_0 can stop later than any one-step rule, the direction the record favours.

### 2.3 Learning the layers

**The head.** Q_θ is a two-layer transformer head on `own1`'s frozen token outputs, memory included. For each candidate in K(h) it outputs:
- layers 0 to 3, each as the critic's three outcome probabilities (choices use P(win) − P(loss) − P(cap); forecasts convert per outcome);
- the π-stopping look-ahead.

It reads the matchup tag. Play uses the self-play tag; forecasts use the `own1` tag.

**The targets** are computed on true-state forks with double Q, cross-fitted so a critic never labels its own fold's states:

    y_j(end) = C(x∘end)
    y_j(a*)  = Σ_o p_{a*}(o|x) Q̄_j(h'_{a*,o}, argmax Q_θ,j)
    y_j(b)   = Σ_o p_b(o|x) Q̄_{j−1}(h'_{b,o}, argmax Q_θ,j−1)

The loss is squared, fitted over four passes; in-turn chains are at most about 14 decisions long.

**Fog.** The minimizer is E[y | h]. Every max is taken over Q_θ at the child's own information state, so this is the information-state backup of the in-turn POMDP with payoff u, and no strategy fusion enters. It is the asymmetric-DQN pattern ([Baisero, Daley & Amato, UAI 2022](https://www.khoury.northeastern.edu/home/abaisero/assets/publications/repository/baisero_asymmetric_2022.slides.pdf), checked on slides only), with privilege only at the turn's end.

**Memory.** The mover writes before acting (`wesnoth_ai/memory.py`), so candidates share the parent's write. The child forward writes the next one, as in play. The opponent's memory does not change during the turn.

**Training states** come from three sources:
- 30 decisions per critic game;
- their children, one deviation deep;
- in-turn episodes from those games' turn starts, played by the ε-greedy layer-3 rule with logged propensities, which reach states two and three deviations deep.

**Cost.** Five candidates have about 12 children. Each child costs one trunk forward (Q̄ at every layer at once) and one critic forward, cached after pass 1. At about $5e-8 a forward that is about $5 for 1.5M decisions, four passes, training included. The 12 forks and 4 enumerations cost about 0.7 ms of CPU against about 7 ms of GPU.

### 2.4 The player

π^(j) keeps a per-turn counter c, set to j at turn start. At each decision it plays the argmax over K_c(h) of Q_θ,c(h, ·), and decrements c on a deviation.

The player is deterministic, reads only its own observation and memory, and makes no fork at play time.

### 2.5 The audit: coins that pick whole turns

**Why whole turns.** Mixing multi-step greedy policies decision by decision breaks their plans. For such policies, soft updates are not monotone "unless the update stepsize is sufficiently large", even without approximation ([Efroni et al., NeurIPS 2018](https://papers.nips.cc/paper/2018/hash/3f998e713a6e02287c374fd26835d87e-Abstract.html)).

**Protocol.**
- Play π^(j) against `own1`. At each side-A turn start, a fair coin c_i picks layer j or layer j−1 for the whole turn. Both re-decide after every roll, and the shadow choice comes from the same forward.
- A turn where the two layers agree at every decision of the realized path follows that path under either coin, so its score term is dropped without bias.

**Estimator.** R̃ is the result corrected for the dice's L terms only:

    Ĵ'(½) = mean over games of R̃ · Σ_i 2(2c_i − 1)

The midpoint rule gives J(1) − J(0) = J'(½) + J'''(ξ)/24, exact up to third-order interactions among turns. The standard error is (4·n_d·σ_r²/n)^½, with n_d the number of disagreeing turns per game. At n_d = 8 and σ_r² = 0.6, 20,000 games give 0.031 result units, about 11 Elo. Splitting the terms by the type of the turn's first deviation shows where the gain comes from.

**Reading a miss.** If layer j's audit falls below Ĝ_plug while layer j−1's lands on its forecast, the critic's errors on positions start being exploited at depth j. No deeper layer is tried. The next critic, trained on that player's games, sees those states.

**Kept from the current design:**
- the critic and P1;
- exploratory coins, β and the nested model, read on the π-stopping output;
- P0, P2 and P5;
- the matchup mix, `parity3` and the league;
- distillation.

**Replaced:**
- the switch cap M = 10 becomes budgets m ≤ 3 per turn;
- the decision-level audit becomes turn coins.

## 3. Regimen

Prices are the current design's:
- a game costs $2.5e-5 under generation numerics, or $1.3e-4 under eval numerics; S0 decides which applies;
- a forward costs about $5e-8;
- a match costs $0.12;
- an audit of 20,000 games costs about $1.1, including the critic's luck forwards.

Every number below is written before any run. Elo ranges are priors for sizing; the kill lines use the forecasts and the audits.

| Stage | Cost | Predicted readout | Kill line |
|---|---|---|---|
| **S0** instruments | $2 | Current design S0 (a)–(d), with P0 in turn-coin form: whole side-turns by `own1` at offset 0 or −1.5, t ∈ {0.211, 0.789}, 20,000 games, reproduce Match A within 2 combined SE. | Outside: a coin, side-sign or propensity bug in the audit; nothing downstream runs. |
| **S1** critic | $11 | As in the current design (P1; curve; ρ_C(50,000) prior 0.50–0.75). | As in the current design; fallback F. |
| **S2** layers 0–1 | $6 | **P2** (village tax) as in the current design. Q_θ on 1.5M decisions: V_1 ≥ V_0 ≥ C on held-out states within 2 SE. **Known answer:** the layer-0 recursion run with offset 0's stop choices in place of its max (the same code evaluating a given rule) forecasts Match A within 2 combined SE. β on the π-stopping output 0.5–1.0. | P2 outside, or known answer missed: the stopping recursion is broken. β < 0.3: fallback F. |
| **S3** round-1 players | $11 | **P3** first (below). Ĝ_old and Ĝ_plug per layer before any match. Priors: layer 0 +10 to +40 Elo; layer-1 marginal +10 to +40; layer-2 marginal 0.3–0.6 of layer 1's audited marginal; layer 3 at most layer 2's. Four turn-coin audits, each within 2 combined SE of its Ĝ_plug. Matches: adopted depth and depth 0 against `own1`; adopted depth against `parity3`. | A layer is adopted only with an audited marginal ≥ 2 SE that lands within 2 combined SE of Ĝ_plug; the first failure ends the depth search. Match outside 2 combined SE of the summed audits: interactions; report. Gain over `parity3` under half that over `own1`: exploitation. |
| **S4** round 2 | $15 | 25,000 π_1 games (¾ self-play, ¼ against `own1`, exploratory coins); warm-started critic; layers; audits; matches. Prior: summed audit at most round 1's. | Two rounds below 1 match SE: S5. |
| **S5** distillation | $5 | P5, then distil π^(m) into the actor by own-trajectory training: level with π^(m) within 2 SE, with no offset and no counter. | As in the current design. |

**P3, the pincer plant**, checks depth 2 or more with a signal that decisions move and one deviation cannot collect.
- **Plant.** At each own end_turn, the side's planted scalar (booked beside the result, like P2's tax) gains r = 0.05 for each enemy unit flanked by own units on two opposite neighbouring hexes. An independent script computes it from the game record.
- **Feasible turns.** A second script finds the turns where both pincer moves are among the top-4 non-end actions at two of π's decisions.
- **Predicted:**
  - layer 2 collects in at least 0.7 of the feasible turns;
  - layer 1 collects no more than the unplanted base rate plus the turns where one pincer move is π's own choice;
  - the layer-2 audit on the planted game exceeds layer 1's by r times the gain in collections, within 2 SE.
- **Kill.** A miss on any of these means the bootstrapping or the budget bookkeeping is broken.
- **Cost.** About $2.5.

**Total.** About $50 on the planned path, and at most $65 with one rerun of S4.

**The $1,000 case** is argued only after two rounds whose matches land within 2 SE of their audits. It buys:
- 25–30 rounds at about $25 each, with a larger critic;
- distillation every two rounds;
- K and m grown by one whenever the deepest layer passes its audit;
- a league check each round.

The section 1 bound is why turn-scale rounds should saturate in tens of rounds rather than hundreds, though it holds only with exact values and a fixed opponent. The predicted gain is the measured per-round decay, extrapolated with its error. No measured link to human ratings exists.

## 4. What is new, and the nearest work

Searched 2026-10-19; absence is not proven.

1. **Swing layers.** A value indexed by the deviations still allowed in this turn. It has free optimal stopping at end_turn and a privileged zero-floor critic as payoff, and it is learned as a fogged Q by fitted iteration with exact chance. Each added coordinated deviation is forecast and audited as its own factor.

   Nearest work:
   - the recursion over remaining rights is swing-option pricing's ([Meinshausen & Hambly 2004](https://ora.ox.ac.uk/objects/uuid:664ee1f1-0ec1-4bc2-8e53-13b16aa15c0a); form standard, full text not checked);
   - multi-step greedy policy iteration with a fixed horizon ([Efroni et al., ICML 2018](https://proceedings.mlr.press/v80/efroni18a.html)), where here the policy's stop action sets the horizon;
   - baseline deviations under a per-state error budget ([Soft-SPIBB](https://arxiv.org/abs/1907.05079));
   - Hearthstone agents that search to the end of their turn under a heuristic evaluation ([Justesen et al. survey](https://arxiv.org/abs/1907.06562)), at play time, with no learned or privileged payoff;
   - optimal stopping by regression (Tsitsiklis & Van Roy 1999; Longstaff & Schwartz 2001).

   A count budget on deviations inside a dynamic program turned up only in bandits with switching budgets.
2. **Per-layer turn-coin audit.** It measures the marginal gain of each look-ahead depth, unbiased up to third-order interactions among turns, and drops agreeing turns exactly. Nearest: the current design's audit, Differences-in-Q ([Farias et al. 2022](https://arxiv.org/abs/2206.02371)), and micro-randomized trials.

## 5. Weakest point

**The curse at depth.**
- Layer j ≥ 2 chooses among many contingent plans against a critic whose errors on positions (the features decisions move) are its least trained.
- Its bootstrapping reads Q_θ two or three deviations off π's trajectories, covered only by the ε-greedy in-turn episodes.
- If the layer-2 audit falls short, the method reduces to layers 0 and 1: the current design plus optimal stopping.

**The stop layer** rests on C at over-continued states, past where π would end. The critic sees these only through about 1.5 stop coins per side-game; S2's known answer is the guard.

**The turn-scale headroom** was measured under `terrain`, from the true state, with 2 of 7 confirmations luck-tainted.

**The play-time counter** is extra state that distillation must absorb.

## 6. Notes on research targets

- **The schedule between the critic's two rules** is [Baird's residual algorithms (1995)](http://www.leemon.com/papers/1995b.pdf). That family contains the direct and residual-gradient updates and keeps convergence; its abstract says pure residual gradient "may learn very slowly". Proposal: start near the semi-gradient end, move toward the residual as ρ_C falls, and read ρ_C at each step.
- **Optimal stopping** is answered by layer 0, which dominates "end versus best act".
- **The advantage spread s.** The offset's 0.58 result units are spread over the 20–40 stop decisions it flips per game. That count is round-1 invent's and was not re-measured. It implies 0.015–0.03 per flipped stop decision. Nothing on disk measures the spread for decisions to act.

## 7. Decision record

- **Rejected: per-state pessimism from a learned E[R_t² | x_t] head**, the zero-floor analogue of KataGo's [value-error head](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md) and [DEUP](https://arxiv.org/abs/2102.08501). E[R_t² | x_t] = e(x_t)² + Σ_k E Var_o e(x_k^o). The future-branch sum dominates and is common to a decision's candidates, so the penalty barely separates them. It stays as an S1 diagnostic.
- **Rejected: decision-level coins for multi-step greedy players**, because per-decision mixing breaks their plans and is not monotone.
- **Rejected: unbudgeted turn-greedy improvement as the first step**, because the curse grows with the number of plans, and no audit could then say at which depth it began.
- **Rejected: a learned world model in the look-ahead (Stochastic MuZero)**, because a fork plus an enumeration costs under 1% of the network forward that each child needs anyway.
- **Rejected: in-turn search at play time**, because the player must be a network picking single actions.
