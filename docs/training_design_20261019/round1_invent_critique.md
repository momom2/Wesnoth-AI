# Round 1 critique of "invent" (chance-made contrasts and turn-closure labels)

**Verdict: repairable.** The structure obeys the rulings: single decisions, re-decision after every roll, fogged deciders, and a privileged evaluator used only in training. But three readouts do not estimate what they claim, and the main channel is the one the proposal lists as a side risk.

## Strongest part

The prediction protocol of 2.4. Because `own1` is deterministic, the performance-difference identity reduces exactly to the sum of `own1`'s advantages at the candidate's own changed decisions. The labels come from the candidate's trajectories, with `own1`'s memory carried in shadow. "Predict the match before it runs" becomes a measurable number. Second: Q̂ integrates the decision's own fight exactly over its outcomes, which removes the record's dice-selection flaw by construction.

## Where it fails

**1. V's bias depends on the action, and decisions are graded mostly by that bias.** Section 2.3 writes E[Q̂ | s] = Q_π(s,a) + b as if b were constant. It is e(x_T^a), V's error at that candidate's end state, so the pair bias e(x_T^a) − e(x_T^a′) does not cancel.

S2 gives V little decision-relevant information:
- turn twins: 120,000 pairs × 0.0009 / 2 ≈ 54 χ²;
- stop pairs: at most 40,000 × 0.0009 / 2 ≈ 18 χ².

That is about 72 χ², or about 8 directions by the proposal's own 9d rule, less than the seed's design it calls too expensive. The other ~6,700 χ² price dice-moved features. Section 5's "weakest point" is therefore the main channel for every move and stop decision.

The update then flips the argmax where Q̂'s gap beats the logit gap. So flips are selected on V's favourable errors, the optimizer's curse ([Smith & Winkler 2006](https://doi.org/10.1287/mnsc.1050.0451)). "Fresh labels carry no selection bias" is true of the dice, not of V. Only the full-playout calibration is unbiased.

**2. S2b's slope cannot read 0.8, even for a perfect V.** Its regressor, a turn-closure label, has noise variance 0.022 (the proposal's own figure, one draw of the later in-turn dice). The true spread of advantages is about δ² ≈ 0.0009, so regression dilution multiplies the slope by 0.0009 / 0.0229 ≈ 0.04. The stated standard error, sqrt(0.0625 / (2,000 · Var x)) = 0.12, implies Var x ≈ 0.0022, a regressor with no noise. As specified, S2b kills a correct V.

Fix: average at least 200 later-dice draws per regressor (about $1.8), or use an independent second draw as an instrument.

**3. The prediction is unfalsifiable at its budget.** With 2,000 changed decisions × 16 pairs at ρ = 0.15:
- per-decision variance 2(1−ρ)/16 ≈ 0.106;
- standard error of the mean advantage 0.0073;
- ĝ's standard error: 11 flips × 0.0073 × 347 ≈ 28 Elo.

The kill band, 2 combined standard errors, is then about ±60 Elo around a +35 prior. A slope fitted without an intercept inherits item 1's bias; with an intercept it is the playout mean again.

Fix: about 12,000 calibrated changed decisions (about $5), which brings ĝ to about 12 Elo.

**4. ĝ is in the wrong unit.** The yardstick fits decisive games only. The memory model needed 944 games for 800 decisive ones, so about 15% are capped. The PDL estimates E[R] with capped games scored 0. Then Elo ≈ 347·ĝ / P(decisive), but only while P(decisive) stays fixed, and S3's stop rule moves exactly that (stalling to the cap is on record). Make the playout reward match the match's estimand, or split ĝ into a win term and a cap term.

**5. Rounds 2–4 predict against the wrong opponent.** J(π_k vs `own1`) − J(π_{k−1} vs `own1`) sums the advantages of π_{k−1} with `own1` on side B. A V refitted on π_k self-play prices π_k as the opponent instead. Fix: run calibration playouts with `own1` on side B.

**6. V(x) on the true state is ill-defined.** Both continuations depend on memories that x omits. A state-only critic of history-dependent policies is biased in general; the unbiased form conditions on both history and state ([Baisero & Amato, AAMAS 2022](https://arxiv.org/abs/2105.11674), Theorem 5.1). Turn-closure states are counterfactual, so their memory distribution differs from V's training mix. The forks already carry both memories, so V should read them.

Two more shifts:
- "End now" states with unmoved units are rare in V's data (18 χ²), and S3 rests on V's reading of them.
- Treating `own1`'s stored memory as fixed input ignores that the trained network writes its own memory in play. That is the causal-confusion mechanism the record measured at 105 Elo.

**7. The planted checks bypass the label generator.** P-A and P-B add the planted term after the playouts, so they certify the regression and the update arithmetic. P-B's prediction is the update formula itself. None of these bugs would trip P-A, P-B or P-C:
- a sign error in the mover's perspective at the opponent's end_turn;
- a broken memory copy in forks;
- a coupling that breaks a branch's marginal law (P-C tests strikes, not branches);
- a fog-off encoding that keeps fog.

No check plants a decision-moved signal, which is where item 1 locates the weakness.

Fix: a modified game whose result includes a term accrued during play from a decision-made fact, for example +0.01 per own unit on a village at each own end_turn. Its Q effect at each candidate is computable, so its recovery rate is predicted from its χ², and it measures V's transfer before S4 spends money.

**8. Coupling is overstated.** Strikes consume draws in strike order. A different defender changes the retaliation count, the weapon or firststrike, and the draw indices then misalign. "Agree with probability 1 − |cth₁ − cth₂|" holds only for aligned sequences. S1 measures ρ and the costs assume ρ = 0, so only section 2.5's pitch suffers.

**9. The gates are weak.**
- S3's kill at +5, with a standard error of 12.3, passes a null 34% of the time.
- S4's kill at +12 passes a null 16% of the time.
- S2's benchmark gate grades V against `obs8` continuing at temperature 0.5 with luck-adjusted truth (docs/selfplay_program_20261008.md), not against V's estimand (`own1` at argmax). Its candidates also mix dice-made and decision-made differences, so it cannot certify item 1's transfer.

## Costs

They add up from the seed's prices (S2 $10, S2b $1.6, audit $0.8, match $0.12). The 60–115 forwards per turn-closure decision assume few fight candidates. With four attack candidates of about 8 outcomes each it is closer to 150–200, which puts S4 near $8. That is not decisive. The repairs add about $9.

## Novelty

- **Forced-outcome contrasts:** my searches found no prior that manufactures value-training pairs at chance nodes either. The nearest are TD-Gammon's pre-roll pseudo-states and the common random numbers and control variates of [Veness et al. 2011](https://proceedings.neurips.cc/paper_files/paper/2011/file/d736bb10d83a904aefc1d6ce93dc54b8-Paper.pdf). The claim stands, weakly.
- **Turn-closure labels:** every component is known:
  - truncated rollout with a terminal cost approximation ([Bertsekas](https://web.mit.edu/dimitrib/www/RLTopics_2024_Lect6.pdf));
  - afterstates (TD-Gammon's horizon is the same one, in a game without in-turn chance);
  - expectimax at the decision's chance node;
  - end-of-own-turn evaluation in multi-action games ([Justesen et al. 2016](https://pure.itu.dk/en/publications/online-evolution-for-multi-action-adversarial-games/)).

  The combination is new.
- **Paired branches with common random numbers:** TRPO's vine procedure ([Schulman et al. 2015](https://arxiv.org/abs/1502.05477)).
- Both 2026 citations exist as described.

## Repairs, in order

1. Plant a decision-moved signal end to end (item 7).
2. Denoise or instrument S2b (item 2).
3. Let V read both memories (item 6).
4. Size a decision-contrast budget on the directions decisions move.
5. Calibrate ĝ on about 12,000 changed decisions, in the match's estimand (items 3–4).
6. Predict rounds 2 and later against `own1` (item 5).
7. Put every gate at two standard errors.
