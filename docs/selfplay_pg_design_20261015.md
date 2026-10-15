# Design: self-play policy gradient from game results (2026-10-15, revised after review)

A design by the project lead for the user's decision. Nothing in it is built or
run. It answers the user's ruling of 2026-10-06: before any self-play training
launches, the algorithm must handle the reward's sparsity, multi-step turns
whose plans depend on dice rolled inside the turn, and the simulator's speed
without being bottlenecked on the network.

**This version** replaces the first draft of the same day, after an independent
review found that its measurement could not measure what it claimed and its
pilot could not decide. The review's findings, and what this version does with
each, are in "Decision record" at the end.

## The bottom line first

On the records, a resolved gain at argmax from outcome-driven self-play likely
needs a million games or more:
- the per-turn ceiling a perfect choice among sampled turns reaches is a few
  hundredths of a game result (+0.033 +- 0.016);
- fight luck explains only 7% of outcome variance at temperature 0.5
  (docs/turn_value_prereg_20260925.md, 36,209 playouts);
- critics a fair player could use do no better than material;
- AlphaGo's RL policy needed about 1.3 million games for its gain.

A million games costs about $80-145 at today's rates (below). That is the
user's possible $1,000, not today's $84. This design spends about $2-3 to
measure the numbers that decide it, before any training.

## Why policy gradient, still

Every evaluator a fair player can use has measured level with the policy, six
times in a week. So no search-based improvement operator is available at this
scale. The game's result is the one signal that carries information the policy
lacks, and policy gradient learns from it with no evaluator. A baseline only
reduces variance, and any baseline leaves the gradient unbiased. It costs one
forward per decision, so every speedup of this week carries over.

The family that improved imitation-initialized policies is AlphaGo's RL policy
(Silver et al. 2016) and AlphaStar's league (Vinyals et al. 2019). AlphaStar,
OpenAI Five and DeepNash bootstrapped their targets, at many orders of magnitude
more games, and OpenAI Five and DeepNash started from scratch. AlphaGo is the
one terminal-reward precedent near this scale.

## Two behaviour policies to choose between

The evaluated player is the argmax of the reference decode. The policy that
generates the games decides what the gradient can change.

- **T: sampling at temperature τ at every decision.** The played policy is
  p(a) ∝ (prior_θ(a) · e^{o·[a = end_turn]})^{1/τ} with o = -1.5. That is
  tools/raw_player.py's order: the end_turn shift, then the temperature.
  - The gradient improves J(π_τ), which is not the argmax player's value.
  - Most of the signal lowers the probability of bad samples, which leaves the
    argmax where it is.
- **D: argmax play with sparse sampled deviations.** At each decision, with
  probability ρ (about one decision in 16), the player draws from π_τ instead
  of taking the argmax. Every other decision is the evaluated player's own.
  - Each deviation is an experiment on the argmax player, and the game's other
    noise is only the dice and the other deviations, so the luck correction
    removes a larger share. The share at argmax is being measured now
    (branch `exp/match-luck`).
  - Only the windows that hold a deviation need a backward pass (the trainer's
    `--train-every` machinery generalized), which cuts the trainer's share of a
    run.
  - The gradient estimator is the policy gradient of the deviation's sampled
    action, with the argmax player's continuation value as baseline.

R0 measures both, and the choice between them is the user's after R0.

## The return and the baseline

- **Reward:** the game's result at the game-side's last decision, +1 or -1.
  A game capped at the turn limit scores -1 for both sides, so stalling a lost
  game never beats playing it out. A 0 at the cap would reward stalling, and
  the PURE fit, which reads decisive games only, would count it as a gain. A
  cap-rate barrier and a caps-as-draws reading run beside every PURE reading.
- **Luck correction (AIVAT):** for every attack of either side, its exact
  outcome distribution from tools/combat_outcomes.py gives
  L = Q~(realized) - Σ_o p(o)·Q~(o), with zero mean given the pre-event state.
  - Q~ is the cost-weighted material margin, with a leader kill scored at the
    game's result.
  - Fights the enumeration does not cover, trait rolls and advancement choices
    take L = 0, which stays unbiased.
  - Q~'s scale is fitted by cross-fitting, never on the game it corrects.
  - AIVAT's term for a player's own known strategy, applied to the later
    decisions of either side, is measured in R0 beside it.
- **Baseline:** a full-size critic of the true state, cross-fitted: the
  previous iteration's, or K folds. With Monte-Carlo returns a state-only
  baseline is unbiased, since the action is drawn from the history alone
  (Baisero and Amato 2022, section 5.1). A critic of history and state is a
  variance choice, built only if R0 shows its gain.
- **λ:** measured, not asserted. R0 tabulates the advantage's variance and its
  TD-error share at λ ∈ {0.95, 0.99, 0.997, 1}.

## The update (fixed in the pilot's pre-registration, from R0)

- A clipped importance ratio against the behaviour log-probability recorded at
  generation (new per-decision fields in the game record). A barrier on the
  log-ratio distribution before the first update checks that generation's
  arithmetic matches the trainer's; if it fails, the eager server generates.
- Gradient accumulation to a batch set from R0's noise scale: tens of updates
  per iteration, not thousands of 512-position steps of noise.
- A KL penalty to `own1` (human-regularized RL, Bakhtin et al., ICLR 2023):
  its direction, temperature and coefficient schedule pre-registered.
- Barriers per iteration:
  - the clip fraction;
  - the KL to the behaviour policy and to `own1`;
  - the prior's entropy;
  - the cap rate;
  - decisions per side-turn (the memory's known failure ended turns early);
  - the holdout imitation loss at 0 and 64 slots;
  - the share of decisions whose argmax changed.

## R0: the measurements, before any training (about $2-3)

1. **Games:** 10,000 games of `own1` under T at τ = 0.5, and 10,000 under D,
   with the behaviour log-probabilities recorded (about $0.8).
2. **What sampling costs:** `own1` at τ = 0.5 against `own1` at argmax, 800
   decisive games (about $0.2).
3. **The critic:** state-only, full size, cross-fitted on these games. Its
   holdout explained variance of the result.
4. **The variance table,** for each behaviour: the advantage's variance under
   the raw result, the critic, the luck correction, both, and the λ grid.
5. **The gradient's signal:**
   - From split-half gradient dot products at 1,000, 2,500 and 5,000 games per
     half: E[g1·g2] = |G|², and E|g_N|² = |G|² + tr(Σ)/N (McCandlish et al.
     2018's B_simple).
   - Each against a sign-flip null: the same games, each game-side's advantages
     multiplied by a random ±1, which has a gradient of exactly zero mean.
6. **Operating characteristics:** from 5, a simulation of what a run of N games
   would show in a match, as step 1 calibrated its critics. That tells the
   budget each reading needs.

Predictions (the lead's, from the records):
- the luck correction cuts the variance under T by 5-10%;
- under D it cuts it by more, the size given by `exp/match-luck`'s measurement;
- the critic cuts the variance by under 5%;
- sampling at τ = 0.5 costs `own1` 50-150 Elo;
- |G| is resolvable from zero at 5,000 games per half under D, not under T.

**Kill:** if neither behaviour's gradient is resolvable from the sign-flip null
at 5,000 games per half, outcome-driven policy gradient is out at any budget
the project has. Otherwise R0's operating characteristics give the games a
resolved match gain needs, and the decision goes to the user with that price.

## Build (before R0)

About a week of agent work, no box:
- per-decision behaviour log-probabilities in the game record;
- behaviour D in the raw player;
- the luck pass over records;
- the state-only critic (ported from `exp/value-policy-iteration`) with
  cross-fitting;
- the split-half gradient tooling on the sequence trainer;
- the tempered played-policy log-probability with its normalizer over all
  legal actions, and its memory cost measured. The trainer's 1,477 positions a
  second was measured for a different loss.

## Cost per million games (recorded rates; the PG loss and generation under T or D unmeasured)

| stage | box-hours |
|---|---|
| generation | 60-85 |
| training | 90-110, less under D, where most windows run forward only |
| `own1`'s forward for the KL anchor | 20-30 |
| the critic | 10-20 |
| record walk, matches, bring-up | 10-15 |
| **total** | **about 180-230 ($80-145)** |

## Decision record

The first draft's review (independent, 2026-10-15), each finding and its
disposition:

| finding | what this version does |
|---|---|
| R0 on argmax games cannot estimate the gradient | fresh sampled games under T and D |
| the records predict both variance levers weak | written as predictions; the deviation behaviour added |
| the pilot cannot decide at 3×10^4 games | replaced by R0's operating characteristics |
| the objective trained is not the evaluated player | behaviour D, and the sampling-cost match |
| the decode formula differed from the code | corrected |
| the tempered normalizer is unpriced | in the build list |
| steps far below the noise scale | accumulation set from R0 |
| the cost table missing stages | rebuilt |
| Baisero and Amato misapplied | state-only, cross-fitted baseline |
| a critic fitted on its own batch leaks the outcome | cross-fitting |
| the null arm broke the advantages' structure | sign-flip per game-side |
| the ruling's compliance | the user's to rule |
| a capped game at 0 rewards stalling | cap scoring and barrier |
| generation arithmetic against the trainer's | log-ratio barrier |
| the kill threshold not derived | derived from operating characteristics |

Rejected: the first draft's R0 on recorded argmax games and its 3×10^4-game
pilot, because neither can decide what it claimed to.

## References

- Silver et al. 2016, Nature 529.
- Vinyals et al. 2019, Nature 575.
- Burch, Schmid, Moravcik, Morrill and Bowling 2018, "AIVAT", AAAI; arXiv
  1612.06915.
- Baisero and Amato 2022, "Unbiased Asymmetric Reinforcement Learning under
  Partial Observability", AAMAS; arXiv 2105.11674.
- Schulman et al. 2017, "Proximal Policy Optimization Algorithms", arXiv
  1707.06347.
- Bakhtin et al. 2023, "Mastering the Game of No-Press Diplomacy via
  Human-Regularized Reinforcement Learning and Planning", ICLR.
- McCandlish, Kaplan, Amodei et al. 2018, "An Empirical Model of Large-Batch
  Training", arXiv 1812.06162.
