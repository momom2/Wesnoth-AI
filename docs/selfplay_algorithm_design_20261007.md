# A CPU-bound turn planner as the self-play teacher (design, 2026-10-07)

**Status: proposal, awaiting the user's decision.** Nothing here is built.

The user's ruling of 2026-10-06: before any self-play training, a new
algorithm must handle the reward's sparsity, multi-step turns whose plans
depend on dice rolled inside the turn, and the simulator's speed without
being bottlenecked on the network. Self-play keeps the memory (ruling of
2026-10-05).

The proposal moves the search off the GPU. A small evaluator and a small
action scorer run inside the Rust core at CPU speed. A turn planner built on
them re-plans after every attack, takes each attack's outcomes from the exact
enumerator, and looks past the opponent's reply. The big network, memory
included, supplies the planner's prior and its guess of the hidden units once
per decision, and learns by imitating the planner once the planner beats the
reference. The first test measures whether the planner's evaluation ranks
alternative turns at all; it costs no training run and kills the design if
it fails.

## What the record constrains

- **One outcome bit per game.** Decisive games end at a median of 24-28
  turns (docs/box_specs.md); at the reference decode that is about 9
  decisions per side-turn, so roughly 450 decisions share one bit.
- **No grader ranks alternative turns at one position well enough.** On 200
  holdout positions with 5 candidate turns each (docs/turn_value_prereg_20260925.md,
  "Measured"), corrected correlations with the playout truth: the
  reference's value head 0.22-0.23, a fitted linear grader 0.27, the HP
  margin right after the turn 0.40, and the mean over 8 rollouts by the
  reference of its value head read at the mover's next turn start (after
  the opponent's reply) 0.27, two of the mover's turns later 0.41-0.42,
  four turns later 0.53. The truth itself has a reliability of 0.58 at 28
  playouts: turn-quality differences are small against the dice.
- **Material read after real rollouts ranks better at short horizons.**
  From the same playouts, against the raw outcomes (a basis that reads
  0.02-0.07 above the verdict's luck-adjusted one): the HP margin at the
  mover's next turn start ranks the candidates at 0.49 +- 0.08 where the
  value head reads 0.31 +- 0.05; after the mover's next turn, at the
  opponent's turn start, both read 0.52; four turns later the head reads
  0.59 and the margin 0.54, the margin then averaging only the 85% of
  playouts still running (`tools/analysis/turn_reads_by_depth.py`, record
  `training/metrics/turn_value_20260925/reads_by_depth_20261007.txt`,
  2026-10-07). The head reads markedly better at the opponent's turn
  starts than at the mover's own (0.52 against 0.31, 0.59 against 0.45).
- **Better turns exist but are rare.** 7 of 60 positions have a sampled
  alternative turn confirmed better by at least 0.25; in 6 of the 7 the
  better turn takes more decisions (14.4 against 10.3)
  (docs/turn_gap_ref_prereg_20260921.md). The largest single gain on record
  is the decode rule "act more": +229 Elo for an end_turn logit offset.
- **Single attacks are already right.** The exact enumerator, checking the
  attacks of 300 human games for a strictly better one, finds 0.98 per
  1,000 decisions, a few Elo at most (branch `exp/xod-dominance`, f3aecd9).
- **The network is the expensive call.** A 4090 serves about 3,200 leaf
  evaluations per second at the pool's batch cap of 64
  (docs/serve_batch_prereg_20260920.md); the trainer costs 2.2 ms per
  experience (docs/box_specs.md). A search that calls the network at every
  node is bound by it: Gumbel MCTS at 32 simulations lost to plain argmax,
  13-27 (docs/raw_argmax_control_20260904.md).
- **The simulator is cheap.** In the Rust core: a fork 0.019 ms, a step
  0.050 ms (2026-09-12); an attack's exact outcome distribution 63 µs
  median, 258 µs at the 90th percentile, with 3 distinct outcomes at the
  median and 9 at the 90th (measured 2026-10-07 through the Python binding
  on the outcome test's boards, `tests/test_rust_outcomes.py`, 139 fights).
  A box whose GPU serves 3,200 calls per second has CPU for a few hundred
  simulator steps per network call.
- **A search on the true state sees through fog** (docs/hidden_information_20260926.md).
  The memory model's belief head gives, per hex, the probability that a
  hidden enemy unit stands there (docs/parity_memory_design_20260929.md).

## The algorithm

### Two fast components, in Rust

- **The action scorer** gives every legal atomic action a score from a short
  feature vector: action kind, the unit's type, level and HP fraction, the
  destination's defense, village capture, distance to the enemy leader,
  and, for an attack, expected damage dealt and taken and the kill
  probability. It is a linear model or a two-layer network, a few
  microseconds per action. It is trained by imitating the big network's
  choices on corpus and self-play positions; the labels cost one batched
  forward per position.
- **The evaluator** scores a side-turn boundary from the mover's view: per
  side, unit value weighted by HP, unit count, gold, income and villages,
  the leader's HP and the enemy units that can reach it, terrain defense
  held, the time of day now and next, the turn number. It is a two-layer
  network of about 10^4 weights, a few microseconds per call. It starts as
  a regression on corpus boundaries to game outcomes and to the big value
  head, and then learns from the planner (below).

This is the pattern of TD-Gammon, a small network trained by self-play
that played backgammon, a dice game, near the best humans after 1.5
million games and played with a 2-ply search (Tesauro 1995), and of NNUE,
a small evaluator updated incrementally inside a deep search (Nasu 2018,
in Stockfish since version 12).

### The turn planner, at each of our decisions

1. One forward of the big network at the current state, memory included:
   the policy prior, the value, the belief over hidden units. This is the
   only network call per decision, as in raw play.
2. Candidates: the k most probable actions under the prior, joined by the
   scorer's top actions (k about 8).
3. For each candidate, M sampled futures to the start of our next turn: the
   candidate itself, the rest of our turn and the opponent's whole reply,
   both played by the scorer. The candidate's own attack is not sampled:
   its outcome distribution is enumerated exactly and each outcome is
   followed with its probability. Later attacks are sampled, with the
   random numbers of each fight keyed by the fight (attacker, defender,
   strike), so two candidates that lead to the same later fight see the
   same dice in it. Each future is scored by the evaluator; Q(a) is the
   probability-weighted mean.
4. Play argmax over candidates of Q(a) + λ log π(a), π being the big
   network's prior: the planner leaves the prior only where Q says the
   difference is larger than its noise (the piKL rule, a λ sweep chooses
   it).
5. The engine resolves the dice of the action played. The planner re-plans
   from the new state at the next decision.

Plans conditional on the dice come from step 5: nothing is committed past
one action, so every roll is answered by a new plan. Within one plan, the
candidate's attack carries all its outcomes (an expectimax chance node,
Ballard 1983) and the rest of the turn its sampled ones.

The horizon (one turn pair above) is a knob. At exactly that horizon, after
rollouts by the reference itself, the value head ranks candidate turns at
0.31 and the plain HP margin at 0.49 (raw outcomes); one turn further, after
our own next turn, both read 0.52. The design's bet is an evaluator at
least as good as the HP margin at the default horizon, with the next read as
the fallback at about twice the cost. What the record cannot say is how much
of that ranking survives when the scorer, not the reference, plays the
rollouts.

### Fog

When the side has hidden enemies, the planner runs in S sampled worlds
(S = 2-4) and averages Q: hidden units are placed by the belief head's
probabilities, and their types are drawn from the enemy's recruit list
weighted by the units already seen. This is perfect-information Monte
Carlo, whose known weakness is strategy fusion (docs/hidden_information_20260926.md).
The type draw is the crudest piece; a type head on the memory would
replace it.

### Learning

- **The evaluator** learns from the planner's own games by temporal
  differences across turn boundaries. Its target at a boundary is the
  planner's expected value at that position, which already averages the
  dice of the candidate's attack exactly, mixed with the game's outcome by
  a λ-return. Expected targets carry less variance than sampled ones
  (Expected Sarsa, van Seijen et al. 2009), and the outcome bit anchors the
  chain. This is where the sparse reward is handled: the bit reaches the
  evaluator through bootstrapped expected values, not through 450 noisy
  credit assignments.
- **The scorer** learns to imitate the planner's chosen actions.
- **The big network** learns, once the planner beats the reference in an
  800-game match, by imitating the planner's decisions through the existing
  memory-stream trainer, anchored to the reference by a KL term, and its
  value head from the planner games' outcomes. Its acceptance is its own
  800-game match, raw, against the reference.

The loop's expensive part is CPU: the planner's playouts. The GPU does one
forward per decision, as in raw play, plus the distillation steps.

## Cost, estimated from the measured per-call costs

| quantity | estimate | from |
|---|---|---|
| one playout decision (scorer over about 350 legal actions, one step) | about 0.5 ms | scorer µs per action, step 0.05 ms |
| one planned decision, k = 8, M = 8, to the next turn start (about 14 playout decisions) | about 0.45 s per core | 64 x 14 x 0.5 ms |
| one game, one side planned (about 225 planned decisions) | about 100 s per core | |
| a 32-core box, one side planned | about 1,100 games per hour | |
| an 800-game gate | under an hour of such a box beside the reference's GPU, about $0.5-1 | |

Every row is to be measured on the first build; a horizon two or four
turns deep costs about two or four times as much.

## Experiment ladder

Each rung is pre-registered (predictions, bars, kill) before it runs.

0. **Build and fit the fast components** (development plus about an hour of
   a GPU box for the big network's labels, about $1). Report the scorer's
   top-1 agreement with the big network on holdout positions, and the
   evaluator's same-turn AUC by phase beside the big head's (0.65-0.93).
1. **The ranking test, no training run.** On the 200 positions of the
   turn-value validation set, each candidate turn scored by the fitted
   evaluator after the opponent's reply played by the scorer, the
   candidate's attacks enumerated exactly, M = 8. Basis: the raw outcomes
   of the stored playouts, measured as `tools/analysis/turn_reads_by_depth.py`
   measures. Bar: corrected correlation 0.45 or more, the reference
   rollouts' HP margin being 0.49 at this read. Kill: below 0.30, the value
   head's level at this read. Between the two, the next read (after our
   own next turn) is measured. This rung is CPU only. Its first half, the
   readings of the reference's own rollouts by depth, is done (above).
2. **The planner against the reference**, 800 decisive games, the planner on
   side A with the reference's own network as its prior, determinized
   under fog. Pass: p 0.535 or more. Kill: below. About $1.
3. **Self-improvement.** Rounds of planner self-play, the evaluator trained
   by temporal differences and the scorer by imitation, each round gated
   by the planner's own 800-game match against the previous round.
4. **Distillation into the big network**, accepted only by the raw
   network's 800-game match against the reference.

## What would prove it wrong

- Rung 1 below 0.30 at every affordable horizon: no evaluation the planner
  can afford ranks turns, and it would be a noisy perturbation of the
  prior.
- Rung 2 failing with rung 1 passing: the scorer's playouts are too unlike
  real play, or the determinization misleads the planner; the measured
  horizon curve and an S sweep tell which.
- Rung 3 not improving: the evaluator does not learn from its own
  planner's targets; the targets would then be mixed more toward outcomes.

## Decision record

- Rejected: a search that calls the big network at its nodes (MCTS,
  Gumbel, turn search graded by the value head), because its cost is the
  network's and its signal the value head's, which ranks alternatives at
  0.22; MCTS-32 lost to argmax 13-27.
- Rejected: policy-gradient self-play of the big network on game outcomes,
  because one bit spreads over about 450 decisions, the project's earlier
  legs never beat their prior, and every experience costs 2.2 ms of
  training; affordable runs would see too few games.
- Rejected: luck-adjusted returns as the main lever, because the lucks
  explained 0.07 of the outcome variance with the graders on record
  (docs/turn_value_prereg_20260925.md).

## References

- G. Tesauro, "Temporal Difference Learning and TD-Gammon", Communications
  of the ACM 38(3), 1995.
- G. Tesauro and G. Galperin, "On-line Policy Improvement using Monte-Carlo
  Search", NIPS 9, 1996: rollouts of a base player reduced its error rate by
  a factor of 5 or more, from a random policy to TD-Gammon.
- Y. Nasu, "Efficiently Updatable Neural-Network-based Evaluation Functions
  for Computer Shogi", 2018.
- B. W. Ballard, "The *-minimax search procedure for trees containing chance
  nodes", Artificial Intelligence 21(3), 1983.
- H. van Seijen, H. van Hasselt, S. Whiteson and M. Wiering, "A Theoretical
  and Empirical Analysis of Expected Sarsa", IEEE ADPRL 2009.
- D. Silver et al., "Mastering the game of Go with deep neural networks and
  tree search", Nature 529, 2016: a fast rollout policy about 24% accurate
  and a thousand times faster than the policy network, mixed with the value
  network at λ = 0.5.
- A. P. Jacob et al., "Modeling Strong and Human-Like Gameplay with
  KL-Regularized Search" (piKL), ICML 2022.
