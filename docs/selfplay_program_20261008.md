# Self-play program: value-guided policy iteration with exact chance (2026-10-08)

**Status: program adopted by the project lead (2026-10-08); step 1
pre-registered below, not run.** It answers the user's ruling of
2026-10-06: before any self-play training, an algorithm must handle the
reward's sparsity, multi-step turns whose plans depend on dice rolled inside
the turn, and the simulator's speed without being bottlenecked on the
network. Self-play keeps the memory (ruling of 2026-10-05).

## The program in one paragraph

A critic, a network separate from the policy, learns from self-play games
what a position is worth. At each decision the policy proposes its few most
probable actions; the simulator builds, for each, every state it can lead to
with the exact probability of each combat outcome; the critic values those
states; the policy's prior is tilted toward the actions whose expected value
is higher, by a bounded amount. That tilted policy is first measured as a
player against `parity3`, then distilled into the policy network, and the
critic is retrained on the new player's games. Each round is gated by a
match against the reference. This is policy iteration with a one-step
look-ahead (Tesauro and Galperin 1996; TD-Gammon), with Muesli's clipped
target (Hessel et al. 2021) and the exact chance nodes Wesnoth's combat
offers. Its prerequisite, and its first measurement, is a critic that ranks
alternatives at one position, which no evaluator in this project has done.

## Why this program

**What the record says** (CLAUDE.md "Self-play"; docs/turn_value_prereg_20260925.md):
- Every self-play attempt distilled a search whose evaluations came from the
  policy's own value head, trained on the outcomes of 17,000 human games.
  That head ranks candidate turns at 0.28 (corrected within-position
  correlation, the state right after the turn), below the static HP margin
  (0.46). Read after two to seven half-turns of `obs8`'s own play, the same
  head ranks at 0.52-0.62 (`tools/analysis/turn_reads_by_depth.py`, record
  `training/metrics/turn_value_20260925/reads_by_depth_20261007.txt`). It
  judges who is ahead; it does not anticipate. Anticipation is what a value
  function learns from bootstrapped or many-game targets.
- AlphaGo's value network overfit when trained on every position of human
  games (0.19 train against 0.37 test MSE) and generalized when trained on
  one position from each of 30 million self-play games (Silver et al. 2016).
  Self-play games are cheap here: acting alone, about 8,700 games per dollar
  (`parity3` against `parity2`, 944 games in 929 s on a $0.42/h 4090,
  `training/metrics/imitation_anneal_20261003/box/`).
- Training the memory policy costs 4.24 ms per position (`parity3`'s hold
  pass 1). A policy-gradient leg that trains on every decision costs about
  1,800 games per dollar at one epoch, a fifth of acting; most of that buys
  noisy scalar advantages.

**What the literature says** (survey of 2026-10-08, sources in the decision
record):
- Imitation-seeded RL keeps its seed close or loses it: plain fine-tuning of
  a NetHack imitation agent fell from over 5,000 to about 1,000 points within
  10% of its budget, while a KL to the seed doubled it (Wolczyk et al. 2024);
  VPT needed the KL to keep its skills; Bridge bidding went from -0.11 to
  +0.57 IMPs per deal against WBridge5 by policy iteration with
  new policy ∝ prior × exp(V/τ) over sampled hidden hands (Lockhart et al.
  2020).
- An exact expectation over chance beats a sampled one when the branching is
  affordable: in 2048, temporal-difference learning over the tile spawns'
  expectation reached the 2048 tile in 0.87 of games against 0.50 for
  model-free Q-learning (Szubert and Jaśkowski 2014). Subtracting each chance
  event's exact expectation from a return leaves it unbiased whatever the
  value function (AIVAT, Burch et al. 2018: -68% standard deviation in poker).
- Muesli's clipped target π' ∝ π · exp(clip(Â/σ, -c, c)) moves the policy by
  at most tanh(c/2) in total variation per update whatever the advantage
  errors, and without the clip performance degraded quickly.
- A critic that sees the true state alone is biased under partial
  observability; feeding it the agent's history as well removes the bias
  (Baisero and Amato 2022, theorems 4.2 and 5.1).

**How it meets the ruling of 2026-10-06:**
- *Sparse reward:* the critic turns one outcome per game, over many cheap
  games, into a value for every candidate at every decision; chance events
  enter its targets as exact expectations, not as sampled luck.
- *Dice inside the turn:* the operator works one atomic decision at a time,
  closed loop. Each candidate attack is valued over its exact outcome
  distribution, and the next decision sees the dice the engine rolled.
- *Simulator speed:* per decision the policy network runs once, for its
  prior; the rest is forks, exact outcome states and critic forwards, and the
  critic is sized to be cheap. Generation costs acting only, and distillation
  trains only on the decisions where the tilted policy differs from the
  prior.

**Scaling.** More self-play makes a better critic, which makes each round's
improvement larger; once the critic ranks well the look-ahead can deepen (the
opponent's reply, a beam over whole turns), and the program becomes expert
iteration with exact chance nodes. The critic is the asset that compounds.

## The rounds

Round k starts from the policy π_k (π_0 = `parity3`) and its critic C_k.

1. **Games.** π_k plays itself, raw, sampling at a low temperature under the
   reference decode, every game recorded whole (`tools/game_record.py`).
2. **Critic.** C_k is trained on those games: the true state of both sides
   (no fog) and the mover's memory as input; targets the outcome, with every
   later chance event's luck removed by its exact expectation; capped games
   bootstrap from the critic at the cap (Pardo et al. 2018); auxiliary heads
   for material and villages a few turns on (KataGo's dense targets); a few
   dozen positions per game.
3. **Operator.** At each decision: the prior's top k actions and end_turn;
   for each, Q̂ = Σ_o p_o C(s_o) over its exact outcome states (one state for
   a move or a recruit); under fog, averaged over S worlds whose hidden units
   are drawn from the policy's belief head; π' ∝ π · exp(clip(Â/σ, -c, c)).
4. **Procedure gate.** π' at argmax against `parity3`, 800 decisive games
   (rule 2 of the plan: nothing is distilled before it wins).
5. **Distillation.** Cross-entropy of π_{k+1} to π' where they differ, a KL
   to `parity3` that decays, and an imitation loss on the human corpus; then
   the raw policy's own 800-game match against `parity3`.

Each step is one factor with its own pre-registration. The operator's
details (k, S, c, the temperature, whether the operator acts on every
decision) are chosen in step 2's pre-registration from step 1's numbers.

## Step 1: does a critic trained on self-play rank alternatives? (pre-registration)

**Question.** At the 199 validation positions of the turn-value experiment,
does a critic trained on game outcomes rank the candidate turns' resulting
states better than `obs8`'s value head (0.28) and the static HP margin
(0.46), and does it improve with the number of games?

**Benchmark** (docs/turn_value_prereg_20260925.md, HF
`tier-b/turn_value_20260925/`): 199 positions from `obs8` against `terrain`
games; 4 or 5 candidate turns each (the base turn at `raw:t0+eo-1.5`, two
turns at temperature 1, a "continue" edit); 28 playouts per candidate by
`obs8` at `raw:t0.5+eo-1.5`, the first 8 used by graders that read
playouts and the last 20 as truth. The critic reads the state right after the
candidate turn (read 0), signed to the mover. Statistic: the corrected
within-position correlation of that experiment, computed by its own code,
with a paired bootstrap over positions for differences between arms. The
candidate's realized dice are part of the state the critic reads and of the
truth, so the measure is a ranking of realized states; it does not credit
the critic with choosing dice.

**Arms.** One recipe, three data sets:
- **H**: the human corpus (`CORPUS_VERSION` 5, the manifest's training
  split), about 14,000 games.
- **M**: every whole-game record on HF from the matches of 2026-10-01 to
  2026-10-04 (`parity_memory_20261001`, `parity_memory_pass2_20261002`,
  `imitation_anneal_20261003`): nine matches, about 8,000 games of engine
  players at argmax. The `obs8`-against-`terrain` games the benchmark draws
  from are excluded.
- **S**: 20,000 new games, `parity3` against itself at `raw:t0.5+eo-1.5`
  with 64 memory slots, the Ladder pool with fog and uniform factions, the
  match's 200-turn cap, recorded whole. A second point at 100,000 games
  (**S100**) runs only if S reads between the kill and the pass, and with the
  user's budget.

**Recipe.** `WesnothModel(observation_parity=True, memory_slots=0)` loaded
from `parity3` with its memory dropped; input the true state of both sides
(the core's encoder with fog off, on a fork); the value head (C51) trained on
the outcome signed to the side to move, capped games as 0 (the benchmark's
own convention), with auxiliary heads for the mover's HP-weighted material
margin and village margin at its next two turn starts; the policy heads
unused. Positions: every side-turn start of either side (the state type the
benchmark reads), at most 24 per game, drawn with the game's own seed. 5% of
each arm's games held out; the checkpoint is chosen by holdout value loss,
never by the benchmark. Step 1 leaves the mover's memory out of the critic
(the state-only form) and leaves luck removal out of the targets: both are
step 2 factors, measured on the same benchmark.

**Predictions** (the lead's, written before any number): H 0.30-0.40, M
0.32-0.45, S 0.35-0.50, each arm at least the one before it.

**Readings.**
- **Pass:** S at 0.50 or more. A critic read at the state after the turn then
  ranks as well as `obs8`'s head read after two to seven half-turns of
  rollouts, at the cost of one forward; step 2 builds the operator.
- **Continue:** S between 0.40 and 0.50 and above H by more than its paired
  standard error: the data helps and the curve is not done; S100 runs.
- **Kill:** S below 0.40, or S not above H by its paired standard error. A
  critic trained on outcomes at this scale does not rank alternatives; the
  program stops at step 1 and the operator is not built. The fallback is
  rollout-graded selection (docs/selfplay_algorithm_design_20261007.md, its
  rung 0 with fresh dice) or a policy-gradient leg whose critic is only a
  baseline.

**Cost.** Arms H and M: data on hand; building the encodings is CPU work and
training is about an hour of a 4090 each (about $1 together). Arm S: 20,000
games at about 3,600 games an hour is 5.5 box-hours, about $2.3, plus its
training. About $3.5 for the three arms; S100 about $12 more. The Vast
balance on 2026-10-08 is $4.35.

## Decision record

- Adopted: one-step look-ahead with a learned critic as the improvement
  operator, because it uses the exact chance nodes and the cheap simulator
  and trains the policy on dense targets; its failure mode (a critic that
  cannot rank) is measured first and cheaply.
- Parked, not rejected: the CPU turn planner
  (docs/selfplay_algorithm_design_20261007.md, landed on `main` with this
  document as a record). Its measured gain (+0.05 to +0.07 outcome units per
  turn with `obs8`'s rollouts) keeps each candidate turn's own dice in the
  truth, so part of it is selection of lucky realizations; its fresh-dice
  gain test is the fallback named in step 1's kill.
- Parked: a policy-gradient leg (PPO or V-trace from `parity3`). It needs no
  critic that ranks, only a baseline, but it trains the 4.24 ms memory policy
  on every decision for a scalar signal: about 1,800 games per dollar. It
  shares this program's critic and luck removal, and becomes the operator if
  step 1 shows a critic good enough for a baseline and not for ranking.
- Rejected: training the policy's own value head further (shared trunk), the
  pathology of the 2026-09 legs (the value gradient owned 94-99% of each
  update).
- Rejected: atomic MCTS with network evaluations at every node, measured
  -124 Elo against argmax at 32 simulations.

Sources: R. Wolczyk et al., "Fine-tuning Reinforcement Learning Models is
Secretly a Forgetting Mitigation Problem", ICML 2024, figure 3a; B. Baker et
al., "Video PreTraining", 2022; E. Lockhart et al., "Human-Agent Cooperation
in Bridge Bidding", 2020, table 1; M. Szubert and W. Jaśkowski, "Temporal
Difference Learning of N-Tuple Networks for the Game 2048", CIG 2014, tables
I-II; N. Burch et al., "AIVAT", AAAI 2018; M. Hessel et al., "Muesli", ICML
2021, theorem 4.1 and figure 4; A. Baisero and C. Amato, "Unbiased Asymmetric
Reinforcement Learning under Partial Observability", AAMAS 2022; F. Pardo et
al., "Time Limits in Reinforcement Learning", ICML 2018; D. Silver et al.,
"Mastering the game of Go with deep neural networks and tree search", Nature
2016; G. Tesauro and G. Galperin, "On-line Policy Improvement using
Monte-Carlo Search", NIPS 1996.
