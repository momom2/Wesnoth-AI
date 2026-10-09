# Self-play program: value-guided policy iteration with exact chance (2026-10-08)

**Status: program adopted by the project lead (2026-10-08), revised the same
day after an independent review (decision record at the end); step 1
pre-registered below, not run.** It answers the user's ruling of 2026-10-06:
before any self-play training, an algorithm must handle the reward's
sparsity, multi-step turns whose plans depend on dice rolled inside the
turn, and the simulator's speed without being bottlenecked on the network.
Self-play keeps the memory (ruling of 2026-10-05).

## The program in one paragraph

At each decision the policy proposes its few most probable actions; the
simulator builds, for each, every state it can lead to, with the exact
probability of each combat outcome; an evaluator values those states; the
policy's prior is tilted toward the actions whose expected value is higher,
by a bounded amount. The tilted player is first matched against `parity3`,
then distilled into the policy network; the evaluator is retrained on the new
player's games, and the round repeats, each round gated by a match against
the reference. This is policy iteration with a one-step look-ahead (Tesauro
and Galperin 1996), with Muesli's clipped target (Hessel et al. 2021) and the
exact chance nodes Wesnoth's combat offers. The evaluator is the open
question: a critic learned from games, which would make the operator cheap
and let it compound, or short rollouts scored by material, the only evaluator
on record with a resolved gain. Step 1 measures the learned critic on data
already on hand.

## Why this program

**What the record says** (CLAUDE.md "Self-play"; docs/turn_value_prereg_20260925.md
on branch `exp/turn-value`):
- Every self-play attempt distilled a search whose evaluations came from the
  policy's own value head, trained on the outcomes of 17,000 human games. On
  the turn-value benchmark that head ranks candidate turns at 0.28 (corrected
  within-position correlation, raw outcomes; 0.22 on luck-adjusted outcomes)
  read right after the turn, below the static HP margin (0.46 raw, 0.40
  adjusted). Read after two to seven half-turns of `obs8`'s own play, the same
  head ranks at 0.45-0.62 (raw;
  `training/metrics/turn_value_20260925/reads_by_depth_20261007.txt`). It
  judges who is ahead and anticipates little.
- On the same records a ranking turns into a selection gain over the base
  turn only from about 0.59: the HP margin (0.46) gains +0.015 +- 0.018, the
  head read two half-turns on (0.52) +0.022 +- 0.019, read four half-turns on
  (0.59) +0.053 +- 0.020. Rollouts scored by the HP margin gain +0.038 +-
  0.015 over the static margin from two half-turns on, the only resolved
  evaluator gain on record.
- AlphaGo's value network overfit when trained on every position of human
  games (0.19 train against 0.37 test MSE) and generalized when trained on one
  position from each of 30 million self-play games (Silver et al. 2016).
  Self-play games cost about 8,700 per dollar of acting (`parity3` against
  `parity2`, 944 games in 929 s on a $0.42/h 4090,
  `training/metrics/imitation_anneal_20261003/box/`), but box prices vary:
  the turn-value box billed $0.63/h.
- Training the memory policy costs 4.24 ms per position (`parity3`'s hold
  pass 1). A policy-gradient leg that trains on every decision comes to about
  1,800 games per dollar at one epoch, most of it spent on noisy scalar
  advantages.

**What the literature says** (survey of 2026-10-08, sources at the end):
- Imitation-seeded RL keeps its seed close or loses it: plain fine-tuning of
  a NetHack imitation agent fell from over 5,000 to about 1,000 points early
  in its run, while a KL to the seed doubled it (Wolczyk et al. 2024); VPT
  needed the KL to keep its skills.
- Bridge bidding went from -0.11 to +0.57 IMPs per deal against WBridge5 by
  policy iteration with new policy ∝ prior × exp(V/τ), where V came from
  rollouts scored double-dummy over sampled hidden hands (Lockhart et al.
  2020). Rollout improvement of a backgammon player cut its error 2.5-6.6x
  (Tesauro and Galperin 1996). Both support rollouts as the evaluator; a
  learned critic in that role is this program's own bet.
- In 2048, TD learning with the tile spawn's exact expectation at one ply
  reached the 2048 tile in 0.87 of games and TD on afterstates (the value of
  the state before the spawn, learned from sampled spawns) in 0.91 at a third
  of the CPU, against 0.50 for Q-learning (Szubert and Jaśkowski 2014). An
  afterstate critic, one that values an attack declared and not yet resolved,
  would need one forward per attack instead of one per outcome.
- Subtracting each chance event's exact expectation from a return leaves it
  unbiased whatever the value function (AIVAT, Burch et al. 2018: -68%
  standard deviation in poker). The project's own measurement bounds it here:
  with the graders of the turn-value experiment the lucks explained 0.07 of
  the outcome variance (its verdict record).
- Muesli's clipped target π' ∝ π · exp(clip(Â/σ, -c, c)) moves the policy by
  at most tanh(c/2) in total variation per update, and stays robust when the
  advantages' scale is off by 100x or more (theorem 4.1, figure 4). Its σ is a
  running estimate of the advantages' scale across updates, not a per-decision
  one. Under it an argmax flips only where the prior's top action leads the
  candidate by less than 2c nats.
- A critic that sees the true state alone is biased under partial
  observability; feeding it the agent's history as well removes the bias
  (Baisero and Amato 2022, theorems 4.2 and 5.1). That bears on a critic used
  as a training baseline; a critic that chooses actions in play must read the
  player's observation or worlds sampled from its belief.

**How it meets the ruling of 2026-10-06:**
- *Sparse reward:* the evaluator turns outcomes into a value for every
  candidate at every decision; chance enters as exact expectations.
- *Dice inside the turn:* the operator works one atomic decision at a time,
  closed loop. Each candidate attack is valued over its exact outcome
  distribution, and the next decision sees the dice the engine rolled.
- *Simulator speed:* per decision the policy network runs once, for its
  prior; the operator's work is forks, outcome states and evaluator calls.
  With a learned critic the evaluator's cost is the open number: about k x
  (outcomes per attack) forwards per decision at full size, which step 1
  prices with a small critic; an afterstate critic cuts it to k.

**Scaling.** If a learned critic ranks, more self-play makes it better and
each round's improvement larger, and the look-ahead can deepen (the
opponent's reply, a beam over whole turns) toward expert iteration with exact
chance nodes. If only rollouts rank, the program still runs, at the
rollouts' price.

## The rounds

Round k starts from the policy π_k (π_0 = `parity3`).

1. **Games.** π_k plays itself, raw, under the reference decode, every game
   recorded whole (`tools/game_record.py`).
2. **Evaluator.** The critic is trained on those games (or the rollouts are
   set up with π_k as their player).
3. **Operator.** At each decision: the prior's top k actions and end_turn;
   for each, Q̂ = Σ_o p_o E(s_o) over its exact outcome states; π' ∝ π ·
   exp(clip(Â/σ, -c, c)). Under fog the evaluator reads the player's
   observation, or worlds sampled from the belief head once a joint sampler
   exists (the belief head gives per-hex marginals only).
4. **Procedure gate.** π' at argmax against `parity3`, 800 decisive games
   (rule 2 of the plan: nothing is distilled before it wins).
5. **Distillation.** Cross-entropy of π_{k+1} to π' where they differ, a KL
   to `parity3` that decays, and an imitation loss on the human corpus; then
   the raw policy's own 800-game match against `parity3`.

Each step is one factor with its own pre-registration.

## Step 1: does a critic learned from games on hand rank better than material? (pre-registration)

**Question.** On the turn-value benchmark, does a critic trained on game
outcomes rank candidate turns, and select among them, better than the static
HP margin, and does it improve with the number of games?

**Benchmark** (docs/turn_value_prereg_20260925.md "Validation", HF
`tier-b/turn_value_20260925/`): 199 positions, side 2's turn starts in 200
holdout human games (`configs/bench_states.json`); 966 candidate turns, at
most five per position (`obs8`'s turn at `raw:t0+eo-1.5`, three turns at
temperature 1, a "continue" edit); 28 playouts per candidate by `obs8` at
`raw:t0.5+eo-1.5`, the last 20 the truth. **Truth basis: luck-adjusted
outcomes**, the turn-value rule's own; the raw basis is reported beside it.
The critic reads each candidate's state right after its end_turn (read 0) and
before it (the pre-end_turn state, recorded with its digest); both are
reported, and a pass at either counts.

**Statistics**, all paired over positions with a bootstrap, against the
static HP margin read at the same state:
1. the corrected within-position correlation (reported; its difference with
   the HP margin's is the ranking test);
2. the selection gain over the base turn (the candidate the critic ranks
   first, against `obs8`'s own turn; `selection_gains` in
   `tools/analysis/turn_reads_by_depth.py`) and its difference with the HP
   margin's selection gain (the selection test).
Before any box runs, the implementation simulates the readings' operating
characteristics from the recorded standard errors and reports them beside
this section.

**Data on hand.** **M**: every whole-game record on HF from the matches of
2026-10-01 to 2026-10-04 (`parity_memory_20261001`,
`parity_memory_pass2_20261002`, `imitation_anneal_20261003`): nine matches
of engine players at argmax, about 8,000 games, split 95/5 by game for
training and holdout. **H**: the human corpus at `CORPUS_VERSION` 5, its
manifest's training split; the benchmark's 200 games are in the holdout
split (to be asserted by the builder). The benchmark positions are human,
which favours H; the size curve inside M is the clean test of data volume.

**Critics** (one recipe; the checkpoint chosen by holdout value loss, never
by the benchmark):
- **T25, T50, T100**: true state of both sides (the core's encoder with fog
  off, on a fork), 25%, 50% and 100% of M's training games;
- **O100**: the mover's own observation, no memory, 100% of M;
- **TH**: true state, H;
- **Tsmall**: true state, 100% of M, a network about a quarter of the
  reference's width and depth, to price the operator.
Recipe: `WesnothModel(observation_parity=True, memory_slots=0)` (the full
critics loaded from `parity3` with its memory dropped; the small one from
scratch), the C51 value head on the outcome signed to the side to move, an
auxiliary head on the mover's HP-weighted material margin at its next turn
start; positions drawn from every decision point and every side-turn start,
at most 24 per game with the game's own seed; capped games left out (their
label is undefined; the benchmark's own playouts capped 3.4% of the time).
The auxiliary margin is a raw hit-point sum, the turn-value experiment's HP
margin over the two player sides only: the mover's units' hit points less its
opponent's, neither cost-weighted nor counting a neutral side's units.

**Free readouts** (the same box, no training): at `parity3`'s decisions in 100
of its recorded games, the distribution of the log-prior gap between its top
action and each of its next seven under the reference decode, the share of
decisions where a clip c of 0.5, 1 or 2 could flip the argmax, and the share
of decisions whose top eight hold two or more attacks. They set step 2's k
and c.

**Predictions** (the lead's, before any number): T100 above the HP margin in
correlation by 0.05-0.15 and in selection gain by 0 to +0.03; the size curve
rising (T100 above T25 by 0.03-0.08); O100 below T100 by 0.03-0.10; TH
between T50 and T100; Tsmall within 0.05 of T100.

**Readings**, each with its next action:
- **Pass:** for some critic at one read (read 0 or before end_turn), both its
  selection gain and its corrected correlation exceed the HP margin's by 2
  paired standard errors, on the luck-adjusted basis. Step 2 builds an attack-only
  operator with that critic (its observation form if O100 passes; the true
  form needs the world sampler first) and pre-registers its 800-game gate
  against `parity3`.
- **Data-limited:** otherwise, if T100 beats T25 in correlation by 2 paired
  standard errors. More games are the lever: 20,000 `parity3` self-play games
  and their critic, priced for the user (about $3.5 at $0.63/h).
- **Kill:** otherwise. A learned critic at this scale does not select better
  than material and more data is not shown to help; the evaluator is
  rollouts, and the next measurement is the CPU planner's rung 0 with fresh
  dice (docs/selfplay_algorithm_design_20261007.md), priced for the user.
  The correlations are reported either way.

**Operating characteristics** (simulated 2026-10-08, before any box run:
`tools/critic_oc.py`, record
`training/metrics/critic_step1_20261008/operating_characteristics.json`).
A synthetic benchmark of the real one's 199 positions and 966 candidates,
calibrated on its records: candidate values with the luck-adjusted truth's
within-position spread (sd 0.167), concentrated on a few positions as the
records' is (the HP margin's simulated standard errors 0.09 in correlation
and 0.020 in selection gain, against 0.098 and 0.017 recorded), the base
turn 0.042 ahead of the alternatives, each candidate's own playout noise,
the HP margin's correlation 0.377; error correlations of 0.12 between a
learned head and the margin and 0.27 between a head's two reads (`obs8`'s,
measured), and 0.5 between two critics (assumed; 0.2 and 0.8 in the first
row). Each draw goes through the readout's own statistics and readings, 80
draws a row, under the Pass reading above:

| critics' true correlation minus the margin's | Pass | Data-limited | Kill |
|---|---|---|---|
| 0 for all (0.2 to 0.8 between critics) | 0.00-0.05 | 0.04-0.10 | 0.85-0.96 |
| +0.05 for all | 0.14 | 0.09 | 0.78 |
| the predictions (T100 +0.10, T25 +0.05) | 0.17 | 0.24 | 0.59 |
| +0.20 for T100, +0.12 for T25 | 0.57 | 0.14 | 0.29 |
| below it, size +0.10 (T100 -0.05, T25 -0.15) | 0.01 | 0.40 | 0.59 |
| -0.13 for all | 0.00 | 0.04 | 0.96 |

Pass takes the best of twelve (critic, read) pairs, each needing both tests:
with no critic better than the margin it fires on at most 0.05 of the draws
(4 of 80 at 0.8 between critics, about +-0.025 from the draw count), so its
bar stays at 2 paired standard errors, the smallest step of 0.25 that keeps
it there. A critic 0.10 above the margin in correlation selects about 0.02
better, one paired standard error (0.019): the predicted effects pass 0.17
of the time, +0.20 passes 0.57. A size effect of 0.10 in correlation reads
Data-limited 0.40 of the time (the paired standard error of T100 against T25
is 0.06-0.10). The simulation treats the HP margin as continuous, while in the
benchmark 83 of the 199 positions tie at the margin's top (74 of them with the
base turn among the tied), where the margin picks the base turn.

**Cost.** One box session: rebuilding and encoding about 22,000 games' sampled
positions (CPU), six critic trainings of 10-40 minutes each, the readouts in
minutes: about 3 box-hours, $1.3-1.9 at $0.42-0.63/h. The Vast balance on
2026-10-08 is $4.35.

## Step 1, measured (2026-10-09): Kill

Box 55000517 (RTX 4090, EPYC 7B13, $0.447/h; the first box, 54996163,
stopped at its test step on a PyTorch 2.5 incompatibility fixed in 0.18.1),
stage `tier-b/staging/stage_20261009_q1b.tar.gz`, records in
`training/metrics/critic_step1_20261009/` (`readout.md`, `readout.json`,
`prior_gaps.json`, the critics' summaries; checkpoints and positions on HF
`tier-b/critic_step1_20261009/`). 509,989 positions from 21,485 games (M:
7,200 engine-match games, 217 capped left out; H: 14,064 human games); the
benchmark rebuilt in full (966 candidates, no error, no digest mismatch, no
benchmark game in H's training rows). Every critic ended by holdout patience
early (T100 in 14 minutes, its best holdout loss half way through its
first epoch).

**The reading is Kill:** no critic clears both paired tests against the HP
margin. On the luck-adjusted basis, at the state after the candidate turn:

| grader | corrected r | minus HP margin | selection gain | minus HP margin |
|---|---|---|---|---|
| HP margin | 0.380 +- 0.097 | - | -0.010 +- 0.017 | - |
| T100 (true state, all of M) | 0.518 +- 0.056 | +0.137 +- 0.093 | +0.031 +- 0.019 | +0.041 +- 0.017 |
| T50 | 0.493 +- 0.071 | +0.113 +- 0.064 | +0.024 +- 0.018 | +0.034 +- 0.016 |
| T25 | 0.498 +- 0.058 | +0.117 +- 0.077 | +0.030 +- 0.018 | +0.040 +- 0.018 |
| O100 (the mover's observation) | 0.396 +- 0.056 | +0.015 +- 0.077 | +0.014 +- 0.017 | +0.024 +- 0.017 |
| TH (true state, human corpus) | 0.351 +- 0.047 | -0.029 +- 0.106 | +0.024 +- 0.018 | +0.034 +- 0.018 |
| Tsmall | 0.218 +- 0.067 | -0.162 +- 0.082 | -0.012 +- 0.018 | -0.002 +- 0.019 |

T100 against T25: correlation +0.020 +- 0.047, selection gain +0.001 +- 0.016.
Before end_turn every critic reads lower (`readout.md`).

What the numbers say beyond the reading:
- The true-state critics trained on engine games select better than the HP
  margin by 2.1 to 2.4 paired standard errors (T25, T50, T100, nested data,
  not independent); their correlation differences are 1.5 to 1.8 standard
  errors, held under 2 by the HP margin's own spread (+-0.097). A selection
  test alone, the rule before the operating characteristics tightened it,
  would have read Pass. The binding reading is the pre-registered one.
- The critic a fair player can use does not beat material: O100, the
  mover's own observation, selects +0.024 +- 0.017 over the HP margin and
  ranks with it. What the true-state critics see that O100 does not is the
  hidden units; a fair operator would need them from a belief.
- More games of the same kind do not help at this scale: the size curve is
  flat from 1,700 to 6,800 training games, while the holdout value loss on
  the critics' own games falls with them (0.492, 0.471, 0.460 for T25, T50,
  T100). More data predicts outcomes better and ranks alternatives no
  better.
- Human games make a worse critic for this benchmark than engine games (TH
  below T25 on eight times the games): the truth is `obs8`'s continuation,
  which engine games predict and human games do not.
- The prior-gap readout (27,171 decisions of `parity3` in 100 of its match
  games, its replayed choice agreeing with the recorded one at 98.2%): the
  median log-prior gap between the top action and the second is 0.59 nats,
  the eighth 2.18; a clip of c = 1 could flip the argmax at 88% of decisions;
  25% of decisions hold two or more attacks in their top eight; the top
  action is end_turn at 15%.

**Consequence, as pre-registered:** the learned-critic operator is not
built at this scale; the evaluator is rollouts, and the next measurement is
the CPU planner's rung 0 with fresh dice, priced for the user. Under $1 for the box.

## Decision record

- Adopted: one-step look-ahead over the prior's top actions with exact chance
  nodes and a clipped target as the improvement operator, its evaluator chosen
  by measurement.
- Revised after the review of 2026-10-08:
  - The first draft described the benchmark's positions as `obs8`-against-
    `terrain` games and its candidates as four; they are human holdout games
    and five. The arm comparison it built on that (human corpus, engine
    matches, new self-play games) confounded the source with the size, so the
    size curve now runs inside one source and no new games are generated in
    step 1.
  - Rejected: a pass bar on the absolute correlation (0.50), because the
    static HP margin reads 0.46 for free and correlations at that level have
    not turned into selection gains on these records. Replaced by paired tests
    against the HP margin, on the turn-value rule's luck-adjusted basis.
  - Added: the observation-input critic, the small critic, the pre-end_turn
    reading and mid-turn training positions, because the operator reads
    mid-turn states, under fog, at a price; and the free prior-gap readouts,
    because an argmax flips under the clipped target only where the prior's
    gap is under 2c.
  - Corrected: the citations of Bridge and Tesauro and Galperin support
    rollouts, not a learned critic; the 2048 result favours a pre-chance
    afterstate value; the luck term's measured share here is 0.07.
- Revised before any run, from the operating characteristics: Pass needed
  only the selection-gain test, and fired on 0.09-0.14 of the null draws (the
  best of twelve paired tests). It now needs, for one critic at one read, both
  the selection-gain and the correlation difference with the HP margin above
  2 paired standard errors: 0.00-0.05 under the null, 0.17 at the predictions,
  0.57 at T100 +0.20. Data-limited and Kill are unchanged.
- Adopted: the critic's auxiliary head trains its trunk (it reads the global
  token with its gradient). The 2026-09-01 detach protects the policy's trunk
  from the auxiliary heads; a critic is a network of its own with no policy to
  disturb, and an auxiliary head that cannot reach the trunk would leave the
  critic's value unchanged.
- Parked, not rejected: the CPU turn planner
  (docs/selfplay_algorithm_design_20261007.md, landed on `main` with this
  document). Its rung 0 with fresh dice is step 1's kill branch.
- Parked: a policy-gradient leg (PPO or V-trace from `parity3`). It needs a
  critic only as a baseline, but trains the 4.24 ms memory policy on every
  decision for a scalar signal: about 1,800 games per dollar.
- Rejected: training the policy's own value head further (shared trunk), the
  pathology of the 2026-09 legs (the value gradient owned 94-99% of each
  update).
- Rejected: atomic MCTS with network evaluations at every node, measured
  -124 Elo against argmax at 32 simulations.

Sources: R. Wolczyk et al., "Fine-tuning Reinforcement Learning Models is
Secretly a Forgetting Mitigation Problem", ICML 2024, figure 3a; B. Baker et
al., "Video PreTraining", 2022; E. Lockhart et al., "Human-Agent Cooperation
in Bridge Bidding", 2020, table 1 and algorithm 2; G. Tesauro and G.
Galperin, "On-line Policy Improvement using Monte-Carlo Search", NIPS 1996;
M. Szubert and W. Jaśkowski, "Temporal Difference Learning of N-Tuple
Networks for the Game 2048", CIG 2014, tables I-II; R. S. Sutton and A. G.
Barto, Reinforcement Learning: An Introduction, 2nd edition, section 8.5; N.
Burch et al., "AIVAT", AAAI 2018; M. Hessel et al., "Muesli", ICML 2021,
theorem 4.1, section 4.5 and figure 4; A. Baisero and C. Amato, "Unbiased
Asymmetric Reinforcement Learning under Partial Observability", AAMAS 2022;
D. Silver et al., "Mastering the game of Go with deep neural networks and tree
search", Nature 2016.
