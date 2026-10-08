# A CPU-bound turn planner as the self-play teacher (design, 2026-10-07)

**Status: proposal, revised after an independent review the same day;
awaiting the user's decision.** Nothing here is built. The review's findings
and what each changed are in the decision record at the end.

The user's ruling of 2026-10-06: before any self-play training, a new
algorithm must handle the reward's sparsity, multi-step turns whose plans
depend on dice rolled inside the turn, and the simulator's speed without
being bottlenecked on the network. Self-play keeps the memory (ruling of
2026-10-05).

The proposal moves the search off the GPU. A small action scorer and a
material evaluator run inside the Rust core at CPU speed. A turn planner
built on them re-plans after every action, plays each candidate's future out
past the opponent's reply, and leaves the network's choice only when a paired
test says its candidate is better by more than the noise. The big network,
memory included, supplies the candidates, the decode and the hidden units'
whereabouts once per decision, and learns by imitating the planner once the
planner beats the reference. The first rungs are cheap gain tests with fresh
dice, the first of them needing no new code; they bound what any planner of
this kind can gain before the Rust is written.

## What the record constrains

Each number names its player. The reference is `parity3` at `raw:t0+eo-1.5`
with 64 memory slots (ruling 2026-10-04), about +270 Elo over `obs8` by chain
(+191, then +82), not measured directly. The turn-level evidence predates it:
`obs8` (turn value, 2026-09-25) and `terrain` (turn gaps, 2026-09-23).

- **One outcome bit per game.** About 9 decisions per side-turn at the
  reference decode (docs/box_specs.md, the end_turn offset sweep); decisive
  games ended at a median of 24-28 turns in a 2026-09-05 seed-era match, and
  144 of the 944 games of `parity3`'s acceptance match reached the turn cap
  (docs/imitation_anneal_prereg_20261003.md).
- **Ranking alternatives at one position** (`obs8`, 200 holdout positions, 5
  candidate turns each, 28 playouts per candidate by `obs8` at temperature
  0.5 on the true state; docs/turn_value_prereg_20260925.md, "Measured",
  luck-adjusted basis): the value head 0.22, a fitted linear grader 0.27, the
  HP margin right after the turn 0.40, the head after `obs8`'s own rollouts
  read at the mover's next turn start 0.27, two turns later 0.41-0.42, four
  turns later 0.53. The truth's reliability is 0.58 at 28 playouts.
- **Choosing by material after the network's rollouts gains, from two reads
  on.** Same playouts, the candidate the read ranks first against the base
  turn (the player's own), truth the other 20 playouts with their own luck
  removed: the HP margin gains +0.011 +- 0.017 right after the turn, +0.031
  +- 0.020 at the mover's next turn start (read 1), +0.049 +- 0.017 at read
  2, +0.055 +- 0.017 at read 3, +0.070 +- 0.016 at read 7, in outcome units
  per turn; its gain over the static read is resolved from read 2 on (+0.038
  +- 0.015). The value head gains nothing at reads 0-1 and +0.02 at reads
  2-3 (`tools/analysis/turn_reads_by_depth.py`, record
  `training/metrics/turn_value_20260925/reads_by_depth_20261007.txt`). Three
  caveats bound these numbers from above: every playout starts from the
  candidate turn's realized dice, so the turn's own luck stays in the truth;
  the margin counts hidden units; the player is `obs8`. The head also reads
  markedly worse at the mover's own turn starts than at the opponent's
  (correlation 0.31 at read 1 against 0.52 at read 2), with and without
  hidden units at the root by the review's split; the reason is not known.
- **Better turns exist but are rare** (`terrain`): 7 of 60 positions have a
  sampled alternative confirmed better by at least 0.25, and in 6 of the 7
  the better turn takes more decisions (14.4 against 10.3)
  (docs/turn_gap_ref_prereg_20260921.md; positions 15 and 57 carry luck and
  are to be re-realized, BACKLOG.md). Acting more is the largest single gain
  on record: +229 Elo for the end_turn offset of -1.5.
- **Dominating attack rewrites are rare.** The exact enumerator finds an
  attack that dominates the human's on every count 0.98 times per 1,000
  decisions in 300 human games, worth a few Elo at most (branch
  `exp/xod-dominance`, f3aecd9). Trade-off choices and the reference's own
  play are not measured.
- **The network's cost.** About 3,200 leaf evaluations per second per 4090
  at batch cap 64, measured on `relset` without memory
  (docs/serve_batch_prereg_20260920.md); the trainer costs 2.2 ms per
  experience. A search that calls the network at every node lost to plain
  argmax at the budget tried (MCTS-32, 13-27; docs/raw_argmax_control_20260904.md).
- **The simulator's cost.** In the Rust core a fork costs 0.019 ms and a
  step 0.050 ms (2026-09-12). An attack's exact outcomes: 63 µs median
  through the Python binding on the outcome test's boards (random HP; 3
  distinct outcomes at the median), and 16 outcomes on a level-1 fight in
  the 2026-09-04 microbenchmark. A midgame decision has 662-677 legal actions, whose
  enumeration costs 1.07-1.35 ms in Python (docs/box_specs.md, 2026-09-13).
- **Fog.** A search on the true state sees through fog
  (docs/hidden_information_20260926.md); fog was on in 197 of the 200
  positions above. The memory model's belief head gives independent per-hex
  probabilities of a hidden enemy unit on the relevant-set hexes, better
  than the last-seen baseline (holdout 0.0156 against 0.0489 nats, the
  anneal run's probe records).
- **Common random numbers were measured dead.** Across branches, the median
  number of shared downstream fights was 0 under every keying tried
  (docs/selfplay_redesign_20260904.md, its Q8 entry), measured on a collapsed
  checkpoint that rolled a median of 0 dice per turn.

## The algorithm

### Fast components, in Rust

- **The evaluator is material first.** The HP margin, the read with the
  measured gain, is the default. A fitted evaluator (unit value weighted by
  HP, villages, gold and income, leader safety, terrain held, time of day)
  replaces it only after beating it on the gain test below, paired.
- **The action scorer** scores every legal action from a short feature
  vector (action kind; unit type, level and HP; the destination's defense;
  village capture; distance to the enemy leader; for attacks, damage dealt
  and taken from the fight's expected values, the exact outcomes being kept
  for the root). It plays both sides in the playouts: our side
  continues the turn, the opponent replies. As the opponent it takes every
  attack whose expected exchange favours it, so exposures are priced. Its
  turn length is matched to the reference decode's (about 9-11 decisions per
  side-turn). It is judged by the gain its playouts give the planner, not by
  agreement with the network: a rollout policy's balance matters more than
  its strength (Silver and Tesauro 2009).

### The turn planner, at each of our decisions

1. **One forward of the big network**, memory included: the prior under the
   reference decode (end_turn offset -1.5 included), the value, the belief.
2. **Candidates**: the prior's choice, end_turn, and the next most probable
   actions up to k (about 8).
3. **Worlds**: when hidden enemies exist, S worlds (2-4). Each known hidden
   unit is placed once, by the belief head's probabilities restricted to the
   hexes it could have reached since it was seen, with the type and HP the
   sighting record holds; unseen recruits are drawn from the enemy's recruit
   list and its gold, which is sampled around the income the side can infer.
4. **Futures**: for each candidate and world, M futures to the horizon,
   default the opponent's next turn start after our next turn (read 2), the
   horizon chosen by the gain test. Every candidate's chance is treated
   alike: its first fight, if it has one, is followed through each exact
   outcome with its probability, and every later fight is sampled. Random
   numbers are keyed by world, future and fight, so the M futures differ;
   no candidate shares dice with another.
5. **Selection**: each candidate's mean and standard error over its futures
   and worlds. The planner plays the prior's choice unless a candidate beats
   it by more than z standard errors of the difference (z pre-registered).
   This replaces a fixed-λ piKL rule, which ignores the estimates' noise and
   favours the noisiest candidates.
6. **Re-planning**: the action is played and the engine rolls its dice. At
   the next decision the planner keeps the futures whose dice the engine
   matched and draws new ones for the rest.

Plans conditional on the dice come from steps 4 and 6: the next fight's
outcomes are branched exactly, the rest are sampled, and nothing is committed
past one action.

### Learning

- **The evaluator**, once fitted ones are allowed, learns from every node the
  planner searched, not only the turn boundaries of the games (TreeStrap,
  Veness et al. 2009). Its targets are expectations under the planner's own
  policy, or the played action's value re-estimated on fresh futures, never
  the maximum over noisy candidates, which is biased upward. Expected updates
  carry less variance than sampled ones (Sutton and Barto 2018, section
  8.5); the game's outcome anchors the chain by a λ-return. This is where the
  sparse reward is handled.
- **The scorer** learns from the planner's choices, accepted by the gain
  test.
- **The big network** learns, once the planner beats the reference, by
  imitating the planner's decisions through the memory-stream trainer,
  anchored to the reference by a KL term, its value head on the planner
  games' outcomes. It is accepted by its own raw 800-game match against the
  reference.

## Cost

The playout decision dominates: about 670 legal actions to enumerate and
score, in Rust; fights resolved by their expected values; one step. Each
planned decision runs k x S x M futures of about 20-30 playout decisions to
read 2 or 3, plus the root's exact branches. With k = 8, S = 2-4, M = 8 that
is 130-260 futures, 3-7 seconds per planned decision on one core if a
playout decision costs about 1 ms. At about 225 planned decisions per game
side, a 32-core box plans roughly 80-160 games per hour with one side
planned, fewer once the games that run to the turn cap are counted: about an
order of magnitude below the first draft's table. The figure is to be
measured on recorded `parity3` positions before anything is priced. The
levers are the selection rule's cut (most decisions keep the prior's choice
and can stop early), reused futures, and fewer worlds where little is
hidden.

## Experiment ladder

Each rung is pre-registered (predictions, bars, kill, multiplicity) before
it runs.

0. **The control, with no new code: the network as its own planner.** The
   planner above with `parity3` in place of the scorer (its proposals, its
   rollouts) and plain material as the evaluator, at a reduced budget
   (k = 4, M = 4), measured by the gain test below. Its gain bounds the CPU
   planner's from above, since the CPU planner plays the same rule with
   weaker rollouts: if it gains nothing against `parity3`'s own turns, the
   CPU planner will not, and the design stops before the Rust is written. It
   is network-bound by construction and cannot be the product. Its cost
   depends on how much of the turn it plans: planning every decision of the
   turn costs about 8,000 network calls per playout, several hours of a
   4090 (a few dollars) at 200 positions and 40 playouts per arm; planning
   the turn's first decision only costs about a tenth. The pre-registration
   prices the choice.
1. **The gain test**, the protocol of every rung that grades a planner
   before a match. At the 200 validation positions, re-realized with
   `parity3`: the planner's whole turn against `parity3`'s own turn, each
   played closed-loop with fresh dice per playout, the game continued by
   `parity3` for both sides, the two arms seeded alike. Truth: the mean
   outcome over N playouts per arm (N about 40, a standard error of about
   0.013-0.015 per turn by the review's estimate). Pass: a gain of at least
   +0.03 and at least the static margin's choice's; kill: a gain at or below
   zero, or below the static margin's. The turn-value playouts bound what to
   expect (+0.03 to +0.06 per turn, with the turn's own dice credited).
2. **Build, time, and test the CPU planner.** The Rust scorer and planner;
   the full planned step timed on recorded `parity3` positions; then the
   gain test with paired controls under the same protocol: the static
   margin's choice, the network-rollout control (rung 0), and the HP margin
   after the scorer's rollouts. The horizon is chosen here by gain. Its CPU
   cost is priced from the timing.
3. **The planner against the reference**, 800 decisive games, the planner on
   side A with `parity3` as its prior. Pass: p 0.535 or more. Decisions per
   side-turn reported. The selection threshold, the horizon and S are chosen
   on disjoint seeds beforehand, and the result is confirmed on fresh seeds.
4. **Self-improvement**, with a games budget per round fixed in advance;
   every round is gated against the reference, not against the previous
   round.
5. **Distillation into the big network**, accepted by the raw network's
   800-game match against the reference.

## Alternatives

- **Whole turns proposed by the network, evaluated on the CPU** (a hybrid):
  the network samples k whole turns, and the CPU branches their fights and
  plays the scorer's reply. Multi-action successes elsewhere searched sampled
  whole turns (docs/literature_sparse_signal_20260921.md); it costs about k
  times plain play on the GPU, which the ruling weighs against. Rung 0
  measures its upper bound too.

## What would prove it wrong

- Rung 0 gaining nothing: no planner of this kind beats `parity3`'s own
  turns at this horizon, whatever plays its rollouts.
- Rung 2's gain below the static margin's with rung 0 above it: the
  scorer's playouts lose what the network's rollouts carry.
- Rung 3 failing with rung 2 passing: the per-turn gain does not add up over
  a game, or the worlds mislead the planner; the S curve and decisions per
  side-turn tell which.
- Rung 4 not improving: the evaluator does not learn from its own
  planner's targets.

## Decision record

From the review of 2026-10-07, which changed the first draft as follows:
- Rejected: a first test that correlates grades with raw outcomes, because
  the static post-turn margin passes its 0.45 bar with no rollout (0.456) and
  raw outcomes credit the turn's own dice, which the turn-value rules
  already rejected. Replaced by the gain test with fresh dice.
- Rejected: one turn pair as the default horizon, because material's gain
  over the static read is not resolved there (+0.021 +- 0.012) and is from
  read 2 on, and the value head reads poorly at the mover's own turn starts
  for reasons not understood.
- Rejected: a fixed-λ piKL selection, because it ignores the estimates' noise
  and, with the first fight enumerated for attacks only, favours the
  noisiest candidates. Replaced by a paired deviation test with equal
  treatment of chance.
- Rejected: per-fight common random numbers across candidates, because
  common random numbers were measured dead under every keying and the draft
  did not answer that record. They may be re-measured on `parity3` games
  before being reconsidered.
- Rejected: accepting the scorer by its agreement with the network, because
  rollout quality is about balance, not strength.
- Rejected: an evaluator fitted to the value head at the mover's turn
  starts, until the head's weakness there is explained.
- Rejected: gating self-improvement rounds against the previous round,
  because chained comparisons are outside the standing rules and an 800-game
  match cannot resolve per-round gains of 10-20 Elo.

Standing:
- Rejected: a search that calls the big network at its nodes as the product,
  because its cost is the network's; the network-rollout control of rung 0
  is kept as a bound, not as the design.
- Rejected: policy-gradient self-play of the big network on game outcomes,
  because one bit spreads over hundreds of decisions, the project's earlier
  legs never beat their prior, and every experience costs 2.2 ms of
  training.
- Rejected: luck-adjusted returns as the main lever, because the lucks
  explained 0.07 of the outcome variance with the graders on record.

## References

- G. Tesauro, "Temporal Difference Learning and TD-Gammon", Communications
  of the ACM 38(3), 1995.
- G. Tesauro and G. Galperin, "On-line Policy Improvement using Monte-Carlo
  Search", NIPS 9, 1996: rollouts of a base player reduced its error rate,
  by as much as a factor of 5, for base players from a random policy to
  TD-Gammon.
- D. Silver and G. Tesauro, "Monte-Carlo Simulation Balancing", ICML 2009.
- J. Veness, D. Silver, W. Uther and A. Blair, "Bootstrapping from Game
  Tree Search" (TreeStrap), NIPS 2009.
- B. W. Ballard, "The *-minimax search procedure for trees containing chance
  nodes", Artificial Intelligence 21(3), 1983.
- R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction,
  2nd edition, 2018, section 8.5 (expected against sample updates).
- D. Silver et al., "Mastering the game of Go with deep neural networks and
  tree search", Nature 529, 2016: a fast rollout policy about 24% accurate,
  mixed with the value network at λ = 0.5.
- A. P. Jacob et al., "Modeling Strong and Human-Like Gameplay with
  KL-Regularized Search" (piKL), ICML 2022.
- Y. Nasu, "Efficiently Updatable Neural-Network-based Evaluation Functions
  for Computer Shogi", 2018: an evaluator trained on search scores and
  updated incrementally inside the search.
