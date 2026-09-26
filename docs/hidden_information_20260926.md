# Hidden information in the search and the playouts (2026-09-26)

The legality mask and the encoder respect fog: every action type reads
only visible units, visibly occupied hexes and the side's own rejections,
and the encoder sees only visible units and the village owners a player
would see. The search and the playouts do not. Every search, turn-search
and playout path runs on the true game state, with no step that samples
what the side cannot see, so a searched player, and a measurement that
grades turns by playouts, see through fog. Raw players, the reference
included, never read the simulator's state and are unaffected; so is every
strength verdict since 2026-09-04.

Found by a read-only crawl on 2026-09-26; the code claims below were
checked against the source, and the counts come from
`tools/analysis/hidden_at_boundary.py` (record:
`training/metrics/hidden_information_20260926/boundary_hidden.jsonl`).

## What reads the true state

- The match harness and the self-play loop hand each player the live
  simulator (`tools/eval_players.py:132,147`, `tools/selfplay_game.py:331`,
  `tools/turn_gap.py:276`). Raw players ignore it.
- MCTS and the Gumbel root fork it at the root (`tools/mcts.py:1608`) and
  on every edge (`:959`). A node is evaluated from its side to move's own
  observation (`:742`), so at a node where the opponent moves the network
  reads the opponent's view, its own units included, and the value is
  backed up with the sign flipped. The end_turn child is such a node: the
  choice between acting and passing reads what the opponent knows. The
  "determinization" named in `mcts.py` samples dice, not hidden units.
- The turn-commit search (`tools/turn_search.py:115-123,166,363`) and the
  plan tournament (`tools/plan_tournament.py:421-488`) do the same.
- The turn-gap playouts (`tools/turn_gap.py:257-268,441-449`) and the
  turn-value experiment's playout reads (`exp/turn-value`,
  `tools/playout_reads.py:58-97`) grade from the true post-turn state.
- Dynamics as well as values: a simulated move next to a hidden ambusher
  is cut short and reveals it, and a simulated end_turn plays the
  opponent's reply with its real hidden units and real gold.

## Where it is live

- Self-play targets: `az_loop`'s policy targets are this search's visit
  counts (`tools/az_loop.py:437-441`).
- Eval procedures `mcts:`, `puct:`, `tcs:` and the plan tournament (TCS
  and the plan tournament are quarantined; the MCTS core is not).
- Not the verdict path: the reference player (`raw:t0+eo-1.5`) and every
  match against it are raw players.

## What it means for numbers on record

- **Searched players against raw ones on fogged games** carried the
  searcher's view through fog. Where the searched side still lost, the
  conclusion holds more strongly: the seed with Gumbel-MCTS-32 lost to
  the seed at argmax 13-27 (-124 +- 58 Elo, CLAUDE.md, 2026-09-04).
- **The turn-level gap, RICH (7 of 60 confirmed, bar 6)**
  (docs/turn_gap_ref_prereg_20260921.md). The candidate turns were
  proposed fairly, but graded by playouts from the true post-turn state.
  In the 60 positions, all fogged, 198 of 628 enemy units were hidden
  from the mover at the boundary, in 51 positions, and the enemy leader
  in 36. Five of the seven confirmed positions hold 2 to 4 hidden units
  (the leader at 15 and 42); positions 11 and 47 hold none. Confirmations
  do not concentrate where units are hidden (2 of the 9 positions without
  a hidden unit confirmed, 5 of the 51 with one), but the verdict stands
  on 7 against a bar of 6, and losing two turns RICH into SPARSE. The
  direction is an overstatement of what a searcher limited to its own
  view could find; the records cannot bound its size. With the luck
  caveat on positions 15 and 57, the verdict now carries two.
- **The turn-ranking value function, FAIL**
  (docs/turn_value_prereg_20260925.md). The graders read the mover's
  observation before end_turn; the truth came from playouts on the true
  state. Fog was on in 197 of the 200 positions, and 686 of 2,061 enemy
  units were hidden (in 161 positions; the leader in 120). A grader
  limited to the mover's view cannot reach a correlation of 1 even with
  exact playouts, and the rollout grader, which rolls out from the same
  true state as the truth, shares hidden information with it. The FAIL
  verdicts stand; the rollout's lead over the learned graders is partly
  this shared information.

## The fix, and the decision it needs

Every search and grading root is to be determinized: what the side
cannot see (hidden units, the enemy's gold, the owners of fogged
villages) is drawn from a belief, and the search plays that sampled
world. Two standard forms: perfect-information Monte Carlo, one search
per sampled world, averaged (Ginsberg, "GIB: Imperfect Information in a
Computationally Challenging Game", JAIR 14, 2001; Long, Sturtevant, Buro
and Furtak, "Understanding the Success of Perfect Information Monte
Carlo Sampling in Game Tree Search", AAAI 2010, on when it works), and
information-set MCTS, one sampled world per simulation (Cowling, Powley
and Whitehouse, "Information Set Monte Carlo Tree Search", IEEE TCIAIG
4(2), 2012). Both inherit strategy fusion: a plan that is right in every
sampled world can still be wrong in the information set (Frank and
Basin, "Search in games with incomplete information: a case study using
Bridge card play", Artificial Intelligence 100, 1998).

The belief model is the design decision: hidden units at their last seen
hexes advanced by their reach, spread uniformly over the fogged hexes
they could reach, or a model learned from the corpus. Until it exists,
a searched procedure on fogged games can be tagged so its numbers are
not read as fair strength. Phase 2's turn search needs the determinized
root before its 800-game gate: a searcher that sees through fog can pass
the gate for the wrong reason and then fail to distill, since the
student cannot see what the teacher used.

## Smaller findings

- The enemy's faction is observed from turn 1 even when the opponent
  chose Random, which the game does not show a player until a sighting
  (also docs/observation_parity_20260926.md). Both players get it, so no
  recorded Elo is biased.
- `Observation` (`wesnoth_ai/observe.py:257-288`) keeps rows for hidden
  units, and `detached()` (`:202-208`) keeps them on the encoded
  observation; every consumer reads only own acting units and visible
  ids, and the leaf wire omits the observation. Latent.
- Shaping terms in `wesnoth_ai/rewards.py` that read hidden state (the
  distance to the enemy leader, commented as a deliberate breach; the
  nearest-enemy distance; village majority; units lost) sit on the
  quarantined REINFORCE path with weight 0 or absent from
  `configs/reward_selfplay.json`. Inert today.

## The contract as written

CLAUDE.md principle 6 bars god view in the legality mask and says
nothing of search or playouts that choose or grade actions; the survey
docs/literature_sparse_signal_20260921.md reads it that way, which is
right for a training-time critic and wrong for a search that plays. The
statements below were wrong about the setting and are corrected with
this document: the survey and its search notes called the game
perfect-information (the Ladder games have fog); the turn-search config
and `sim_self_play`'s help called the mover frame "the mover's own
information set" (only the encoding is; the state was reached against
the true hidden units); the turn-gap record described `value_post` as
read from the mover's side (the head reads the opponent's observation,
and only the sign is the mover's); the encoder said the state collector
already drops hidden units from `gs.map.units` (true on the live bridge
only; in the simulator the observation filters them).
