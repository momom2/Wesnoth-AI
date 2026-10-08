# Pre-registration draft: the look-ahead player against `parity3` (step 2's gate)

**Status: skeleton written by the implementing session on 2026-10-09 from the
built player (`feature/lookahead-player`). Predictions, bars, k, c, sigma and
the arms to run are the lead's, marked TODO. Nothing is run or rented.** It
is step 2 of docs/selfplay_program_20261008.md: the procedure gate of round 0
(step 4 of "The rounds"), after step 1 selects the evaluator.

## Question

Does the look-ahead player, with the evaluator step 1 selects and k and c set
from step 1's prior-gap readout, beat `parity3` at `raw:t0+eo-1.5` with 64
memory slots over 800 decisive games, PURE?

## The player

`tools/lookahead_player.py`, configured by a file in the form of
`configs/lookahead.json`:

- **Prior:** `parity3` with its memory at 64 slots under the reference decode
  (`configs/reference_player.json`), served by the shared inference server as
  in every reference match. The look-ahead calls the policy once per decision,
  on the game's position (tested: the memory equals a raw player's fed the same
  positions).
- **Candidates:** the prior's top k. Step 1's Pass reading asks for an
  attack-only operator: `kinds: "attacks"`, whose candidates are the prior's
  argmax and the attacks among the top k, so the operator runs only at
  decisions whose top k hold an attack, and can play only the argmax or an
  attack.
- **Outcomes:** a move, a recruit (one trait roll on the look-ahead's own
  dice stream, salted per decision) or end_turn (the state after the
  opponent's init_side) gives one state; an attack gives every distinct state
  its fight can end in, with exact probabilities (tested against
  `enumerate_attack_outcomes`). A fight with more than `max_attack_leaves`
  hit/miss sequences (berserk) keeps its prior score, counted.
- **The world it expands in** (`determinization: "observed"`,
  wesnoth_ai/lookahead_world.py): every unit of another side the deciding side
  does not see is removed (its sighting record keeps what it saw); villages are
  owned as the side's display shows them; the opponent's gold and base income
  are set to the side's own (both sides start equal in every pool scenario; the
  side cannot see the opponent's gold under fog), its upkeep and income follow
  from that world; the opponent's fog is recomputed from its visible units and
  its sighting record emptied; a removed leader's side stays alive. The side's
  observation of this world equals its observation of the game, byte for byte
  (tested).
- **Value:** Q(a) = sum over outcomes of p * v, v the evaluator's value for the
  deciding side (the critic's value of an end_turn state, where the opponent
  moves, negated); a state where a leader died takes the game's result.
- **Choice:** argmax of log pi(a) + clip((Q(a) - V) / sigma, -c, c), V the
  prior-weighted mean of Q over the candidates; a candidate whose log-prior gap
  to the argmax exceeds 2c never wins (tested).
- **Evaluator:** the critic step 1 selects, under its own view. TODO (lead):
  if step 1 selects a true-view critic, decide between waiting for the world
  sampler (the program's ruling) and reading it on the observed world, where
  the hidden units it was trained with are absent (686 of 2,061 enemy units
  were hidden at the turn-value benchmark's 200 positions,
  docs/hidden_information_20260926.md).
- **Procedure tag:** `la:critic.<view>:k<k>c<c>s<sigma>+atk+eo-1.5`; the whole
  configuration and the critic's SHA-256 are recorded as `lookahead_a`.

**Parameters** (TODO, lead):
- k: from the readout's log-prior gaps between the top action and the next
  seven (`tools/prior_gaps.py`).
- c: from the readout's share of decisions a clip of 0.5, 1 or 2 could flip.
- sigma: a fixed scale. One way to set it without the gate's seeds: the spread
  of Q - V over a calibration run of the player on other seeds.
- Seed base: TODO, disjoint from every seed step 1 and any calibration used.

## The match

```
python tools/run_elo_batch.py \
    --label-a la_critic --spec-a training/checkpoints/parity3.pt \
    --raw-end-turn-offset-a -1.5 --lookahead-a configs/lookahead_gate.json \
    $(python tools/reference_player.py --flags b) \
    --outdir eval_games/lookahead_gate --games 800 --mcts-sims 0 \
    --raw-temperature-a 0 --raw-temperature-b 0 --seed-base TODO \
    --device cuda --jobs 20 --persistent-workers --shared-inference
python tools/elo_collect.py eval_games/lookahead_gate --no-catalog
```

800 decisive games (capped games are absences), sides alternated, the 21-map
Ladder pool with fog, both factions drawn uniformly, per-game luck. The
look-ahead side's label must differ from the reference's even though both play
`parity3`'s checkpoint.

Recorded per game beside the outcome (`lookahead_telemetry_a`): decisions, the
decisions the operator ran on, the decisions played otherwise than the prior's
argmax by kind (attack, move, recruit, end_turn) and the kind played instead,
candidates by kind, evaluator states and forwards per decision, terminal
states, failed expansions by reason, seconds per decision split into the
prior, the expansions and the evaluator.

## Controls

- **c = 0, the null:** the same configuration with c = 0. The operator runs and
  plays the prior's argmax at every decision (tested decision for decision on
  CPU), so the arm is `parity3` against itself through the look-ahead's
  plumbing: the self-pin `parity3` has not had. TODO (lead): its size, or
  whether the decision-for-decision test is enough.
- **Material, the arm the critic must beat:** the same k, c, kinds and
  determinization with the material evaluator (tanh of the HP margin over the
  two player sides divided by `hp_scale`), at its own sigma. TODO (lead): two
  matches against `parity3`, compared through their p's (standard error of
  the difference about 0.025 at 800 decisive each), or one match between the
  two look-ahead players.
- **God view (optional, an upper bound):** the selected evaluator on the true
  state (`determinization: "godview"`, tag `+godview`). It measures what the
  fog costs the operator; it pools with no other procedure and never enters the
  Elo catalog. TODO (lead): run or not.

## Predictions, bars, kill

TODO (lead): p against `parity3` for each arm; the pass bar and the kill; how
the critic arm is compared with the material arm; the share of decisions
played otherwise than the argmax expected from step 1's readout (bounded by
the share whose gap to a candidate is under 2c).

## Cost

**Measured on the laptop** (CPU, 2 threads; 15 decisions at turns 9 and 10 of
a `multiplayer_Hamlets` game after 100 raw decisions, k = 8, `kinds: "all"`,
the observed world). The prior is `obs8` (`parity3` is not on the laptop);
the critics are untrained at the reference's architecture (full) and step 1's
small one, the same forwards as trained ones (`tools/lookahead_timing.py`):

| evaluator | evaluator states per decision | expansions | evaluator | look-ahead total |
|---|---|---|---|---|
| material | 15.3 (max 38) | 32 ms | 3 ms | 35 ms |
| small critic, CPU | 10.0 | 34 ms | 111 ms | 145 ms |
| full critic, CPU | 10.0 | 25 ms | 2.7 s | 2.7 s |

Per decision and per state, a critic costs about 1-2 ms of encoding through
the core and, on this CPU, 11 ms (small) or 265 ms (full) of forward, with or
without batching (a 16-state microbenchmark at turn 9 of the same game). The
prior's own CPU forward (0.26-0.29 s) does not count on the box, where the
server serves it.

**Scaled to a 20-worker 4090 box**, from the anneal match's records
(`training/metrics/imitation_anneal_20261003/box/`: 944 games in 929 s for
800 decisive, 20 workers, so 19.7 worker-seconds per game; the candidate's
server answered 253,010 requests, 268 decisions per game for one side), with
the look-ahead's work added to the worker's time and the laptop's per-decision
figures taken as the box's:

| arm (944 games) | look-ahead per game | match wall | at $0.42-0.63/h |
|---|---|---|---|
| material | +9 s | about 23 min | $0.16-0.24 |
| small critic in each worker, CPU | +39 s | about 46 min | $0.32-0.48 |
| full critic in each worker, CPU | +12 min | about 10 h | not affordable |
| full critic in each worker, GPU | not measured | not measured | |

These are `kinds: "all"` figures: an attack-only operator runs only where the
top k hold an attack (TODO: the share from step 1's readout) and values fewer
candidates, so it costs less. The full critic on the GPU is the case the gate
most likely needs, and nothing here prices it: the policy server's device time
in the anneal match was 2.7 ms per position at its mean batch of 4.2
(`cand64_vs_ref.server_0.json`), which at 10-15 states per decision would be
7-11 s of GPU per game, 2-3 GPU-hours over the match if the twenty workers'
critics did not overlap. Before the gate, time it on the box:
`tools/lookahead_timing.py --checkpoint training/checkpoints/parity3.pt
--critic-arch full --critic-device cuda` (about a minute; the VRAM of twenty
critic processes is not measured either). If it is too slow, the lever is a
shared critic server batching the workers' states, on the pattern of
`tools/eval_inference_server.py` (not built).
