# The memory everywhere (2026-10-05)

User ruling 2026-10-05: self-play keeps the memory; its size is open to
discussion, its existence is not. This plan carries a side's memory into
every place that runs the network for that side's decisions, as the match
player already does (docs/parity_memory_design_20260929.md, "Serving and
play").

**The rule.** A side's memory at a decision is the memory its previous
decision wrote, or the learned initial memory at its first decision; a
refused decision (a bounced recruit) is undone. Whatever runs a memory
model for a decision carries that state or refuses the model by name;
nothing plays it at 0 slots unless told to.

## The holes (main, 0.13.0)

| # | where | today | fix |
|---|---|---|---|
| 1 | `RawPolicyPlayer` | `memory_slots=None` means no memory, so a caller that forgets it plays a memory model at 0 slots | a memory model needs its slot count named; 0 only when asked |
| 2 | `tools/turn_gap.py` | builds its players without the memory (hole 1) | the memory at a corpus position is the sequence trainer's: the side's earlier positions run in order from the initial memory; each playout carries both sides' memories on |
| 3 | MCTS (`tools/mcts.py`, `mcts_policy.py`) | refuses | each node keeps the memory its evaluation wrote; a leaf reads the last one of its side on its path, else the player's for that side; the root's write is the decision's; no transposition table for a memory model (one position reached by two paths holds two memories); the player keeps each side's memory per game |
| 4 | turn search (`turn_search.py`, `turn_policy.py`) | refuses | each materialized turn carries its side's memory decision by decision; boundary values and playouts read the memory of the side they evaluate |
| 5 | the self-play pool (`actor_pool.py`, `actor_worker.py`) | refuses | actors run hole 3's search over the served model; `RemoteModel.forward_batch` sends each leaf's memory (the server already takes them, `inference_seam.InferenceServer`) |
| 6 | the learner (`az_loop`, `trainer.py`) | experiences carry no memory | experiences carry game, side and decision index; the learner trains game-side streams with truncated back-propagation through time (the sequence trainer's scheme: windows of 16 decisions, the memory carried across them without gradient) on the search targets and the outcome |
| 7 | the graphed serve path | refuses | serves a memory model on the eager path with a warning (an opt-in speed path, docs/box_specs.md) |
| 8 | analysis and benchmark tools that run the network | silently at 0 slots | carry it where the number depends on play (value and prior probes over games); otherwise refuse a memory model by name |

## Order

1, 2 first: the next phase-2 measurement (the turn gap under `parity3`)
needs them. Then 3 and 4 (search), 5 (the pool), 6 (the learner), 7 and 8.
Each lands with tests that fail without it; a search test compares a
search's root decision memory with the match player's on the same game.

## Built (2026-10-05)

Every hole above is closed (0.14.0); each fix has tests that fail without
it (mutation-checked).

- **Search.** MCTS and the turn search carry both sides' memories along
  every line they walk; a materialized turn advances the mover's memory
  only when something reads it after the turn (a mover-frame boundary or a
  projection). The players drop a game's memories when it ends.
- **The pool.** Each leaf's memory rides the priors protocol's request
  (`wesnoth_ai/leaf_wire.py`), so a memory model needs server priors. The
  PLAY command also carries the parity observation and its relevant-set
  version, which the actors' encoder lacked.
- **The learner** (`wesnoth_ai/memory_step.py`, `tools/memory_trace.py`).
  An actor ships every decision of a memory player's game, the ones
  without a search target as "carry" positions with zero weights, each
  encoded in the actor: a state's binding to its core does not cross
  processes, and the parity observation is built only by the core. The
  step runs `train_batch_size` game-sides side by side in windows of
  `memory_window` (16) decisions, back-propagates through the memory
  within a window and carries it across without gradient, and remains one
  optimizer update per iteration. The belief head trains on every
  position at the recipe's weight (`TrainerConfig.belief_coef`, 1), its
  targets read from the actor's god view and the view dropped before
  shipping. Telemetry reads each state with its memory from one in-order
  pass over its games under the weights being probed.
- **Refused by name.** The value-memory reservoir, the replay buffer,
  value grounding and the plan tournament (they sample positions without
  their game-sides or carry no memory), the offline deep profiler
  (`signal_profiler/run_profile_v2.py`; az_loop skips it for a memory
  model), and the legacy sampler in eval.
- **Graphed serve.** The pool's and the eval server's graphed switches
  serve a model with the parity observation or a memory on the eager path,
  with a warning.

Open: the actors play one slot count per run (`az_loop --memory-slots`);
the imitation recipe draws it per game-side (nested dropout), which
self-play does not yet do.
