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
