# Raw-argmax control (2026-09-04)

Question: every evaluation before this date compared search (argmax
of visit counts after 30 decisions) against the raw seed SAMPLING
from its full distribution over ~350 legal actions. Does "search
improves the seed" survive a raw player that plays the policy's
argmax?

## Protocol

- Seed: HF `tier-b/a3/seed_imit_tierb_start.pt` (the catalog's +223
  seed; identical to the imitation checkpoint except the value
  head). Same weights on both sides of every game.
- Raw-argmax player: `tools/raw_player.py`, the joint prior over
  every legal action (what search reads), temperature 0. Procedure
  tag `raw:t0`; the legacy sampler keeps `raw`.
- Search player: the catalog procedure, `--mcts-sims 32
  --no-turn-search` (Gumbel root, leaf batch 1).
- 40 games per match, ladder maps, sides alternated, max_turns 200,
  PURE fit (`tools/elo_collect.py --no-catalog`).
- Box 49838860: RTX 4090, 24 Ryzen 9 7900X3D cores, 30 GB RAM,
  $0.334/h; `--device cuda --jobs 10`, bf16 + compile defaults.
  Script: `scripts/raw_argmax_control.sh`. Game files:
  `eval_games/raw_argmax_control/`.

## Results

| match | A over B, W-D-L | Elo of A over B | turns (median) | decisions per side-turn |
|---|---|---|---|---|
| A: seed argmax vs seed sampling | 37-0-3 | +412 ± 97 | 17 | 9.4 vs 9.5 |
| B: seed MCTS-32 vs seed argmax | 13-0-27 | -124 ± 58 | 21.5 | 453 forwards vs 9.9 |

Both matches: 40 of 40 decided by leader kill, no timeouts, wins on
both sides of the board (match A: 20-0 as side 1, 17-3 as side 2;
match B: 6-14 as side 1, 7-13 as side 2).

Reference: seed MCTS-32 vs seed sampling was 9-0-1 with 10 of 20
games timed out (`eval_games/engine_seed_search_vs_raw`, the
"+321 for search" number).

## Reading

1. Sampling from the imitation policy costs about 400 Elo against
   playing its mode, at the same number of actions per turn. The
   whole "+321 from search" was smaller than the argmax effect: it
   was argmax play with a search tax on top.
2. Search at 32 simulations makes the seed WORSE than its own argmax
   by about 120 Elo. The distillation program's teacher was never
   better than the prior it was meant to improve, which is
   consistent with the visit-target and search-depth measurements of
   the 2026-09-04 review (32 leaf evaluations over ~350 actions look
   1-2 actions ahead; the target is a truncation of the prior).
3. Every `mcts:32` number on the Elo board measures the procedure as
   much as the weights. The strongest measured player is the raw
   seed at temperature 0.

## Not settled here

- Whether more simulations, plain PUCT at leaf batch 1, or a lower
  positive temperature beat `raw:t0`. A temperature sweep (0, 0.25,
  0.5) at 40 games each costs about 5 minutes of box time.
- Whether argmax play is exploitable by an opponent that adapts;
  both sides here were the same fixed network.

## Next

- Re-baseline the board with `raw:t0` as the reference procedure.
- Any search procedure must beat `raw:t0` on the seed before it is
  used as a teacher for distillation.
