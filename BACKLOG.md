# BACKLOG

Live backlog for `docs/plan_20260904.md`. The pre-restart backlog
(1,055 lines of rulings and open items, 2026-05 to 2026-09-04) is
archived verbatim at `docs/archive/backlog_20260904.md`.

## NEXT ACTIONS (phase 1: engineering, in order)

1. **Benchmark harness** (`tools/bench_pipeline.py`, plan 1.1): a
   fixed set of 200 mid-game states and 20 game seeds; ms per forward
   by batch size and token count; leaf evaluations per second under
   the actor pool; seconds per raw game and per searched game.
   Record the baseline row in `docs/box_specs.md` before changing
   anything. Baseline to beat: 3.7 ms per forward single stream,
   300-370 leaf evaluations per second, 12 s per raw game, 146 s per
   Gumbel-MCTS-32 game on a 4090 box.
2. **Rust simulator core** (plan 1.2; `docs/rust_port_plan.md`
   phases 2b-4): raw encoding, combat and step, Rust-owned GameState
   with a cheap fork. Certification: byte-identical encodings,
   24,796-replay corpus sweep clean, strike-level parity. Targets:
   step and fork under 0.1 ms, encode under 0.2 ms.
3. **Batched inference server** (plan 1.3): leaves from all games
   batched by token bucket; bf16 and compile validated on the
   training path (reproduce or clear the 2026-08-29 deadlock); CUDA
   graphs where shapes allow. Target: 3,000 leaf evaluations per
   second per 4090 for the 15M net.
4. **Defects** (plan 1.6):
   - `tools/az_loop.py` `_probe`: pins at sims 0 must pass
     `--raw-temperature-a 0 --raw-temperature-b 0`; every az pin,
     including the 11-29, compared the sampling player on both sides.
   - `tools/step_control.py`: `_clone_weights`/`publish_weights`
     cover `_model` only; the encoder's parameters (same AdamW) keep
     the full step, even when the step is skipped.
   - `tests/test_actor_pool_watchdog.py`: five failures since the az4
     change (the test's pool stub lacks `value_center`).
   - Eval search procedure: `elo_eval_game` plays the Gumbel root by
     default while `az_loop` trained plain PUCT at leaf batch 16;
     record and match the procedure per player.
5. **Model cost study** (plan 1.4) and **eval at scale** (plan 1.5).

## Phase 2 prerequisite (measure before designing)

- Turn-level value gap: four alternative whole turns at 60 boundary
  positions, 40 playouts each; fraction of positions where the best
  alternative differs from the base's turn by at least 0.25 in
  expected outcome. About $1.3 at current efficiency. Under 5% kills
  rollout-graded turn search.

## Cheap measurements worth taking

- Temperature sweep for the deployed raw player: 0, 0.25, 0.5 vs
  `raw:t0`, 40 games each, about 5 minutes of box time.
- Re-baseline the Elo catalog (`training/metrics/elo_catalog.json`):
  every edge is `mcts:32`; the board needs `raw:t0` edges.
- `raw:t0` vs `raw:t0` self-play: decisive rate, turns, decisions per
  game (never played; needed for every play-out cost model).

## Uncommitted work (2026-09-04)

`tools/raw_player.py`, the `--raw-temperature-a/-b` flags in
`tools/elo_eval_game.py` / `tools/run_elo_batch.py` /
`tools/eval_procedure.py`, `tests/test_raw_player.py`,
`scripts/raw_argmax_control.sh`, `eval_games/raw_argmax_control/`,
`docs/raw_argmax_control_20260904.md`,
`docs/selfplay_redesign_20260904.md`, `docs/plan_20260904.md`, the
`docs/archive/` move, and this file.

## Ops notes that are still true

- Boxes: propose specs and cost, wait for a yes, `vms_enabled=false`,
  CPU model EPYC or Ryzen, destroy at the end (`yes | vastai destroy
  instance <id>`; the CLI prompts).
- Launch a detached job over ssh in one call and verify in another;
  the launching session hangs while the child runs.
- No compute on the laptop beyond sub-minute microbenchmarks.
