# BACKLOG

Live backlog for `docs/plan_20260904.md`. The pre-restart backlog
(1,055 lines of rulings and open items, 2026-05 to 2026-09-04) is
archived verbatim at `docs/archive/backlog_20260904.md`.

## NEXT ACTIONS (phase 1: engineering, in order)

1. **Benchmark harness** (plan 1.1): DONE 2026-09-04, baseline in
   `docs/box_specs.md` and `training/metrics/bench_pipeline/`. Per
   decision: 12.4 ms of Python (enumerate priors 6.9, masks 2.2 on
   the Python path, encoding 2.5, sim step 0.7) against 5.2 ms per
   forward; batched forwards plateau at ~600 samples/s per process
   from batch 16 (CPU-side ceiling). Raw game 35 s, searched game
   160 s at 10 jobs, one process per game. Open: run it on a second
   box shape with the Rust wheel built (`scripts/bench_box.sh` now
   builds it) to pin the reproducibility band.
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
  game. The benchmark's 20 such games ran 48 turns median (outcomes
  not recorded); deterministic self-play may stall, and every
  play-out cost model depends on this number.

## Ops notes that are still true

- Boxes: propose specs and cost, wait for a yes, `vms_enabled=false`,
  CPU model EPYC or Ryzen, destroy at the end (`yes | vastai destroy
  instance <id>`; the CLI prompts).
- Launch a detached job over ssh in one call and verify in another;
  the launching session hangs while the child runs.
- No compute on the laptop beyond sub-minute microbenchmarks.
