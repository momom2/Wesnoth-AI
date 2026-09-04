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
2. **Per-decision Python work** (plan 1.2), ordered by the measured
   cost: enumeration 6.9 ms, masks 2.2 (Python path) / ~0.5 (Rust),
   encoding 2.5, sim step 0.7; fork and deepcopy are 0.05 ms already
   (the Rust plan's phase 4 "cheap fork" is not where the time is).
   - DONE 2026-09-04: vectorized enumeration
     (`enumerate_legal_actions_with_priors`, reference kept as
     `_enumerate_legal_actions_reference`, `WESNOTH_ENUM_REFERENCE=1`
     forces it; differential test `tests/test_enumerate_vectorized.py`):
     7.3 -> 3.8 ms on the laptop including the mask build; the
     combat-oracle damage computation in the mask builder is skipped
     when both oracle alphas are 0 (they are). The floor is now the
     construction of ~600-700 action objects per state; the
     struct-of-arrays output that removes it belongs to the server
     step below (arrays are what cross processes cheaply).
   - NEXT: Rust `encode_raw` (rust_port_plan phase 2b, byte-identical
     arrays), then combat and step (phase 3, full corpus sweep).
3. **Batched inference server** (plan 1.3). Measured ceiling: ~600
   samples/s per process from batch 16 regardless of batch 64, i.e.
   CPU-side. Suspects, in order: `WesnothModel.forward_batch` pads
   with per-sample Python loops and rebuilds a per-sample ModelOutput
   (target logits [A, H] sliced per leaf); `output_to_wire` pickles
   ~120 KB of target logits per leaf through a multiprocessing queue
   (72 MB/s at 600 leaves/s); the actor then runs the masked softmax
   itself. Design to measure on a GPU box: actors ship the legality
   masks (bit-packed, ~5 KB) with the request, the server computes
   the masked joint priors on the GPU in the same batch, and replies
   with the compact legal-action arrays (~12 KB) plus value. Target:
   3,000 leaf evaluations per second per 4090 for the 15M net; bf16
   and compile validated on the training path (the 2026-08-29
   deadlock reproduced or cleared).
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
