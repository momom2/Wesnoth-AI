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
   - Measured on the box with the Rust wheel: masks 0.76 ms,
     enumeration incl. masks 2.53 ms (was 6.90).
   - NEXT: Rust `encode_raw` (rust_port_plan phase 2b, byte-identical
     arrays; 1.35 ms per leaf now the largest actor-side item), then
     combat and step (phase 3, full corpus sweep).
3. **Batched inference server** (plan 1.3). Measured 2026-09-04
   (docs/box_specs.md "Phase-1 iterations"): the batched forward ran
   fp32 eager (only the single-sample path had bf16 and compile);
   with bf16 it does 1,584 samples/s at batch 16 (was 602). The
   server-side priors protocol (`wesnoth_ai/server_priors.py`: actors
   ship packed masks, the server returns compact legal actions,
   `ActorPool.server_priors`, default off) serves ~1,080 leaves/s
   per thread on token-sorted batches with 9-15 KB per leaf on the
   wire (was 60-87 KB). Certified: parity tests through seam and
   wire, pool smoke end to end.
   - MEASURED through the real pool (docs/box_specs.md "Generation
     throughput"): az-leg configuration 141 leaves/s -> priors on 172
     -> priors + bf16 320 leaves/s, 33 -> 64 games/h, same box. The
     serve threads share one GIL and were busy 60-75% of the time:
     the server is still the ceiling.
   - NEXT: serve from several processes (or move the per-batch Python
     off the GIL); torch.compile the padded path; length-bucketed
     batches; then flip `server_priors` and bf16 on by default in the
     generation path.
   - Persistent eval workers shipped (`run_elo_batch
     --persistent-workers`, tools/eval_workers.py): 20 seed-vs-seed
     games at 10 concurrent in 97 s against 408 s one-process
     (docs/box_specs.md, evaluation section). An 800-game gate is
     about 65 minutes of one 4090 box. Open: 3 of 20 argmax games
     ended differently between the two modes (numeric noise under
     bf16/compile flips near-ties); run-to-run agreement check pending.
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
