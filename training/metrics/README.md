# Measurement records

Every number the docs quote from a run should have its record here;
the document that quotes it names the directory.

## Index

- `bench_pipeline/<run>_<date>/` — one directory per box run of a
  `scripts/*_box.sh`, named by the run and its date; the
  pre-registration in docs/ that the run answers names it too. The
  record is: `box.txt` (the host: cores, CPU, GPU, memory, torch, the
  Rust kernels live), `*.fit.json` (a match's Elo fit,
  `tools/elo_collect.py --save-json`), `verdict.txt` and `readout_*.txt`
  (the pre-registered rule applied to the games), `match.walls`,
  `pool.walls` or `eval.walls` (the wall of each match or arm), the
  pool benchmarks' JSON rows, and `ALL_DONE` when the run finished.
  Game records and checkpoints are not kept here; the recent runs
  upload them to Hugging Face under the run's own directory.
- `elo/<run>/` — match fits and tallies from before that convention
  (seed2 against the seed, the relevant-set arms, `relset`'s first
  self-pin, the `raw:t0` catalog edges).
- `elo_catalog.json`, `elo_catalog_raw_t0.json` — the Elo catalogs
  (`tools/elo_catalog.py`).
- `fidelity/` — the engine oracles and rule censuses (hidden units,
  scenario init, vision, the counter weapon), one JSON per run.
- `value_head/` — the value-head study (docs/value_head_study_20260907.md).
- `turn_gap/`, `turn_gap_ref_20260921/` — the turn-level gap under the
  seed and under the reference (docs/turn_gap_prereg_20260904.md,
  docs/turn_gap_ref_prereg_20260921.md).
- `player_ratings/` — the corpus players' rating fit
  (`tools/player_ratings.py`).
- `imitation_15m/` — holdout curves of the 15M imitation runs.
- `sweeps/` — the 2026-09-05 temperature sweep and relevant-set probe.
- `corpus_census.json` — the replay headers' census
  (`tools/analysis/corpus_census.py`).
- `box_49875606/` — the escrow of the 2026-09-04 to 09-06 box.
- The remaining top-level files (`policy_shape_*.json`, `e2_*.json`,
  `luck_probe.csv`, `l4_metrics.html`, `a3_results.html`,
  `step_scale_20260903/`) are records of the legs and probes of
  2026-08 and early 2026-09 (docs/archive/).
- `history_5m.csv`, `history_15m.csv` (untracked) — the per-net-size
  training histories below.

## Per-net-size metric archives (standing rule, user order 2026-08-06)

One CSV per net size, accumulating EVERY training iteration's
trainer-history row across campaigns. **CSVs are untracked** (see
.gitignore); this README is tracked so the convention survives.

**The rule:** any new performance-metric analysis must be presented in
context — compared against the past metrics of the SAME net size from
these files. No more single-leg readings ("plateau looks benign")
without the same-size baseline next to it; the 2026-08-06 lesson was
that the 15M "benign plateau" was only exposed as floor-relative
regression by comparison against the 5M campaigns.

Files:
- `history_5m.csv`  — d256/L6 5.0M lineage
- `history_15m.csv` — d384/L8 15M lineage (Tier-b)

Schema: `net, run, <trainer_history columns>` (union schema; rows
predating a column carry an empty cell — e.g. `distill_*` stats exist
only from 2026-08-04). `run` is the provenance label of the
leg/campaign segment; iteration indices restart within runs (relaunch
boundaries are where `iter` resets).

Maintenance: at each harvest (leg or campaign checkpoint), append the
new `trainer_history_local.csv` rows with `net` + `run` filled in.
Backfill note: only the tier-a campaign's FINAL stretch (07-29..31,
41 iters) is loaded; earlier tier-a history lives in the HF revision
history of `trainer_history_local.csv` (repo
`momom2/wesnoth-model-checkpoints`, one revision per escrow) and in
`trainer_history_3090_20260712.csv` / `..._curriculum_20260714.csv`,
assembleable on demand.

Reading reminders (memory: fresh-ce-default-success-metric): fresh CE
is read FLOOR-RELATIVE (`fresh_value_ce - fresh_ce_floor`); skip
iter-0-after-restart rows; decisive rate = (s1_wins+s2_wins)/n_games.
