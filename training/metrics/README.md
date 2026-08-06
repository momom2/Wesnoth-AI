# Per-net-size metric archives (standing rule, user order 2026-08-06)

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
