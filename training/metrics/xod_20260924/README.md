# XOD blocked-rewrite measurement (2026-09-24)

Scripts and text outputs of the measurement summarized in
docs/xod_dominance_design_20260924.md section 10.4: every candidate
rewrite of the first 300 training-split corpus games, admitted or not,
with its comparison vector, the added class Q2, the blocking families
and the candidate relaxations.

Scripts as run on the laptop; their paths are the laptop's
(`ROOT`, `CORPUS` in `census.py` and `load.py`). The per-candidate rows
(`rows_000.jsonl.gz`, `rows_150.jsonl.gz`, 8.8 MB, derived from corpus
games) are not committed; regenerate them with

    python census.py            # about 5 minutes on three cores for 300 games

then `analyze.py` (-> `relaxations.txt`), `context.py` (-> `context.txt`),
`extras.py` (-> `extras.txt`), `families.py`, `magnitudes.py`,
`fixcheck.py`, `base_rates.py`, `class_blocks.py`. `check20.py` checks
the driver against `tools/analysis/dominance_count.py` on the first 20
games (7,310 decisions, 3 attacks admitted at R0, 28 under the loosest
combination). `vectors.py` rereads status dimensions with a dead unit as
the extreme value.
