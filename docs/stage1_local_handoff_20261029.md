# Stage 1 played locally, handed to stage C (plan, 2026-10-29)

Branch `feature/stage1-local-handoff`, cut from `feature/local-gpu-stage1`.
Nothing here is run: no upload, no box. Stage C costs $4.3-7.8 on a Vast
4090 and waits on the user's approval, and so does the upload.

Stage 1 of the training design (`docs/training_design_20261019/current_design.md`,
section 6) is being played on the desktop's RX 6600
(`scripts/selfplay_t1_local.py`, `docs/selfplay_t1_local_prereg_20261028.md`),
not on a box. Stage C (`scripts/stage_c_box.sh`) was written to read the
box's stage 1 from Hugging Face. This note covers four things: what stage C
reads, what the local run leaves, what differs and how each difference is
handled, and the decisions that belong to the user.

## 1. What stage C reads from stage 1

`scripts/stage_c_box.sh` (unchanged on this branch), `restore_stage1`
(lines 174-228), then the tools it runs:

| What | Where | What it requires |
|---|---|---|
| The HF folder | `STAGE1_HF_DIR`, default `tier-b/selfplay_t1_20261022` (line 75); `box_upload.py --restore` (lines 177-178, 194-195) | Each name sits flat in the folder of `momom2/wesnoth-model-checkpoints` (`scripts/box/box_upload.py:56, 200, 400`) |
| `plan.txt` | line 179 | `<games> <held out from> ...`, two positive integers (lines 180-183). Otherwise `STAGE1_UNUSABLE` |
| The match names | lines 184-189 | `s00`, `s01`, ...: one per `CHUNK_GAMES` (5000, line 77) of `<games>` |
| `games_sNN.tar.gz` | lines 192, 196-206 | It extracts to `games_sNN/` at the tarball's top (line 202); box_upload.py's tarball puts it there (`send_dir`, lines 298-311) |
| `walk_sNN.jsonl` | lines 191, 208, 218-219 | Joined into `walk.jsonl`, which the plant reads |
| `labels.jsonl`, `readout.json` | lines 211-213 | Present. `fits.json` is restored too (line 175) but read by nothing |
| The readout's gates | lines 220-227 | `i_sample`, `legal_mass`, `i_count`, `dice_control`, `i_v` each `pass: true` |
| The labelled games' players | `tools/pg_positions.py` `refuse_foreign` (now lines 335-366), from `tools/pg_pass.py:223` and `tools/pg_eval.py:123` | Each game-side: procedure `raw:t1`, memory 64, a result naming the checkpoint's SHA-256, no per-turn memory reset |
| The serving precision | the precision floor, `tools/pg_pass.py` and `tools/pg_floor.py` (section 4) | Before this branch: nothing read it, and the floor ran at `bfloat16` by default |
| Seeds | `SP_SEED_BASE` 800000 (line 76): `pg_plant.py delta --seed-base` (line 342) and `selfplay_rows.py --seed-base` (line 350) | A walk row's `seed` less the base is the game's index |
| The held-out split | `plan.txt`'s second field, `pg_labels.py --heldout-from` (line 352) | The plant relabels with stage 1's split |
| The records | `pg_positions.positions_of` replays every decision through the Rust core and checks the record's turn digests (`verify`, on by default) | The core's rules equal those that played the games |

Stage C reads nothing else of stage 1: not `rates.json` (only a comment
names it, line 45), not D0 or the memory report, not the server stats.

## 2. What the local run leaves

In `C:\Users\momom2\wesnoth_runs\selfplay_t1_local\`, by
`scripts/selfplay_t1_local.py`:

- `games_s00` to `games_s05`: 5,000 self-play games each, each game a
  result (`game_*.json`) and a record (`*.game.jsonl.gz`, with the decision
  log), plus the server's and workers' hidden stats and logs;
- `walk_sNN.jsonl` and `rows_sNN.jsonl` per match; then `rows.jsonl`,
  `labels.jsonl`, `fits.json`, `readout.json`, `readout.done`;
- `games_d0`, `d0.fit.json`; `games_memory`, `memory.fit.json`;
- `chunk_sNN.done`, `measure_d0.done`, `measure_memory.done`, `wall_*.txt`,
  `matches.txt`, `progress.txt`, the logs (`match_*.log`, `rows.log`,
  `labels.log`, `readout.log`, `fit_*.log`, `runner*.log`), and `ALL_DONE`
  when the runner ends.

As of 10:30 on 2026-10-29: `games_s00` (5,000 games, 224 MB), `walk_s00.jsonl`,
`rows_s00.jsonl`, and D0 in progress (2,256 results at the dry run below).

## 3. Each difference, and how it is handled

| Difference | Effect on stage C | Handled by |
|---|---|---|
| No `plan.txt`. The local runner plays every planned match and never sizes the run (`selfplay_t1_local.py` docstring); the box writes it in `size_plan` (`selfplay_t1_box.sh:376-418`) | `STAGE1_UNUSABLE: no plan.txt` (line 179) | The upload writes `30000 20000 unknown`: the box's line for a run that played its plan. The third field, the first match's games an hour, is unread by stage C (`read -r PLAN_N HELDOUT_FROM _`). The local first match's walls miss its interrupted attempts, so the field is `unknown`, as the box writes it when unread |
| No HF folder. The local run uploads nothing | Stage C cannot restore | `scripts/stage1_local_upload.py`: the box's names and tarballs, into a new folder (section 5) |
| **Served in fp16** on the HIP kernels, not bf16 on flash; every game records `infer_fp16: true`, `infer_bf16: false` | The floor ran at bfloat16 by default (`pg_pass.py:192`, CLI `:495`, at 66218e4b), and `stage_c_box.sh` passes no `--floor-dtype` (lines 254-256, 305-306, 353-355): a bf16 floor on fp16 logs | `--floor-dtype auto`, now the command line's default, reads the floor at the games' recorded precision. A floor of the other precision is refused (section 4) |
| The pool: 96 games in flight on one server, not 120 | None: no stage C reader sees it | — |
| No stage C rates (`rates.json`) | None: stage C reads none of stage 1's rates; C0 measures its own on the box (lines 252-272) and sizes the run from them (`c0_plan`) | — |
| D0 and the memory report served in fp16, the reference included | None for stage C. D0 is the claims' origin (design section 3.2): see the user's decisions | — |
| Seeds, match size, held-out split, procedure, memory, checkpoint | None: the box's values (`selfplay_t1_local.py:46-54`; the census below) | The upload refuses records that depart from them |
| The code that played the games: 115be5d1, then fc560ab9 (the run's resume) | The records replay through the core of stage C's code stage | `rust/` is unchanged from 115be5d1 to this branch (core phase 38): stage C must be staged from a commit with the same `rust/` |

## 4. The precision floor

**What it is.** C0's replay identity compares the logged log p of each
sampled action with the trainer's, recomputed in fp32 on the padded trunk.
`pg_pass` estimates the per-decision KL from the logged law to the
trainer's as the mean of (logged − trainer) and fails (exit 3) above
`limit = max(base, base + 2 (mean + 2 SE))` (`tools/pg_floor.py`
`replay_limit`, now lines 238-245). The base, 1e-4, is ε/20, a twentieth
of the step search's smallest radius (`pg_pass.py:147`, design section 5).
The floor's mean is the rounding of the serving precision: KL(packed trunk
at that precision ‖ packed trunk in fp32) on the decisions of 400 game-sides,
each forward taking the trainer's own memory. Its SE is clustered by game.
The pass refuses (exit 4) when the limit passes 1e-3
(`FLOOR_LIMIT_CAP`, now `pg_floor.py:95`), and (exit 5) when the packed and
padded trunks differ in fp32 (`PATH_KL_LIMIT`, 1e-6, line 99).

**What `--floor-dtype` accepted.** `bfloat16` (the default), `float16`
and `off` (`pg_pass.py:495` at 66218e4b). `FloorReader` autocasts the
packed trunk to `getattr(torch, dtype)` (`pg_floor.py:68-73, 94` at
66218e4b). So an fp16 floor existed, but nothing chose it: stage C would
have read a bf16 floor on fp16 logs.

**fp16 against bf16, by the formats.** bfloat16 keeps 8 significant bits
and float16 keeps 11, so their unit roundoffs are 2^-8 and 2^-11: fp16's
is an eighth of bf16's. A law's KL is quadratic in a small perturbation of
its logits (Var_p(d)/2), so the fp16 floor should be about 1/64 of the
bf16 floor on the same positions. A bf16 floor on fp16 logs is therefore
too loose: it lifts the limit by about 64 times the rounding the logs carry.

**Measured on the CPU** (2026-10-29, `own2`, the local stage 1's own logs;
the CPU autocast stands in for the GPU's). Two probes, both scripts in the
session scratchpad:
- `floor_probe.py`: `pg_pass` on the first game (seed 800000, 507
  decisions), with floor readers at both precisions;
- `replay_probe.py`: the first five games (seeds 800000-800008, 3,021
  decisions), each replayed by `pg_eval`'s evaluator in fp32 and compared
  decision by decision with the log.

| Reading | Value |
|---|---|
| Floor at float16 (CPU autocast, packed trunk, the trainer's memory), game 1 | KL 3.77e-6 a decision |
| Floor at bfloat16 (the same) | KL 2.44e-4 |
| Their ratio | 65, against 64 from the formats |
| Path (packed against padded, fp32) | -8.5e-11 |
| The logs against fp32, second-order reading (`half_mean_square`), game 1 | KL 5.5e-6: about 1.5 times the float16 floor |
| The logs against fp32, first-order estimate (`kl`, what the gate compares) | 5.0e-5 ± 5.3e-5 over five games. Per game, from -1.6e-4 to 2.1e-4: noise |
| \|logged − fp32\| of the sampled action's log p | mean 0.0018, largest 0.019 |
| The same by place in the game-side: decisions 0-9, 10-49, 50-99, 100-199, 200+ | 0.0015, 0.0018, 0.0018, 0.0019, 0.0017: no growth |
| \|logged − fp32\| of the class masses (end_turn, attack, recruit) | mean 2.9e-5, 5.6e-5, 2.7e-5; largest 1.6e-3 |

What this shows:
- **The floor.** The formats' argument holds: the fp16 floor is about 1/65 of
  the bf16 floor.
- **The size of the logs' error.** The logs carry about what fp16 rounding
  explains: 1.5 times the CPU autocast's floor on game 1, the HIP kernels
  and the other device adding the rest. Over the five games the mean
  |logged − fp32| is alike, so nothing of bf16's size.
- **The memory.** The error does not grow along a game-side, so the memory
  carried through hundreds of fp16 forwards adds no measurable drift.
- **The gate's estimate.** The first-order estimate the gate compares has
  an SE of about 1.3e-4 on one game. A pass over a game or two can
  therefore cross 1e-4 by chance: the first game alone read 2.13e-4,
  which is a "fail". C0's 200 games, about 120,000 decisions, read it to
  about 1e-5.

**Predicted at C0:**

| Floor | Limit | Logs' KL against it |
|---|---|---|
| float16 (the box's CUDA autocast, expected near the CPU's 4e-6) | about 1.1e-4 | about 5e-6, estimated to ±1e-5: passes by about 10 SE |
| bfloat16 (the old default) | about 5.9e-4 | Passes too, but a real departure up to about six times the base limit would pass with it |

**The change** (this branch):
- `pg_positions.serving_precision` reads each match result's precision:
  `float16` under `infer_fp16`, `bfloat16` under `infer_bf16`, else
  `float32` (absent fields read False, as `elo_eval_game`'s own resume guard
  reads them). `record_facts` carries it as `serving`. A game served on one
  of the server's other numerics paths (`infer_bf16_residual`,
  `infer_fused_relu`, `infer_fp8_a`/`_b`) reads as its precision followed by
  them, `bfloat16+infer_bf16_residual` say, and the floor refuses it under
  every choice but `off`: those paths round otherwise than the plain
  autocast the floor reads (a first version read them as plain bf16; found
  on review, 2026-10-29). Stage 1's scripts, box and local, serve none of
  them.
- `pg_floor.resolve_floor_dtype`:
  - `auto` reads the games' precision. For games served in float32 it
    reads the path alone (`float32`, `pg_floor.PATH_ONLY`): no rounding is
    read, so the limit is the base, and the packed-against-padded check
    (exit 5) still runs;
  - a named precision is refused when the games were served in the other
    one;
  - games of more than one serving precision in one pass are refused under
    `auto`, float32 among them: a reduced precision's floor would lift the
    limit for the decisions of fp32-served games, which carry no rounding.
    A named precision still takes fp32-served games, alone or beside games
    of that precision, as the suite's CPU games do (the caller's choice);
  - `off` reads neither the floor nor the path.
- `pg_pass --floor-dtype` now defaults to `auto`. pass.pt and pass.json
  record `floor_dtype`, the precision read. The API's default stays
  `bfloat16`.
- No behaviour changes for a box's bf16 records: they resolve to the same
  bfloat16 floor, and a test checks that the floor and the limit are
  identical.
- Records served in float32 under the command line read the path alone;
  before this branch they read a bf16 floor, the path included. A first
  version of this branch resolved them to `off`, which dropped the path
  check with the rounding (found on review, 2026-10-29): a stage 1
  re-served in fp32, as C0's exit 4 advises, would have lost it, since
  `stage_c_box.sh` passes no `--floor-dtype`.
- The derivation is in `tools/pg_floor.py`'s docstring, "The serving
  precision".

**What it means for the local records.** With the fp16 floor, C0's
replay identity should pass on these records, by a wide margin, and at
the limit the design intends: ε/20 plus about twice fp16's rounding.

The old default would also have let them through, but at a limit about six
times looser than the base. That limit is the one the floor exists to
avoid: it would let a real departure (memory carry, masks, rejection
replay; design section 5) pass as rounding.

## 5. The upload: `scripts/stage1_local_upload.py`

It reads the run directory and writes nothing to it. It sends, flat in
`--hf-dir`, the names the box's stage 1 sends:
- every regular file of the run directory, except `STOP`, `.tmp` files and
  `upload.log`;
- each `games_<name>` as `games_<name>.tar.gz`, built by box_upload.py's
  own tarball;
- the server stats copied out as `<name>.inference_server_0.json` and
  `<name>.worker_stages.json`;
- `plan.txt`, and `local_provenance.json` (the census, for the reviewer).

The sending itself is `box_upload.Round`: the box's retries, timeouts and
landed-version state, kept in a staging directory beside the run,
`selfplay_t1_local_upload/`. The match markers wait until their tarballs
land, and `ALL_DONE` goes in a second round only when everything landed.

`--dry-run` is the default and lists everything. `--upload` refuses,
sending nothing, unless all of these hold:
- the runner ended (`ALL_DONE`, no `STOP`);
- the six matches are whole;
- the labels, fits and readout are there, and the five gates passed;
- the labels' held-out split is the plan's;
- every self-play result is `own2` at 64 slots on both seats, under
  `raw:t1`, served in float16 and with its decisions logged.

It also refuses a folder holding files this staging directory did not
send, and always refuses the box stage 1's folder. It uses
huggingface_hub's stored login and never reads the token.

The staging directory must be new, empty, or one the script filled: its
record, `tmp/stage1_local_upload.json`, names the files the script wrote
there, and only those are ever deleted. Any other existing directory is
refused with nothing touched, and so is a file the record does not name,
since the uploader sends every file directly in the directory. (A first
version deleted every top-level file not in the upload, so a `--staging`
pointed at the repository's root would have deleted CLAUDE.md and
README.md; found on review, 2026-10-29.) The record also names the run, by
a hash of `ALL_DONE`'s content and the census: the uploader's landed state
is keyed by the HF folder alone, so another run sent through the same
staging directory to the same folder would have passed the first run's
landed files as its own. It is refused; a run whose directory moved is the
same run.

Dry run on the real directory at 10:32, s00 only (abridged):

```
destination: momom2/wesnoth-model-checkpoints:tier-b/selfplay_t1_local_20261028/
staging: C:\Users\momom2\wesnoth_runs\selfplay_t1_local_upload

  local_provenance.json        1.5 kB, written at upload
  match_d0.log                 309.6 kB from ...\selfplay_t1_local\match_d0.log
  match_s00.log                679.0 kB from ...\selfplay_t1_local\match_s00.log
  matches.txt                  349 B from ...
  plan.txt                     20 B, written at upload: 30000 20000 unknown
  progress.txt, rows.log, runner*.log, wall_s00.txt   (small, from the run directory)
  rows_s00.jsonl               3.3 MB from ...
  s00.inference_server_0.json  2.6 kB from ...\games_s00\.inference_server_0.json
  s00.worker_stages.json       1.5 kB from ...\games_s00\.worker_stages.json
  walk_s00.jsonl               5.0 MB from ...
  games_d0.tar.gz              15.7 MB in 4547 files, packed at upload, from ...\games_d0
  games_s00.tar.gz             223.7 MB in 10052 files, packed at upload, from ...\games_s00
  chunk_s00.done               21 B from ...; held until games_s00.tar.gz lands
  upload.log                   the upload's own log, after the files

25 names, 248.8 MB before compression

census of the match results:
  games_d0: 2256 results
    2256: {"checkpoint": ["aea027b2696a"], "decisions_logged": false, "forced_faction": "none",
           "memory": [64, 64], "packed": true, "procedure": ["raw:t1", "raw:t0+eo-1.5"], "served": "float16"}
  games_s00: 5000 results
    5000: {"checkpoint": ["aea027b2696a"], "decisions_logged": true, "forced_faction": "none",
           "memory": [64, 64], "packed": true, "procedure": ["raw:t1", "raw:t1"], "served": "float16"}

NOT READY: an upload would be refused (26): ALL_DONE: the runner has not ended;
  chunk_s01..s05.done, games_s01..s05, walk_s01..s05.jsonl, rows_s01..s05.jsonl: missing;
  rows.jsonl, labels.jsonl, fits.json, readout.json, readout.done: missing

stage C on these records: STAGE1_HF_DIR=tier-b/selfplay_t1_local_20261028 SP_SEED_BASE=800000 CHUNK_GAMES=5000
```

When the run is whole, the upload holds about 1.4 GB of self-play games
(six tarballs of about 224 MB), D0's and the memory report's, and about
70 MB of walks and rows. The runner's smoke run
(`wesnoth_runs/stage1_smoke`, `--smoke`) is whole, and its dry run passes
every check but one: its `dice_control` gate failed on 48 games. With
`--smoke` the upload's folder is `tier-b/selfplay_t1_local_smoke` unless one
is named, and the stage 1 folder is refused (a first version kept the
stage 1 folder as the smoke run's default; found on review, 2026-10-29).

## 6. Commands (the user's or the coordinator's to run)

```
# 1. once ALL_DONE is in the run directory: the dry run, then the upload
C:/Users/momom2/Desktop/perso/projects/wesnoth_ai/.venv/Scripts/python.exe scripts/stage1_local_upload.py ^
    --run C:/Users/momom2/wesnoth_runs/selfplay_t1_local
C:/Users/momom2/Desktop/perso/projects/wesnoth_ai/.venv/Scripts/python.exe scripts/stage1_local_upload.py ^
    --run C:/Users/momom2/wesnoth_runs/selfplay_t1_local --upload
#    (a second --upload resumes; nothing that landed is sent again)

# 2. stage C, after the user's rulings (section 9) and approval: a green CI run on the
#    commit to stage, then (docs/box_runbook.md)
python tools/stage_code.py --out <tmp>/stage_YYYYMMDDx.tar.gz --script scripts/stage_c_box.sh ^
    --upload tier-b/staging/stage_YYYYMMDDx.tar.gz
python scripts/rent_box.py search --min-hours 15
python scripts/rent_box.py create OFFER_ID --onstart stage_c_box.sh ^
    --stage tier-b/staging/stage_YYYYMMDDx.tar.gz --hours 13.6 --disk 120 ^
    --env STAGE1_HF_DIR=tier-b/selfplay_t1_local_20261028 --env HF_DIR=tier-b/stage_c_local_YYYYMMDD
```

`SP_SEED_BASE` (800000) and `CHUNK_GAMES` (5000) are stage C's defaults
and the local run's values, so they need no `--env`. A stage C `HF_DIR`
of its own keeps its records apart from the default
`tier-b/stage_c_20261022`, whose name implies the box's stage 1.

## 7. What a reviewer should check

- **Before the upload:** the dry run says READY. Its census shows one kind
  per self-play match: `raw:t1` on both seats, `[64, 64]`, `aea027b2696a`
  (`configs/reference_player.json`), served `float16`, decisions logged.
  `plan.txt` reads `30000 20000 unknown`.
- **After the upload:** the folder's listing
  (`HfApi().list_repo_files`) holds `games_s00..s05.tar.gz`,
  `walk_s00..s05.jsonl`, `labels.jsonl`, `readout.json`, `plan.txt` and
  `ALL_DONE`. `upload.log` names no failure.
- **Stage C's C0** (`c0_pass.json`, `c0.log`):
  - `floor_dtype` is `float16`;
  - `floor.kl` is near the CPU's 4e-6;
  - `replay.half_mean_square` is near 5e-6;
  - `replay.kl` is within about ±3e-5 of zero, under `replay.limit`,
    about 1.1e-4.

  A `replay.kl` near 1e-4 or above would be a departure to find, not
  noise, at 200 games. A `PositionError` (a record's turn digests) would
  mean the staged core is not the one that played the games.
- **Stage C's own gate:** its CPU tests on the box include the new
  `test_the_floor_reads_the_precision_the_games_were_served_in` in
  `tests/test_pg_pass.py`. The box image's torch (2.5.1) may not support
  CPU fp16 autocast; it then disables it with a warning, and the test's
  assertions still hold.

## 8. Are stage C's own rates (C0) unaffected?

Yes. C0 times the trainer and the evaluator on the box, on the records
(`stage_c_box.sh:252-272`), and `stage_c_plan.py` sizes the run from those
rates, the labels' counts and the switch. None of these depends on where
or in what precision the games were played.

The game length at temperature 1 enters the rate per game. It is the
policy's, measured by stage 1 wherever it runs. fp16 against bf16 sampling
changes no game's length beyond chance.

The fp16 floor costs what the bf16 floor costs: three forwards on 1 % of
the decisions.

The one thing that changes is the replay limit C0 holds the logs to:
about 1.1e-4 in place of about 5.9e-4 (section 4).

## 9. Decisions for the user

**Decided (user, 2026-10-29 evening): stage C may train on the local stage
1's fp16-served games, and the upload goes ahead.**
- Items 1, 2 and 4 are settled by it: the fp16 games, the float16 floor
  and the upload.
- Item 3 follows: a claim chains onto D0 only when played in fp16, as the
  desktop plays it.
- Item 5 stands: stage C is staged from a commit with a green CI run.
  That waits on a GitHub sign-in on this machine, and the box on a Vast key
  (`vastai set api-key`, in `%USERPROFILE%\venvs\wesnoth_vast`, the box
  tooling's own venv on the desktop).

1. **May stage C consume fp16-served games at all?** π_k's games were
   served by the HIP kernels in fp16 on the RX 6600. The trainer
   recomputes in fp32 on a 4090. The design asks that actions be drawn
   from the logged law and checks it by the replay identity at ε/20. On
   the CPU probe the logs sit about 5.5e-6 nats a decision from fp32,
   within that. π_0 is then `own2` as served in fp16: a law within
   rounding of the box's bf16 `own2`, but not the same one.
2. **The floor at the served precision.** This branch reads the floor at
   float16 for these records, `--floor-dtype auto`, now the command line's
   default. The limit falls from about 5.9e-4 under the old bf16 default
   to about 1.1e-4. Approving the branch approves that.
   - The probes are on one to five games, on the CPU's autocast.
   - The box's CUDA fp16 floor and C0's own reading on 200 games are the
     measurement.
3. **D0 and the memory report** were played in fp16 on both seats, the
   reference included. D0 is the debt the claims measure against
   (`claim_k − claim_0`). A claim played on a box in bf16 does not chain
   onto it. The claims would have to be played in the same numerics, or D0
   played again.
4. **The upload itself:** about 1.5 GB to `momom2/wesnoth-model-checkpoints`,
   including the run's logs, which name local paths. The folder name is
   `tier-b/selfplay_t1_local_20261028`.
5. **CI.** The branch has not been pushed from this machine, so it has no
   CI run. Stage C is staged only from a commit with a green one.
