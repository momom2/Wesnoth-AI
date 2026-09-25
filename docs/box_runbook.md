# Box runbook

How a run goes on a rented Vast.ai box, from the checks before renting to
the destroy. Box classes, throughput and costs measured on past boxes are
in docs/box_specs.md. Scripts that source the box library
(`scripts/box/boxlib.sh`) follow this page; the other `scripts/*_box.sh`
are records of the runs they ran.

## Quickstart

    # 1. stage the code, its box library and the run script (one HF commit)
    python tools/stage_code.py --out /tmp/stage_20260926a.tar.gz \
        --script scripts/unit_vocab_retrain_box.sh \
        --upload tier-b/staging/stage_20260926a.tar.gz

    # 2. pick an offer; rent it with the run's estimate in hours
    python scripts/rent_box.py search --min-hours 7
    python scripts/rent_box.py create OFFER_ID --onstart unit_vocab_retrain_box.sh \
        --stage tier-b/staging/stage_20260926a.tar.gz --hours 4.5 --disk 60

    # 3. watch
    python scripts/rent_box.py status INSTANCE_ID
    python tools/pull_box_records.py tier-b/unit_vocab_retrain_20260925 /tmp/peek

    # 4. once ALL_DONE is on HF: check the stop, keep the records, destroy
    python scripts/rent_box.py status INSTANCE_ID
    python tools/pull_box_records.py tier-b/unit_vocab_retrain_20260925 \
        training/metrics/bench_pipeline/unit_vocab_retrain_20260925
    python scripts/rent_box.py destroy INSTANCE_ID

## Secrets

- **HF token.** The laptop's (`huggingface_hub.get_token()`).
  `rent_box.py create` passes it to the instance as `HF_TOKEN`; the onstart
  writes it to `/workspace/.hf_token` (mode 600); `box_init` reads that file
  with `tr -d '\r\n'` (a file written on Windows ends in CR) and exports it.
- **Instance key.** Vast puts `CONTAINER_ID` and `CONTAINER_API_KEY` in the
  container. `scripts/box/box_stop.py` reads them, from PID 1's environment
  when its own lacks them, and prints only status codes, Vast's `msg` with
  the key blanked, and exception type names.
- **Account key.** The vastai SDK's configuration on the laptop.
  `rent_box.py` blanks it from every answer and error it prints.
- No key or token in a script, an argument, a URL, a commit or a log; no
  `set -x` in a box script (`box_init` turns xtrace off). `.gitignore`
  covers `.hf_token`, `*vast_api_key*` and `box.env`.

## Before renting

1. A pre-registration (`docs/*_prereg_*.md`) with the bars, the steps and
   their estimated durations, the box class and the cost.
2. The script sources the library, bounds every step from those estimates,
   and sets `BOX_MAX_H`, the dead-man's switch, at about twice the
   estimated box-hours.
3. A green CI run on the commit to stage: `gh run list --branch BRANCH --limit 1`.
4. The user's explicit yes, given the box's specs, its price, the estimate
   and the account balance. `rent_box.py create --hours H` refuses when the
   funds do not cover 1.5 x H hours of the offer, or when the offer leaves
   the market within 1.5 x H hours; Vast stops an instance when the balance
   crosses -$0.01 (2026-09-24: a run lost at 3.5 h of 6).

## Staging

`tools/stage_code.py --out TARBALL --script scripts/RUN_box.sh --upload
tier-b/staging/stage_DATE.tar.gz` writes one HF commit holding:

- the code stage: the tracked files of the working tree, uncommitted edits
  included and listed as "NOT IN ANY COMMIT"; data directories are left
  out, untracked files come only with `--extra`;
- the box library of that stage, `tier-b/staging/stage_DATE.box/`, read
  from the tarball;
- the run script, `tier-b/staging/RUN_box.sh`, read from the tarball.

It refuses a payload without the library or without a `--require` or
`--script` file, and a shell file with CR line endings.

## Bring-up

What `rent_box.py create` sets going, for a script on the library:

1. The onstart installs huggingface_hub, fetches `box_onstart.sh` from the
   stage's library folder, writes the token file, and starts
   `bash /workspace/box_onstart.sh SCRIPT LIBRARY_FOLDER` detached; its
   output, then the script's, goes to `/workspace/onstart_script.log`.
2. `box_onstart.sh` fetches `box_stop.py` first, then the rest of the
   library into `/workspace/box/`, then the script, and becomes the script.
   When a file does not arrive, it stops the instance with `box_stop.py`;
   when `box_stop.py` itself did not arrive, nothing on the box can stop
   it, and the laptop must (see Watching).
3. `box_init` takes the entry lock (one entry per records directory),
   installs the traps, starts the dead-man's switch, reads the token,
   deletes the previous entry's `ALL_DONE` and `FAILED` here and on HF, and
   restores `status.txt`, `stages.txt` and `walls.txt` when this machine
   lacks them. HF unreachable at this point finishes the entry.
4. `box_stage_code` downloads the stage into a new directory and swaps it
   in whole, records it in `.staged_from`, and refuses a stage whose
   `scripts/box/` differs from the library that is running.

A script that does not source the library gets the older onstart: the
script is fetched from `tier-b/staging/` and run.

## Renting

- `rent_box.py search` leaves out VM hosts (they never deliver the ssh
  key: four rentals lost 2026-08-24/25) and shows each offer's remaining
  rental window. Take the box class the pre-registration names.
- `rent_box.py create` checks the script, the stage and the stage's
  library on HF, the funds and the window, then creates. The stage reaches
  the box as `STAGE`.
- `success: false` with a contract id: the instance exists, STOPPED.
  `rent_box.py start ID` brings it up (the onstart runs then), or destroy it.
- A start answered "state change queued" starts later on its own and bills
  idle: cancel it at once with `rent_box.py stop ID`.
- No container after 10 minutes (`actual_status` None, no disk activity):
  destroy and take another offer.

## Watching

- On HF, under the run's `HF_DIR`: `stages.txt` (one line per entry),
  `status.txt` (one line per finish), `walls.txt` (one line per step: its
  exit code, `ok`, `cut`, `stalled`, `until` or `failed`, and its seconds),
  `box.txt`, the step logs, `progress.txt`, `upload.log` (every file's size,
  seconds and rate; a checkpoint under 1 MB/s means a slow uplink),
  `deadman.log`, `watchdog.log`, `onstart_script.log`. A round goes up every
  30 minutes and after milestones.
- `rent_box.py status ID` for the instance's state, `rent_box.py logs ID
  --tail 50` for the container's output.
- Nothing on HF 20 minutes after the create: read `rent_box.py logs ID`,
  and stop the box if its bring-up failed.
- A job past 1.5 x its estimate is inspected and cut
  (`rent_box.py stop ID`), not waited on. The dead-man's switch finishes
  the entry at `BOX_MAX_H` in any case.

## Recovery

- **Re-entry** is the script running again: after `rent_box.py start ID`
  (the onstart runs at every start and fetches the script as it is on HF
  staging then), or on a new rental with the same script and `--stage`.
  The entry clears the previous `ALL_DONE` and `FAILED`; the files the
  script lists come back from HF when this machine lacks them; markers
  skip finished steps; an imitation pass continues where it was cut
  (`tools/supervised_train.py` `PassPosition`).
- **Markers** in the records directory (`DONE`, `STOPPED_AFTER_EPOCH`)
  name the stage that wrote them. The wheel and the tests are redone for
  each stage on each machine (their markers live in `BOX_STATE`, which
  never goes to HF).
- **Another machine.** A stopped instance keeps its disk but not its GPU:
  the host can rent the GPU out. Plan around HF, not the disk: a new
  rental resumes from what HF holds.

## Finishing

- Every finish writes the reason to `status.txt` and `ALL_DONE` (and to
  `FAILED` when it failed), uploads the records with `ALL_DONE` last, then
  stops the instance through `box_stop.py`, which counts a stop only when
  Vast answers `success: true`.
- A stop Vast refused goes to HF: `stop.log` ends in "the instance is NOT
  stopped" and the last line of `stop.jsonl` reads `"stopped": false`.
  Stop the instance from the laptop (`rent_box.py stop ID`); the dead-man's
  switch stays armed and tries again at `BOX_MAX_H`.
- Check the stop with `rent_box.py status ID`.
- Pull the records into the repository with `tools/pull_box_records.py
  HF_DIR DESTINATION` (it leaves files over `--max-mb` on HF), then
  `rent_box.py destroy ID`: a stopped instance bills its storage.

## Writing a box script

    #!/usr/bin/env bash
    # shellcheck source-path=SCRIPTDIR
    set -uo pipefail
    WORKDIR=/workspace
    BOX_OUT=$WORKDIR/myrun
    export HF_DIR="${HF_DIR:-tier-b/myrun_20260926}"
    STAGE="${STAGE:-}"
    BOX_MAX_H="${BOX_MAX_H:-4}"
    # shellcheck source=box/boxlib.sh
    . "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || { echo "no box library"; exit 1; }
    box_init
    [ -n "$STAGE" ] || box_finish "NO_STAGE" 1
    box_restore result.json || box_finish "RESTORE_FAILED (restore.log)" 1
    box_stage_code || box_finish "CODE_STAGING_FAILED (staging.log)" 1
    cd "$BOX_REPO" || box_finish "CODE_STAGING_FAILED" 1
    box_build_wheel || box_finish "BUILD_FAILED (build.log)" 1
    box_monitor_start
    box_bounded measure 90 measure.log python tools/YOUR_TOOL.py --out "$BOX_OUT/result.json" \
        || box_finish "MEASURE_$BOX_WHY rc=$BOX_RC (measure.log)" 1
    box_finish "MYRUN_DONE"

`scripts/unit_vocab_retrain_box.sh` is a complete example: restores,
markers, a watched training step, a match whose games go up as a tarball.

| Function | Does |
|---|---|
| `box_init` | the entry's bring-up (see Bring-up, step 3) |
| `box_finish REASON [RC]` | records, uploads with ALL_DONE last, stops the instance, exits RC |
| `box_bounded [--stall FILE MIN] [--until FILE TEXT] NAME CUT_MIN LOG CMD...` | runs CMD cut at CUT_MIN, output to LOG; ends it when FILE stops growing or TEXT appears; sets `BOX_RC`, `BOX_WHY` |
| `box_restore NAME...` | files absent here come back from HF; fails when HF cannot answer |
| `box_upload_dir NAME PATH` | PATH goes up as NAME.tar.gz (a match's games) |
| `box_upload_hold NAME DEP...` | NAME goes up only once each DEP has landed in its current version |
| `box_upload_skip NAME` / `box_upload_extra PATH` | keep NAME on the box / send a file outside BOX_OUT |
| `box_upload`, `box_upload_async`, `box_monitor_start` | one round; one in the background; one every 30 minutes, after the script's `box_on_round` |
| `box_stage_code`, `box_build_wheel`, `box_facts` | the stage, the Rust wheel checked against lib.rs's phase, box.txt |
| `box_mark PATH`, `box_marked_this_stage PATH` | a marker naming its stage; whether this stage wrote it |

The library expects `set -uo pipefail`, not `set -e`. Call `box_finish` in
the script's own shell, never in `$(...)` or a pipeline, where its `exit`
would end only a subshell.
