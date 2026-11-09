# Stage 1 on the local GPU: the run's small records (2026-10-29)

The pre-registration and the reading are in
`docs/selfplay_t1_local_prereg_20261028.md` ("Results"). The run's output
directory, `C:\Users\momom2\wesnoth_runs\selfplay_t1_local` (1.7 GB: the
46,000 game records with their decision logs, the luck walks, rows.jsonl and
labels.jsonl), stays on the desktop until the user decides where it goes.

The runner served from the frozen checkout `wesnoth_ai-run-stage1` at
fc560ab9; the code that played the games is 115be5d1's.

| File | What it is |
|---|---|
| `matches.txt` | One line per match attempt: games, seeds, exit code, wall, results |
| `walls.txt` | Each match's wall in seconds, one per attempt. s02's 11 s is its resumed attempt only; its games were played 13:44-15:02, 4,690 s of server wall |
| `server_stats.txt` | The inference server's summary line at the end of each match |
| `readout.json`, `readout.log` | `tools/selfplay_readout.py`: the census, the five gates, label noise by correction, the sharpening slope, P1's targets |
| `fits.json`, `labels.log` | `tools/pg_labels.py`: the luck baseline fitted on the held-out games (`train`) and on each held-out half |
| `d0.fit.json`, `fit_d0.log` | D0: `own2t1` against the reference, 12,800 decisive games |
| `memory.fit.json`, `fit_memory.log` | The memory report: `own2m64` against `own2m0`, both at temperature 1, 3,200 decisive games |
| `runner.log`, `progress.txt` | The runner's log, with the hand-written notes of every stop and resume |

## The server, match by match

Requests served a second over the server's wall (requests / wall, from
`server_stats.txt`); one server, 96 games in flight, fp16:

| Match | Requests | Wall (s) | Requests a second | Note |
|---|---|---|---|---|
| s00 | 438,479 | 1,083 | 404.9 | the resumed part (1,512 games), ramp included |
| D0 | 4,289,436 | 9,978 | 429.9 | games at argmax on side B are longer |
| s01 | 1,910,926 | 4,091 | 467.1 | |
| s02 | 1,890,221 | 4,690 | 403.0 | the two display resets and other sessions' benchmarks |
| s03 | 1,907,155 | 6,116 | 311.9 | the machine shared with other sessions' CPU work (see below) |
| s04 | 1,882,524 | 4,158 | 452.7 | |
| s05 | 1,864,572 | 4,139 | 450.5 | |
| memory | 1,192,552 | 2,425 | 491.8 | half the games at 0 slots |

- s03 ran 15:35-17:17, while other sessions on the machine ran CPU
  benchmarks and tests. I believe that is the slowdown; the run's own
  logs do not show it.
- The server's `gpu` seconds are not read. They went from 3,413 (s01) to
  472 (s04) between two matches with the same code and rate, so on this
  path they do not measure GPU time.
- `t_finish`, the wait for the forward to finish, is 0.83-0.85 of each
  match's wall, and 0.865 and 0.896 in the two contended ones (s02, s03).
