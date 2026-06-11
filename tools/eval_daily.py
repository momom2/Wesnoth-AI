"""Daily checkpoint-vs-RCA eval wrapper with rolling history.

Picks the freshest self-play checkpoint, runs a quick eval against
Wesnoth's built-in RCA AI, and appends one row to a rolling CSV +
markdown history. Daily workflow:

    1. sim_self_play.py trains (overnight / in the background)
    2. eval_daily.py runs a 30-game preset against RCA    [this]
    3. operator inspects training/eval_history.md

The eval itself goes through `tools/eval_vs_builtin.py` -- this script
just wraps it with sensible "daily quick read" defaults (a small map
+ pair subset, no swap, all in one Wesnoth-launch session) and the
history-writing step. Default preset is roughly:

    --maps caves den --pairs cross --no-swap --parallel 2

which is 2 maps x 15 cross-pairs = 30 games. At ~2-3 min/game on
2 parallel that's ~30-45 minutes locally -- fits inside a coffee
break, big enough to detect a real WR shift across a day's training.

History format:
  training/eval_history.csv  -- one row per eval, machine-readable
  training/eval_history.md   -- same data, human-readable table

Both grow forever (an eval row is ~200 bytes), so the operator can
plot WR over training time without re-running old evals.

Usage:
    python tools/eval_daily.py
    python tools/eval_daily.py --checkpoint training/checkpoints/foo.pt
    python tools/eval_daily.py --preset full   # full grid, ~hours
    python tools/eval_daily.py --dry-run       # print plan, don't run
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from constants import WESNOTH_USERDATA_PATH


log = logging.getLogger("eval_daily")


# ---------------------------------------------------------------------
# Sound muting
# ---------------------------------------------------------------------

# Wesnoth preferences keys to mute for the eval. The file is WML-ish
# `key=value` lines at top level (sectioned blocks like
# `[multiplayer]` exist but the sound prefs are not nested). The
# values below match Wesnoth's parser: `yes/no` for booleans, integer
# strings for volumes (range 0-100). Restored to the user's saved
# values after the eval finishes.
_MUTE_KEYS: Dict[str, str] = {
    "sound":         "no",
    "music":         "no",
    "UI_sound":      "no",
    "turn_bell":     "no",
    "sound_volume":  "0",
    "music_volume":  "0",
    "UI_volume":     "0",
    "bell_volume":   "0",
}


@contextlib.contextmanager
def _muted_wesnoth_sound(prefs_path: Path):
    """Mute Wesnoth's preferences for the duration of the with-block.

    Why this exists: the daily eval spawns many Wesnoth processes in
    sequence; without muting, the operator's machine plays ~30
    games' worth of opening fanfares + battle SFX. The `--nosound`
    CLI flag would be simpler IF we controlled every spawn site, but
    `tools/eval_runner.py` builds the command line and threading
    flags through that layer would be invasive. Editing the user's
    preferences file is universal: every Wesnoth process spawned
    while the file is muted comes up silent, regardless of how it's
    launched.

    Safety:
      - Wesnoth reads `preferences` at startup and writes it back on
        clean exit. While Wesnoth is running it does NOT re-read,
        so flipping the file between spawns is safe.
      - `try/finally` guarantees restoration even on
        KeyboardInterrupt / crash inside the with-block.
      - If restore fails (disk full, locked file), the original text
        is saved as `preferences.restore_failed` next to the
        original so the user can recover by hand.
      - Atomic write via tmp + os.replace.

    Not thread-safe: don't run two evals in parallel against the
    same prefs file -- the second "save" would read the first's
    muted state and "restore" to silence.
    """
    if not prefs_path.exists():
        log.warning(f"[sound] preferences file not at {prefs_path}; "
                    f"running without muting")
        yield
        return

    # Read+write as BYTES to preserve the file exactly. write_text /
    # read_text on Windows go through text-mode newline translation,
    # which silently turns `\r\n` into `\r\r\n` on write and the
    # reverse on read -- the round-trip looks identical to Python
    # but the on-disk bytes diverge, and Wesnoth's WML parser is
    # sensitive to stray `\r`s. write_bytes/read_bytes bypass that.
    original_bytes = prefs_path.read_bytes()
    original_text = original_bytes.decode("utf-8", errors="replace")
    # Track top-level keys only. Sectioned blocks (`[name]` ... `[/name]`)
    # increase depth; we only modify keys at depth 0. Wesnoth's prefs
    # historically keeps sound flags top-level, but if a future version
    # moves them this guard prevents corrupting a nested block.
    original_lines = original_text.splitlines(keepends=True)
    found: Dict[str, int] = {}     # key -> line index
    depth = 0
    for i, line in enumerate(original_lines):
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            if s.startswith("[/"):
                depth -= 1
            else:
                depth += 1
            continue
        if depth != 0 or "=" not in s:
            continue
        k = s.split("=", 1)[0].strip()
        if k in _MUTE_KEYS and k not in found:
            found[k] = i

    # Detect dominant line ending so appended keys match the file's
    # style. If the file is empty we default to "\n".
    line_end = "\n"
    for ln in original_lines:
        if ln.endswith("\r\n"):     line_end = "\r\n"; break
        elif ln.endswith("\n"):     line_end = "\n";   break

    muted_lines = list(original_lines)
    # Ensure the file ends with a newline before appending new keys
    # so we don't produce `lastkey=valNEWKEY=val`.
    if muted_lines and not muted_lines[-1].endswith(("\n", "\r\n")):
        muted_lines[-1] = muted_lines[-1] + line_end
    for key, muted_val in _MUTE_KEYS.items():
        if key in found:
            muted_lines[found[key]] = f"{key}={muted_val}{line_end}"
        else:
            muted_lines.append(f"{key}={muted_val}{line_end}")
    muted_text = "".join(muted_lines)

    tmp = prefs_path.with_suffix(prefs_path.suffix + ".eval_daily_tmp")

    def _atomic_write(text: str) -> None:
        # write_bytes to skip Windows newline translation; see comment
        # on the read step above for the same reason.
        tmp.write_bytes(text.encode("utf-8"))
        os.replace(tmp, prefs_path)

    try:
        _atomic_write(muted_text)
        log.info(
            f"[sound] muted Wesnoth ({len(found)}/{len(_MUTE_KEYS)} "
            f"existing keys updated, "
            f"{len(_MUTE_KEYS) - len(found)} added)"
        )
        yield
    finally:
        # Always-fires restore. If even THIS fails, dump the original
        # text to a backup file next to the prefs and surface the
        # error loudly -- the user can hand-restore from the backup.
        try:
            _atomic_write(original_text)
            log.info("[sound] restored original Wesnoth preferences")
        except OSError as e:
            backup = prefs_path.with_suffix(
                prefs_path.suffix + ".restore_failed")
            try:
                backup.write_bytes(original_bytes)
            except OSError:
                pass
            log.error(f"[sound] FAILED to restore preferences ({e}); "
                      f"saved original text to {backup}. Hand-restore "
                      f"by moving that file over the current "
                      f"`preferences` file.")
        finally:
            # Clean up the tmp file if it lingered.
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


# Two backends:
#   - sim:     in-process simulator, candidate-vs-reference. Fast
#              (~30-60s for 30 games on DML/CUDA). The default.
#   - wesnoth: spawn real Wesnoth subprocesses vs the built-in RCA
#              AI. Slow (~30-45min for 30 games). Cross-check.
#
# Presets are backend-specific because the game count / wall-time
# trade-off differs by orders of magnitude. `quick` / `standard` /
# `full` are the same labels but mean different things per backend.
SIM_PRESETS: Dict[str, List[str]] = {
    "quick":    ["--games", "30",  "--max-turns", "60"],
    "standard": ["--games", "100", "--max-turns", "60"],
    "full":     ["--games", "300", "--max-turns", "60"],
}
WESNOTH_PRESETS: Dict[str, List[str]] = {
    "quick":    ["--maps", "caves", "den",
                 "--pairs", "cross", "--no-swap", "--parallel", "2"],
    "standard": ["--maps", "caves", "den", "freeport",
                 "--pairs", "cross", "--parallel", "3"],
    "full":     ["--parallel", "4"],
}


def _presets_for(backend: str) -> Dict[str, List[str]]:
    return SIM_PRESETS if backend == "sim" else WESNOTH_PRESETS


# Backwards-compatibility alias: legacy callers / GUI code that
# reads PRESETS keys for dropdown population still works.
PRESETS = SIM_PRESETS


def _pick_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Pick the freshest self-play .pt under `ckpt_dir`. Prefers
    `sim_selfplay.pt` (the rolling self-play target) if present;
    falls back to the highest-mtime *.pt. Returns None if none found
    so the caller can warn instead of crashing."""
    sp = ckpt_dir / "sim_selfplay.pt"
    if sp.exists():
        return sp
    cands = sorted(ckpt_dir.glob("*.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _parse_summary(summary_path: Path) -> Dict:
    """Pull the headline numbers from `eval_vs_builtin --save-json`
    output. Keys we extract:

      n_games, wins, losses, draws, timeouts, errored,
      decisive_n, win_rate (None if no decisive games),
      checkpoint, wall_seconds.

    The rest of the JSON (per-faction / per-map / per-matchup) we
    don't summarize into the history row -- it's still on disk under
    `training/eval_runs/<timestamp>.json` if the operator wants a
    deeper look. Keep the history file narrow so plotting stays easy.
    """
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    overall = data.get("overall") or {}
    wins      = int(overall.get("win",     0))
    losses    = int(overall.get("loss",    0))
    draws     = int(overall.get("draw",    0))
    timeouts  = int(overall.get("timeout", 0))
    errored   = int(overall.get("errored", 0))
    decisive  = wins + losses
    win_rate  = (wins / decisive) if decisive > 0 else None
    # Backend tag + reference checkpoint are only present in sim
    # backend output. Defaults keep the wesnoth path unchanged.
    backend   = str(data.get("backend", "wesnoth"))
    reference = data.get("reference")
    return {
        "n_games":      int(data.get("n_games", 0)),
        "wins":         wins,
        "losses":       losses,
        "draws":        draws,
        "timeouts":     timeouts,
        "errored":      errored,
        "decisive_n":   decisive,
        "win_rate":     win_rate,
        "checkpoint":   str(data.get("checkpoint", "")),
        "wall_seconds": float(data.get("wall_seconds", 0.0)),
        "backend":      backend,
        "reference":    str(reference) if reference else "",
    }


def _migrate_legacy_history(csv_path: Path, md_path: Path) -> None:
    """Archive pre-backend-aware history files when present.

    The schema gained `backend` + `reference` columns when the
    sim-backend eval landed. Old files have a narrower header; new
    rows are wider. To avoid jagged tables, on first run after the
    upgrade we rename existing files with a `.legacy.<suffix>` tail
    and start fresh.

    Detection: any file whose CSV header doesn't include `backend`
    (or markdown header lacking the `backend` column) is a legacy
    file. Idempotent: once renamed, the next run sees no file at
    the canonical path and writes fresh.

    The legacy data isn't lost -- the operator can still read /
    plot from `eval_history.legacy.csv`.
    """
    for path, sentinel in ((csv_path, "backend"),
                           (md_path,  "| backend ")):
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        # Sentinel can appear anywhere in the file -- CSV header
        # is the first line, markdown table header is several
        # lines in (after the # Eval history title + intro
        # paragraph + Backends list). A whole-file substring
        # check covers both without false negatives.
        if sentinel in text:
            continue       # already new-format
        legacy = path.with_suffix(path.suffix + ".legacy")
        # If a .legacy from a previous migration is sitting there,
        # don't double-rename (rename fails on Windows when target
        # exists). The previous migration already preserved the
        # old data; leave it in place and just delete the stale
        # old-format file so the next-run write creates a fresh
        # new-format file from scratch.
        try:
            if legacy.exists():
                path.unlink()
                log.info(f"removed stale old-format {path.name} "
                         f"(prior {legacy.name} already preserved)")
            else:
                path.rename(legacy)
                log.info(f"archived legacy {path.name} -> {legacy.name}")
        except OSError as e:
            log.warning(f"couldn't archive {path}: {e}")


def _append_csv(csv_path: Path, row: Dict) -> None:
    """Append `row` to `csv_path`, writing the header on first call.
    Column order is fixed so external plotters/spreadsheets stay
    happy across eval runs.
    """
    header = [
        "timestamp", "backend", "preset", "checkpoint", "reference",
        "n_games", "decisive_n",
        "wins", "losses", "draws", "timeouts", "errored",
        "win_rate", "wall_seconds",
    ]
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


def _append_markdown(md_path: Path, row: Dict) -> None:
    """Append `row` to `md_path` as a table row. Writes header + the
    intro blurb on first call so the file is self-explanatory if a
    teammate opens it without context."""
    new_file = not md_path.exists()
    md_path.parent.mkdir(parents=True, exist_ok=True)
    wr_cell = (
        f"{100.0 * row['win_rate']:.1f}%"
        if row["win_rate"] is not None else "n/a"
    )
    ckpt_short = Path(row["checkpoint"]).name if row["checkpoint"] else ""
    ref_short  = (Path(row.get("reference", "")).name
                  if row.get("reference") else "")
    backend    = row.get("backend", "")
    with md_path.open("a", encoding="utf-8") as f:
        if new_file:
            f.write(
                "# Eval history\n\n"
                "One row per `tools/eval_daily.py` run. Win rate is "
                "wins / decisive games (excludes draws + timeouts + "
                "errors). Per-faction / per-matchup cuts live in "
                "the JSON files under `training/eval_runs/`.\n\n"
                "Backends:\n"
                "- **sim**: candidate checkpoint vs reference "
                "checkpoint, in-process simulator. Fast.\n"
                "- **wesnoth**: candidate vs Wesnoth's built-in RCA "
                "AI, real Wesnoth subprocesses. Slow but tests "
                "against an external opponent.\n\n"
                "| timestamp | backend | preset | checkpoint | reference "
                "| n | decisive | W | L | D | T | E | WR | wall |\n"
                "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
            )
        f.write(
            f"| {row['timestamp']} | {backend} | {row['preset']} | "
            f"{ckpt_short} | {ref_short} | "
            f"{row['n_games']} | {row['decisive_n']} | "
            f"{row['wins']} | {row['losses']} | {row['draws']} | "
            f"{row['timeouts']} | {row['errored']} | {wr_cell} | "
            f"{row['wall_seconds']:.0f}s |\n"
        )


def main(argv: List[str]) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Specific checkpoint to eval. Default: "
                         "freshest *.pt under training/checkpoints/ "
                         "(prefers sim_selfplay.pt).")
    ap.add_argument("--ckpt-dir", type=Path,
                    default=Path("training/checkpoints"),
                    help="Where to look for checkpoints.")
    ap.add_argument("--history-csv", type=Path,
                    default=Path("training/eval_history.csv"),
                    help="Where to append the per-run summary row.")
    ap.add_argument("--history-md", type=Path,
                    default=Path("training/eval_history.md"),
                    help="Where to append the per-run markdown row.")
    ap.add_argument("--runs-dir", type=Path,
                    default=Path("training/eval_runs"),
                    help="Where to keep the full per-run JSON dumps "
                         "(per-faction / per-map / per-matchup cuts).")
    ap.add_argument("--backend", choices=("sim", "wesnoth"),
                    default="sim",
                    help="Eval backend. 'sim' (default): in-process "
                         "simulator, candidate vs reference "
                         "checkpoint; fast (~30-60s for 30 games). "
                         "'wesnoth': spawn real Wesnoth subprocesses "
                         "vs the built-in RCA AI; slow but tests "
                         "against an external opponent.")
    ap.add_argument("--reference", default="auto",
                    help="(sim backend only) Reference checkpoint "
                         "to play against. 'auto' = freshest "
                         "sim_selfplay_archive_*.pt (falls back to "
                         "freshest supervised_epoch). 'random' = "
                         "random-init baseline. Or pass a path.")
    ap.add_argument("--preset", choices=sorted(SIM_PRESETS),
                    default="quick",
                    help="Eval scope. Meaning is backend-specific. "
                         "sim: quick=30 / standard=100 / full=300 "
                         "games. wesnoth: quick=30 games "
                         "(~30-45min) / standard=90 (~2h) / "
                         "full=252 (~21h).")
    ap.add_argument("--device", default="auto",
                    help="Torch device. 'auto' = DML (discrete) > "
                         "CUDA > CPU; passed through to the eval "
                         "sub-runner.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the eval_vs_builtin invocation, "
                         "DON'T launch Wesnoth or update history.")
    ap.add_argument("--note", default="",
                    help="Optional free-text note recorded alongside "
                         "the eval row (e.g. 'reward v3' or 'after "
                         "lr drop'). Lands in the markdown only.")
    args = ap.parse_args(argv[1:])

    # -- 1. Pick the checkpoint --
    if args.checkpoint is not None:
        ckpt = args.checkpoint
        if not ckpt.exists():
            log.error(f"checkpoint not found: {ckpt}")
            return 1
    else:
        ckpt = _pick_latest_checkpoint(args.ckpt_dir)
        if ckpt is None:
            log.error(f"no checkpoints under {args.ckpt_dir} -- train "
                      f"one via tools/sim_self_play.py first, or pass "
                      f"--checkpoint explicitly")
            return 1
        log.info(f"auto-picked checkpoint: {ckpt}")

    # -- 2. Build invocation (per backend) --
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.runs_dir / f"eval_{args.backend}_{ts}.json"
    presets = _presets_for(args.backend)
    # Pick python.exe for the child even if WE were launched via
    # pythonw.exe -- pythonw's NUL stdio handles get inherited and
    # silently crash subprocess output paths (a direct
    # `pythonw eval_daily.py` invocation would bite without this).
    py_exe = sys.executable
    if os.name == "nt":
        _d, _b = os.path.split(py_exe)
        if _b.lower() == "pythonw.exe":
            _cand = os.path.join(_d, "python.exe")
            if os.path.exists(_cand):
                py_exe = _cand
    if args.backend == "sim":
        cmd = [
            py_exe, "-u", str(_THIS.parent / "eval_sim.py"),
            "--checkpoint", str(ckpt),
            "--reference",  args.reference,
            "--device",     args.device,
            "--save-json",  str(out_json),
            *presets[args.preset],
        ]
    else:  # wesnoth
        cmd = [
            py_exe, "-u", str(_THIS.parent / "eval_vs_builtin.py"),
            "--checkpoint", str(ckpt),
            "--device",     args.device,
            "--save-json",  str(out_json),
            *presets[args.preset],
        ]
    log.info(f"backend={args.backend} preset={args.preset}: "
             f"{' '.join(cmd)}")
    if args.dry_run:
        log.info("--dry-run: skipping eval launch + history update")
        return 0

    # -- 3. Run the sub-eval --
    # Wesnoth backend wraps in the sound-mute context (the only path
    # that actually spawns Wesnoth processes). Sim backend doesn't
    # touch the preferences file -- no Wesnoth involved.
    if args.backend == "wesnoth":
        prefs_path = WESNOTH_USERDATA_PATH / "preferences"
        with _muted_wesnoth_sound(prefs_path):
            rc = subprocess.run(cmd).returncode
    else:
        rc = subprocess.run(cmd).returncode
    if rc != 0:
        log.error(f"sub-eval exited with code {rc}; not appending "
                  f"to history")
        return rc

    if not out_json.exists():
        log.error(f"sub-eval finished but {out_json} is missing; "
                  f"not appending to history")
        return 1

    # -- 4. Append history --
    # If the history files predate the backend column, archive
    # them so the new rows don't jam jagged into the old table.
    _migrate_legacy_history(args.history_csv, args.history_md)
    summary = _parse_summary(out_json)
    row = {
        "timestamp":  ts,
        "preset":     args.preset,
        "backend":    args.backend,
        **summary,
    }
    _append_csv(args.history_csv, row)
    _append_markdown(args.history_md, row)
    log.info(
        f"history updated: "
        f"WR={100.0*summary['win_rate']:.1f}% "
        f"({summary['wins']}/{summary['decisive_n']}); "
        f"row appended to {args.history_csv} and {args.history_md}"
        if summary["win_rate"] is not None else
        f"history updated (no decisive games this run); row "
        f"appended to {args.history_csv} and {args.history_md}"
    )
    if args.note:
        # Tack the note on a separate line so it doesn't break the
        # markdown table's column count.
        with args.history_md.open("a", encoding="utf-8") as f:
            f.write(f"  - note: {args.note}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
