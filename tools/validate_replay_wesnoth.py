"""Layer 2 replay validation: play a sim-exported replay back in REAL
Wesnoth and fail on out-of-sync markers in the engine log.

This is the ground-truth check layer 1 approximates: Wesnoth itself
re-simulates the recorded commands during replay playback and logs
desyncs.

Two findings shape the launch procedure (2026-07-06):

1. `--load <replay> --with-replay` opens the replay viewer STOPPED.
   1.18.4 src/replay_controller.cpp constructs with the base
   `replay_stop_condition` (should_stop() == true), so playback only
   starts when something calls play_replay() — i.e. the theme's Play
   button or the `playreplay` hotkey. Every "working" harness run
   before this fix had a human pressing Play; unattended runs sat at
   replay action 1 forever.

2. `playreplay` has NO default key binding (hotkey_command.cpp:
   `{ HOTKEY_REPLAY_PLAY, "playreplay", ..., "" }`), so we launch
   with an isolated `--userdata-dir` whose preferences bind
   key "p" -> playreplay, and PostMessage that key to the (minimized,
   never-focused) window until the log shows playback progressing.
   The isolated userdata also keeps the user's real preferences and
   saves untouched; its WML cache persists across runs.

Used by the opt-in pytest (WESNOTH_E2E=1) and manually:

    python tools/validate_replay_wesnoth.py logs/validate_mini.bz2
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import WESNOTH_PATH

log = logging.getLogger("validate_replay_wesnoth")

# Isolated userdata: survives across runs so the WML cache is only
# built once (~1 min); deliberately NOT the session scratchpad.
USERDATA_DIR = Path(tempfile.gettempdir()) / "wesnoth_ai_validate_userdata"

# preferences WML format per 1.18.4 src/hotkey/hotkey_item.cpp
# save_helper: [hotkey] command=..., key=<lowercased key name>.
# Speed prefs (2026-07-15, user request -- playback was wasting
# minutes on animations): turbo x16 (max UI choice), no map/water
# animation, no action scrolling, no audio; skip_ai_moves is
# harmless belt-and-braces (only fires for AI-controller sides;
# our exported sides are human). `replayskipanimation` (bound to
# "s", unbound by default like playreplay) TOGGLES
# play_controller::skip_replay_ -- posted exactly ONCE after
# playback visibly starts, since a second press would toggle the
# skip back off. All of this lives in the ISOLATED userdata
# profile, so the user's own Wesnoth preferences are untouched.
_PREFERENCES = (
    '[hotkey]\n\tcommand="playreplay"\n\tkey="p"\n[/hotkey]\n'
    '[hotkey]\n\tcommand="replayskipanimation"\n\tkey="s"\n[/hotkey]\n'
    'turbo=yes\n'
    'turbo_speed=16\n'
    'skip_ai_moves=yes\n'
    'animate_map=no\n'
    'animate_water=no\n'
    'scroll_to_action=no\n'
    'sound=no\n'
    'music=no\n'
)

# Engine log signatures of replay desync, calibrated 2026-07-06 by
# playing a KNOWN-BAD replay with --log-debug=replay and reading what
# the engine actually wrote (don't guess — the first marker list
# false-PASSED the bad replay):
#   error replay: Recruiting leader not found at 6,5.
#   error replay: unfound location for source of movement: 5,5 -> 4,4
#   error replay: unfound location for source of attack
#   error replay: received a synced [command] from side 1. Expacted...
#   error replay: found dependent command in replay while is_synced=false
# " error replay:" prefixes them all; keep a couple of generic OOS
# phrases as belt-and-suspenders.
_OOS_MARKERS = (" error replay:", "out of sync", "desync",
                "checksum mismatch", "verification failed")
# Playback-progress signature: "up to replay action X/Y" (info-level,
# needs --log-debug=replay). X==Y means the replay played to the end.
_PROGRESS_RE = re.compile(r"up to replay action (\d+)/(\d+)")
# A run that never progressed past the start events never actually
# played the replay — report that as a FAILURE, not a pass.
_MIN_PROGRESS = 2


if os.name == "nt":
    _u32 = ctypes.windll.user32
    from ctypes import wintypes
    _ENUM = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND,
                               wintypes.LPARAM)
    _WM_KEYDOWN, _WM_KEYUP = 0x100, 0x101
    _VK_P, _VK_S, _VK_RETURN = 0x50, 0x53, 0x0D
    _SW_SHOWMINNOACTIVE = 7

    def _windows_of(pid: int) -> list:
        got = []

        def _cb(hwnd, _lparam):
            p = wintypes.DWORD()
            _u32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
            if p.value == pid and _u32.IsWindowVisible(hwnd):
                got.append(hwnd)
            return True

        _u32.EnumWindows(_ENUM(_cb), 0)
        return got

    def _minimize(pid: int) -> bool:
        found = False
        for h in _windows_of(pid):
            _u32.ShowWindow(h, _SW_SHOWMINNOACTIVE)
            found = True
        return found

    def _post_hotkey(pid: int, vk: int) -> None:
        """PostMessage a bound key — reaches SDL's message pump
        without focusing/raising the window (no focus stealing)."""
        scan = _u32.MapVirtualKeyW(vk, 0)
        lp_down = 1 | (scan << 16)
        lp_up = 1 | (scan << 16) | (1 << 30) | (1 << 31)
        for h in _windows_of(pid):
            _u32.PostMessageW(h, _WM_KEYDOWN, vk, lp_down)
            _u32.PostMessageW(h, _WM_KEYUP, vk, lp_up)

    def _post_play_hotkey(pid: int) -> None:
        _post_hotkey(pid, _VK_P)

    def _post_skip_hotkey(pid: int) -> None:
        _post_hotkey(pid, _VK_S)

    def _post_enter(pid: int) -> None:
        _post_hotkey(pid, _VK_RETURN)
else:
    def _minimize(pid: int) -> bool:      # pragma: no cover
        return False

    def _post_play_hotkey(pid: int) -> None:  # pragma: no cover
        pass

    def _post_skip_hotkey(pid: int) -> None:  # pragma: no cover
        pass

    def _post_enter(pid: int) -> None:  # pragma: no cover
        pass


def _engine_logs() -> set:
    # Engine logs go to `wesnoth-*.log`; the `.out.log` twins carry
    # only Lua std_print output (learned 2026-07-06 — tailing the
    # wrong one made every verdict vacuous).
    d = USERDATA_DIR / "logs"
    if not d.exists():
        return set()
    return {p for p in d.glob("wesnoth-*.log")
            if not p.name.endswith(".out.log")}


def _validate_once(replay: Path, timeout: float = 420.0,
                   settle: float = 30.0) -> list:
    """One playback pass; see validate_in_wesnoth for the contract."""
    if not WESNOTH_PATH.exists():
        raise RuntimeError(f"Wesnoth not found at {WESNOTH_PATH}")
    saves = USERDATA_DIR / "saves"
    saves.mkdir(parents=True, exist_ok=True)
    prefs = USERDATA_DIR / "preferences"
    if (not prefs.exists()
            or "replayskipanimation"
            not in prefs.read_text(encoding="utf-8")):
        # rewrite also upgrades pre-2026-07-15 isolated profiles
        # (play hotkey only, no turbo/skip prefs).
        prefs.write_text(_PREFERENCES, encoding="utf-8")
    shutil.copy2(replay, saves / replay.name)

    pre_logs = _engine_logs()
    cmd = [str(WESNOTH_PATH), "--userdata-dir", str(USERDATA_DIR),
           "--load", replay.name, "--with-replay", "--nodelay",
           "--log-debug=replay"]
    log.info("launching: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd)

    offending: list = []
    progress = 0          # highest X seen in "action X/Y"
    total = None          # Y
    minimized = False
    skip_sent = False
    log_file = None
    pos = 0
    t0 = time.time()
    last_growth = t0
    _last_nudge = 0.0
    _last_enter = 0.0
    try:
        while time.time() - t0 < timeout:
            if not minimized:
                minimized = _minimize(proc.pid)
            if log_file is None:
                cand = [p for p in _engine_logs() if p not in pre_logs]
                if cand:
                    log_file = max(cand,
                                   key=lambda p: p.stat().st_mtime)
                else:
                    time.sleep(1.0)
                    continue
            try:
                with log_file.open("r", encoding="utf-8",
                                   errors="replace") as f:
                    f.seek(pos)
                    new = f.read()
                    pos = f.tell()
            except OSError:
                time.sleep(1.0)
                continue
            if new:
                last_growth = time.time()
                for line in new.splitlines():
                    low = line.lower()
                    m = _PROGRESS_RE.search(low)
                    if m:
                        progress = max(progress, int(m.group(1)))
                        total = int(m.group(2))
                    if any(mk in low for mk in _OOS_MARKERS):
                        offending.append(line.strip())
            if offending:
                return offending          # fail fast on first OOS
            # The viewer opens STOPPED (see module docstring); keep
            # nudging the play hotkey until playback visibly moves.
            if (progress < _MIN_PROGRESS
                    and time.time() - _last_nudge >= 5.0):
                # Nudge sparingly: a 'p' landing on the wrong UI
                # state can flip the viewer into take-control mode,
                # whose locally-generated [choose] answers mismatch
                # the recorded stream -- a FALSE OOS (2026-07-19
                # false-positive class; see validate_in_wesnoth's
                # retry).
                _post_play_hotkey(proc.pid)
                _last_nudge = time.time()
            elif not skip_sent:
                # Playback confirmed running -> hotkeys reach the
                # replay controller. Toggle skip-animation exactly
                # once (it's a TOGGLE; repeats would turn it off).
                _post_skip_hotkey(proc.pid)
                skip_sent = True
            if total is not None and progress >= total:
                log.info("replay played to the end (%d/%d)",
                         progress, total)
                break
            if proc.poll() is not None:
                break                     # Wesnoth exited
            # Quiet-settle exit only applies once playback has
            # visibly STARTED — boot + asset load takes ~40s with
            # sparse logging, and bailing during it produced
            # inconclusive runs (2026-07-06 calibration).
            # Story [message] dialogs block unattended playback
            # (Aethermaw's turn-4 gates dialog stalled midgame
            # validation at action 38/163, 2026-07-19). When
            # playback has visibly started but the log has been
            # quiet >=10s and we're short of the final action, post
            # ENTER to dismiss a dialog. Sparse (10s cadence) and
            # only mid-stall -- ENTER on the dialog-less viewer is a
            # no-op, and the [choose]-first retry guard still
            # backstops any derailment.
            if (progress >= _MIN_PROGRESS
                    and (total is None or progress < total)
                    and time.time() - last_growth >= 10.0
                    and time.time() - _last_enter >= 10.0):
                _post_enter(proc.pid)
                _last_enter = time.time()
            if (progress >= _MIN_PROGRESS
                    and time.time() - last_growth > settle):
                break                     # playback done / stalled
            time.sleep(1.0)
        else:
            offending.append(
                f"TIMEOUT after {timeout}s "
                f"(progress {progress}/{total})")
        if progress < _MIN_PROGRESS:
            offending.append(
                f"INCONCLUSIVE: playback never progressed past "
                f"action {progress} — the play hotkey didn't take "
                f"(window state? preferences?); refusing to report "
                f"success")
        elif total is not None and progress < total:
            offending.append(
                f"INCONCLUSIVE: playback stopped at action "
                f"{progress}/{total} without an OOS line; refusing "
                f"to report success")
        return offending
    finally:
        if proc.poll() is None:
            proc.terminate()


_CHOOSE_ANSWER_RE = re.compile(
    r"answer[^\n]*\[choose\]|\[choose\][^\n]*answer", re.I)


def validate_in_wesnoth(replay: Path, timeout: float = 420.0,
                        settle: float = 30.0) -> list:
    """Returns a list of offending log lines (empty = no OOS seen).

    `settle`: extra seconds to keep scanning after the last log
    growth, so slow playback isn't cut short. A replay that plays to
    its final action OR goes quiet without OOS lines (after having
    visibly progressed) counts as clean.

    FALSE-POSITIVE guard (2026-07-19): the play-hotkey nudge can
    land on the wrong UI state and flip the viewer into take-control
    mode, whose locally-generated [choose] answers mismatch the
    recorded stream. A GENUINE desync cascade shows damage-SYNC
    overrides before any [choose] complaint (verified on the
    Silverhead null-side case); a [choose]-FIRST verdict with no
    damage-SYNC line is therefore retried once, and the retry's
    verdict stands. Same-file OOS-then-clean flips motivated this
    (camp0719_fog_2, 2026-07-18/19).
    """
    verdict = _validate_once(replay, timeout=timeout, settle=settle)
    if verdict and _CHOOSE_ANSWER_RE.search(verdict[0])             and not any("SYNC: In attack" in l for l in verdict):
        log.warning(
            "[choose]-first OOS with no damage-SYNC lines -- "
            "possible hotkey-derailment false positive; retrying "
            "once (%s)", replay.name)
        verdict = _validate_once(replay, timeout=timeout,
                                 settle=settle)
    return verdict


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("replay", type=Path)
    ap.add_argument("--timeout", type=float, default=420.0)
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=logging.INFO)
    bad = validate_in_wesnoth(args.replay, timeout=args.timeout)
    if bad:
        print(f"OOS/errors in Wesnoth playback of {args.replay.name}:")
        for line in bad[:20]:
            print(f"  {line}")
        return 1
    print(f"clean playback: {args.replay.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
