"""Tiny Tk GUI to one-click the cluster ops.

What it does
============
Wraps the existing PowerShell / Python scripts so the common workflows
(sync code, pull a checkpoint, start a self-play smoke, run eval) are
buttons instead of memorized command lines. The cluster ops authenticate
non-interactively by feeding the password to OpenSSH's `SSH_ASKPASS`
hook -- so for the duration of one GUI session you type the password
ONCE at the top of the window and every button afterwards is silent.

Password handling
=================
- Stored only as a `tk.StringVar` instance attribute. Never written to
  any persistent state (no config file, no .ssh/credentials).
- On a cluster-op button click, written to a per-op temp file + a
  one-line askpass .bat that `type`s it back. SSH_ASKPASS is pointed
  at that .bat. ssh reads the password through the helper for every
  prompt of the proxy chain (relais -> istanbul -> ...) and the temp
  files are deleted as soon as the op exits.
- On window close: the StringVar is overwritten with empty, leftover
  temp files are deleted, any in-flight subprocess is terminated.
- Python str memory cannot be securely scrubbed (the GC frees it when
  the last reference drops, but other copies may persist). For luxury-
  grade convenience this is the right tradeoff; a real secret-handling
  app would use OS keyrings.

Launching
=========
Double-click `cluster/gui.pyw` (Python file association on Windows
maps `.pyw` to `pythonw.exe`, no console window appears) or run
`pythonw cluster/gui.pyw` from anywhere. Use the included
`cluster/gui.bat` shortcut if your file association doesn't pick up
.pyw automatically.

Stdlib only -- no pip installs.
"""

from __future__ import annotations

import os
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import tkinter.scrolledtext as scrolledtext
from tkinter import ttk
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ttkthemes provides ThemedTk + a library of nicer ttk themes than
# the stock vista/winnative. `arc` is a flat clean light theme that
# fits the rest of Windows 11. Falls back to stock tk.Tk if the
# import fails -- the project's "no required deps" stance still
# works, just with the default vista theme.
try:
    from ttkthemes import ThemedTk
    _HAVE_TTKTHEMES = True
except ImportError:
    ThemedTk = None  # type: ignore[assignment]
    _HAVE_TTKTHEMES = False

# Default theme. `arc` is light-flat, looks native-ish on Windows.
# Alternatives: `equilux` (dark), `breeze` (KDE-style), `clam`
# (built-in fallback). Make it Settings-tunable later (Phase 10).
GUI_THEME = "arc"


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
# This file lives at <PROJECT>/cluster/gui.pyw -- compute the project
# root so the GUI works no matter what cwd the user launches from.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUSTER_DIR  = PROJECT_ROOT / "cluster"

POWERSHELL = "powershell.exe"


def _python_exe_for_subprocess() -> str:
    """Return python.exe (NOT pythonw.exe) for spawning subprocesses.

    When this GUI is launched via `cluster/gui.pyw`, Windows file
    association maps `.pyw` to `pythonw.exe` so `sys.executable`
    points at pythonw.exe. That's correct for the GUI process
    (no console flash, GUI thread runs cleanly) but WRONG for
    spawned child Python processes: pythonw.exe at runtime sets
    sys.stdout/sys.stderr to NUL when there's no console, and
    when subprocess.Popen inherits those handles the child's
    writes go nowhere -- or, worse, crash the child outright
    (observed 2026-05-13: eval_sim.py exited 1 within 3 seconds
    under pythonw, working under python; the GUI showed exit-1
    with no error because the error never reached the pipe).

    Workaround: derive python.exe from sys.executable. The
    subprocess.CREATE_NO_WINDOW flag on Windows still prevents
    a console window from flashing, so we get the best of both:
    a quiet GUI and working subprocess stdio.

    On non-Windows (or if sys.executable already points at
    python.exe), this is a no-op.
    """
    if os.name == "nt":
        d, base = os.path.split(sys.executable)
        if base.lower() == "pythonw.exe":
            cand = os.path.join(d, "python.exe")
            if os.path.exists(cand):
                return cand
    return sys.executable


PYTHON     = _python_exe_for_subprocess()

SCRIPT_SYNC          = CLUSTER_DIR / "sync.ps1"
SCRIPT_PULL          = CLUSTER_DIR / "pull_checkpoint.ps1"
SCRIPT_TDR           = CLUSTER_DIR / "setup_tdr.ps1"
# Self-play + one-game demo run via the in-process Wesnoth simulator
# (`tools/sim_self_play.py` / `tools/sim_demo_game.py`). The legacy
# subprocess-driven path through `main.py --display` / `game_manager.py`
# was retired 2026-05-11 (game_manager + the --display flag were
# removed entirely). The simulator is bit-exact against Wesnoth for
# combat strikes (731/731 verified) and full-replay reconstruction
# (100% on the 5,484-replay competitive-2p corpus), ~1000x faster
# than driving a Wesnoth subprocess, and from-scratch save export
# produces Wesnoth-loadable .bz2 replays for visual inspection.
SCRIPT_SIM_SELFPLAY  = PROJECT_ROOT / "tools" / "sim_self_play.py"
SCRIPT_SIM_DEMO      = PROJECT_ROOT / "tools" / "sim_demo_game.py"
SCRIPT_DIAGNOSE      = PROJECT_ROOT / "tools" / "diagnose_selfplay.py"
EVAL_SCRIPT          = PROJECT_ROOT / "tools" / "eval_vs_builtin.py"

REMOTE_HOST = "mesogip_outside"
REMOTE_PATH = "~/wesnoth-ai"

# Persistent location for "last-picked dialog values become the new
# defaults" behavior. Per-user under ~/.wesnoth_ai/ rather than in
# the project tree so it survives bundle re-extracts on the cluster
# AND doesn't show up in git diffs. Schema: top-level dict keyed by
# dialog name ("selfplay_cluster", "selfplay_local"), each value a
# dict of param-name -> serializable value.
GUI_STATE_PATH = Path.home() / ".wesnoth_ai" / "gui_state.json"


def _load_gui_state() -> dict:
    """Best-effort: a corrupt / missing file just returns {}. The
    dialog code falls back to its hardcoded defaults for any missing
    keys, so a wiped state file doesn't break the GUI."""
    try:
        with GUI_STATE_PATH.open(encoding="utf-8") as f:
            import json as _json
            return _json.load(f)
    except (OSError, ValueError):
        return {}


def _save_gui_state(state: dict) -> None:
    try:
        GUI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        with GUI_STATE_PATH.open("w", encoding="utf-8") as f:
            _json.dump(state, f, indent=2)
    except OSError:
        pass  # don't crash the GUI on disk errors


# ---------------------------------------------------------------------
# Subprocess + askpass plumbing
# ---------------------------------------------------------------------

class AskpassFiles:
    """One-shot pair of (password file, askpass.bat) that lives only
    for the duration of a single subprocess. `cleanup()` is idempotent
    and safe to call from atexit / window-close paths.
    """

    def __init__(self, password: str):
        # Use a unique tmpdir per pair so we never collide with another
        # in-flight op or with stale debris from a prior run.
        self._dir = Path(tempfile.mkdtemp(prefix="wai_gui_pw_"))
        self.pw_file  = self._dir / "pw.txt"
        self.bat_file = self._dir / "askpass.bat"
        # Write the password without a trailing newline -- ssh strips
        # newlines but some prompts do not, and we want exact bytes.
        self.pw_file.write_text(password, encoding="utf-8", newline="")
        # The askpass helper is invoked by ssh with the prompt as
        # argv[1]; it must echo the password to stdout and exit. `type`
        # is cmd.exe's `cat` and produces no extra newline if the
        # source file has none.
        self.bat_file.write_text(
            f'@echo off\r\ntype "{self.pw_file}"\r\n',
            encoding="utf-8", newline="",
        )

    def env(self) -> dict:
        """Return an env dict with SSH_ASKPASS configured. The caller
        merges this into os.environ before spawning the subprocess."""
        e = os.environ.copy()
        e["SSH_ASKPASS"] = str(self.bat_file)
        # `force` works on OpenSSH 8.4+ (Windows OpenSSH ships modern
        # versions). Falls back to nothing on older versions, which
        # would just prompt on the missing TTY -- diagnosable and
        # rare enough not to special-case.
        e["SSH_ASKPASS_REQUIRE"] = "force"
        # DISPLAY=:0 is the legacy gate that makes pre-8.4 OpenSSH
        # consider SSH_ASKPASS at all. Setting both covers more
        # OpenSSH builds than either alone.
        e.setdefault("DISPLAY", ":0")
        return e

    def cleanup(self) -> None:
        for p in (self.bat_file, self.pw_file):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            self._dir.rmdir()
        except OSError:
            pass


def _purge_stale_askpass_dirs() -> int:
    """Walk `%TEMP%` for orphan `wai_gui_pw_*` directories from
    prior GUI runs that crashed or got force-killed before
    `AskpassFiles.cleanup()` could fire. Returns the count
    removed.

    `tempfile.mkdtemp(prefix='wai_gui_pw_')` always creates the
    directory under `tempfile.gettempdir()`, which is what
    `gettempdir()` returns -- the same location we walk here.
    `rmtree(ignore_errors=True)` swallows permission / locked-file
    issues; another concurrent GUI process holding that dir's
    handle just means we leave it for next time.

    Called once at GUI startup. Idempotent and cheap.
    """
    import shutil
    tmp = Path(tempfile.gettempdir())
    if not tmp.is_dir():
        return 0
    removed = 0
    for p in tmp.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith("wai_gui_pw_"):
            continue
        shutil.rmtree(p, ignore_errors=True)
        if not p.exists():
            removed += 1
    return removed


def _stream_to_queue(pipe, q: queue.Queue, tag: str) -> None:
    """Read `pipe` line by line and push (tag, line) onto q. Runs in
    a worker thread; the GUI thread drains q via tk.after()."""
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            q.put((tag, line))
    finally:
        try:
            pipe.close()
        except Exception:
            pass


# ---------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------

class App:

    OUTPUT_POLL_MS  = 80     # how often the GUI thread drains the stdout queue
    FRESHNESS_POLL_MS = 30_000  # how often the freshness row re-stats local files
    # How often the auto-pull watcher probes squeue for the
    # self-play job. 120s balances "operator sees the pull start
    # within a couple minutes of the job ending" against "we
    # don't hammer the cluster". Each probe is one ssh +
    # one-line squeue -- cheap, but not free.
    AUTO_PULL_POLL_MS = 120_000
    # Initial delay before the first probe so the GUI doesn't
    # fire a poll before the operator has typed their password.
    AUTO_PULL_FIRST_TICK_MS = 15_000

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Wesnoth AI -- Cluster GUI")
        # New side-by-side layout (Phase 1 of the 2026-05-16
        # redesign): wider window to accommodate the output panel
        # moving from the bottom to the right side. Tabs on the
        # left, output on the right via a ttk.PanedWindow; the
        # operator can drag the divider.
        root.geometry("1180x720")
        root.minsize(900, 560)

        self._password    = tk.StringVar(value="")
        self._proc:        Optional[subprocess.Popen] = None
        self._proc_lock   = threading.Lock()
        self._askpass:    Optional[AskpassFiles] = None
        self._stdout_q:   queue.Queue            = queue.Queue()
        # Pull-checkpoint mode. Default empty -> sim_selfplay.pt (the
        # NEW default, 2026-05-13). Other accepted values: 'supervised'
        # (highest supervised_epoch), 'rolling' (mid-epoch supervised),
        # a number (specific supervised_epoch<N>.pt), or 'list'.
        self._epoch_var   = tk.StringVar(value="")
        # Per-op line capture buffer + on-complete callback. Set in
        # `_spawn(... on_complete=...)`; cleared in `_end_op`. The
        # buffer accumulates every output line the op produced so the
        # callback can parse cluster-status / squeue output and update
        # the freshness row without a second ssh round-trip.
        self._captured_lines: Optional[List[str]] = None
        self._on_complete: Optional[Callable[[List[str]], None]] = None
        # Auto-pull on cluster completion (Phase 5 of the
        # redesign). Polls squeue every AUTO_PULL_POLL_MS; when the
        # self-play job we last saw RUNNING disappears (or moves
        # to COMPLETED/FAILED/CANCELLED), we kick off
        # _op_pull_bundle automatically so the operator's local
        # checkpoint matches the cluster before they next push.
        # This is the anti-overwrite key: pre-redesign, an
        # operator who started a local Train without pulling
        # would silently overwrite cluster progress on the next
        # push. Now the pull happens passively, in the background.
        gs = _load_gui_state()
        self._auto_pull_enabled = tk.BooleanVar(
            value=bool(gs.get("auto_pull_on_completion", True)))
        # `armed` = "we have seen this jid RUNNING/PENDING, so
        # when it next disappears, fire the pull". Stored as a
        # string jid (not bool) so we can survive transient
        # squeue blips without re-firing on the same job.
        self._auto_pull_armed_jid: Optional[str] = None
        # Re-entrance guard: don't fire a second probe while a
        # previous one is still in flight (slow cluster network
        # could otherwise stack probes).
        self._auto_pull_probe_in_flight = False
        # Edge-trigger logging: only log a probe failure when it
        # transitions ok->fail (or fail->ok), so a sustained
        # outage doesn't spam the output panel every 2 minutes.
        self._auto_pull_last_probe_ok = True
        # Persist the toggle when the operator flips it.
        self._auto_pull_enabled.trace_add(
            "write", self._on_auto_pull_toggle)
        # Auto-eval after pull (Phase 6 of the redesign). When a
        # pull-bundle op lands a new sim_selfplay.pt locally AND
        # this toggle is on, fire `tools/eval_daily.py` with the
        # quick preset so eval_history.{csv,md} stays in lockstep
        # with the rolling checkpoint -- the operator never has to
        # remember "did I eval the new model yet?".
        self._auto_eval_after_pull = tk.BooleanVar(
            value=bool(gs.get("auto_eval_after_pull", True)))
        self._auto_eval_after_pull.trace_add(
            "write", self._on_auto_eval_toggle)
        # Stale-push guard (Phase 7 of the redesign). When the
        # auto-pull watcher knows there's cluster work the local
        # copy doesn't yet have, the Push buttons grey out so the
        # operator literally can't fire an overwriting push by
        # accident. Mechanism:
        #   * `_last_pulled_jid` -- the SLURM jid of the cluster
        #     self-play job whose checkpoint our local was last
        #     pulled from. Persisted across GUI restarts.
        #   * `_push_blocked_reason` -- None when push is safe; a
        #     human-readable reason string when blocked. Set by
        #     the auto-pull probe; cleared by a successful pull.
        # The buttons in the Daily/Train tabs register themselves
        # via `_register_push_button` so the central refresh can
        # flip them all at once without each call site duplicating
        # the state logic.
        self._last_pulled_jid: Optional[str] = gs.get(
            "last_pulled_jid") or None
        self._push_blocked_reason: Optional[str] = None
        self._push_buttons: List[ttk.Button] = []
        self._push_block_lbl_var = tk.StringVar(value="")
        # Disabled-state polish (Phase 11). Two more button
        # categories whose enablement is a pure function of
        # observable state:
        #   * `_cluster_buttons` -- need a non-empty ssh password
        #     to do anything useful. Refreshed on every keystroke
        #     in the password entry via a trace on _password.
        #   * `_model_buttons` -- need a local sim_selfplay.pt to
        #     evaluate / display / push. Refreshed by the
        #     existing freshness tick (which already stats the
        #     file).
        # Each list collects ttk.Button widgets registered by the
        # tab builders. The refresh helpers flip widget state at
        # once so the operator's first signal that something is
        # impossible is the widget itself, not an error popup.
        self._cluster_buttons: List = []
        self._model_buttons:   List = []
        # Re-greying triggered on password edits. The trace_add
        # write callback is cheap (one strip + len check + N
        # widget.configure calls); fires on every keystroke,
        # which is what we want -- typing the first character
        # immediately enables the cluster panel.
        self._password.trace_add(
            "write", lambda *_a: self._refresh_cluster_buttons())
        self._build_widgets()
        # Periodic pump from worker threads' queue -> output panel.
        root.after(self.OUTPUT_POLL_MS, self._drain_output)
        # Periodic local-side freshness refresh (cheap, no ssh).
        # Cluster-job state is refreshed only after explicit ssh
        # ops (see _scan_for_cluster_job_state).
        self._refresh_local_freshness()
        root.after(self.FRESHNESS_POLL_MS, self._tick_freshness)
        # Kick off the auto-pull watcher. First tick is delayed
        # so the operator has time to type the password.
        root.after(self.AUTO_PULL_FIRST_TICK_MS, self._tick_auto_pull)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        # Power-user command palette (Phase 12). Ctrl-K (and the
        # alternate Ctrl-Shift-P, à la VS Code) opens a fuzzy
        # search modal over every named action. Keeps the daily
        # mouse path uncluttered while giving the keyboard-
        # oriented operator a single shortcut for every op.
        root.bind_all("<Control-k>",
                      lambda _e: self._open_command_palette())
        root.bind_all("<Control-K>",
                      lambda _e: self._open_command_palette())

    # -- widgets --------------------------------------------------------
    #
    # Layout (Phase 1 of the 2026-05-16 redesign):
    #
    #   ┌──────────────────────────────────────────────────────────────┐
    #   │  Status row (cluster / local / archive / [⚙])                │
    #   ├───────────────────────────────────────┬──────────────────────┤
    #   │  Notebook                             │  Output panel        │
    #   │   [All ops]   <- Phase 1: 1 tab       │  ScrolledText        │
    #   │     all 29 buttons live here          │                      │
    #   │     temporarily, will be split into   │                      │
    #   │     Daily/Train/Maintenance in        │                      │
    #   │     phases 4/8/9                      │                      │
    #   ├───────────────────────────────────────┴──────────────────────┤
    #   │  Status bar  [Stop training (save)]  [Force cancel] [Clear]  │
    #   └──────────────────────────────────────────────────────────────┘
    #
    # The PanedWindow lets the operator drag the divider; sash sits
    # at ~60% initially (tabs wider than output).

    def _build_widgets(self):
        pad = {"padx": 8, "pady": 4}

        # 1. Top status row -- promoted from the old freshness row
        #    near the bottom. Now first-glance state for the
        #    operator: cluster job state, local ckpt freshness,
        #    archive age, plus a Settings affordance.
        self._build_status_row(pad)

        # 2. Main pane: a horizontal PanedWindow with a Notebook on
        #    the left and the output panel on the right. The user
        #    can resize the split. The Notebook starts with one
        #    placeholder tab "All ops" that holds the legacy layout;
        #    phases 4/8/9 split it into Daily/Train/Maintenance.
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, **pad)
        self._notebook = ttk.Notebook(paned)
        # Three task-themed tabs. Every action that used to live
        # on the legacy "All ops" panel has a home in exactly one
        # of these:
        #   * Daily       -- the Pull/Inspect/Push loop, plus
        #                    quick Status check and eval ops.
        #   * Train       -- start/stop cluster jobs, sync+
        #                    continue chains, local training,
        #                    reward-config edits.
        #   * Maintenance -- rarer/legacy ops: specialised pulls,
        #                    archive restore, profile pipeline,
        #                    code-only sync, TDR setup, etc.
        # Added in user-facing-priority order: Daily is selected
        # by default (it's the most common loop).
        daily_tab = ttk.Frame(self._notebook)
        self._notebook.add(daily_tab, text="Daily")
        train_tab = ttk.Frame(self._notebook)
        self._notebook.add(train_tab, text="Train")
        maint_tab = ttk.Frame(self._notebook)
        self._notebook.add(maint_tab, text="Maintenance")
        paned.add(self._notebook, weight=3)

        # Output panel goes to the right side. The ScrolledText
        # keeps its existing self._output reference so every log()
        # call elsewhere keeps working unchanged.
        out_frame = ttk.LabelFrame(paned, text="Output")
        paned.add(out_frame, weight=2)
        self._output = scrolledtext.ScrolledText(
            out_frame, wrap="word", font=("Consolas", 9))
        self._output.pack(fill="both", expand=True, padx=4, pady=4)
        self._output.configure(state="disabled")

        # 3. Bottom bar (unchanged from Phase 0): cancel + clear +
        #    status label. Built last so it docks to the bottom.
        #    Built BEFORE the all-ops content because the all-ops
        #    builder's bottom-bar section gets dropped by the
        #    refactor (it's replaced by this).
        self._build_bottom_bar(pad)

        # 4. Fill each tab. All three are recomposed views of the
        #    same set of self._op_* commands; the legacy All-ops
        #    panel was removed entirely so the operator never sees
        #    a duplicate button. Daily is selected on launch.
        self._build_daily_tab(daily_tab, pad)
        self._build_train_tab(train_tab, pad)
        self._build_maintenance_tab(maint_tab, pad)
        try:
            self._notebook.select(daily_tab)
        except tk.TclError:
            pass

    def _build_daily_tab(self, parent: ttk.Frame,
                         pad: Dict[str, int]) -> None:
        """The 'Daily' tab: the most-common workflow surfaced as a
        linear top-to-bottom flow.

        Three sections, each a LabelFrame for visual grouping:
          1. PULL  -- atomic pull from cluster (model + log + history)
          2. INSPECT -- open the local trainer/eval history files,
                        display a game, run diagnose. Merges what
                        used to be a separate "Inspect" mental
                        category into the daily loop per the
                        redesign brief ("merge inspect and daily").
          3. PUSH  -- push local checkpoint + code; or push + restart
                      the cluster chain link.

        The buttons here reuse the same _op_* commands as the All-ops
        tab. The Daily tab is just a recomposed view of the same
        actions, optimised for the operator who isn't memorising the
        legacy button grid. Power-user / one-off ops (Pull mode
        archive, supervised epoch, profile artifacts, TDR setup, etc.)
        stay on the All-ops tab.

        Password chrome (entry, status pill) lives in the All-ops
        tab and is shared globally -- typed once, every cluster
        button below uses it silently.
        """
        # Tiny hint at the top of the tab so a fresh operator knows
        # what the columns mean. Gray, single line, doesn't compete
        # for attention with the action buttons.
        hint = ttk.Label(
            parent,
            text=("Pull from cluster -> inspect what came down -> "
                  "push back. ssh password goes in the header above."),
            foreground="gray",
        )
        hint.pack(fill="x", padx=pad["padx"], pady=(pad["pady"], 8))

        # ---- 0. STATUS -------------------------------------------------
        # Quick "what's happening" probe. Surfaced as its own section
        # so the operator's first instinct ("am I about to step on a
        # running job?") has a one-click answer. Refreshes the
        # freshness header above; doesn't kick off anything destructive.
        st_box = ttk.LabelFrame(parent, text="0. Status")
        st_box.pack(fill="x", **pad)
        st_row = ttk.Frame(st_box)
        st_row.pack(fill="x", padx=8, pady=6)
        btn_status = ttk.Button(st_row, text="Check cluster status",
                                width=22, command=self._op_status)
        btn_status.pack(side="left", padx=4)
        self._register_cluster_button(btn_status)
        ttk.Label(
            st_row,
            text=("squeue + sacct probe; updates the "
                  "header dashboard above"),
            foreground="gray",
        ).pack(side="left", padx=8)

        # ---- 1. PULL ---------------------------------------------------
        pull_box = ttk.LabelFrame(parent, text="1. Pull from cluster")
        pull_box.pack(fill="x", **pad)
        pull_row = ttk.Frame(pull_box)
        pull_row.pack(fill="x", padx=8, pady=6)
        btn_pull_bundle = ttk.Button(
            pull_row, text="Pull from cluster",
            width=22, command=self._op_pull_bundle)
        btn_pull_bundle.pack(side="left", padx=4)
        self._register_cluster_button(btn_pull_bundle)
        ttk.Label(
            pull_row,
            text=("self-play checkpoint + freshest SLURM log + "
                  "trainer_history CSV, one password"),
            foreground="gray",
        ).pack(side="left", padx=8)
        # Auto-pull toggle row. On by default; persisted to
        # gui_state.json via the trace_add binding in __init__.
        # When ticked, the GUI polls squeue every 2 min and
        # automatically fires _op_pull_bundle when the
        # wai-selfplay job ends. The anti-overwrite key feature:
        # operator never has to remember to pull before they
        # start a local training run.
        auto_row = ttk.Frame(pull_box)
        auto_row.pack(fill="x", padx=8, pady=(0, 2))
        ttk.Checkbutton(
            auto_row,
            text="Auto-pull when the cluster self-play job ends",
            variable=self._auto_pull_enabled,
        ).pack(side="left", padx=4)
        ttk.Label(
            auto_row,
            text=("polls squeue every 2 min; "
                  "prevents push-stale-checkpoint overwrites"),
            foreground="gray",
        ).pack(side="left", padx=8)
        # Auto-eval after pull. Same persistence pattern as the
        # auto-pull toggle above. Pair: every pull that lands a
        # fresh model also drops a quick-preset eval row into
        # eval_history -- so the trend graph stays continuous
        # without manual button-clicks.
        eval_row = ttk.Frame(pull_box)
        eval_row.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Checkbutton(
            eval_row,
            text="Auto-eval after every pull that lands a new model",
            variable=self._auto_eval_after_pull,
        ).pack(side="left", padx=4)
        ttk.Label(
            eval_row,
            text=("quick preset; appends a row to eval_history"),
            foreground="gray",
        ).pack(side="left", padx=8)

        # ---- 2. INSPECT ------------------------------------------------
        # Inspect groups the local "look at what we have" actions.
        # All read-only conceptually (View opens files; Display
        # renders a game; Diagnose prints state; eval probes the
        # model strength) -- the operator uses this section to
        # decide whether to push back or keep iterating.
        ins_box = ttk.LabelFrame(parent, text="2. Inspect")
        ins_box.pack(fill="x", **pad)
        ins_r1 = ttk.Frame(ins_box)
        ins_r1.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Button(ins_r1, text="View train history",
                   width=22,
                   command=self._op_view_train_history).pack(
                       side="left", padx=4)
        ttk.Button(ins_r1, text="View eval history",
                   width=22,
                   command=self._op_view_eval_history).pack(
                       side="left", padx=4)
        ttk.Label(ins_r1, text="(rolling .csv / .md, system editor)",
                  foreground="gray").pack(side="left", padx=8)
        ins_r2 = ttk.Frame(ins_box)
        ins_r2.pack(fill="x", padx=8, pady=(2, 2))
        btn_display = ttk.Button(
            ins_r2, text="Display 1 game", width=22,
            command=self._op_display_selfplay)
        btn_display.pack(side="left", padx=4)
        self._register_model_button(btn_display)
        # Diagnose runs against the local checkpoint too, but it
        # has a useful fallback path (probes encoder/sampler with
        # a synthetic state) when no model exists -- keep it
        # always-enabled so the operator can investigate even on
        # a fresh install.
        ttk.Button(ins_r2, text="Diagnose ...",
                   width=22,
                   command=self._op_diagnose).pack(side="left", padx=4)
        ttk.Label(ins_r2, text="(render or probe local state)",
                  foreground="gray").pack(side="left", padx=8)
        ins_r3 = ttk.Frame(ins_box)
        ins_r3.pack(fill="x", padx=8, pady=(2, 6))
        btn_eval_daily = ttk.Button(
            ins_r3, text="Daily eval", width=22,
            command=self._op_eval_daily)
        btn_eval_daily.pack(side="left", padx=4)
        self._register_model_button(btn_eval_daily)
        btn_eval_dlg = ttk.Button(
            ins_r3, text="Run eval ...", width=22,
            command=self._op_eval_dialog)
        btn_eval_dlg.pack(side="left", padx=4)
        self._register_model_button(btn_eval_dlg)
        ttk.Label(
            ins_r3,
            text=("Daily eval = preset matrix vs built-in RCA AI; "
                  "Run eval... lets you tune"),
            foreground="gray").pack(side="left", padx=8)

        # ---- 3. PUSH ---------------------------------------------------
        # Two push intents:
        #   * Push code + local checkpoint, leave any running job
        #     alone (chains into next link automatically).
        #   * Push + restart chain link now (Sync + Continue).
        # The rarer "code-only sync" lives on the Maintenance tab.
        push_box = ttk.LabelFrame(parent, text="3. Push back to cluster")
        push_box.pack(fill="x", **pad)
        push_row = ttk.Frame(push_box)
        push_row.pack(fill="x", padx=8, pady=6)
        btn_push = ttk.Button(push_row, text="Push local + code",
                              width=22,
                              command=self._op_push_checkpoint)
        btn_push.pack(side="left", padx=4)
        btn_sync_cont = ttk.Button(
            push_row, text="Sync + Continue (self-play)",
            width=24, command=self._op_sync_continue_selfplay)
        btn_sync_cont.pack(side="left", padx=4)
        ttk.Label(push_row,
                  text=("Push uploads; Sync+Continue uploads and "
                        "restarts the cluster chain link"),
                  foreground="gray").pack(side="left", padx=8)
        # Both buttons are stale-push-guarded. When the auto-pull
        # watcher sees a cluster job whose work we haven't yet
        # pulled, these grey out + the reason label below shows
        # why. Re-enable happens automatically after the pull
        # lands a fresh sim_selfplay.pt. They're ALSO cluster-
        # button-gated (no password = no push) and model-gated
        # (no local checkpoint = nothing to push), so all three
        # constraints stack via the unified state engine.
        self._register_push_button(btn_push)
        self._register_cluster_button(btn_push)
        self._register_model_button(btn_push)
        self._register_push_button(btn_sync_cont)
        self._register_cluster_button(btn_sync_cont)
        self._register_model_button(btn_sync_cont)
        # Block-reason label. Empty string = invisible (zero
        # height); populated string = "why are buttons greyed".
        # Single label shared across all Push-button rows on
        # this tab so the operator sees the explanation right
        # under the disabled buttons.
        block_row = ttk.Frame(push_box)
        block_row.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Label(
            block_row,
            textvariable=self._push_block_lbl_var,
            foreground="#cc6600",  # warning-amber
            wraplength=720, justify="left",
        ).pack(side="left", padx=4)

    def _build_train_tab(self, parent: ttk.Frame,
                         pad: Dict[str, int]) -> None:
        """The 'Train' tab: start, stop, and configure training jobs.

        Cluster section:
          * Start / Stop  -- submit or scancel the self-play or
            supervised SLURM job. run.sh start refuses to double-
            submit, so re-clicking is safe; stop scancels the
            recorded jobid.
          * Sync + Continue -- the post-walltime restart pattern:
            sync code, scancel any running job, sbatch a fresh
            one that auto-resumes from the latest checkpoint.

        Local section:
          * Train (self-play) -- start a local self-play training
            run; the pop-up picks ladder vs mini-maps.
          * Edit rewards ... -- modify reward shaping; cluster
            jobs pick up the new config on the NEXT chain link.

        Push back / Daily-loop actions stay on the Daily tab
        (the operator who's in Train tab is managing jobs, not
        the daily push loop)."""
        hint = ttk.Label(
            parent,
            text=("Start, stop and configure training jobs. "
                  "Cluster ops use the ssh password from the "
                  "header above."),
            foreground="gray",
        )
        hint.pack(fill="x", padx=pad["padx"], pady=(pad["pady"], 8))

        # ---- Cluster: self-play (primary path) -------------------------
        sp_box = ttk.LabelFrame(parent, text="Cluster — self-play")
        sp_box.pack(fill="x", **pad)

        # Row 1: Configure-only entry point. Opens the settings
        # dialog the same way Start self-play does, but the dialog's
        # "Save only" button (added 2026-05-20) persists the chosen
        # values to gui_state.json without sbatching. The next
        # Start self-play / Sync + Continue picks them up. This is
        # the discoverable "where do I change settings" entry --
        # the dialog has knobs for iterations, games per iter,
        # max-turns, workers, seed, time budget, forced faction,
        # reward config, MCTS toggles, AND mini-maps + mini-mix.
        sp_cfg_row = ttk.Frame(sp_box)
        sp_cfg_row.pack(fill="x", padx=8, pady=(6, 0))
        btn_sp_configure = ttk.Button(
            sp_cfg_row, text="Configure training settings...",
            width=30,
            command=lambda: self._open_selfplay_dialog(
                title="Self-play settings (cluster)",
                on_run=self._run_cluster_selfplay,
                cluster_mode=True,
            ))
        btn_sp_configure.pack(side="left", padx=4)
        ttk.Label(sp_cfg_row,
                  text="(iterations, games/iter, mini-mix, etc.; "
                       "Save only to persist without starting)",
                  foreground="gray").pack(side="left", padx=8)

        # Row 2: Start / Stop / Sync+Continue actions.
        sp_row = ttk.Frame(sp_box); sp_row.pack(fill="x", padx=8, pady=6)
        btn_sp_start = ttk.Button(
            sp_row, text="Start self-play", width=18,
            command=self._op_start_selfplay)
        btn_sp_start.pack(side="left", padx=4)
        self._register_cluster_button(btn_sp_start)
        btn_sp_stop = ttk.Button(
            sp_row, text="Stop self-play", width=18,
            command=self._op_stop_selfplay)
        btn_sp_stop.pack(side="left", padx=4)
        self._register_cluster_button(btn_sp_stop)
        btn_sp_sync = ttk.Button(
            sp_row, text="Sync + Continue (self-play)",
            width=24, command=self._op_sync_continue_selfplay)
        btn_sp_sync.pack(side="left", padx=4)
        # Sync + Continue uploads the local checkpoint AND restarts
        # the cluster chain link -- destructive in the same way as
        # the Daily-tab Push, so the stale-push guard + cluster +
        # model gates all apply.
        self._register_push_button(btn_sp_sync)
        self._register_cluster_button(btn_sp_sync)
        self._register_model_button(btn_sp_sync)
        ttk.Label(
            sp_row,
            text=("Start = sbatch (no-op if running); "
                  "Sync+Continue = restart chain link"),
            foreground="gray").pack(side="left", padx=8)

        # ---- Cluster: supervised (warm-start path) ---------------------
        # Kept separate from self-play so the operator can't accidentally
        # `Stop supervised` when they meant to stop self-play (the most
        # common confusion before the redesign).
        sup_box = ttk.LabelFrame(parent, text="Cluster — supervised")
        sup_box.pack(fill="x", **pad)
        sup_row = ttk.Frame(sup_box); sup_row.pack(fill="x", padx=8, pady=6)
        btn_sup_start = ttk.Button(
            sup_row, text="Start supervised", width=18,
            command=self._op_start_supervised)
        btn_sup_start.pack(side="left", padx=4)
        self._register_cluster_button(btn_sup_start)
        btn_sup_stop = ttk.Button(
            sup_row, text="Stop supervised", width=18,
            command=self._op_stop_supervised)
        btn_sup_stop.pack(side="left", padx=4)
        self._register_cluster_button(btn_sup_stop)
        btn_sup_sync = ttk.Button(
            sup_row, text="Sync + Continue (sup.)",
            width=22, command=self._op_sync_continue_supervised)
        btn_sup_sync.pack(side="left", padx=4)
        # Supervised Sync+Continue doesn't touch sim_selfplay.pt;
        # only the supervised chain is restarted. We still register
        # it as a push button (code edits get uploaded; a running
        # self-play job would notice on its next checkpoint write)
        # and as a cluster button. Not a model button -- supervised
        # mode is decoupled from sim_selfplay.pt.
        self._register_push_button(btn_sup_sync)
        self._register_cluster_button(btn_sup_sync)
        ttk.Label(
            sup_row,
            text=("warm-start path; self-play is the primary "
                  "production workflow"),
            foreground="gray").pack(side="left", padx=8)

        # ---- Local + config --------------------------------------------
        loc_box = ttk.LabelFrame(parent, text="Local + training config")
        loc_box.pack(fill="x", **pad)
        # Row 1: start + stop, paired side-by-side so the operator
        # who's running local training has the cancel button right
        # next to the start button (rather than buried in the
        # bottom-bar where it lives across all tabs). The bottom-
        # bar copy stays for visibility during cluster ops.
        loc_r1 = ttk.Frame(loc_box); loc_r1.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Button(loc_r1, text="Train (self-play)",
                   width=22,
                   command=self._op_train_selfplay).pack(
                       side="left", padx=4)
        # Graceful cancel: writes the sentinel file the trainer
        # polls between iters. Trainer saves the current
        # sim_selfplay.pt + exits cleanly. Worst-case latency is
        # one iter (~1-5 min depending on iter size and hardware).
        # Same _op_graceful_cancel as the bottom-bar button; we
        # surface it here too so the operator who's just clicked
        # Train doesn't have to hunt for the cancel.
        ttk.Button(loc_r1, text="Stop training (save)",
                   width=22,
                   command=self._op_graceful_cancel).pack(
                       side="left", padx=4)
        ttk.Label(
            loc_r1,
            text=("Stop = graceful sentinel; trainer saves "
                  "sim_selfplay.pt + exits at next iter boundary"),
            foreground="gray").pack(side="left", padx=8)
        # Row 2: training config. "Configure training settings..."
        # opens the same dialog as Train (self-play) but with a
        # "Save only" footer button -- lets the operator tune
        # knobs (iterations, games/iter, mini-mix, MCTS, etc.)
        # without immediately starting a run. The next Train
        # (self-play) click picks up the persisted values.
        loc_r2 = ttk.Frame(loc_box); loc_r2.pack(fill="x", padx=8, pady=(2, 6))
        ttk.Button(
            loc_r2, text="Configure training settings...",
            width=30,
            command=lambda: self._open_selfplay_dialog(
                title="Self-play settings (local)",
                on_run=self._run_local_selfplay,
                cluster_mode=False,
            )).pack(side="left", padx=4)
        ttk.Button(loc_r2, text="Edit rewards ...",
                   width=22,
                   command=self._op_edit_rewards).pack(
                       side="left", padx=4)
        ttk.Label(
            loc_r2,
            text="(reward edits apply on next chain link)",
            foreground="gray").pack(side="left", padx=8)

    def _build_maintenance_tab(self, parent: ttk.Frame,
                               pad: Dict[str, int]) -> None:
        """The 'Maintenance' tab: rarely-used and danger-zone ops.

        Three groups, ordered by frequency-of-use:
          * Specialised pull -- the legacy Pull dropdown (specific
            archive, supervised epoch, rolling target) and Pull
            logs. The atomic "Pull from cluster" on the Daily tab
            covers 95% of pulls; these stay for the remaining 5%.
          * Profile pipeline -- submit a profile job, pull its
            artifacts, view them. Rare (used when investigating
            performance regressions).
          * Danger zone / setup -- archive restore (recover from
            an accidental push), code-only sync, cluster-budget
            probe, and the one-time Windows TDR registry edit.

        Buttons reuse the same _op_* commands; the operator never
        sees them on Daily/Train so the common paths stay
        uncluttered."""
        hint = ttk.Label(
            parent,
            text=("Less-common ops: specialised pulls, profile "
                  "pipeline, recovery, one-time setup."),
            foreground="gray",
        )
        hint.pack(fill="x", padx=pad["padx"], pady=(pad["pady"], 8))

        # ---- Specialised pull ------------------------------------------
        sp_box = ttk.LabelFrame(parent, text="Specialised pull")
        sp_box.pack(fill="x", **pad)
        sp_row = ttk.Frame(sp_box); sp_row.pack(fill="x", padx=8, pady=6)
        ttk.Label(sp_row, text="Pull mode:").pack(side="left", padx=4)
        ttk.Entry(sp_row, textvariable=self._epoch_var,
                  width=12).pack(side="left", padx=4)
        btn_m_pull = ttk.Button(
            sp_row, text="Pull", width=10, command=self._op_pull)
        btn_m_pull.pack(side="left", padx=4)
        self._register_cluster_button(btn_m_pull)
        btn_m_pull_logs = ttk.Button(
            sp_row, text="Pull logs", width=12,
            command=self._op_pull_logs)
        btn_m_pull_logs.pack(side="left", padx=4)
        self._register_cluster_button(btn_m_pull_logs)
        sp_hint = ttk.Frame(sp_box); sp_hint.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Label(
            sp_hint,
            text=("mode: empty=self-play, supervised, rolling, "
                  "archive[:stamp], N (epoch), list"),
            foreground="gray").pack(side="left", padx=4)

        # ---- Profile pipeline ------------------------------------------
        prof_box = ttk.LabelFrame(parent, text="Profile")
        prof_box.pack(fill="x", **pad)
        prof_row = ttk.Frame(prof_box); prof_row.pack(fill="x", padx=8, pady=6)
        btn_m_profile = ttk.Button(
            prof_row, text="Profile ...", width=14,
            command=self._op_profile_cluster)
        btn_m_profile.pack(side="left", padx=4)
        self._register_cluster_button(btn_m_profile)
        btn_m_pull_profile = ttk.Button(
            prof_row, text="Pull profile", width=14,
            command=self._op_pull_profile)
        btn_m_pull_profile.pack(side="left", padx=4)
        self._register_cluster_button(btn_m_pull_profile)
        # View profile reads from local training/profiles/, no
        # ssh -- always enabled.
        ttk.Button(prof_row, text="View profile",
                   width=14,
                   command=self._op_view_profile).pack(
                       side="left", padx=4)
        ttk.Label(prof_row, text="(one-shot ~10min cluster run)",
                  foreground="gray").pack(side="left", padx=8)

        # ---- Danger zone & one-time setup ------------------------------
        dz_box = ttk.LabelFrame(parent, text="Danger zone & one-time setup")
        dz_box.pack(fill="x", **pad)
        dz_r1 = ttk.Frame(dz_box); dz_r1.pack(fill="x", padx=8, pady=(6, 2))
        btn_m_restore = ttk.Button(
            dz_r1, text="Restore from archive ...", width=24,
            command=self._op_restore_from_archive)
        btn_m_restore.pack(side="left", padx=4)
        self._register_cluster_button(btn_m_restore)
        # Sync code is itself a push of source files to the
        # cluster. Not stale-push-guarded (it doesn't touch
        # sim_selfplay.pt), but does need the password.
        btn_m_sync = ttk.Button(
            dz_r1, text="Sync code (no checkpoint)", width=24,
            command=self._op_sync)
        btn_m_sync.pack(side="left", padx=4)
        self._register_cluster_button(btn_m_sync)
        ttk.Label(dz_r1, text="(recover; or push code-only)",
                  foreground="gray").pack(side="left", padx=8)
        dz_r2 = ttk.Frame(dz_box); dz_r2.pack(fill="x", padx=8, pady=(2, 6))
        btn_m_budget = ttk.Button(
            dz_r2, text="Check cluster budget", width=24,
            command=self._op_budget)
        btn_m_budget.pack(side="left", padx=4)
        self._register_cluster_button(btn_m_budget)
        # Setup TDR opens an elevated PowerShell window locally;
        # no ssh, no cluster dependency. Always enabled.
        ttk.Button(dz_r2, text="Setup TDR (admin)",
                   width=24,
                   command=self._op_setup_tdr).pack(side="left", padx=4)
        ttk.Label(dz_r2, text="(SLURM quota; one-time Windows reg edit)",
                  foreground="gray").pack(side="left", padx=8)

    # -- top status row -------------------------------------------------

    def _build_status_row(self, pad: Dict[str, int]) -> None:
        """The persistent header area pinned above the Notebook.

        Two stacked rows that stay visible regardless of which tab
        is active:

          * Row A: ssh password entry + Forget button. Always
            visible because EVERY cluster op needs it; burying it
            inside one tab means the operator can't tell at a
            glance whether the password is set, and tab-switching
            doesn't change "do I need to type the password".

          * Row B: freshness dashboard (cluster job, local ckpt
            age, last archive age) + the ⚙ Settings gear. Same
            three StringVars as before the redesign, just stacked
            under the password row.

        The password lives in `self._password` (a tk.StringVar);
        the entry widget is `self._pw_entry` so external callers
        (e.g. ssh-failure handlers) can re-focus it if a wrong
        password was entered."""
        # Row A: password.
        row_pw = ttk.Frame(self.root)
        row_pw.pack(fill="x", padx=pad["padx"],
                    pady=(pad["pady"], 2))
        ttk.Label(row_pw, text="ssh password:").pack(
            side="left", padx=(4, 6))
        # `show="*"` masks the field; the value still lives in
        # self._password (StringVar) just like in the legacy
        # All-ops layout. tk.Entry kept (not ttk.Entry) because
        # we use the show= mask which both honour but tk.Entry
        # is slightly more predictable across Tk versions.
        self._pw_entry = tk.Entry(
            row_pw, textvariable=self._password,
            show="*", width=32,
        )
        self._pw_entry.pack(side="left", padx=4)
        ttk.Button(row_pw, text="Forget",
                   command=self._forget_password).pack(
                       side="left", padx=4)
        ttk.Label(
            row_pw,
            text=("kept in memory only; cleared on Forget or "
                  "window close"),
            foreground="gray",
        ).pack(side="left", padx=8)

        # Row B: freshness labels + ⚙. Same semantics as the
        # pre-refactor freshness row; promoted to the top so it's
        # first-glance state. ttk.Separator pipes between the
        # three labels keep them visually distinct without spam-
        # ming dashes.
        row = ttk.Frame(self.root)
        row.pack(fill="x", padx=pad["padx"], pady=(0, 2))
        self._fresh_job_var     = tk.StringVar(value="Cluster job: (unknown — click Status)")
        self._fresh_ckpt_var    = tk.StringVar(value="Local ckpt: (none)")
        self._fresh_archive_var = tk.StringVar(value="Last archive: (none)")
        ttk.Label(row, textvariable=self._fresh_job_var,
                  foreground="gray").pack(side="left", padx=4)
        ttk.Separator(row, orient="vertical").pack(
            side="left", fill="y", padx=8)
        ttk.Label(row, textvariable=self._fresh_ckpt_var,
                  foreground="gray").pack(side="left", padx=4)
        ttk.Separator(row, orient="vertical").pack(
            side="left", fill="y", padx=8)
        ttk.Label(row, textvariable=self._fresh_archive_var,
                  foreground="gray").pack(side="left", padx=4)
        # Settings gear, top-right. Placeholder for Phase 10 --
        # when implemented it opens the unified Settings dialog.
        # For now it pops a "not yet implemented" stub.
        ttk.Button(row, text="⚙", width=3,
                   command=self._op_settings_stub).pack(
                       side="right", padx=4)

    # -- command palette ----------------------------------------------

    def _palette_actions(self) -> List[Tuple[str, str, Callable[[], None]]]:
        """The canonical (name, description, command) list the
        command palette searches over. Built lazily so each call
        reflects the live `self._op_*` bindings (handy when a
        future refactor renames a method).

        Names mirror what's printed on the corresponding button;
        descriptions are short hints the operator sees in the
        palette listbox. Both are searched on substring match."""
        return [
            # Daily-loop ops
            ("Pull from cluster",
             "atomic checkpoint + log + history",
             self._op_pull_bundle),
            ("Check cluster status",
             "squeue + sacct probe",
             self._op_status),
            ("View train history",
             "open trainer_history_*.csv",
             self._op_view_train_history),
            ("View eval history",
             "open eval_history.md / .csv",
             self._op_view_eval_history),
            ("Display 1 game",
             "render a self-play game",
             self._op_display_selfplay),
            ("Diagnose",
             "probe encoder/sampler against a synthetic state",
             self._op_diagnose),
            ("Daily eval",
             "tools/eval_daily.py --preset quick",
             self._op_eval_daily),
            ("Run eval",
             "open the eval dialog (tune knobs)",
             self._op_eval_dialog),
            ("Push local + code",
             "upload sim_selfplay.pt + sync code",
             self._op_push_checkpoint),
            ("Sync + Continue (self-play)",
             "upload + restart self-play chain link",
             self._op_sync_continue_selfplay),
            # Train / cluster job management
            ("Start self-play",
             "sbatch wai-selfplay (no-op if running)",
             self._op_start_selfplay),
            ("Stop self-play",
             "scancel the self-play job",
             self._op_stop_selfplay),
            ("Start supervised",
             "sbatch wai-supervised (no-op if running)",
             self._op_start_supervised),
            ("Stop supervised",
             "scancel the supervised job",
             self._op_stop_supervised),
            ("Sync + Continue (supervised)",
             "restart supervised chain link",
             self._op_sync_continue_supervised),
            ("Train (self-play) locally",
             "local trainer run (no password needed)",
             self._op_train_selfplay),
            ("Stop local training (save)",
             "graceful sentinel; trainer exits at next iter",
             self._op_graceful_cancel),
            ("Force cancel current op",
             "taskkill the running subprocess immediately",
             self._cancel_op),
            ("Edit rewards",
             "open the reward-config editor",
             self._op_edit_rewards),
            # Maintenance / rarer
            ("Pull (specialised)",
             "pull a specific archive / epoch (mode entry)",
             self._op_pull),
            ("Pull logs",
             "pull the latest cluster SLURM log",
             self._op_pull_logs),
            ("Profile (cluster)",
             "submit a one-shot profile job",
             self._op_profile_cluster),
            ("Pull profile",
             "scp the latest profile artifact dir",
             self._op_pull_profile),
            ("View profile",
             "open the latest local profile artifact",
             self._op_view_profile),
            ("Restore from archive",
             "revert cluster's sim_selfplay.pt to a snapshot",
             self._op_restore_from_archive),
            ("Sync code",
             "push code only (no checkpoint, no restart)",
             self._op_sync),
            ("Check cluster budget",
             "squeue + sacct + sshare summary",
             self._op_budget),
            ("Setup TDR (admin)",
             "raise Windows GPU TDR threshold (reboot needed)",
             self._op_setup_tdr),
            # Meta
            ("Open Settings",
             "automation toggles + theme + advanced",
             self._op_settings_stub),
        ]

    def _open_command_palette(self) -> None:
        """Modal Ctrl-K palette: search Entry on top, filtered
        Listbox below.

          * Type to filter (substring, case-insensitive, matches
            either the action name or its description).
          * Up/Down to navigate the list.
          * Enter (or Return) to execute the highlighted action.
          * Escape to cancel without firing anything.

        Re-entrant guard: pressing Ctrl-K with the palette
        already open just refocuses the existing one instead
        of spawning a stacked clone. Tracked via a weak
        instance attribute.
        """
        existing = getattr(self, "_palette_dlg", None)
        if existing is not None and existing.winfo_exists():
            try:
                existing.lift()
                existing.focus_force()
                return
            except tk.TclError:
                pass

        dlg = tk.Toplevel(self.root)
        dlg.title("Command palette")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.geometry("520x360")

        actions = self._palette_actions()
        # Pre-compute lowercase search corpus so filtering is
        # cheap on every keystroke (~30 actions; negligible).
        corpus = [(name, desc, cmd,
                   (name + " " + desc).lower())
                  for name, desc, cmd in actions]

        query_var = tk.StringVar(value="")
        entry = ttk.Entry(dlg, textvariable=query_var,
                          font=("Segoe UI", 11))
        entry.pack(fill="x", padx=10, pady=(10, 4))

        # Listbox + scrollbar. Listbox chosen over ttk.Treeview
        # because the data is one column ("name -- desc") and a
        # plain Listbox is keyboard-friendly out of the box
        # (Up/Down navigate, Enter binds cleanly).
        lst_frame = ttk.Frame(dlg)
        lst_frame.pack(fill="both", expand=True, padx=10, pady=(0, 4))
        scrl = ttk.Scrollbar(lst_frame, orient="vertical")
        lst = tk.Listbox(lst_frame, font=("Segoe UI", 10),
                         yscrollcommand=scrl.set,
                         activestyle="dotbox",
                         exportselection=False)
        scrl.config(command=lst.yview)
        scrl.pack(side="right", fill="y")
        lst.pack(side="left", fill="both", expand=True)

        # Filtered subset of `corpus` for the current query.
        filtered: List = []

        def refresh_list(*_a):
            q = query_var.get().strip().lower()
            lst.delete(0, "end")
            filtered.clear()
            for entry_tuple in corpus:
                name, desc, cmd, hay = entry_tuple
                if not q or q in hay:
                    filtered.append(entry_tuple)
                    lst.insert("end", f"{name}  —  {desc}")
            if filtered:
                lst.selection_clear(0, "end")
                lst.selection_set(0)
                lst.activate(0)

        def run_selected(_event=None):
            sel = lst.curselection()
            if not sel:
                return
            _, _, cmd, _ = filtered[sel[0]]
            dlg.destroy()
            # Defer the call so the modal is fully torn down
            # before the action fires -- some ops open their own
            # modals (eval dialog, restore-from-archive), and
            # stacking a dialog inside a destroying parent
            # produces flicker.
            self.root.after(20, cmd)

        # Bindings. Up/Down on the Entry navigate the list;
        # Enter runs; Escape closes. The "down arrow from
        # Entry" pattern means the operator never has to move
        # focus off the search box -- type, Up/Down, Enter.
        def move(delta):
            if not filtered:
                return "break"
            cur = lst.curselection()
            idx = cur[0] if cur else 0
            idx = max(0, min(len(filtered) - 1, idx + delta))
            lst.selection_clear(0, "end")
            lst.selection_set(idx)
            lst.activate(idx)
            lst.see(idx)
            return "break"

        entry.bind("<Down>", lambda _e: move(+1))
        entry.bind("<Up>", lambda _e: move(-1))
        entry.bind("<Return>", run_selected)
        lst.bind("<Return>", run_selected)
        lst.bind("<Double-Button-1>", run_selected)
        dlg.bind("<Escape>", lambda _e: dlg.destroy())
        query_var.trace_add("write", refresh_list)

        # Initial population + focus.
        refresh_list()
        entry.focus_set()

        # Centre on parent.
        dlg.update_idletasks()
        try:
            px = self.root.winfo_rootx()
            py = self.root.winfo_rooty()
            pw = self.root.winfo_width()
            ph = self.root.winfo_height()
            dw = dlg.winfo_width()
            dh = dlg.winfo_height()
            x = px + max(0, (pw - dw) // 2)
            y = py + max(0, (ph - dh) // 3)
            dlg.geometry(f"+{x}+{y}")
        except tk.TclError:
            pass

        # Track for re-entrance guard. Cleared on destroy via
        # the lambda below; the WeakRef-style attribute lets
        # _open_command_palette short-circuit subsequent Ctrl-K
        # presses while the palette is up.
        self._palette_dlg = dlg
        dlg.bind("<Destroy>",
                 lambda _e: setattr(self, "_palette_dlg", None))

    def _op_settings_stub(self) -> None:
        """Open the unified Settings dialog. Kept under the old name
        so the ⚙ button binding (and any cross-reference) doesn't
        need a rename; the body is the real implementation now.

        Sections:
          * Automation -- the two anti-overwrite toggles
            (auto-pull on completion, auto-eval after pull).
            Mirrored in the Daily-tab Pull section; flipping in
            either place updates the other instantly because
            they share the BooleanVars.
          * Appearance -- ttk theme picker. Live-applied on
            Save without restarting the GUI (ttkthemes'
            `set_theme` handles widget restyling).
          * Advanced -- the auto-pull poll cadence, for the
            rare operator who wants a faster or slower probe.
            Default 120s is the recommended balance.

        Save persists every field via `gui_state.json`; Cancel
        reverts to the on-disk values without touching them. No
        confirmation modal -- this dialog doesn't do anything
        destructive."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Settings")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)

        # Snapshot the live values so Cancel can revert. The
        # BooleanVars persist via trace_add on `write`, so a
        # naive "always write through" would commit changes
        # immediately -- we work around that by detaching the
        # trace inside the dialog and re-attaching on close.
        gs = _load_gui_state()
        prev_auto_pull = self._auto_pull_enabled.get()
        prev_auto_eval = self._auto_eval_after_pull.get()
        prev_theme = gs.get("theme", GUI_THEME)
        prev_poll  = gs.get("auto_pull_poll_seconds",
                            self.AUTO_PULL_POLL_MS // 1000)

        v_auto_pull = tk.BooleanVar(value=prev_auto_pull)
        v_auto_eval = tk.BooleanVar(value=prev_auto_eval)
        v_theme     = tk.StringVar(value=prev_theme)
        v_poll      = tk.IntVar(value=int(prev_poll))

        # ---- Automation ----------------------------------------------
        auto_box = ttk.LabelFrame(dlg, text="Automation")
        auto_box.pack(fill="x", padx=12, pady=(10, 4))
        ttk.Checkbutton(
            auto_box,
            text=("Auto-pull bundle (checkpoint + log + history) "
                  "when the cluster self-play job ends"),
            variable=v_auto_pull,
        ).pack(anchor="w", padx=8, pady=(6, 2))
        ttk.Checkbutton(
            auto_box,
            text=("Auto-run Daily eval after every pull that "
                  "lands a new sim_selfplay.pt"),
            variable=v_auto_eval,
        ).pack(anchor="w", padx=8, pady=(0, 6))

        # ---- Appearance ----------------------------------------------
        app_box = ttk.LabelFrame(dlg, text="Appearance")
        app_box.pack(fill="x", padx=12, pady=4)
        app_row = ttk.Frame(app_box)
        app_row.pack(fill="x", padx=8, pady=6)
        ttk.Label(app_row, text="Theme:").pack(side="left", padx=4)
        # Theme list: only enumerate themes ttkthemes actually
        # exposes on this install. Fall back to a tiny built-in
        # subset if ttkthemes isn't installed (stock tk has
        # 'clam', 'alt', 'default'; safer to just show those).
        if _HAVE_TTKTHEMES:
            try:
                themes = sorted(self.root.get_themes())   # type: ignore[attr-defined]
            except Exception:
                themes = ["arc", "equilux", "breeze", "clam"]
        else:
            themes = ["clam", "alt", "default"]
        ttk.Combobox(
            app_row, textvariable=v_theme,
            values=themes, state="readonly",
            width=18,
        ).pack(side="left", padx=4)
        ttk.Label(
            app_row,
            text="(applies on Save without restart)",
            foreground="gray",
        ).pack(side="left", padx=8)

        # ---- Advanced ------------------------------------------------
        adv_box = ttk.LabelFrame(dlg, text="Advanced")
        adv_box.pack(fill="x", padx=12, pady=4)
        adv_row = ttk.Frame(adv_box)
        adv_row.pack(fill="x", padx=8, pady=6)
        ttk.Label(
            adv_row, text="Auto-pull probe interval (seconds):",
        ).pack(side="left", padx=4)
        # 30s minimum because each probe is one full ssh round-
        # trip + handshake; sub-30s thrashes both the cluster
        # login node and the operator's network. 600s max because
        # past that the "respond within a couple minutes" promise
        # breaks down -- a 10min idle window is plenty stale.
        ttk.Spinbox(
            adv_row, from_=30, to=600, increment=30,
            textvariable=v_poll, width=6,
        ).pack(side="left", padx=4)
        ttk.Label(
            adv_row,
            text="(default 120; lower = snappier auto-pull, "
                 "higher = lighter on cluster)",
            foreground="gray",
        ).pack(side="left", padx=8)

        # ---- Buttons -------------------------------------------------
        btns = ttk.Frame(dlg)
        btns.pack(fill="x", padx=12, pady=(8, 12))

        def on_save():
            # Validate poll value -- Spinbox returns a string in
            # the var even with from_/to clamps, so a deliberate
            # type-and-paste of "abc" would crash int() below.
            try:
                poll_sec = int(v_poll.get())
            except (tk.TclError, ValueError):
                messagebox.showerror(
                    "Invalid value",
                    "Auto-pull probe interval must be an integer "
                    "between 30 and 600 seconds.")
                return
            poll_sec = max(30, min(600, poll_sec))
            # Persist + live-apply.
            state = _load_gui_state()
            state["auto_pull_on_completion"] = bool(v_auto_pull.get())
            state["auto_eval_after_pull"]    = bool(v_auto_eval.get())
            state["theme"] = v_theme.get()
            state["auto_pull_poll_seconds"] = poll_sec
            _save_gui_state(state)
            # BooleanVar trace handlers re-write the same keys --
            # idempotent, but they also log the toggle change.
            # Set them via the shared vars so the Daily-tab
            # checkboxes update visually too.
            self._auto_pull_enabled.set(bool(v_auto_pull.get()))
            self._auto_eval_after_pull.set(bool(v_auto_eval.get()))
            # Apply theme. ttkthemes' ThemedTk exposes set_theme;
            # stock tk.Tk doesn't. Either way the persisted value
            # is honoured on next launch.
            if _HAVE_TTKTHEMES:
                try:
                    self.root.set_theme(v_theme.get())   # type: ignore[attr-defined]
                except Exception as e:
                    self._log(f"[settings] theme '{v_theme.get()}' "
                              f"didn't apply live: {e}. It will be "
                              f"loaded on next launch.")
            # Cadence: AUTO_PULL_POLL_MS is read at next tick, so
            # we just update the class-level attribute. Currently-
            # scheduled tick still uses the old value; that's
            # fine (the very next reschedule picks up the new).
            self.AUTO_PULL_POLL_MS = poll_sec * 1000
            self._log(
                f"[settings] saved -- auto_pull="
                f"{bool(v_auto_pull.get())}, auto_eval="
                f"{bool(v_auto_eval.get())}, theme="
                f"{v_theme.get()}, poll={poll_sec}s.")
            dlg.destroy()

        ttk.Button(btns, text="Cancel", width=12,
                   command=dlg.destroy).pack(side="right", padx=4)
        ttk.Button(btns, text="Save", width=12,
                   command=on_save).pack(side="right", padx=4)

        # Centre on the parent window.
        dlg.update_idletasks()
        try:
            px = self.root.winfo_rootx()
            py = self.root.winfo_rooty()
            pw = self.root.winfo_width()
            ph = self.root.winfo_height()
            dw = dlg.winfo_width()
            dh = dlg.winfo_height()
            x = px + max(0, (pw - dw) // 2)
            y = py + max(0, (ph - dh) // 3)
            dlg.geometry(f"+{x}+{y}")
        except tk.TclError:
            pass

    # -- bottom bar -----------------------------------------------------

    def _build_bottom_bar(self, pad: Dict[str, int]) -> None:
        """The two cancel buttons + clear + status label. Built as
        its own method so the layout is clear: the bar is `pack`ed
        AFTER the PanedWindow (which has expand=True), so it docks
        to the bottom of the window across all Notebook tabs.

        Behavior unchanged from the Phase-0 monolithic
        _build_widgets:
          "Stop training (save)" -- writes a sentinel file the
            trainer detects between iters. Trainer saves the
            current sim_selfplay.pt + exits cleanly. Worst-case
            latency = one iter (~3-5 min on CPU, less on DML).
          "Force cancel" -- taskkill /T /F. Immediate kill of the
            subprocess tree. Use when graceful is too slow or for
            non-training ops that don't honor the sentinel.
        """
        bar = ttk.Frame(self.root); bar.pack(fill="x", **pad)
        self._graceful_btn = ttk.Button(
            bar, text="Stop training (save)",
            command=self._op_graceful_cancel,
        )
        self._graceful_btn.pack(side="left", padx=4)
        self._cancel_btn = ttk.Button(
            bar, text="Force cancel",
            command=self._cancel_op, state="disabled",
        )
        self._cancel_btn.pack(side="left", padx=4)
        ttk.Button(bar, text="Clear output",
                   command=self._clear_output).pack(side="left", padx=4)
        self._status_var = tk.StringVar(value="Idle.")
        ttk.Label(bar, textvariable=self._status_var,
                  foreground="gray").pack(side="left",
                                          fill="x", expand=True, padx=8)

    # -- output stream --------------------------------------------------

    def _log(self, line: str, tag: str = "info") -> None:
        """Append a line to the output panel from the main thread."""
        self._output.configure(state="normal")
        self._output.insert("end", line if line.endswith("\n") else line + "\n")
        self._output.see("end")
        self._output.configure(state="disabled")

    def _drain_output(self) -> None:
        """Pull queued lines from worker threads onto the output panel.
        Re-arms via tk.after; runs forever. Also feeds the per-op
        capture buffer when one is active so on-complete callbacks
        can inspect the full output."""
        try:
            while True:
                tag, line = self._stdout_q.get_nowait()
                stripped = line.rstrip("\r\n")
                self._log(stripped, tag)
                if self._captured_lines is not None:
                    self._captured_lines.append(stripped)
        except queue.Empty:
            pass
        self.root.after(self.OUTPUT_POLL_MS, self._drain_output)

    def _clear_output(self) -> None:
        self._output.configure(state="normal")
        self._output.delete("1.0", "end")
        self._output.configure(state="disabled")

    # -- freshness row --------------------------------------------------

    @staticmethod
    def _humanize_age(seconds: float) -> str:
        """Coarse age string: '3m ago', '2h ago', '4d ago'. Coarser
        than strftime but readable at a glance, which is the point of
        the freshness row."""
        if seconds < 60:        return f"{int(seconds)}s ago"
        if seconds < 3_600:     return f"{int(seconds / 60)}m ago"
        if seconds < 86_400:    return f"{int(seconds / 3_600)}h ago"
        return                       f"{int(seconds / 86_400)}d ago"

    def _refresh_local_freshness(self) -> None:
        """Re-stat the local checkpoint + freshest archive. Cheap
        (two stat() calls + one glob). Also refreshes the model-
        dependent button states (Display 1 game, Daily eval,
        Push, etc.) since the checkpoint we just stat'd is the
        gate for them."""
        import time as _t
        ckpt_dir = PROJECT_ROOT / "training" / "checkpoints"
        sp = ckpt_dir / "sim_selfplay.pt"
        now = _t.time()
        if sp.exists():
            age = now - sp.stat().st_mtime
            self._fresh_ckpt_var.set(
                f"Local ckpt: sim_selfplay.pt {self._humanize_age(age)}")
        else:
            self._fresh_ckpt_var.set("Local ckpt: (none)")
        archives = sorted(ckpt_dir.glob("sim_selfplay_archive_*.pt"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if archives:
            age = now - archives[0].stat().st_mtime
            self._fresh_archive_var.set(
                f"Last archive: {self._humanize_age(age)} "
                f"({len(archives)} kept)")
        else:
            self._fresh_archive_var.set("Last archive: (none)")
        # Same stat() informs the model-button enablement. Guarded
        # for the first call (before _build_widgets has registered
        # any buttons) -- the list is empty then, so the iter is
        # a no-op.
        if hasattr(self, "_model_buttons"):
            self._refresh_model_buttons()

    def _tick_freshness(self) -> None:
        """Periodic refresh handler -- re-arms via tk.after."""
        try:
            self._refresh_local_freshness()
        except Exception:
            # Don't let a transient file-stat error kill the loop.
            pass
        self.root.after(self.FRESHNESS_POLL_MS, self._tick_freshness)

    def _scan_for_cluster_job_state(self, lines: List[str]) -> None:
        """Parse `lines` from a finished cluster-status op and update
        `_fresh_job_var`. Called after Status / Start / Stop / Sync+
        Continue finish. We avoid a periodic ssh poll because each
        would fire a password prompt; the trade-off is a slightly
        stale "Cluster job" label until the operator clicks Status.

        Format we look for, from `cluster/run.sh status`:
            --- our selfplay job 17999 ---
                  17999    ENSTA-l40s wai-selfp alcim  R    1:23:45      1 ...
        or:
                  (job 17999 no longer in the queue)
        """
        # `run.sh status` emits one labelled block per mode. We
        # collect all of them and prefer a RUNNING job over a DONE
        # one when displaying. Per-block state is built up while
        # scanning; we anchor on the `--- our (selfplay|supervised)
        # job <jid> ---` header (regex captures both names).
        import re as _re
        header_re = _re.compile(
            r"^---\s+our\s+(selfplay|supervised)\s+job\s+(\d+)\s+---\s*$",
            _re.IGNORECASE,
        )
        # Per-mode collected state.
        per_mode: Dict[str, Dict[str, Optional[str]]] = {}
        current_mode: Optional[str] = None
        for raw in lines:
            stripped = raw.strip()
            m = header_re.match(stripped)
            if m:
                current_mode = m.group(1).lower()
                per_mode[current_mode] = {
                    "jid":     m.group(2),
                    "state":   None,
                    "elapsed": None,
                }
                continue
            if current_mode is None:
                continue
            slot = per_mode[current_mode]
            jid = slot["jid"]
            if jid and "no longer in the queue" in stripped:
                slot["state"] = "DONE"
                continue
            # squeue row: <jid> <partition> <name> <user> <st> <elapsed> ...
            parts = stripped.split()
            if (jid and len(parts) >= 6 and parts[0] == jid
                    and len(parts[4]) <= 2):
                slot["state"]   = parts[4]
                slot["elapsed"] = parts[5]
        if not per_mode:
            return
        # Pick which mode's state to show. Prefer a job whose
        # status is "R" (running), then "PD" (queued), then DONE,
        # then unknown. If both are running (rare on a fair-share
        # cluster), prefer selfplay since that's the primary
        # workflow.
        order = ("selfplay", "supervised")
        priority = {"R": 0, "PD": 1, "DONE": 2, None: 3}
        chosen = sorted(
            per_mode.items(),
            key=lambda kv: (priority.get(kv[1]["state"], 4),
                            order.index(kv[0])),
        )[0]
        mode, slot = chosen
        jid     = slot["jid"]
        state   = slot["state"]
        elapsed = slot["elapsed"]
        if state == "R":
            self._fresh_job_var.set(
                f"Cluster job: {mode} {jid} RUNNING ({elapsed})")
        elif state == "PD":
            self._fresh_job_var.set(f"Cluster job: {mode} {jid} QUEUED")
        elif state == "DONE":
            self._fresh_job_var.set(f"Cluster job: {mode} {jid} done")
        elif state:
            self._fresh_job_var.set(
                f"Cluster job: {mode} {jid} {state}")
        else:
            self._fresh_job_var.set(
                f"Cluster job: {mode} {jid} (unknown state)")

    # -- auto-pull watcher ---------------------------------------------

    # -- stale-push guard ---------------------------------------------

    def _register_push_button(self, btn) -> None:
        """Track a Push-flavoured button so the stale-push guard can
        grey it out centrally. Called from each tab builder that
        creates a button which would overwrite cluster state. The
        guard's initial state is applied immediately so the button
        spawns disabled if we already know push is unsafe (e.g.,
        GUI restart while a cluster job is mid-run)."""
        self._push_buttons.append(btn)
        # Apply current state to the just-added button. Cheap
        # one-button refresh -- the caller doesn't have to know
        # whether the central state was already set.
        if self._push_blocked_reason:
            try:
                btn.configure(state="disabled")
            except tk.TclError:
                pass

    def _refresh_push_buttons(self) -> None:
        """Apply the stale-push block to the explanatory label
        below the Push buttons + delegate to the unified state
        engine to grey/ungrey the buttons themselves.

        Why split it out from `_refresh_button_states`: the label
        ONLY exists when push is blocked, and the text comes from
        `_push_blocked_reason`. The button state is one of several
        constraints the unified engine considers."""
        self._push_block_lbl_var.set(self._push_blocked_reason or "")
        self._refresh_button_states()

    def _refresh_button_states(self) -> None:
        """Single source-of-truth for "is button X clickable right
        now?". Each registered button can sit in one or more of
        the three constraint lists:
            push_buttons:    blocked by stale-push guard
            cluster_buttons: needs ssh password
            model_buttons:   needs local sim_selfplay.pt
        The button's effective state is enabled iff EVERY
        constraint it's subject to is satisfied (logical AND).

        Cheap enough to call from any state-change source --
        password keystroke, freshness tick, pull-bundle finalize,
        auto-pull probe handler. Idempotent."""
        have_pw = bool(self._password.get())
        have_ckpt = (PROJECT_ROOT / "training" / "checkpoints"
                     / "sim_selfplay.pt").exists()
        push_blocked = bool(self._push_blocked_reason)
        # Build the set of all registered buttons exactly once,
        # then resolve each one's constraints by set membership.
        # Set-membership is O(1) per check; the whole pass is
        # linear in (cluster + model + push) buttons -- ~15 total.
        push_set    = set(map(id, self._push_buttons))
        cluster_set = set(map(id, self._cluster_buttons))
        model_set   = set(map(id, self._model_buttons))
        all_btns = {id(b): b for b in (self._push_buttons
                                       + self._cluster_buttons
                                       + self._model_buttons)}
        for bid, btn in all_btns.items():
            enabled = True
            if bid in push_set and push_blocked:
                enabled = False
            if bid in cluster_set and not have_pw:
                enabled = False
            if bid in model_set and not have_ckpt:
                enabled = False
            try:
                btn.configure(state="normal" if enabled else "disabled")
            except tk.TclError:
                pass

    def _set_push_blocked(self, reason: Optional[str]) -> None:
        """Set the block reason and refresh the UI. Pulled out as
        a helper so the half-dozen state-machine transitions stay
        readable (each is one call to set/clear)."""
        if reason == self._push_blocked_reason:
            return                       # idempotent; avoid log spam
        self._push_blocked_reason = reason
        if reason:
            self._log(f"[push-guard] Push disabled: {reason}")
        else:
            self._log("[push-guard] Push re-enabled.")
        self._refresh_push_buttons()

    def _register_cluster_button(self, btn) -> None:
        """Register a button that needs the ssh password to do
        anything useful. Disabled until the password entry has
        text. Called from each tab builder right after creating
        the widget."""
        self._cluster_buttons.append(btn)
        if not self._password.get():
            try:
                btn.configure(state="disabled")
            except tk.TclError:
                pass

    def _refresh_cluster_buttons(self) -> None:
        """Trigger a unified button-state refresh on a password
        change. Thin wrapper that exists so the trace_add binding
        in __init__ has a stable name to call.
        """
        self._refresh_button_states()

    def _register_model_button(self, btn) -> None:
        """Register a button that requires a local sim_selfplay.pt
        to do anything useful (evaluate, display a game, push to
        cluster). Disabled when no local checkpoint exists.

        The freshness tick (every 30s) already stats the file,
        so the refresh piggy-backs on that loop -- no extra
        cost. Operators who drop a checkpoint into the directory
        by hand wait at most one tick for the button to enable."""
        self._model_buttons.append(btn)
        if not (PROJECT_ROOT / "training" / "checkpoints"
                / "sim_selfplay.pt").exists():
            try:
                btn.configure(state="disabled")
            except tk.TclError:
                pass

    def _refresh_model_buttons(self) -> None:
        """Trigger a unified button-state refresh after a local-
        file change. Thin wrapper called from the freshness tick;
        named for clarity at call sites."""
        self._refresh_button_states()

    def _record_pulled_jid(self, jid: Optional[str]) -> None:
        """Persist the jid we just pulled from. Side-effect:
        clears the push block if it was set for THIS jid (the
        operator just brought the local copy up to date)."""
        if not jid:
            return
        self._last_pulled_jid = jid
        gs = _load_gui_state()
        gs["last_pulled_jid"] = jid
        _save_gui_state(gs)

    def _on_auto_eval_toggle(self, *_a) -> None:
        """Persist the auto-eval toggle. Symmetric with the auto-
        pull toggle: same `gui_state.json` store, same log/notice
        when flipped so background behaviour is never invisible."""
        state = _load_gui_state()
        state["auto_eval_after_pull"] = bool(
            self._auto_eval_after_pull.get())
        _save_gui_state(state)
        if self._auto_eval_after_pull.get():
            self._log("[auto-eval] enabled -- will run "
                      "tools/eval_daily.py (quick preset) "
                      "automatically after every pull that lands "
                      "a new sim_selfplay.pt.")
        else:
            self._log("[auto-eval] disabled -- eval the new model "
                      "manually via `Daily eval` if you want a "
                      "fresh row in eval_history.")

    def _on_auto_pull_toggle(self, *_a) -> None:
        """Persist the toggle to gui_state.json + log the new state.
        Bound via `trace_add('write', ...)` on the BooleanVar so a
        click on the Daily-tab checkbox immediately survives a GUI
        restart.

        Why log: the feature changes background behaviour invisibly
        otherwise. The operator wants to know "did I just turn this
        off?" without inspecting a config file."""
        state = _load_gui_state()
        state["auto_pull_on_completion"] = bool(
            self._auto_pull_enabled.get())
        _save_gui_state(state)
        if self._auto_pull_enabled.get():
            self._log("[auto-pull] enabled -- will pull bundle "
                      "automatically when the cluster self-play "
                      "job finishes.")
        else:
            self._log("[auto-pull] disabled -- you'll need to "
                      "click `Pull from cluster` manually.")
            # Disarm any pending watch so re-enabling later
            # doesn't immediately fire for a stale jid.
            self._auto_pull_armed_jid = None

    def _tick_auto_pull(self) -> None:
        """Periodic kick. Reschedules itself unconditionally so the
        watcher keeps running even if the operator toggles auto-pull
        off and back on. Probe is skipped when:
          * The feature is disabled (operator unticked the box).
          * No ssh password is set yet (probe would prompt forever).
          * Another op is in flight (the user is actively doing
            something; probing would either compete or be redundant).
          * A prior probe is still running (slow cluster network).

        The probe itself runs in a background thread so the GUI
        stays responsive even on a sluggish ssh round-trip; the
        result is posted back to the GUI thread via
        `root.after(0, ...)` for thread-safe state updates."""
        try:
            if (self._auto_pull_enabled.get()
                    and self._password.get()
                    and not self._is_busy()
                    and not self._auto_pull_probe_in_flight):
                self._auto_pull_probe_in_flight = True
                threading.Thread(
                    target=self._auto_pull_probe_thread_fn,
                    daemon=True,
                ).start()
        finally:
            # Always re-schedule, even on exceptions. The watcher
            # MUST keep running for as long as the GUI is open --
            # losing it silently would defeat the whole anti-
            # overwrite purpose.
            self.root.after(
                self.AUTO_PULL_POLL_MS, self._tick_auto_pull)

    def _auto_pull_probe_thread_fn(self) -> None:
        """Background worker: one ssh + one squeue, captures result,
        posts back to the GUI thread.

        Probe command: `squeue --me -h -o "%i %j %T"`
          %i = jobid, %j = job-name, %T = state (RUNNING/PENDING/
          COMPLETING/etc).  `-h` strips the header; `--me` filters
          to the current user. Output is one line per active job.

        Only the FIRST `wai-selfplay` line is considered. If none,
        the self-play job is absent (either never submitted, or
        already finished). Sacct could distinguish those two cases
        but adds latency; we just record "absent" and let the
        state machine decide whether that's a terminal transition."""
        pw = self._password.get()
        # The password could have been cleared between the
        # tick's guard check and this thread starting; double-
        # check here to avoid a stuck askpass prompt.
        if not pw:
            self.root.after(
                0, lambda: self._auto_pull_on_probe_result(
                    ok=False, jid=None, state=None,
                    err="password cleared mid-probe"))
            return
        askpass = AskpassFiles(pw)
        try:
            creationflags = (subprocess.CREATE_NO_WINDOW
                             if os.name == "nt" else 0)
            res = subprocess.run(
                ["ssh", REMOTE_HOST,
                 'squeue --me -h -o "%i %j %T" 2>/dev/null'],
                capture_output=True, text=True, env=askpass.env(),
                creationflags=creationflags, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            self.root.after(
                0, lambda e=e: self._auto_pull_on_probe_result(
                    ok=False, jid=None, state=None, err=str(e)))
            return
        finally:
            askpass.cleanup()
        if res.returncode != 0:
            err = res.stderr.strip() or f"exit {res.returncode}"
            self.root.after(
                0, lambda err=err: self._auto_pull_on_probe_result(
                    ok=False, jid=None, state=None, err=err))
            return
        # Parse: one line per job, looking for wai-selfplay.
        # Job-name truncation: squeue defaults clamp %j to 8 chars
        # so the visible name is "wai-self" -- our format string
        # forces full names by overriding %j explicitly, but we
        # match by prefix for safety in case the cluster admin
        # ever tightens the width on the server side.
        sp_jid: Optional[str] = None
        sp_state: Optional[str] = None
        for line in res.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            jid, name, state = parts[0], parts[1], parts[2]
            if name.startswith("wai-selfplay") or name == "wai-self":
                sp_jid = jid
                sp_state = state
                break
        self.root.after(
            0, lambda: self._auto_pull_on_probe_result(
                ok=True, jid=sp_jid, state=sp_state, err=None))

    def _auto_pull_on_probe_result(
            self, *, ok: bool, jid: Optional[str],
            state: Optional[str], err: Optional[str]) -> None:
        """Handle a probe result on the GUI thread.

        Edge-triggered logging:
          * On the first failure after a stretch of success, we
            log the error and the cause. Subsequent consecutive
            failures stay silent.
          * On the first success after failures, we log "recovered".
        This keeps a 2-hour cluster outage from spamming the
        output panel 60 times.

        State machine:
          * ok && jid && state in {R, PD, CF, CG} -> ARM on jid
              ("currently watching this job; pull when it goes").
              CF/CG = configuring/completing transition states
              -- still "active", don't treat as done.
          * ok && (no jid OR state in {CD, F, CA, TO, OOM, BF, NF})
              AND armed_jid is set -> FIRE pull.
              Terminal SLURM states: COMPLETED, FAILED, CANCELLED,
              TIMEOUT, OUT_OF_MEMORY, BOOT_FAIL, NODE_FAIL.
          * else -> no action.

        We re-arm on every running observation so a long job
        produces exactly one auto-pull on its (eventual) end."""
        self._auto_pull_probe_in_flight = False
        if not ok:
            if self._auto_pull_last_probe_ok:
                self._log(f"[auto-pull] probe failed: {err}. "
                          f"Will keep retrying every "
                          f"{self.AUTO_PULL_POLL_MS // 1000}s; "
                          f"subsequent failures stay silent.")
            self._auto_pull_last_probe_ok = False
            return
        if not self._auto_pull_last_probe_ok:
            self._log("[auto-pull] probe recovered; watcher active.")
        self._auto_pull_last_probe_ok = True

        # SLURM state codes we treat as "still active" -- the job
        # is in the queue, so we ARM the watch. Includes the
        # short transitional ones the operator might encounter.
        ACTIVE = {"R", "RUNNING", "PD", "PENDING",
                  "CG", "COMPLETING", "CF", "CONFIGURING",
                  "S", "SUSPENDED"}
        # Terminal states. squeue normally drops terminal jobs
        # entirely, but on some clusters they linger briefly with
        # state=COMPLETED/FAILED before disappearing. We treat
        # both "gone from squeue" and "in squeue with terminal
        # state" as the trigger condition.
        TERMINAL = {"CD", "COMPLETED", "F", "FAILED",
                    "CA", "CANCELLED", "TO", "TIMEOUT",
                    "OOM", "OUT_OF_MEMORY", "BF", "BOOT_FAIL",
                    "NF", "NODE_FAIL", "DL", "DEADLINE"}

        if jid and state in ACTIVE:
            # Job is alive. Arm (or re-arm on the same jid).
            if self._auto_pull_armed_jid != jid:
                self._log(
                    f"[auto-pull] now watching cluster self-play "
                    f"job {jid} ({state}); will auto-pull when "
                    f"it ends.")
            self._auto_pull_armed_jid = jid
            # Stale-push guard: a running cluster job WITH a jid
            # we haven't pulled from means our local copy is
            # behind whatever progress the cluster is producing.
            # Block push until the operator pulls (manually or
            # via auto-pull when the job ends).
            if jid != self._last_pulled_jid:
                self._set_push_blocked(
                    f"cluster self-play job {jid} is running "
                    f"and produced work we haven't pulled. "
                    f"Pull first, or wait for auto-pull.")
            return

        # Either the job is gone (jid is None) OR squeue is still
        # showing it but in a terminal state.
        is_terminal = (state is not None and state in TERMINAL)
        is_absent   = (jid is None)
        if not (is_terminal or is_absent):
            # Some other state (e.g. "REVOKED", "STOPPED"
            # cluster-specific code). Be conservative: don't
            # fire, but log once so we know it happened.
            if self._auto_pull_armed_jid:
                self._log(
                    f"[auto-pull] job "
                    f"{self._auto_pull_armed_jid} now in state "
                    f"{state!r}; not a recognised terminal "
                    f"state, holding off.")
            return

        # Fire only if we have something armed. Operator may have
        # opened the GUI AFTER the job completed -- in that case
        # we have no record of seeing it run, so we don't pull
        # (we'd just be guessing). They can click manually.
        if not self._auto_pull_armed_jid:
            # Job is gone AND we never saw it run. Cluster may be
            # idle -- safe to push. Clear any stale block.
            if (jid is None
                    and self._push_blocked_reason
                    and "is running" in self._push_blocked_reason):
                self._set_push_blocked(None)
            return
        prev_jid = self._auto_pull_armed_jid
        self._auto_pull_armed_jid = None
        # If we're currently busy, skip the auto-pull THIS tick
        # (the operator is doing something). Don't re-arm --
        # they'll either pull manually or we'll catch the next
        # cluster job.
        if self._is_busy():
            self._log(
                f"[auto-pull] cluster job {prev_jid} ended but "
                f"another op is running; skipping auto-pull. "
                f"Click `Pull from cluster` manually when free.")
            return
        reason = ("ended" if is_absent
                  else f"finished with state {state}")
        self._log(
            f"[auto-pull] cluster self-play job {prev_jid} "
            f"{reason}; pulling bundle (checkpoint + log + "
            f"trainer_history).")
        # Update the freshness label so the operator sees the
        # transition even if they weren't looking at the output
        # panel.
        self._fresh_job_var.set(
            f"Cluster job: selfplay {prev_jid} done -- auto-pulling")
        # The job ended with unmerged work; push is unsafe until
        # we actually land the new checkpoint. (The block was
        # likely already set when we saw the job RUNNING; this
        # is a belt-and-braces in case the watcher missed the
        # running phase, e.g. short job that completed between
        # ticks.)
        if prev_jid != self._last_pulled_jid:
            self._set_push_blocked(
                f"cluster self-play job {prev_jid} {reason}. "
                f"Auto-pull is fetching the new checkpoint -- "
                f"push will re-enable when it lands.")
        # Fire the pull. _op_pull_bundle has its own busy/password
        # guards so even if we raced something, the worst case is
        # a no-op + a log line.
        self._op_pull_bundle()

    # -- op runner ------------------------------------------------------

    def _is_busy(self) -> bool:
        with self._proc_lock:
            return self._proc is not None

    def _begin_op(self, label: str) -> bool:
        """Common pre-flight. Returns False if we shouldn't start."""
        if self._is_busy():
            messagebox.showinfo("Busy", "Another operation is running.")
            return False
        self._status_var.set(f"Running: {label}")
        self._log(f"\n--- {label} ---")
        self._cancel_btn.configure(state="normal")
        return True

    def _end_op(self, returncode: int) -> None:
        # Drain any straggler lines the reader queued between the
        # process exit and the GUI tick before firing the
        # on_complete callback. Otherwise the buffer might be
        # missing the final few lines (squeue often ends with one).
        try:
            while True:
                _, line = self._stdout_q.get_nowait()
                stripped = line.rstrip("\r\n")
                self._log(stripped)
                if self._captured_lines is not None:
                    self._captured_lines.append(stripped)
        except queue.Empty:
            pass

        with self._proc_lock:
            self._proc = None
        if self._askpass is not None:
            self._askpass.cleanup()
            self._askpass = None
        self._cancel_btn.configure(state="disabled")
        if returncode == 0:
            self._status_var.set("Idle.")
            self._log(f"--- done (exit 0) ---")
        else:
            self._status_var.set(f"Last op exited {returncode}.")
            self._log(f"--- done (exit {returncode}) ---")

        # Fire the per-op completion callback (if any) and clear the
        # capture state for the next op. Wrapped so a buggy callback
        # can't strand the GUI in a half-ended state.
        cb = self._on_complete
        captured = self._captured_lines or []
        self._on_complete = None
        self._captured_lines = None
        if cb is not None:
            try:
                cb(captured)
            except Exception as e:
                self._log(f"on_complete callback raised: {e}")

        # Local-side files may have changed (a pull just landed a new
        # checkpoint; a cleanup just added an archive). Refresh
        # cheaply -- single stat + glob.
        try:
            self._refresh_local_freshness()
        except Exception:
            pass

    def _spawn(self,
               argv: List[str],
               *,
               needs_password: bool,
               label: str,
               cwd: Optional[Path] = None,
               on_complete: Optional[Callable[[List[str]], None]] = None) -> None:
        """Spawn a subprocess in a background thread, stream output.

        `on_complete(captured_lines)` -- optional. Fires on the GUI
        thread once the process exits and `_end_op` has cleaned up.
        Lines are the stripped (no trailing newline) stdout+stderr
        the op produced, in order. Used by the cluster-status ops
        to parse squeue output and update the freshness row.
        """
        if not self._begin_op(label):
            return
        # Start the per-op capture buffer + register the callback.
        # Cleared by _end_op after the callback fires.
        self._captured_lines = []
        self._on_complete = on_complete

        env = os.environ.copy()
        if needs_password:
            pw = self._password.get()
            if not pw:
                messagebox.showwarning(
                    "No password", "Cluster ops need an ssh password.")
                self._end_op(returncode=-1)
                return
            self._askpass = AskpassFiles(pw)
            env = self._askpass.env()

        # Hide subprocess console windows on Windows -- otherwise each
        # spawned ps1/python flashes a console. The GUI's output pane
        # captures everything anyway.
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW   # type: ignore[attr-defined]

        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(cwd or PROJECT_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,   # merge so we see auth errors inline
                stdin=subprocess.DEVNULL,   # prevent ssh from grabbing a TTY
                text=True,
                bufsize=1,                  # line-buffered
                creationflags=creationflags,
            )
        except OSError as e:
            self._log(f"FAILED to spawn: {e}", tag="error")
            self._end_op(returncode=-1)
            return

        with self._proc_lock:
            self._proc = proc

        # Reader thread streams output to our queue.
        threading.Thread(
            target=_stream_to_queue, args=(proc.stdout, self._stdout_q, "out"),
            daemon=True,
        ).start()
        # Waiter thread blocks on proc.wait, then schedules end-of-op
        # cleanup back on the GUI thread.
        threading.Thread(
            target=self._waiter, args=(proc,), daemon=True,
        ).start()

    def _waiter(self, proc: subprocess.Popen) -> None:
        rc = proc.wait()
        # End-of-op runs on the GUI thread for thread safety with tk.
        self.root.after(0, lambda: self._end_op(rc))

    def _op_graceful_cancel(self) -> None:
        """Write the graceful-cancel sentinel that the local trainer
        polls between iters. Trainer saves the current
        sim_selfplay.pt + exits cleanly at the next iter boundary.

        Doesn't immediately kill anything -- the trainer's loop has
        to come around to the next save_every / time-budget check.
        Worst-case latency = one iter (~3-5 min CPU, ~1-2 min DML).

        Safe to call when nothing is running: the sentinel is
        always cleared at trainer startup, so a stale write
        doesn't trap a future run. The button is always enabled
        (not gated on _is_busy) so an operator can flip it
        slightly before a known-good iter completes -- the
        polling check is idempotent.
        """
        sentinel = (PROJECT_ROOT / "training" / "checkpoints"
                    / ".cancel_local")
        try:
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("requested", encoding="utf-8")
        except OSError as e:
            self._log(f"[cancel] failed to write sentinel "
                      f"{sentinel}: {e}")
            messagebox.showerror(
                "Write failed",
                f"Couldn't write graceful-cancel sentinel:\n{e}")
            return
        if self._is_busy():
            self._log(
                "[cancel] graceful-stop sentinel written; the "
                "trainer will save sim_selfplay.pt + exit cleanly "
                "after the current iter completes. Use Force "
                "cancel if you need an immediate kill instead.")
        else:
            # Sentinel persists until either a trainer detects+
            # consumes it OR the operator launches a new run
            # (which CLEARS it at startup without honoring -- a
            # stale "cancel me" from yesterday shouldn't terminate
            # tomorrow's training).
            self._log(
                "[cancel] no training currently running. The "
                "sentinel was written but a future training run "
                "will clear it at startup WITHOUT exiting (we "
                "treat a stale sentinel as not-applicable). If "
                "you wanted to interrupt a specific run, launch "
                "it first then click this button again.")

    def _cancel_op(self) -> None:
        """Terminate the running op. The naive `proc.terminate()` only
        kills the immediate child (powershell.exe) and leaves anything
        it spawned -- python.exe + its child wesnoth.exe processes --
        running. On Windows we issue `taskkill /T /F` to walk the whole
        process tree from `proc.pid` down. Falls back to terminate() on
        non-Windows or if taskkill is unavailable.
        """
        with self._proc_lock:
            proc = self._proc
        if proc is None:
            return
        killed_via_taskkill = False
        if os.name == "nt":
            try:
                # /T = include the entire tree starting at PID.
                # /F = force (Wesnoth doesn't respond to graceful close
                #      on a hidden console anyway).
                # Suppress the spawned taskkill's own console window
                # for the same reason we did on the orchestrating
                # subprocess: keep the GUI uncluttered.
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                    capture_output=True, timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW,  # type: ignore[attr-defined]
                )
                killed_via_taskkill = True
            except (OSError, subprocess.TimeoutExpired) as e:
                self._log(f"taskkill failed ({e}); falling back to terminate()")
        if not killed_via_taskkill:
            try:
                proc.terminate()
            except OSError:
                pass
        self._log("--- cancel requested ---")

    # -- cluster operations ---------------------------------------------

    def _ps(self, script: Path, *args: str) -> List[str]:
        """Build a powershell.exe command line for a .ps1 script."""
        return [POWERSHELL, "-ExecutionPolicy", "Bypass",
                "-File", str(script), *args]

    def _op_status(self) -> None:
        # `bash cluster/run.sh status` runs ON the cluster. Wrap with ssh.
        # The on_complete callback parses the output and updates the
        # "Cluster job: ..." label in the freshness row -- this is the
        # only operator-facing refresh of that label (we avoid
        # periodic ssh polls because they'd each fire a password
        # prompt). Click Status to refresh.
        self._spawn(
            ["ssh", REMOTE_HOST,
             f"cd {REMOTE_PATH} && bash cluster/run.sh status"],
            needs_password=True, label="cluster status",
            on_complete=self._scan_for_cluster_job_state,
        )

    def _op_sync(self) -> None:
        self._spawn(self._ps(SCRIPT_SYNC),
                    needs_password=True, label="sync code")

    def _op_sync_continue_supervised(self) -> None:
        # sync + scancel running supervised job + sbatch a fresh
        # one that auto-resumes from the latest checkpoint. This
        # is the post-walltime workflow now that auto-resubmission
        # has been removed (operator-initiated chain links).
        self._spawn(self._ps(SCRIPT_SYNC, "-Continue",
                             "-Mode", "supervised"),
                    needs_password=True,
                    label="sync code + continue (supervised)")

    def _op_sync_continue_selfplay(self) -> None:
        # Same as the supervised version but targets self-play AND
        # also pushes the local sim_selfplay.pt so the cluster's
        # next chain link starts from local-overnight progress
        # (rather than whatever the cluster left behind, which is
        # typically staler now that DML training is unblocked).
        # Overwrite semantics by design -- the operator chose to
        # push, accepting that any cluster progress that didn't
        # get pulled is replaced. The sync script warns loudly
        # with the local checkpoint's size + mtime so this is
        # never silent. Refuses to push if the local checkpoint
        # is missing (e.g., a fresh install with no training yet).
        if not self._confirm_destructive_push(
                action="Sync code + push local sim_selfplay.pt + "
                       "restart cluster self-play chain",
        ):
            return
        self._spawn(self._ps(SCRIPT_SYNC, "-Continue",
                             "-Mode", "selfplay",
                             "-IncludeCheckpoint"),
                    needs_password=True,
                    label="push local + sync code + continue (self-play)")

    def _confirm_destructive(
            self, *,
            title: str,
            action: str,
            consequences: str = "",
            danger: str = "normal",
            reversible: bool = True,
            proceed_text: str = "Proceed",
            cancel_text: str = "Cancel",
    ) -> bool:
        """Uniform confirmation dialog for destructive ops.

        Centralizes the look + behavior of every "are you sure"
        prompt in the GUI so the operator builds one mental model
        of how confirmation works (rather than 8 slightly different
        `askyesno` flavours). Replaces ad-hoc `messagebox.askyesno`
        calls scattered through the file -- ops that need cluster-
        side state comparison (e.g. push-with-mtime-check) keep
        their specialised wrapper and call this at the end.

        Behaviour contract (the bits that prevent misclicks):
          * Default focus is the Cancel button. Pressing Enter on
            an unfocused dialog therefore CANCELS, not confirms --
            the operator must actually click or Tab-then-Enter to
            proceed. (askyesno defaults focus to Yes; this is the
            exact misclick we lost a training run to on 2026-05-14.)
          * Escape always cancels.
          * `danger="high"` adds a red banner above the action text
            and changes the Proceed button to a red-bordered style
            -- visual cue for the truly destructive ops
            (overwrite-with-older, force-cancel mid-iter, TDR
            registry edit) vs the routine "confirm intent" cases
            (stop a job, save+sync rewards).
          * `reversible=False` appends an explicit "this is NOT
            reversible" note to the consequences block; the
            operator should see undoability up-front.

        Returns True iff the operator clicked Proceed.

        Why a custom Toplevel rather than messagebox.askyesno
        with a wordier prompt: askyesno's default-Yes is a
        footgun, and we want danger styling that the stock
        dialog can't express. Tk Toplevel is cheap; one dialog
        spec per call site reads cleanly.
        """
        dlg = tk.Toplevel(self.root)
        dlg.title(title)
        dlg.transient(self.root)
        dlg.resizable(False, False)
        # grab_set makes the dialog modal -- clicks outside go nowhere
        # until it's dismissed. Same contract as messagebox.askyesno.
        dlg.grab_set()

        result = {"ok": False}

        # Optional danger banner. Bright red bg with white text is
        # the standard "stop and read" pattern -- consistent with
        # OS-level UAC-style warnings.
        if danger == "high":
            banner = tk.Label(
                dlg,
                text=" !!  DESTRUCTIVE OPERATION  !! ",
                bg="#cc2222", fg="white",
                font=("Segoe UI", 9, "bold"),
                padx=10, pady=4,
            )
            banner.pack(fill="x")

        # Action line: one-sentence imperative description. Bold so
        # it stands out from the consequences paragraph. Asymmetric
        # vertical padding MUST live on the pack() call -- the
        # widget's own `pady` option only accepts a single distance,
        # passing a tuple there raises TclError ("bad screen distance
        # '10 4'"), the Toplevel ends up empty with grab_set still
        # held, and the operator gets a blank modal dialog.
        action_lbl = tk.Label(
            dlg, text=action,
            font=("Segoe UI", 10, "bold"),
            justify="left", anchor="w", wraplength=480,
            padx=12,
        )
        action_lbl.pack(fill="x", pady=(10, 4))

        # Consequences block: what the op will actually do. Wraps
        # at 480px so long explanations stay readable in the modal.
        cons_text = consequences
        if not reversible:
            cons_text = (cons_text + "\n\n" if cons_text else "") + \
                "This is NOT reversible."
        if cons_text:
            cons_lbl = tk.Label(
                dlg, text=cons_text,
                justify="left", anchor="w", wraplength=480,
                padx=12,
            )
            cons_lbl.pack(fill="x", pady=(0, 10))

        # Button row at the bottom. Cancel on the right (closer to
        # the operator's natural "X to close" reflex), Proceed to
        # its left. Both width=12 so they align visually.
        btns = tk.Frame(dlg)
        btns.pack(fill="x", padx=12, pady=(6, 10))

        def on_cancel():
            result["ok"] = False
            dlg.destroy()

        def on_proceed():
            result["ok"] = True
            dlg.destroy()

        cancel_btn = tk.Button(btns, text=cancel_text, width=12,
                               command=on_cancel)
        cancel_btn.pack(side="right", padx=4)
        # For danger=high we want the Proceed button visually
        # de-emphasized (not a "default action" colour). Stock Tk
        # doesn't expose easy colour overrides for ttk Buttons on
        # all platforms; use plain tk.Button with explicit fg/bg
        # so the styling is consistent.
        if danger == "high":
            proceed_btn = tk.Button(
                btns, text=proceed_text, width=12,
                fg="#cc2222",
                command=on_proceed,
            )
        else:
            proceed_btn = tk.Button(btns, text=proceed_text, width=12,
                                    command=on_proceed)
        proceed_btn.pack(side="right", padx=4)

        # Default focus on Cancel: pressing Enter on an unfocused
        # dialog cancels rather than confirms. Operator must Tab
        # to Proceed (or click it) to confirm.
        cancel_btn.focus_set()
        # Escape always cancels. WM_DELETE_WINDOW (clicking the
        # window's X) also cancels -- same as the Cancel button.
        dlg.bind("<Escape>", lambda _e: on_cancel())
        dlg.protocol("WM_DELETE_WINDOW", on_cancel)

        # Centre on the parent window. Geometry must be computed
        # after pack so the dialog knows its own size.
        dlg.update_idletasks()
        try:
            px = self.root.winfo_rootx()
            py = self.root.winfo_rooty()
            pw = self.root.winfo_width()
            ph = self.root.winfo_height()
            dw = dlg.winfo_width()
            dh = dlg.winfo_height()
            x = px + max(0, (pw - dw) // 2)
            y = py + max(0, (ph - dh) // 3)
            dlg.geometry(f"+{x}+{y}")
        except tk.TclError:
            pass

        # Block until dismissed.
        self.root.wait_window(dlg)
        return result["ok"]

    def _confirm_destructive_push(self, action: str) -> bool:
        """Pre-flight confirmation for ops that overwrite the
        cluster's sim_selfplay.pt with the local copy. Shows BOTH
        sides' size + mtime side-by-side so the operator can see
        immediately whether they're about to overwrite newer data.

        Process:
          1. Check local file exists -- skip confirm if missing
             (sync.ps1 will refuse with a clearer error).
          2. Single ssh `stat` to query the cluster's mtime + size.
             Uses the existing AskpassFiles plumbing so the
             password prompt is silent.
          3. If cluster file is NEWER than local, the dialog text
             switches to a red-flag warning and the operator must
             click through a stronger message.

        Returns True iff the operator confirmed. The
        cluster-stat step adds ~1 second; well worth it for a
        destructive op (lost 17838's training to this exact
        misclick 2026-05-14).
        """
        local_ckpt = PROJECT_ROOT / "training" / "checkpoints" / "sim_selfplay.pt"
        if not local_ckpt.exists():
            # Don't block the spawn here -- sync.ps1 will refuse
            # with its own clear error. We just skip the confirm.
            return True
        import datetime as _dt
        sz_mb = local_ckpt.stat().st_size / (1024 * 1024)
        local_mtime_ts = local_ckpt.stat().st_mtime
        local_mtime = _dt.datetime.fromtimestamp(
            local_mtime_ts).strftime("%Y-%m-%d %H:%M:%S")
        local_age = _dt.datetime.now() - _dt.datetime.fromtimestamp(
            local_mtime_ts)

        # Query the cluster's mtime + size for comparison. ssh
        # `stat -c '%Y %s' <path>` returns: "<epoch_seconds> <bytes>".
        # On failure (cluster down, file missing, ssh aborts) we
        # fall back to the local-only dialog -- conservative but
        # not a hard blocker.
        cluster_mtime_ts: Optional[float] = None
        cluster_sz_mb: Optional[float] = None
        pw = self._password.get()
        if pw:
            ap = AskpassFiles(pw)
            try:
                creationflags = (subprocess.CREATE_NO_WINDOW
                                 if os.name == "nt" else 0)
                r = subprocess.run(
                    ["ssh", REMOTE_HOST,
                     f"stat -c '%Y %s' "
                     f"{REMOTE_PATH}/training/checkpoints/sim_selfplay.pt "
                     f"2>/dev/null"],
                    capture_output=True, text=True, env=ap.env(),
                    creationflags=creationflags, timeout=15,
                )
                if r.returncode == 0 and r.stdout.strip():
                    parts = r.stdout.strip().split()
                    if len(parts) == 2:
                        cluster_mtime_ts = float(parts[0])
                        cluster_sz_mb = int(parts[1]) / (1024 * 1024)
            except (OSError, subprocess.TimeoutExpired, ValueError):
                pass
            finally:
                ap.cleanup()

        # Build the comparison block.
        if cluster_mtime_ts is None:
            cluster_block = (
                "Cluster's sim_selfplay.pt: (couldn't stat -- "
                "cluster offline or file missing)")
            local_is_older = False
        else:
            cluster_mtime = _dt.datetime.fromtimestamp(
                cluster_mtime_ts).strftime("%Y-%m-%d %H:%M:%S")
            cluster_block = (
                f"Cluster's sim_selfplay.pt:\n"
                f"  {cluster_sz_mb:.1f} MB, mtime {cluster_mtime}")
            # "Local is older" means local mtime < cluster mtime
            # (cluster has newer data). Threshold: 60s slack to
            # account for clock skew + filesystem mtime granularity.
            local_is_older = local_mtime_ts < cluster_mtime_ts - 60

        # Compose the dialog. If local is older, lead with the
        # red-flag warning; otherwise the normal confirm.
        local_age_str = (
            f"{int(local_age.total_seconds() / 86400)}d ago"
            if local_age.total_seconds() > 86400
            else f"{int(local_age.total_seconds() / 3600)}h ago"
        )
        # Route the comparison through the uniform destructive
        # dialog. `danger="high"` is reserved for the older-than-
        # cluster case where the push would destroy newer remote
        # training -- the operator should see a red banner there.
        if local_is_older:
            return self._confirm_destructive(
                title="DESTRUCTIVE: local is OLDER than cluster",
                action=(f"{action}. WARNING: your local checkpoint "
                        f"is OLDER than the cluster's."),
                consequences=(
                    f"Local sim_selfplay.pt to upload:\n"
                    f"  {sz_mb:.1f} MB, mtime {local_mtime} "
                    f"({local_age_str})\n"
                    f"{cluster_block}\n\n"
                    f"This will REPLACE the cluster's newer file "
                    f"with your older local copy. The cluster's "
                    f"training since {local_mtime} would be LOST.\n\n"
                    f"If you actually meant to PULL from the "
                    f"cluster (not push to it), cancel here and "
                    f"click `Pull from cluster` instead."),
                danger="high",
                proceed_text="Push anyway",
            )
        return self._confirm_destructive(
            title="Confirm push",
            action=action,
            consequences=(
                f"Local sim_selfplay.pt to upload:\n"
                f"  {sz_mb:.1f} MB, mtime {local_mtime} "
                f"({local_age_str})\n"
                f"{cluster_block}\n\n"
                f"This OVERWRITES the cluster's sim_selfplay.pt."),
            danger="normal",
            proceed_text="Push",
        )

    def _op_budget(self) -> None:
        # `bash cluster/run.sh budget` prints squeue + today's
        # sacct + 7-day summary + sshare. Best-effort: the cluster
        # may not expose all of these.
        self._spawn(
            ["ssh", REMOTE_HOST,
             f"cd {REMOTE_PATH} && bash cluster/run.sh budget"],
            needs_password=True, label="check cluster budget",
        )

    def _op_start_supervised(self) -> None:
        # `run.sh start supervised` refuses to double-submit (it
        # checks squeue against the recorded jobid before sbatch'ing),
        # so this is a safe no-op when a supervised chain is alive.
        self._spawn(
            ["ssh", REMOTE_HOST,
             f"cd {REMOTE_PATH} && bash cluster/run.sh start supervised"],
            needs_password=True, label="start supervised job",
        )

    def _op_stop_supervised(self) -> None:
        # `run.sh stop supervised` scancels the recorded supervised
        # jobid (best-effort: silent no-op if no job is running).
        # Distinct from the "Cancel running op" button at the
        # bottom of the GUI, which only cancels the LOCAL ssh/scp
        # subprocess driving the current op — it doesn't touch the
        # cluster.
        if not self._confirm_destructive(
                title="Confirm stop",
                action="scancel the supervised cluster job?",
                consequences=(
                    "Any uncheckpointed training in the current iter "
                    "is lost. (Saved checkpoints are kept.) The chain "
                    "is not auto-restarted -- re-launch via Start "
                    "supervised or Sync + Continue (sup.)."),
                proceed_text="Stop job",
        ):
            return
        self._spawn(
            ["ssh", REMOTE_HOST,
             f"cd {REMOTE_PATH} && bash cluster/run.sh stop supervised"],
            needs_password=True, label="stop supervised cluster job",
        )

    def _op_stop_selfplay(self) -> None:
        # Same as above for the self-play job.
        if not self._confirm_destructive(
                title="Confirm stop",
                action="scancel the self-play cluster job?",
                consequences=(
                    "Any uncheckpointed training in the current iter "
                    "is lost (save_every=5, so worst case ~5 iters "
                    "discarded). Saved sim_selfplay.pt is preserved. "
                    "The chain is not auto-restarted -- re-launch via "
                    "Start self-play or Sync + Continue (self-play)."),
                proceed_text="Stop job",
        ):
            return
        self._spawn(
            ["ssh", REMOTE_HOST,
             f"cd {REMOTE_PATH} && bash cluster/run.sh stop selfplay"],
            needs_password=True, label="stop self-play cluster job",
        )

    def _op_start_selfplay(self) -> None:
        # Open the parameter panel; on Run, ssh to the cluster and
        # sbatch with --export overrides for each knob. Idempotent
        # against double-submit (run.sh start checks the recorded
        # jobid against squeue).
        #
        # 2026-05-20: removed the binary "Mini or Ladder?" pop-up
        # that used to fire before this dialog. The dialog now has
        # a full mini/ladder mix slider (in addition to the binary
        # checkbox), so the pre-prompt is obsolete and adds an extra
        # modal click. The dialog's persisted gui_state restores the
        # last-picked values across sessions.
        if self._is_busy():
            messagebox.showinfo("Busy", "Another operation is running.")
            return
        self._open_selfplay_dialog(
            title="Self-play (cluster)",
            on_run=self._run_cluster_selfplay,
            cluster_mode=True,
        )

    def _run_cluster_selfplay(self, params: dict) -> None:
        # Build sbatch --export string. Leaving a value out keeps
        # the sbatch's hardcoded default; passing it overrides.
        export_kvs = [
            "ALL",   # inherit caller's env
            f"ITERATIONS={params['iterations']}",
            f"GAMES_PER_ITER={params['games_per_iter']}",
            f"MAX_TURNS={params['max_turns']}",
            f"WORKERS={params['workers']}",
            f"SAVE_EVERY={params['save_every']}",
            f"SEED={params['seed']}",
        ]
        # Time budget is the primary cluster-job exit criterion;
        # the sbatch defaults to TIME_BUDGET=03:50:00. Forward
        # the dialog value when non-empty so the operator can
        # tighten it for short test runs without re-editing the
        # sbatch.
        if params.get("time_budget"):
            export_kvs.append(f"TIME_BUDGET={params['time_budget']}")
        if params["forced_faction"] == "(none / fully random)":
            export_kvs.append("FORCED_FACTION=none")
        else:
            export_kvs.append(f"FORCED_FACTION={params['forced_faction']}")
        if params["reward_config"]:
            # Strip the project-root prefix so the path is correct
            # cluster-side too. ALWAYS use POSIX separators in the
            # transmitted string -- Path.relative_to(...) returns a
            # WindowsPath here, and `str()` of that yields
            # `cluster\configs\reward_selfplay.json`. The cluster is
            # Linux and can't open backslash-separated paths
            # (FileNotFoundError, observed 2026-05-09 when the first
            # cross-platform self-play submission landed). `.as_posix()`
            # forces forward slashes regardless of the local OS.
            rp = params["reward_config"]
            try:
                rp_path = Path(rp).resolve().relative_to(PROJECT_ROOT)
            except ValueError:
                # absolute / unrelated path: still normalize separators.
                rp_path = Path(rp)
            export_kvs.append(f"REWARD_CONFIG={rp_path.as_posix()}")
        # MCTS knobs. Off by default (REINFORCE training); when
        # checked, the sbatch's USE_MCTS branch flips action
        # selection to AlphaZero-style search. Send the numeric
        # knobs only when MCTS is on -- saves four lines of noise
        # in the [job] params echo when REINFORCE is active.
        if params.get("use_mcts"):
            export_kvs.append("USE_MCTS=1")
            export_kvs.append(f"MCTS_SIMS={params['mcts_sims']}")
            export_kvs.append(f"MCTS_C_PUCT={params['mcts_c_puct']}")
            export_kvs.append(f"MCTS_BATCH_SIZE={params['mcts_batch_size']}")
        if params.get("mini_maps"):
            export_kvs.append("MINI_MAPS=1")
        # Mini/ladder mix ratio. Sent only when non-zero so the
        # [job] params echo stays clean for pure-ladder runs.
        # Ignored cluster-side when MINI_MAPS=1 (already 100% mini).
        mr = float(params.get("mini_ratio") or 0.0)
        if mr > 0.0:
            export_kvs.append(f"MINI_RATIO={mr}")
        # --export uses commas to separate KEY=VAL pairs. Quote the
        # whole thing so shell metacharacters in values don't bite.
        export_arg = ",".join(export_kvs)
        # Pre-flight double-submit guard: check squeue cluster-side
        # for an existing self-play job before we sbatch. `run.sh
        # start` does the same check but the GUI bypasses it (we
        # need --export). Doing the check inside the same ssh
        # command keeps it atomic from the operator's view -- two
        # rapid clicks both see "active job exists" rather than
        # the second one slipping through before squeue catches up.
        # The grep is `head -1` because `squeue --me` lists ALL
        # active jobs; we just need to know "any wai-selfplay
        # alive?".
        remote_cmd = (
            f"cd {REMOTE_PATH} && "
            # Check for an active self-play job. squeue's `--me`
            # filters to this user; `-h` (--noheader) drops the
            # title row; `-o %j` prints only job-names; grep for
            # the sbatch's --job-name (wai-selfplay). If any
            # match, abort with a clear message.
            f"if squeue --me -h -o '%j' 2>/dev/null | "
            f"grep -q '^wai-selfplay$'; then "
            f"echo '[start-selfplay] already a self-play job in queue; refusing to double-submit. Use Stop self-play first, or wait for it to finish.' >&2; "
            f"exit 1; "
            f"fi && "
            f"sbatch --export='{export_arg}' --parsable "
            f"cluster/job_selfplay.sbatch | tee training/logs/selfplay.jobid"
        )
        self._spawn(
            ["ssh", REMOTE_HOST, remote_cmd],
            needs_password=True,
            label=f"start self-play (cluster, {params['iterations']} iters)",
        )

    def _op_pull(self) -> None:
        # Modes (PowerShell side: cluster/pull_checkpoint.ps1):
        #   empty / 'selfplay' / 'sim' -> default: sim_selfplay.pt
        #   'supervised' / 'sup'       -> highest supervised_epoch*.pt
        #   'rolling'                  -> supervised.pt (mid-epoch)
        #   'archive'                  -> freshest sim_selfplay_archive_*.pt
        #   'archive:<YYYYMMDD-HHMMSS>'-> specific archive snapshot
        #   N (integer)                -> supervised_epoch<N>.pt
        #   'list'                     -> list cluster's checkpoints,
        #                                 don't download anything
        #
        # The empty-default flip (2026-05-13) follows the workflow
        # change: self-play is the primary training, so pulling its
        # rolling target is what the operator wants 95% of the time.
        # Previously empty mapped to "highest supervised_epoch",
        # which is now accessible via 'supervised'.
        spec = self._epoch_var.get().strip().lower()
        args: List[str] = []
        if spec in ("", "selfplay", "sim", "self-play"):
            pass                       # default = sim_selfplay.pt
        elif spec in ("supervised", "sup"):
            args = ["-Supervised"]
        elif spec == "rolling":
            args = ["-Rolling"]
        elif spec == "archive":
            args = ["-Archive"]
        elif spec.startswith("archive:"):
            stamp = spec.split(":", 1)[1].strip()
            if not stamp:
                messagebox.showerror(
                    "Bad mode",
                    "`archive:` needs a timestamp after the "
                    "colon, e.g. archive:20260513-180822 "
                    "(use `list` mode to see what's available).")
                return
            args = ["-Archive", "-ArchiveStamp", stamp]
        elif spec == "list":
            args = ["-List"]
        else:
            try:
                n = int(spec)
            except ValueError:
                messagebox.showerror(
                    "Bad mode",
                    f"`{spec}` is not a valid pull mode. Use empty "
                    f"(self-play), `supervised`, `rolling`, "
                    f"`archive` (freshest snapshot), "
                    f"`archive:<stamp>` (specific snapshot), a "
                    f"number (specific supervised epoch), or `list`.")
                return
            args = ["-Epoch", str(n)]
        self._spawn(self._ps(SCRIPT_PULL, *args),
                    needs_password=True,
                    label=f"pull checkpoint ({spec or 'self-play'})")

    def _op_pull_bundle(self) -> None:
        """Atomic 'Pull from cluster': self-play checkpoint + freshest
        SLURM log + freshest trainer_history CSV in a single resolve-
        and-scp round-trip. One password prompt for the whole bundle.

        Why this exists: the Daily workflow always wants ALL THREE
        of (model, log, history) together -- pulling them separately
        was three button clicks, three password prompts, and an easy
        way to forget one (e.g. pulling the model but not the log
        means you're inspecting a model with no idea what reward
        shape produced it). The redesign collapses the trio into
        one click.

        Falls back gracefully on per-artifact misses: if the cluster
        hasn't produced a log or trainer_history yet (job just
        started), pulls whatever exists and logs the absences. Only
        a fully empty cluster (no checkpoint either) aborts the op.

        Specialised pulls (specific archive, supervised epoch,
        rolling) keep the legacy `Pull` button -- they're rare
        enough that conflating them into the atomic op would
        clutter the common path. Profile artifacts stay manual
        too; they're rarer still.

        Implementation: one inline ssh resolves the remote paths
        (synchronous, ~1s), then one scp pulls every source into a
        staging dir. The on_complete callback moves each file from
        staging to its proper local destination (checkpoint to
        training/checkpoints/, log + history to training/logs/).
        scp with multiple sources requires a single destination
        dir, hence the staging step.
        """
        if self._is_busy():
            messagebox.showinfo("Busy", "Another operation is running.")
            return
        pw = self._password.get()
        if not pw:
            messagebox.showwarning(
                "No password", "Cluster ops need an ssh password.")
            return
        # Single ssh: emits up to three lines tagged CKPT/LOG/HIST
        # with the remote path, or *_MISSING for absences. `ls -t`
        # picks freshest by mtime; we don't care about jobid
        # correlation here -- the operator inspects whatever the
        # latest run produced.
        remote_resolve = (
            f"cd {REMOTE_PATH} && "
            f"CKPT=training/checkpoints/sim_selfplay.pt; "
            f"if [ -f \"$CKPT\" ]; then echo \"CKPT $CKPT\"; "
            f"else echo CKPT_MISSING; fi; "
            f"LOG=$(ls -t training/logs/selfplay-slurm-*.log "
            f"2>/dev/null | head -n1); "
            f"if [ -n \"$LOG\" ]; then echo \"LOG $LOG\"; "
            f"else echo LOG_MISSING; fi; "
            f"HIST=$(ls -t training/logs/trainer_history_*.csv "
            f"2>/dev/null | head -n1); "
            f"if [ -n \"$HIST\" ]; then echo \"HIST $HIST\"; "
            f"else echo HIST_MISSING; fi"
        )
        askpass = AskpassFiles(pw)
        try:
            creationflags = (subprocess.CREATE_NO_WINDOW
                             if os.name == "nt" else 0)
            res = subprocess.run(
                ["ssh", REMOTE_HOST, remote_resolve],
                capture_output=True, text=True, env=askpass.env(),
                creationflags=creationflags, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            askpass.cleanup()
            self._log(f"[pull-bundle] ssh resolve failed: {e}")
            return
        finally:
            askpass.cleanup()
        if res.returncode != 0:
            self._log(f"[pull-bundle] resolve exit "
                      f"{res.returncode}: {res.stderr.strip()}")
            return
        # Parse the tagged output. Each `kind` maps to:
        #   remote_rel: path relative to REMOTE_PATH (e.g.
        #               training/checkpoints/sim_selfplay.pt)
        #   final_dir:  local destination directory after staging
        ckpt_dir = PROJECT_ROOT / "training" / "checkpoints"
        log_dir  = PROJECT_ROOT / "training" / "logs"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        plan: List[Tuple[str, str, Path]] = []   # (kind, remote_rel, final_dir)
        missing: List[str] = []
        for line in res.stdout.splitlines():
            ln = line.strip()
            if ln == "CKPT_MISSING":
                missing.append("sim_selfplay.pt")
            elif ln == "LOG_MISSING":
                missing.append("selfplay-slurm-*.log")
            elif ln == "HIST_MISSING":
                missing.append("trainer_history_*.csv")
            elif ln.startswith("CKPT "):
                plan.append(("CKPT", ln[5:], ckpt_dir))
            elif ln.startswith("LOG "):
                plan.append(("LOG",  ln[4:], log_dir))
            elif ln.startswith("HIST "):
                plan.append(("HIST", ln[5:], log_dir))
        for m in missing:
            self._log(f"[pull-bundle] cluster has no {m} (skipping)")
        if not plan:
            self._log(
                "[pull-bundle] nothing to pull -- cluster has no "
                "self-play artifacts yet. Submit a self-play job "
                "first, or use legacy `Pull` to fetch a supervised "
                "checkpoint.")
            return
        # Staging dir: one scp landing zone for all sources, since
        # scp requires a single destination. Created fresh each
        # call; cleaned up in the on_complete callback.
        staging = PROJECT_ROOT / "training" / ".pull_bundle_staging"
        # If a prior run crashed mid-pull, scrub the leftovers --
        # otherwise stale files in staging could shadow the fresh
        # ones. rmtree is idempotent against missing.
        if staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError as e:
                self._log(f"[pull-bundle] couldn't clean stale "
                          f"staging dir {staging}: {e}")
                return
        staging.mkdir(parents=True, exist_ok=True)
        # Build the scp command. Multiple `host:path` sources land
        # into one local dir; OpenSSH's scp handles this natively.
        sources = [f"{REMOTE_HOST}:{REMOTE_PATH}/{rel}"
                   for _, rel, _ in plan]
        self._log(f"[pull-bundle] pulling {len(plan)} file(s):")
        for kind, rel, final in plan:
            base = Path(rel).name
            self._log(
                f"           [{kind}] {rel} -> "
                f"{(final / base).relative_to(PROJECT_ROOT)}")

        def finalize(_lines: List[str]) -> None:
            # Move each file from staging to its final destination.
            # Logged individually so any failure (e.g. permission,
            # zero-byte scp) is visible. Missing files in staging
            # mean the scp itself failed for that source -- scp
            # already wrote the error to the output panel.
            #
            # We also track whether the checkpoint specifically
            # landed: this gates the post-pull auto-eval, which
            # only makes sense when there's actually a fresh
            # sim_selfplay.pt to evaluate. A pull that only
            # captured the log + history (e.g. cluster job hasn't
            # produced a checkpoint yet) shouldn't trigger eval
            # on whatever stale local model exists.
            ckpt_landed = False
            # We also extract the cluster jid from the LOG basename
            # (selfplay-slurm-<jid>.log). Used by the stale-push
            # guard to record "we are now up-to-date with this
            # jid" -- the next time the auto-pull watcher sees a
            # different jid running, the block re-arms.
            pulled_jid: Optional[str] = None
            for kind, rel, final in plan:
                base = Path(rel).name
                src = staging / base
                dst = final / base
                if not src.exists():
                    self._log(
                        f"[pull-bundle] {kind} {base}: not in "
                        f"staging (scp likely failed for this file)")
                    continue
                try:
                    # shutil.move is atomic within a filesystem;
                    # we're staying inside the project tree so the
                    # move is just a rename.
                    if dst.exists():
                        dst.unlink()
                    shutil.move(str(src), str(dst))
                    sz_mb = dst.stat().st_size / (1024 * 1024)
                    self._log(
                        f"[pull-bundle] {kind} -> "
                        f"{dst.relative_to(PROJECT_ROOT)} "
                        f"({sz_mb:.1f} MB)")
                    if kind == "CKPT":
                        ckpt_landed = True
                    if kind == "LOG":
                        # selfplay-slurm-<jid>.log -- the middle
                        # field is the SLURM jobid. Defensive
                        # parse: any anomaly (renamed file, no
                        # jid digits) just leaves pulled_jid
                        # None, which means the stale-push
                        # guard stays on the last-known jid.
                        import re as _re
                        mjid = _re.match(
                            r"selfplay-slurm-(\d+)\.log",
                            base)
                        if mjid:
                            pulled_jid = mjid.group(1)
                except OSError as e:
                    self._log(
                        f"[pull-bundle] {kind} move failed: {e}")
            try:
                shutil.rmtree(staging)
            except OSError:
                pass
            self._log("[pull-bundle] done.")
            # Stale-push guard: we now have whatever the cluster
            # produced. Record the jid + clear the block so Push
            # re-enables. Only when CKPT actually landed -- a
            # log-only pull doesn't bring us up to date model-wise.
            if ckpt_landed:
                self._record_pulled_jid(pulled_jid)
                self._set_push_blocked(None)
            # Post-pull auto-eval. Only when:
            #   * a fresh checkpoint actually landed (no point
            #     eval-ing a model the pull didn't update);
            #   * the operator hasn't opted out of the
            #     automation (toggle in the Daily tab);
            #   * we're not blocked by another op.
            # `root.after(50, ...)` lets _end_op fully unwind --
            # the callback runs inside _end_op AFTER the busy
            # lock is released, but the next _spawn would
            # immediately reacquire it; tiny defer keeps the log
            # ordering "[pull-bundle] done." -> "[auto-eval]..."
            # readable in the output panel.
            if ckpt_landed and self._auto_eval_after_pull.get():
                if self._is_busy():
                    self._log(
                        "[auto-eval] new checkpoint landed but "
                        "another op is in flight; skipping auto-"
                        "eval. Click `Daily eval` manually when "
                        "free.")
                else:
                    self._log(
                        "[auto-eval] new sim_selfplay.pt landed; "
                        "kicking off Daily eval (quick preset).")
                    self.root.after(50, self._op_eval_daily)

        self._spawn(
            ["scp"] + sources + [str(staging) + os.sep],
            needs_password=True,
            label=f"pull bundle ({len(plan)} file"
                  f"{'s' if len(plan) != 1 else ''})",
            on_complete=finalize,
        )

    def _op_push_checkpoint(self) -> None:
        """Push the local sim_selfplay.pt to the cluster, no code
        sync, no chain-link restart. Wraps `sync.ps1 -IncludeCheckpoint`
        with no `-Continue` flag.

        Why this exists separately from `Sync + Continue (self-
        play)`: sometimes you want to land the checkpoint on the
        cluster without immediately starting a job (e.g., a cluster
        job is already running and you want the NEXT chain link
        to pick up local progress without disrupting the current
        run). Distinct mental model: "push my work" vs "push and
        restart."

        Overwrites the cluster's sim_selfplay.pt by design -- same
        contract as Sync + Continue (self-play). Refuses if local
        sim_selfplay.pt is missing.
        """
        local_ckpt = PROJECT_ROOT / "training" / "checkpoints" / "sim_selfplay.pt"
        if not local_ckpt.exists():
            messagebox.showwarning(
                "No local checkpoint",
                f"{local_ckpt.relative_to(PROJECT_ROOT)} doesn't "
                f"exist locally. Run local self-play training "
                f"(Train (self-play) button) first.")
            return
        if not self._confirm_destructive_push(
                action="Push local sim_selfplay.pt to the cluster "
                       "(also re-syncs code; no cluster job is "
                       "started or stopped by this op)",
        ):
            return
        # Surface the size + mtime in the GUI log so the operator
        # confirms it's the right file before the cluster's gets
        # overwritten. The sync.ps1 also logs this, but we want
        # the info visible BEFORE the password prompt fires.
        sz_mb = local_ckpt.stat().st_size / (1024 * 1024)
        import datetime as _dt
        mtime = _dt.datetime.fromtimestamp(
            local_ckpt.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        self._log(f"[push-ckpt] uploading {local_ckpt.name} "
                  f"({sz_mb:.1f} MB, mtime {mtime}) -- will "
                  f"overwrite the cluster's copy")
        self._spawn(self._ps(SCRIPT_SYNC, "-IncludeCheckpoint"),
                    needs_password=True,
                    label="push local sim_selfplay.pt + code to cluster")

    def _op_restore_from_archive(self) -> None:
        """Restore the cluster's `sim_selfplay.pt` from one of its
        tier-retained archives (`sim_selfplay_archive_<stamp>.pt`).

        Flow:
          1. Synchronous ssh `ls` to list archives on the cluster.
          2. Modal dialog showing the list (newest first); operator
             picks one.
          3. Confirm dialog (this overwrites the cluster's current
             sim_selfplay.pt -- destructive in the opposite
             direction from Push: instead of overwriting cluster
             with local, we're overwriting cluster's current with
             cluster's earlier).
          4. ssh `cp <archive> sim_selfplay.pt` on the cluster.
             No network transfer -- it's a cluster-local copy.

        Use case: an operator-mistake push (Sync + Continue with a
        stale local checkpoint) erased the cluster's good
        training state. As long as cleanup.py had archived the
        good state before the bad push landed, the archive is
        still there to restore from.
        """
        if self._is_busy():
            messagebox.showinfo("Busy", "Another operation is running.")
            return
        pw = self._password.get()
        if not pw:
            messagebox.showwarning(
                "No password", "Cluster ops need an ssh password.")
            return
        # Synchronous list of archives (one ssh trip).
        ap = AskpassFiles(pw)
        try:
            creationflags = (subprocess.CREATE_NO_WINDOW
                             if os.name == "nt" else 0)
            r = subprocess.run(
                ["ssh", REMOTE_HOST,
                 f"cd {REMOTE_PATH}/training/checkpoints && "
                 f"ls -lt --time-style='+%Y-%m-%d %H:%M' "
                 f"sim_selfplay_archive_*.pt 2>/dev/null"],
                capture_output=True, text=True, env=ap.env(),
                creationflags=creationflags, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            ap.cleanup()
            self._log(f"[restore] ssh list failed: {e}")
            return
        finally:
            ap.cleanup()
        if r.returncode != 0:
            self._log(f"[restore] ls exit {r.returncode}: "
                      f"{r.stderr.strip()}")
            return
        lines = [ln for ln in r.stdout.strip().splitlines() if ln]
        # Parse each `ls -l` line. Format we asked for:
        #   -rw-r--r-- 1 user grp 6388234 2026-05-13 12:38
        #     sim_selfplay_archive_20260513-123806.pt
        # Extract: size bytes, mtime "YYYY-MM-DD HH:MM", filename.
        entries: List[Tuple[str, str, str]] = []
        # (filename, size_str, mtime_str)
        for ln in lines:
            parts = ln.split(None, 7)
            if len(parts) < 8:
                continue
            size_b = parts[4]
            mtime  = f"{parts[5]} {parts[6]}"
            fname  = parts[7]
            try:
                size_mb = int(size_b) / (1024 * 1024)
            except ValueError:
                continue
            entries.append((fname, f"{size_mb:.1f} MB", mtime))
        if not entries:
            messagebox.showinfo(
                "No archives",
                "No sim_selfplay_archive_*.pt on the cluster. The "
                "cleanup.py archive policy may have pruned them "
                "all, or no self-play job has run with cleanup "
                "enabled yet.")
            return

        # Modal picker dialog. Listbox with mtime + size + name.
        dlg = tk.Toplevel(self.root)
        dlg.title("Restore from archive")
        dlg.transient(self.root)
        dlg.grab_set()
        tk.Label(dlg, text="Pick an archive to restore. The cluster's "
                           "current sim_selfplay.pt will be OVERWRITTEN "
                           "with the chosen archive.",
                 wraplength=520, justify="left", fg="gray").pack(
                     padx=10, pady=(8, 4), anchor="w")
        lst = tk.Listbox(dlg, width=70, height=min(12, len(entries)),
                         font=("Consolas", 9))
        for fname, sz, mt in entries:
            # Strip the path prefix; filenames are bare.
            lst.insert("end", f"{mt}  {sz:>10}  {fname}")
        lst.pack(padx=10, pady=4, fill="both", expand=True)
        lst.selection_set(0)         # default = newest

        def on_restore():
            sel = lst.curselection()
            if not sel:
                messagebox.showwarning("No selection",
                                       "Pick an archive first.")
                return
            fname = entries[sel[0]][0]
            mtime = entries[sel[0]][2]
            dlg.destroy()
            if not self._confirm_destructive(
                    title="Confirm restore",
                    action=(f"Restore cluster's sim_selfplay.pt "
                            f"from {fname}?"),
                    consequences=(
                        f"  mtime: {mtime}\n\n"
                        f"This OVERWRITES the cluster's current "
                        f"sim_selfplay.pt. The archives themselves "
                        f"aren't deleted -- you can revert this by "
                        f"restoring a different one."),
                    proceed_text="Restore",
            ):
                return
            # ssh + cp on the cluster (no network transfer).
            remote_cmd = (
                f"cd {REMOTE_PATH}/training/checkpoints && "
                f"cp -f '{fname}' sim_selfplay.pt && "
                f"echo '[restore] cluster sim_selfplay.pt = {fname}'"
            )
            self._spawn(
                ["ssh", REMOTE_HOST, remote_cmd],
                needs_password=True,
                label=f"restore cluster from {fname}",
            )

        btns = tk.Frame(dlg)
        btns.pack(fill="x", padx=10, pady=(4, 10))
        tk.Button(btns, text="Cancel",
                  command=dlg.destroy).pack(side="right", padx=4)
        tk.Button(btns, text="Restore",
                  command=on_restore).pack(side="right", padx=4)

    def _op_pull_logs(self) -> None:
        """Pull the most recent SLURM log from the cluster.

        Two modes via small picker:
          * `selfplay-slurm-*.log` (the default; what 95% of clicks
            want now that self-play is the primary workflow)
          * `slurm-*.log` (supervised job logs; useful when
            supervised is the active path)

        Cluster-side path: `training/logs/<prefix>-<jobid>.log`,
        one per job. We pick the freshest by mtime and scp it back
        to local `training/logs/`, preserving the filename so the
        operator can correlate by jobid.

        Single ssh round-trip resolves the path, then scp transfers
        the file. Same proxy chain (relais -> istanbul) as the other
        cluster ops; askpass plumbing handles the prompts silently.
        """
        # Pick which log stream: selfplay (default) or supervised.
        # Modal yes/no/cancel-style dialog with three buttons would
        # be cleaner UX but Tk's stock doesn't ship one and writing
        # a custom one for a once-in-a-while click is overkill.
        # askyesnocancel: Yes=selfplay (the default), No=supervised,
        # Cancel=abort.
        choice = messagebox.askyesnocancel(
            "Pull which log?",
            "Pull the freshest cluster SLURM log:\n\n"
            "  Yes  -> self-play (selfplay-slurm-<jobid>.log)\n"
            "  No   -> supervised (slurm-<jobid>.log)\n"
            "  Cancel -> abort",
        )
        if choice is None:
            return
        if choice:
            log_glob = "selfplay-slurm-*.log"
            log_label = "self-play"
        else:
            log_glob = "slurm-*.log"
            log_label = "supervised"
        # Quote both halves of the pipeline so the remote shell sees
        # one command. `ls -t` sorts by mtime (newest first), `head -n1`
        # picks the freshest. -- prevents glob expansion on the local
        # side; the wildcard is interpreted remotely.
        remote_resolve = (
            f"cd {REMOTE_PATH} && "
            f"ls -t training/logs/{log_glob} 2>/dev/null | "
            f"head -n 1"
        )
        # Two-stage: resolve the remote filename, then scp. We do the
        # whole thing in one spawned op (a small bash wrapper) so the
        # operator only types the password once for the whole pull.
        # `set -e` fails the wrapper if `head -n 1` produced nothing
        # (no log files); we propagate that as an obvious error
        # rather than a silent zero-byte scp.
        log_dir_local = PROJECT_ROOT / "training" / "logs"
        log_dir_local.mkdir(parents=True, exist_ok=True)
        # The remote `head -n 1` pipe always exits 0 even on empty
        # input (head reads nothing, returns 0). Guard with `[ -n
        # "$LATEST" ]` so an empty result trips an explicit error
        # instead of an empty scp source argument.
        bash_wrapper = (
            f'LATEST=$({remote_resolve}); '
            f'if [ -z "$LATEST" ]; then '
            f'echo "[pull-logs] no {log_glob} on cluster" >&2; '
            f'exit 2; fi; '
            f'echo "[pull-logs] freshest: $LATEST"; '
            # Print path so the local side learns the basename for scp.
            f'basename "$LATEST"'
        )
        # We need to know the basename BEFORE scp, so do the resolve
        # in a synchronous step (one ssh) and parse stdout. Using
        # subprocess directly here (rather than self._spawn) so we
        # block briefly for the resolve + then chain the scp through
        # the normal _spawn machinery for live output streaming.
        pw = self._password.get()
        if not pw:
            messagebox.showwarning(
                "No password", "Cluster ops need an ssh password.")
            return
        if self._is_busy():
            messagebox.showinfo("Busy", "Another operation is running.")
            return
        # Synchronous resolve (small env, hidden console).
        askpass = AskpassFiles(pw)
        try:
            creationflags = (subprocess.CREATE_NO_WINDOW
                             if os.name == "nt" else 0)
            res = subprocess.run(
                ["ssh", REMOTE_HOST, bash_wrapper],
                capture_output=True, text=True, env=askpass.env(),
                creationflags=creationflags, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            askpass.cleanup()
            self._log(f"[pull-logs] ssh resolve failed: {e}")
            return
        finally:
            askpass.cleanup()
        if res.returncode != 0:
            self._log(f"[pull-logs] resolve exit {res.returncode}: "
                      f"{res.stderr.strip()}")
            return
        # The wrapper prints two lines: "[pull-logs] freshest: ..."
        # and then the basename. Last non-empty line is the basename.
        lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
        if not lines:
            self._log("[pull-logs] empty resolve output")
            return
        basename = lines[-1]
        for prefix_line in lines[:-1]:
            self._log(prefix_line)
        remote_path = f"{REMOTE_PATH}/training/logs/{basename}"
        local_path = log_dir_local / basename
        self._log(f"[pull-logs] scp {REMOTE_HOST}:{remote_path} -> "
                  f"{local_path.relative_to(PROJECT_ROOT)}")
        # The actual transfer goes through the regular streaming
        # spawn so progress (and any auth errors) land in the output
        # panel. scp through this OpenSSH binary picks up
        # SSH_ASKPASS automatically.
        self._spawn(
            ["scp", f"{REMOTE_HOST}:{remote_path}", str(local_path)],
            needs_password=True, label=f"pull {log_label} log {basename}",
        )

    def _op_profile_cluster(self) -> None:
        """Open a small dialog, then submit `cluster/job_profile.sbatch`
        to the cluster. Same `--export=KEY=VAL,...` pattern as the
        self-play submission, so each knob in the dialog maps cleanly
        to an env-var override the sbatch script reads.

        Idempotent against double-submit at SLURM level: profile and
        self-play are different job names, so a profile job queues
        independently of a running self-play job.
        """
        if self._is_busy():
            messagebox.showinfo("Busy", "Another operation is running.")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Profile (cluster)")
        dlg.transient(self.root)
        dlg.grab_set()
        # Persisted defaults via the same gui_state.json the other
        # dialogs use, so last-picked knobs survive a GUI restart.
        persisted = _load_gui_state().get("profile_cluster", {})

        v_iters    = tk.IntVar(value=int(persisted.get("iterations", 3)))
        v_warmup   = tk.IntVar(value=int(persisted.get("warmup", 1)))
        v_games    = tk.IntVar(value=int(persisted.get("games_per_iter", 8)))
        v_turns    = tk.IntVar(value=int(persisted.get("max_turns", 60)))
        v_workers  = tk.IntVar(value=int(persisted.get("workers", 6)))
        v_torchp   = tk.BooleanVar(value=bool(persisted.get("torch_profile", False)))
        v_mcts     = tk.BooleanVar(value=bool(persisted.get("use_mcts", False)))
        v_msims    = tk.IntVar(value=int(persisted.get("mcts_sims", 50)))
        v_smiv     = tk.DoubleVar(value=float(persisted.get(
            "nvidia_smi_interval", 1.0)))

        def grid(label: str, widget, row: int, hint: str = ""):
            tk.Label(dlg, text=label, anchor="e").grid(
                row=row, column=0, padx=6, pady=3, sticky="e")
            widget.grid(row=row, column=1, padx=6, pady=3, sticky="w")
            if hint:
                tk.Label(dlg, text=hint, fg="gray").grid(
                    row=row, column=2, padx=4, pady=3, sticky="w")

        grid("Iterations:",
             tk.Spinbox(dlg, from_=1, to=20, textvariable=v_iters,
                        width=6),
             0, "profiled iters (3 = good signal vs walltime)")
        grid("Warmup:",
             tk.Spinbox(dlg, from_=0, to=5, textvariable=v_warmup,
                        width=6),
             1, "absorbs JIT / vocab build before measurement")
        grid("Games per iter:",
             tk.Spinbox(dlg, from_=1, to=32, textvariable=v_games,
                        width=6),
             2, "match production self-play setting")
        grid("Max turns:",
             tk.Spinbox(dlg, from_=20, to=200, textvariable=v_turns,
                        width=6),
             3, "60 is plenty to exercise mid-game code paths")
        grid("Workers:",
             tk.Spinbox(dlg, from_=0, to=12, textvariable=v_workers,
                        width=6),
             4, "rollout threads; match cluster sbatch (6)")
        grid("nvidia-smi sec:",
             tk.Spinbox(dlg, from_=0.1, to=5.0, increment=0.1,
                        textvariable=v_smiv, width=6),
             5, "GPU sample interval (0 disables)")
        # Flag checkboxes. torch.profiler is expensive (~30-50%
        # overhead and a multi-MB chrome-trace); off by default.
        tk.Checkbutton(dlg, text="Use torch.profiler",
                       variable=v_torchp).grid(
                           row=6, column=0, columnspan=2,
                           padx=6, pady=3, sticky="w")
        tk.Label(dlg, text="(expensive, opt-in)", fg="gray").grid(
            row=6, column=2, padx=4, pady=3, sticky="w")
        # MCTS toggle + sims. Off by default; when on, profile the
        # AlphaZero-style workload which looks completely different
        # (N_sim forwards per move).
        mcts_frame = tk.Frame(dlg)
        tk.Checkbutton(mcts_frame, text="MCTS mode",
                       variable=v_mcts).pack(side="left")
        tk.Label(mcts_frame, text="  sims:").pack(side="left")
        tk.Spinbox(mcts_frame, from_=1, to=500, textvariable=v_msims,
                   width=5).pack(side="left", padx=2)
        mcts_frame.grid(row=7, column=0, columnspan=2,
                        padx=6, pady=3, sticky="w")

        def on_ok():
            try:
                params = {
                    "iterations":      v_iters.get(),
                    "warmup":          v_warmup.get(),
                    "games_per_iter":  v_games.get(),
                    "max_turns":       v_turns.get(),
                    "workers":         v_workers.get(),
                    "torch_profile":   bool(v_torchp.get()),
                    "use_mcts":        bool(v_mcts.get()),
                    "mcts_sims":       v_msims.get(),
                    "nvidia_smi_interval": float(v_smiv.get()),
                }
            except (tk.TclError, ValueError) as e:
                messagebox.showerror(
                    "Invalid value",
                    f"One or more fields has an invalid or empty "
                    f"value:\n\n{e}\n\nFix the field(s) and click "
                    f"Run again.")
                return
            state = _load_gui_state()
            state["profile_cluster"] = params
            _save_gui_state(state)
            dlg.destroy()
            self._run_cluster_profile(params)

        btns = tk.Frame(dlg)
        btns.grid(row=20, column=0, columnspan=3, pady=10)
        tk.Button(btns, text="Cancel", width=10,
                  command=dlg.destroy).pack(side="right", padx=4)
        tk.Button(btns, text="Run", width=10,
                  command=on_ok).pack(side="right", padx=4)

    def _run_cluster_profile(self, params: dict) -> None:
        """Submit `cluster/job_profile.sbatch` with the dialog's
        knobs as `--export` env-var overrides. Mirrors the self-
        play submission's pattern; same caveats about quoting
        space-containing values (none here) and recording the
        jobid for later `Pull profile`."""
        export_kvs = [
            "ALL",
            f"ITERATIONS={params['iterations']}",
            f"WARMUP_ITERS={params['warmup']}",
            f"GAMES_PER_ITER={params['games_per_iter']}",
            f"MAX_TURNS={params['max_turns']}",
            f"WORKERS={params['workers']}",
            f"NVIDIA_SMI_INTERVAL={params['nvidia_smi_interval']}",
            f"USE_TORCH_PROFILE={1 if params['torch_profile'] else 0}",
        ]
        if params["use_mcts"]:
            export_kvs.append("USE_MCTS=1")
            export_kvs.append(f"MCTS_SIMS={params['mcts_sims']}")
        export_arg = ",".join(export_kvs)
        # Stash the jobid in a file so "Pull profile" can later
        # resolve the right artifact directory without a second
        # squeue round-trip. `--parsable` prints just the jobid;
        # tee captures it both for the log AND for the local
        # `selfplay-style` jobid file pattern.
        remote_cmd = (
            f"cd {REMOTE_PATH} && "
            f"sbatch --export='{export_arg}' --parsable "
            f"cluster/job_profile.sbatch | "
            f"tee training/logs/profile.jobid"
        )
        self._spawn(
            ["ssh", REMOTE_HOST, remote_cmd],
            needs_password=True,
            label=f"submit profile job ({params['iterations']} iters)",
        )

    def _op_pull_profile(self) -> None:
        """scp the freshest cluster profile artifact dir to local.

        Two-stage like `_op_pull_logs`: resolve the freshest
        `training/profiles/*/` on the cluster (one ssh), then scp
        the directory recursively (one scp). Both use the same
        askpass plumbing so the operator types the password once
        per GUI session.

        Profile dirs are named by SLURM jobid (`training/profiles/
        <jobid>/`). Freshest = highest mtime, which on the cluster
        also means most recently submitted/completed.
        """
        if self._is_busy():
            messagebox.showinfo("Busy", "Another operation is running.")
            return
        pw = self._password.get()
        if not pw:
            messagebox.showwarning(
                "No password", "Cluster ops need an ssh password.")
            return
        # Resolve the freshest cluster profile dir. `ls -td */`
        # lists subdirs newest-first; the basename is the jobid.
        # We accept either training/profiles/<jobid>/ or any
        # other timestamped subdir layout the user creates.
        remote_resolve = (
            f"cd {REMOTE_PATH}/training/profiles 2>/dev/null && "
            f"ls -td */ 2>/dev/null | head -n 1 | "
            f"sed 's|/$||'"
        )
        askpass = AskpassFiles(pw)
        try:
            creationflags = (subprocess.CREATE_NO_WINDOW
                             if os.name == "nt" else 0)
            res = subprocess.run(
                ["ssh", REMOTE_HOST, remote_resolve],
                capture_output=True, text=True, env=askpass.env(),
                creationflags=creationflags, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            askpass.cleanup()
            self._log(f"[pull-profile] ssh resolve failed: {e}")
            return
        finally:
            askpass.cleanup()
        if res.returncode != 0:
            self._log(f"[pull-profile] resolve exit {res.returncode}: "
                      f"{res.stderr.strip()}")
            return
        jobid = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
        if not jobid:
            self._log("[pull-profile] no profile dirs on cluster "
                      "(submit a profile job first)")
            return
        remote_path = f"{REMOTE_PATH}/training/profiles/{jobid}"
        local_dir = PROJECT_ROOT / "training" / "profiles"
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / jobid
        # If the local dir already exists, scp -r would nest a
        # second copy inside it (Linux scp behavior). Wipe it
        # first so the pull is idempotent.
        if local_path.exists():
            import shutil as _shutil
            _shutil.rmtree(local_path, ignore_errors=True)
        self._log(f"[pull-profile] scp -r {REMOTE_HOST}:{remote_path} -> "
                  f"{local_path.relative_to(PROJECT_ROOT)}")
        # scp -r pulls the whole directory in one transfer; the
        # local target is the parent dir so scp creates <jobid>/
        # underneath it.
        self._spawn(
            ["scp", "-r",
             f"{REMOTE_HOST}:{remote_path}",
             str(local_dir)],
            needs_password=True, label=f"pull profile {jobid}",
        )

    def _op_view_profile(self) -> None:
        """Open the freshest local profile DIRECTORY in the system
        file browser (Explorer on Windows). The directory holds
        multiple artifacts the operator may want:
          summary.txt              -- the headline read
          cprofile.prof            -- open in snakeviz
          cprofile_cumulative.txt  -- top-N by cumtime
          cprofile_self.txt        -- top-N by tottime
          nvidia_smi.csv           -- GPU timeline
          torch_trace.json         -- (if --torch-profile)
          phase_timing.csv         -- rollout vs train per iter
        Opening the directory lets the operator click into whichever
        one they want -- summary.txt is just the most common entry
        point. Previously this opened summary.txt directly, which
        forced an extra File Explorer round-trip if the operator
        wanted any of the other artifacts."""
        profiles_dir = PROJECT_ROOT / "training" / "profiles"
        if not profiles_dir.is_dir():
            messagebox.showinfo(
                "No profiles",
                "No profile artifacts under training/profiles/. "
                "Submit a profile job + 'Pull profile' first.")
            return
        # Pick the freshest subdir by mtime. local_<ts> names sort
        # lexicographically too, but mtime is the source of truth
        # if the operator has both jobid- and timestamp-named dirs.
        dirs = sorted(
            (p for p in profiles_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if not dirs:
            messagebox.showinfo(
                "No profiles",
                "training/profiles/ is empty. Submit a profile "
                "job + 'Pull profile' first.")
            return
        latest = dirs[0]
        # Also print the summary contents inline -- the operator
        # almost always wants to read it FIRST, and surfacing it
        # in the GUI panel avoids the alt-tab to the file viewer
        # just to see "rollout 30%, train 70%". The directory open
        # below lets them click into other artifacts after.
        summary = latest / "summary.txt"
        if summary.exists():
            try:
                self._log(f"[view-profile] {latest.name}/summary.txt:")
                for line in summary.read_text(
                        encoding="utf-8").splitlines():
                    self._log(line)
            except OSError as e:
                self._log(f"[view-profile] read summary failed: {e}")
        self._log(
            f"[view-profile] opening directory "
            f"{latest.relative_to(PROJECT_ROOT)}")
        self._os_open(latest)

    # -- local operations -----------------------------------------------

    def _autoselect_checkpoint(self) -> Optional[Path]:
        """Pick the best checkpoint for inference/training.

        Priority order (matches eval_daily._pick_latest_checkpoint):
          1. `sim_selfplay.pt` -- the active self-play target, what
             Pull writes and what Sync+Continue pushes. THE default
             once self-play is the primary workflow.
          2. Freshest `sim_selfplay_archive_*.pt` -- last cluster
             archive snapshot if no rolling self-play exists yet.
          3. Freshest `supervised_epoch*.pt` -- the warm-start anchor.
          4. `supervised.pt` (mid-epoch rolling supervised) if
             nothing else.
          5. Any other `*.pt` by mtime, last-resort.

        Why NOT "freshest by mtime" wholesale: a stale
        supervised_epoch3.pt that happens to have been touched
        after a Pull would beat the just-pulled sim_selfplay.pt.
        The semantic priority above is what every operator
        actually wants 95% of the time.
        """
        ckpt_dir = PROJECT_ROOT / "training" / "checkpoints"
        if not ckpt_dir.is_dir():
            return None
        sp = ckpt_dir / "sim_selfplay.pt"
        if sp.exists():
            return sp
        archives = sorted(
            ckpt_dir.glob("sim_selfplay_archive_*.pt"),
            key=lambda p: p.stat().st_mtime, reverse=True)
        if archives:
            return archives[0]
        sup_epoch = sorted(
            ckpt_dir.glob("supervised_epoch*.pt"),
            key=lambda p: p.stat().st_mtime, reverse=True)
        if sup_epoch:
            return sup_epoch[0]
        sup = ckpt_dir / "supervised.pt"
        if sup.exists():
            return sup
        any_pt = sorted(ckpt_dir.glob("*.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        return any_pt[0] if any_pt else None

    # Factions for the dropdown. Loaded once at class-init time
    # from tools/scenario_pool (single source of truth) so a new
    # era / faction addition shows up in the GUI without
    # double-maintenance. Falls back to the default-era list if
    # the import fails (fresh checkout where the project deps
    # aren't installed yet).
    _FACTIONS: List[str] = [
        "(none / fully random)",
        # Filled in by _load_factions_at_startup() below. The
        # default-era list is a fallback only.
        "Knalgan Alliance", "Drakes", "Loyalists",
        "Northerners", "Rebels", "Undead",
    ]

    @classmethod
    def _load_factions_at_startup(cls) -> None:
        """Replace the hardcoded faction list with whatever
        scenario_pool.load_factions() returns. Called once from
        main() before any dialog is built so the OptionMenu sees
        the current list. Errors are non-fatal: the hardcoded
        baseline above stays in place if the import fails."""
        try:
            import sys as _sys
            if str(PROJECT_ROOT) not in _sys.path:
                _sys.path.insert(0, str(PROJECT_ROOT))
            from tools.scenario_pool import load_factions
            facs = load_factions()
            # `load_factions` returns dict-like with faction names
            # as keys. Sort for stable display order.
            names = sorted(facs.keys() if hasattr(facs, "keys")
                           else facs)
            if names:
                cls._FACTIONS = ["(none / fully random)"] + names
        except Exception:
            # Import or scrape failure -> keep the hardcoded list.
            # We don't even log this here because the GUI's own
            # logger isn't configured at class-init time; if the
            # operator notices a missing faction in the dropdown
            # they can manually pass --forced-faction.
            pass

    def _op_train_selfplay(self) -> None:
        """Open a parameter panel, then run sim_self_play.py with
        the chosen flags. No live Wesnoth window -- the sim is
        headless. Logs stream to the GUI text panel; checkpoints
        land in `training/checkpoints/sim_selfplay.pt`.

        2026-05-20: removed the binary "Mini or Ladder?" pop-up;
        the dialog has a full mix slider now. See
        `_op_start_selfplay` for the matching cluster-side change."""
        if self._is_busy():
            messagebox.showinfo("Busy", "Another operation is running.")
            return
        self._open_selfplay_dialog(
            title="Self-play (local)",
            on_run=self._run_local_selfplay,
            cluster_mode=False,
        )

    def _ask_map_pool(self) -> Optional[bool]:
        """Pop a small modal asking the operator to pick the scenario
        pool BEFORE the full self-play dialog opens. Returns:
          True  -> use mini-maps (8-scenario engagement-curriculum pool)
          False -> use ladder maps (21-scenario production pool)
          None  -> operator cancelled; caller aborts dialog open

        Pre-dialog rather than buried-in-checkbox because the map
        pool changes the meaning of every other knob (mini maps have
        gold=40/income=0 in scenarios, max_turns of 30-60 makes
        sense vs 200 on ladder, attack% varies 10× between pools).
        Surfacing it first frames the rest of the configuration.

        The choice still lives on the main dialog's checkbox after,
        so the operator can flip it if they change their mind --
        this pop-up just sets the default.
        """
        # askyesnocancel: Yes -> mini-maps; No -> ladder; Cancel -> abort.
        # Tk doesn't ship a generic 3-button modal without rolling
        # a custom Toplevel; askyesnocancel is the standard idiom.
        return messagebox.askyesnocancel(
            "Pick map pool",
            "Which scenario pool should self-play sample from?\n\n"
            "Yes  -> Mini maps  (8 scenarios, 35-273 cells)\n"
            "         engagement-curriculum: leaders ~5 hex apart,\n"
            "         attack% typically 10-30% out of the gate.\n\n"
            "No   -> Ladder maps  (21 scenarios, 690-2352 cells)\n"
            "         production: leaders 12-30 hex apart,\n"
            "         attack% historically ~1% from cold start.\n\n"
            "Cancel -> don't open the dialog\n\n"
            "You can change this on the dialog's `Mini-maps only` "
            "checkbox if needed.",
        )

    def _run_local_selfplay(self, params: dict) -> None:
        """Build argv from the dialog's params dict and spawn
        sim_self_play.py."""
        ckpt = params["checkpoint"]
        argv: List[str] = [PYTHON, str(SCRIPT_SIM_SELFPLAY)]
        if ckpt:
            argv += ["--checkpoint-in", str(ckpt)]
            self._log(f"[selfplay] starting from checkpoint: {Path(ckpt).name}")
        else:
            self._log(
                "[selfplay] no checkpoint -- training from random init")
        # sim_self_play.py defaults --checkpoint-out to
        # training/checkpoints/sim_selfplay.pt. The same file the
        # cluster jobs write to and the Pull button pulls into.
        # Overwrite semantics (Path B): local and cluster share
        # one file; whichever trained more recently is what gets
        # pushed/pulled. The `Sync + Continue (self-play)` button
        # uploads the local sim_selfplay.pt before restarting the
        # cluster chain so the cluster picks up where local left off.
        self._log(
            "[selfplay] checkpoint will be saved to "
            "training/checkpoints/sim_selfplay.pt (overwrites; "
            "the cluster's Pull / Sync+Continue see this file)"
        )
        argv += [
            "--iterations",     str(params["iterations"]),
            "--games-per-iter", str(params["games_per_iter"]),
            "--max-turns",      str(params["max_turns"]),
            "--workers",        str(params["workers"]),
            "--save-every",     str(params["save_every"]),
            "--seed",           str(params["seed"]),
            # Auto-pick the best local device: DML (discrete RX
            # 6600, ~4x CPU since the 2026-05-13 fix) > CUDA > CPU.
            # See tools/device_select.py.
            "--device",         "auto",
        ]
        if params.get("time_budget"):
            argv += ["--time-budget", str(params["time_budget"])]
            self._log(
                f"[selfplay] time-budget: {params['time_budget']} "
                f"(loop exits with a saved checkpoint when elapsed)")
        if params["forced_faction"] == "(none / fully random)":
            argv += ["--forced-faction", "none"]
        else:
            argv += ["--forced-faction", params["forced_faction"]]
        if params["reward_config"]:
            argv += ["--reward-config", str(params["reward_config"])]
        if params.get("use_mcts"):
            argv += [
                "--mcts",
                "--mcts-sims",       str(params["mcts_sims"]),
                "--mcts-c-puct",     str(params["mcts_c_puct"]),
                "--mcts-batch-size", str(params["mcts_batch_size"]),
            ]
            self._log(
                f"[selfplay] MCTS enabled: "
                f"{params['mcts_sims']} sims/move, "
                f"c_puct={params['mcts_c_puct']}")
        if params.get("mini_maps"):
            argv += ["--mini-maps"]
            self._log(
                "[selfplay] mini-maps mode: scenario pool restricted "
                "to 5 smallest Ladder maps (engagement-curriculum)")
        else:
            mr = float(params.get("mini_ratio") or 0.0)
            if mr > 0.0:
                argv += ["--mini-ratio", f"{mr}"]
                self._log(
                    f"[selfplay] mini/ladder mix: {mr*100:.0f}% mini-maps "
                    f"per iter, {(1.0-mr)*100:.0f}% ladder")

        def _post_train_cleanup(captured_lines: List[str]) -> None:
            """After local Train finishes, fire `tools/cleanup.py
            --location local --yes` to archive sim_selfplay.pt
            (subject to --archive-stride-hours) and prune stale
            artifacts. Mirrors what the cluster sbatch tail does
            -- without this, local `training/checkpoints/` grows
            unbounded and the freshness row's `Last archive`
            label stays at `(none)` forever.

            We don't block the GUI on this; it runs as a normal
            spawned op which means the operator sees the cleanup
            log immediately after the train log. Cleanup typically
            takes ~1s. If Train was cancelled or failed (exit
            code != 0) we skip to avoid acting on partial state.
            """
            # _end_op already cleared the busy flag; we can spawn
            # another op now. The captured_lines arg is unused but
            # required by the _spawn(on_complete=...) contract.
            del captured_lines
            cleanup_argv = [
                PYTHON, str(PROJECT_ROOT / "tools" / "cleanup.py"),
                "--location", "local",
                "--yes",
                # Use the same archive stride as the cluster sbatch
                # (6h). For overnight local training that's one new
                # archive per session.
                "--archive-stride-hours", "6",
            ]
            self._log("[cleanup] running tools/cleanup.py "
                      "--location local --yes")
            self._spawn(cleanup_argv, needs_password=False,
                        label="cleanup after local train")

        self._spawn(argv, needs_password=False,
                    label=f"self-play (local, {params['iterations']} iters)",
                    on_complete=_post_train_cleanup)

    def _open_selfplay_dialog(
        self, *, title: str,
        on_run, cluster_mode: bool,
        default_mini_maps: Optional[bool] = None,
    ) -> None:
        """Modal panel with knobs for every self-play parameter.
        `on_run(params: dict)` is called when the user clicks Run.
        `cluster_mode=True` adjusts defaults (more iterations,
        more workers) and hides irrelevant knobs (no checkpoint
        picker -- the sbatch's latest-ckpt logic owns that).

        `default_mini_maps`: if set, OVERRIDES both the persisted
        gui_state.json value and the hardcoded baseline for the
        mini-maps checkbox. The pre-dialog pop-up sets this so the
        operator's pool choice is reflected at dialog open. None
        leaves the existing precedence (persisted > baseline) in
        place.

        Defaults precedence (highest first):
          0. `default_mini_maps` (when provided) for the mini-maps
             field only.
          1. Last-picked values for THIS dialog instance, persisted to
             `~/.wesnoth_ai/gui_state.json` on Run.
          2. Hardcoded baseline (cluster_mode-tuned).
        Missing keys fall through to baseline so adding a new field
        doesn't break a stale state file.
        """
        dlg = tk.Toplevel(self.root)
        dlg.title(title)
        dlg.transient(self.root)
        dlg.grab_set()

        dialog_key = "selfplay_cluster" if cluster_mode else "selfplay_local"

        # Hardcoded baseline. `iterations` is a safety ceiling;
        # `time_budget` is the real exit on long-running jobs (the
        # sim_self_play loop saves a final checkpoint when the budget
        # is exhausted, so wall-time replaces iter-count as the
        # primary stopping criterion). Cluster default 03:50:00
        # matches job_selfplay.sbatch's TIME_BUDGET default (5min
        # headroom under the 03:55:00 SLURM walltime). Local default
        # 10:00:00 lets an overnight DML-on-RX-6600 run get full
        # benefit (~10h * ~25% cluster throughput ~= one cluster
        # day's worth of trajectories), and 100k iters as ceiling
        # matches cluster -- safety, not a target. The operator
        # tunes time_budget down for short tests; gui_state.json
        # remembers the last-picked value across sessions.
        defaults = {
            "iterations":  100000,
            "time_budget": "03:50:00" if cluster_mode else "10:00:00",
            "games_per_iter":  8,
            "max_turns":     200,
            "workers":         6 if cluster_mode else 4,
            # save_every: cluster has stable hardware -> 5 is fine
            # (checkpoint write ~0.5s, every iter is overkill). Local
            # has DML/driver TDR risk (observed 2026-05-13: GPU device
            # suspended after iter 0's train_step, no save = full
            # hour wasted) so save after every iter -- one .pt write
            # per iter is trivial cost, big crash-resilience win.
            "save_every":     5 if cluster_mode else 1,
            "seed":            0,
            "forced_faction": "Knalgan Alliance",
            "reward_config": str(PROJECT_ROOT / "cluster" / "configs"
                                  / "reward_selfplay.json"),
            # MCTS knobs default to OFF / paper-ish values. Toggling
            # the checkbox in the UI flips the action-selection
            # algorithm the cluster sbatch picks via USE_MCTS=1.
            "use_mcts":       False,
            "mcts_sims":      50,
            "mcts_c_puct":    1.5,
            "mcts_batch_size": 1,
            # Mini-maps: restrict to the 5 smallest Ladder maps for
            # engagement-curriculum training. Default off (full pool).
            "mini_maps":      False,
            # Mini/ladder MIX. Float in [0, 1]: fraction of games per
            # iter that sample from the mini pool. 0.0 = pure ladder
            # (default), 1.0 = pure mini (equivalent to mini_maps=True
            # but via the mix code path). 0.3 = ~30% mini for
            # engagement gradient, ~70% ladder for production
            # distribution. Ignored when mini_maps=True.
            "mini_ratio":     0.0,
        }
        # Layer persisted user-picked values on top.
        persisted = _load_gui_state().get(dialog_key, {})
        for k, v in persisted.items():
            if k in defaults:
                defaults[k] = v
        # The pre-dialog map-pool pop-up's choice wins over both
        # baseline and persisted -- that pop-up exists specifically
        # so the operator's mood-of-the-moment pick is what loads.
        if default_mini_maps is not None:
            defaults["mini_maps"] = bool(default_mini_maps)

        v_iters     = tk.IntVar(value=defaults["iterations"])
        v_tbudget   = tk.StringVar(value=defaults["time_budget"])
        v_games     = tk.IntVar(value=defaults["games_per_iter"])
        v_turns     = tk.IntVar(value=defaults["max_turns"])
        v_workers   = tk.IntVar(value=defaults["workers"])
        v_save      = tk.IntVar(value=defaults["save_every"])
        v_seed      = tk.IntVar(value=defaults["seed"])
        v_faction   = tk.StringVar(value=defaults["forced_faction"])
        v_reward    = tk.StringVar(value=defaults["reward_config"])
        v_ckpt      = tk.StringVar(value=str(self._autoselect_checkpoint() or ""))
        v_use_mcts  = tk.BooleanVar(value=bool(defaults["use_mcts"]))
        v_mcts_sims = tk.IntVar(value=int(defaults["mcts_sims"]))
        v_mcts_cpuct= tk.DoubleVar(value=float(defaults["mcts_c_puct"]))
        v_mini_maps = tk.BooleanVar(value=bool(defaults["mini_maps"]))
        v_mini_ratio = tk.DoubleVar(value=float(defaults.get("mini_ratio", 0.0)))
        v_mcts_bs   = tk.IntVar(value=int(defaults["mcts_batch_size"]))

        def grid(label: str, widget, row: int, hint: str = ""):
            tk.Label(dlg, text=label, anchor="e").grid(
                row=row, column=0, padx=6, pady=3, sticky="e")
            widget.grid(row=row, column=1, padx=6, pady=3, sticky="ew")
            if hint:
                tk.Label(dlg, text=hint, fg="gray").grid(
                    row=row, column=2, padx=4, pady=3, sticky="w")

        # Row 0: iteration count (safety ceiling; time budget is the
        # primary exit on cluster jobs).
        grid("Iterations:",
             tk.Spinbox(dlg, from_=1, to=1000000,
                        textvariable=v_iters, width=10),
             0, "safety ceiling (time-budget wins normally)")
        grid("Time budget:",
             tk.Entry(dlg, textvariable=v_tbudget, width=10),
             1, "HH:MM:SS or seconds; empty = no time limit")
        grid("Games per iter:",
             tk.Spinbox(dlg, from_=1, to=64, textvariable=v_games, width=10),
             2, "rollouts before each train_step")
        grid("Max turns:",
             tk.Spinbox(dlg, from_=20, to=500, textvariable=v_turns, width=10),
             3, "per-game turn cap")
        grid("Workers:",
             tk.Spinbox(dlg, from_=0, to=12, textvariable=v_workers, width=10),
             4, "0 = serial, N = parallel rollouts")
        grid("Save every (iters):",
             tk.Spinbox(dlg, from_=1, to=100, textvariable=v_save, width=10),
             5, "checkpoint write cadence")
        grid("Seed:",
             tk.Spinbox(dlg, from_=0, to=999999, textvariable=v_seed, width=10),
             6, "RNG seed for reproducibility")
        grid("Forced faction:",
             tk.OptionMenu(dlg, v_faction, *self._FACTIONS),
             7, "always present on at least one side")

        # Reward config picker. Greyed out when MCTS is enabled
        # (AlphaZero distills the terminal z, ignoring shaping
        # rewards) -- see the trace below the MCTS checkbox.
        rew_frame = tk.Frame(dlg)
        rew_entry = tk.Entry(rew_frame, textvariable=v_reward, width=40)
        rew_entry.pack(side="left", fill="x", expand=True)
        def _pick_reward():
            p = filedialog.askopenfilename(
                title="Pick reward config",
                initialdir=str(PROJECT_ROOT / "cluster" / "configs"),
                filetypes=[("JSON config", "*.json"), ("All", "*.*")],
            )
            if p:
                v_reward.set(p)
        rew_btn = tk.Button(rew_frame, text="...", width=3,
                            command=_pick_reward)
        rew_btn.pack(side="left", padx=(4, 0))
        rew_label = tk.Label(dlg, text="Reward config:", anchor="e")
        rew_label.grid(row=8, column=0, padx=6, pady=3, sticky="e")
        rew_frame.grid(row=8, column=1, padx=6, pady=3, sticky="ew")
        rew_hint = tk.Label(dlg, text="(unused in MCTS mode)", fg="gray")
        rew_hint.grid(row=8, column=2, padx=4, pady=3, sticky="w")

        # MCTS row. Off by default (REINFORCE training); checking
        # the box flips the cluster sbatch's USE_MCTS flag, which
        # in turn passes `--mcts --mcts-sims N --mcts-c-puct C` to
        # sim_self_play.py. Reward config is silently unused in
        # MCTS mode (AlphaZero distills terminal z, not shaping).
        mcts_frame = tk.Frame(dlg)
        tk.Checkbutton(mcts_frame, text="Use MCTS",
                       variable=v_use_mcts).pack(side="left")
        tk.Label(mcts_frame, text="  sims:").pack(side="left")
        tk.Spinbox(mcts_frame, from_=1, to=2000, textvariable=v_mcts_sims,
                   width=6).pack(side="left", padx=2)
        tk.Label(mcts_frame, text="  c_puct:").pack(side="left")
        tk.Spinbox(mcts_frame, from_=0.1, to=5.0, increment=0.1,
                   textvariable=v_mcts_cpuct, width=5).pack(
                       side="left", padx=2)
        tk.Label(mcts_frame, text="  batch:").pack(side="left")
        tk.Spinbox(mcts_frame, from_=1, to=64, textvariable=v_mcts_bs,
                   width=4).pack(side="left", padx=2)
        grid("MCTS:", mcts_frame, 10,
             "off=REINFORCE; on=AlphaZero-style search")
        # Mini-maps toggle + mix ratio. "Mini-maps only" is the
        # binary toggle (100% mini); the mix slider lets the operator
        # blend mini and ladder per-iter. When "only" is checked,
        # the mix slider is ignored (always 100% mini). When unchecked,
        # the slider value (in [0, 1]) is the per-game probability
        # of sampling from mini. Used during the engagement-
        # curriculum phase to reduce the long-march exploration
        # cost while keeping the policy exposed to ladder maps.
        mini_frame = tk.Frame(dlg)
        tk.Checkbutton(mini_frame, text="Mini-maps only",
                       variable=v_mini_maps).pack(side="left")
        tk.Label(mini_frame,
                 text="(8 maps, leaders ~5 hex apart)",
                 fg="gray").pack(side="left", padx=4)
        grid("Map pool:", mini_frame, 11, "")
        # Mix slider. Disabled when "Mini-maps only" is ticked.
        mix_frame = tk.Frame(dlg)
        mix_scale = tk.Scale(mix_frame, from_=0.0, to=1.0,
                             resolution=0.05, orient="horizontal",
                             length=200, variable=v_mini_ratio,
                             showvalue=True)
        mix_scale.pack(side="left")
        mix_hint = tk.Label(mix_frame,
                            text="(fraction of games sampled from mini)",
                            fg="gray")
        mix_hint.pack(side="left", padx=4)
        grid("Mini-mix ratio:", mix_frame, 12, "")

        def _sync_mix_enabled(*_args):
            # When "Mini-maps only" is checked, the mix is fixed at
            # 100% and the slider is meaningless. Disable + grey to
            # make the relationship visible.
            if v_mini_maps.get():
                mix_scale.configure(state="disabled")
                mix_hint.configure(fg="lightgray")
            else:
                mix_scale.configure(state="normal")
                mix_hint.configure(fg="gray")
        v_mini_maps.trace_add("write", _sync_mix_enabled)
        _sync_mix_enabled()

        # Grey out the reward-config widgets when MCTS is on. The
        # trace fires whenever the checkbox flips and on dialog open
        # (we call it once explicitly below to seed the initial
        # state). Disabling the widgets is purely cosmetic --
        # `_run_cluster_selfplay` / `_run_local_selfplay` already
        # ignore reward_config in MCTS mode -- but it's a clear
        # visual cue that the field doesn't matter then.
        def _sync_reward_enabled(*_args):
            state = "disabled" if v_use_mcts.get() else "normal"
            rew_entry.configure(state=state)
            rew_btn.configure(state=state)
            rew_label.configure(fg="gray" if v_use_mcts.get() else "black")
        v_use_mcts.trace_add("write", _sync_reward_enabled)
        _sync_reward_enabled()

        # Checkpoint picker (LOCAL only -- cluster sbatch handles its
        # own latest-ckpt logic).
        if not cluster_mode:
            ckpt_frame = tk.Frame(dlg)
            tk.Entry(ckpt_frame, textvariable=v_ckpt, width=40).pack(
                side="left", fill="x", expand=True)
            def _pick_ckpt():
                p = filedialog.askopenfilename(
                    title="Pick starting checkpoint (or Cancel for random init)",
                    initialdir=str(PROJECT_ROOT / "training" / "checkpoints"),
                    filetypes=[("PyTorch checkpoint", "*.pt"), ("All", "*.*")],
                )
                if p:
                    v_ckpt.set(p)
            tk.Button(ckpt_frame, text="...", width=3,
                      command=_pick_ckpt).pack(side="left", padx=(4, 0))
            grid("Checkpoint:", ckpt_frame, 9,
                 "blank = train from random init")

        def on_ok():
            # Tk Spinbox / Entry .get() raises TclError when the
            # field is empty (e.g., operator cleared a numeric field
            # then clicked Run). Under pythonw stderr goes nowhere
            # so the exception was silently swallowed; the dialog
            # appeared to "do nothing." Wrap the whole param read
            # in a try/except and surface the error via messagebox.
            try:
                params = {
                    "iterations":      v_iters.get(),
                    "time_budget":     v_tbudget.get().strip(),
                    "games_per_iter":  v_games.get(),
                    "max_turns":       v_turns.get(),
                    "workers":         v_workers.get(),
                    "save_every":      v_save.get(),
                    "seed":            v_seed.get(),
                    "forced_faction":  v_faction.get(),
                    "reward_config":   v_reward.get().strip() or None,
                    "checkpoint":      v_ckpt.get().strip() or None,
                    "use_mcts":        bool(v_use_mcts.get()),
                    "mcts_sims":       int(v_mcts_sims.get()),
                    "mcts_c_puct":     float(v_mcts_cpuct.get()),
                    "mcts_batch_size": int(v_mcts_bs.get()),
                    "mini_maps":       bool(v_mini_maps.get()),
                    "mini_ratio":      max(0.0, min(1.0,
                                                    float(v_mini_ratio.get()))),
                }
            except (tk.TclError, ValueError) as e:
                messagebox.showerror(
                    "Invalid value",
                    f"One or more fields has an invalid or empty "
                    f"value:\n\n{e}\n\nFix the field(s) and click "
                    f"Run again.")
                return
            # Persist user-picked values as the new defaults for
            # the next dialog open. Stash everything except
            # `checkpoint` (that's a one-shot path picker; we always
            # autoselect freshest .pt next time).
            persistable = {k: v for k, v in params.items()
                           if k != "checkpoint"}
            state = _load_gui_state()
            state[dialog_key] = persistable
            _save_gui_state(state)
            dlg.destroy()
            on_run(params)

        def on_save_only():
            # Parse + persist the form values WITHOUT firing the
            # actual training submission. The next `Start self-play`
            # / `Sync + Continue` (or `Train (self-play)` locally)
            # will pick up these values from gui_state.json. Lets
            # the operator tune settings ahead of a launch (or
            # between launches) without immediately kicking a job.
            #
            # Same validation as on_ok; errors short-circuit with
            # the same messagebox. Doesn't call on_run.
            try:
                params = {
                    "iterations":      v_iters.get(),
                    "time_budget":     v_tbudget.get().strip(),
                    "games_per_iter":  v_games.get(),
                    "max_turns":       v_turns.get(),
                    "workers":         v_workers.get(),
                    "save_every":      v_save.get(),
                    "seed":            v_seed.get(),
                    "forced_faction":  v_faction.get(),
                    "reward_config":   v_reward.get().strip() or None,
                    "use_mcts":        bool(v_use_mcts.get()),
                    "mcts_sims":       int(v_mcts_sims.get()),
                    "mcts_c_puct":     float(v_mcts_cpuct.get()),
                    "mcts_batch_size": int(v_mcts_bs.get()),
                    "mini_maps":       bool(v_mini_maps.get()),
                    "mini_ratio":      max(0.0, min(1.0,
                                                    float(v_mini_ratio.get()))),
                }
            except (tk.TclError, ValueError) as e:
                messagebox.showerror(
                    "Invalid value",
                    f"One or more fields has an invalid or empty "
                    f"value:\n\n{e}\n\nFix the field(s) and click "
                    f"Save again.")
                return
            state = _load_gui_state()
            state[dialog_key] = params
            _save_gui_state(state)
            self._log(
                f"[selfplay] settings saved to gui_state.json "
                f"(key={dialog_key!r}); next start picks them up.")
            dlg.destroy()

        btns = tk.Frame(dlg)
        btns.grid(row=20, column=0, columnspan=3, pady=10)
        tk.Button(btns, text="Cancel", width=12,
                  command=dlg.destroy).pack(side="right", padx=4)
        tk.Button(btns, text="Save only", width=12,
                  command=on_save_only).pack(side="right", padx=4)
        tk.Button(btns, text="Save & start", width=14,
                  command=on_ok).pack(side="right", padx=4)
        dlg.columnconfigure(1, weight=1)

    def _op_display_selfplay(self) -> None:
        """Watch ONE game with the loaded model: runs one full game
        through the simulator, then writes a Wesnoth-loadable .bz2
        replay so the user can scrub through it in Wesnoth's replay
        viewer (File -> Load Game -> pick the .bz2).

        Output bz2 lands in `logs/sim_demo_<UTC>.bz2`. The path is
        echoed to the GUI log; double-click on Windows opens it in
        Wesnoth via the file association."""
        ckpt = self._autoselect_checkpoint()
        argv: List[str] = [PYTHON, str(SCRIPT_SIM_DEMO)]
        if ckpt is not None:
            argv += ["--checkpoint", str(ckpt)]
            self._log(f"[demo] using checkpoint: {ckpt.name}")
        else:
            self._log(
                "[demo] no *.pt found under training/checkpoints/; "
                "the demo will fail. Pull a checkpoint from the "
                "cluster first.")
            messagebox.showwarning(
                "No checkpoint",
                "No *.pt checkpoint found under "
                "`training/checkpoints/`. Use the Pull button "
                "first (the default mode pulls sim_selfplay.pt), "
                "then retry Display.")
            return
        self._spawn(argv, needs_password=False,
                    label="demo game (sim + .bz2 export)")

    def _op_eval_dialog(self) -> None:
        """Pop a small dialog to pick the checkpoint + matchup config."""
        if self._is_busy():
            messagebox.showinfo("Busy", "Another operation is running.")
            return

        path = filedialog.askopenfilename(
            title="Pick a checkpoint to evaluate",
            initialdir=str(PROJECT_ROOT / "training" / "checkpoints"),
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All", "*.*")],
        )
        if not path:
            return

        # Tiny modal for the rest of the eval knobs.
        dlg = tk.Toplevel(self.root)
        dlg.title("Eval config")
        dlg.transient(self.root)
        dlg.grab_set()

        # Layer last-picked values on top of hardcoded defaults so
        # the operator's previous choice sticks across GUI restarts.
        # Persists to ~/.wesnoth_ai/gui_state.json on Run, same
        # storage the self-play dialog uses.
        persisted = _load_gui_state().get("eval_custom", {})
        maps_var     = tk.StringVar(value=persisted.get("maps", "caves den"))
        pairs_var    = tk.StringVar(value=persisted.get("pairs", "cross"))
        # Default OFF: side-swapping doubles game count but eliminates
        # first-mover bias, which is the standard in eval. The CLI
        # (tools/eval_vs_builtin.py) defaults to swap-on; the GUI
        # silently regressed every eval through it before this flip.
        no_swap_var  = tk.BooleanVar(value=bool(persisted.get("no_swap", False)))
        parallel_var = tk.IntVar(value=int(persisted.get("parallel", 4)))
        max_actions_var = tk.IntVar(value=int(persisted.get("max_actions", 500)))

        def grid(label: str, widget, row: int):
            tk.Label(dlg, text=label).grid(row=row, column=0,
                                           padx=8, pady=4, sticky="w")
            widget.grid(row=row, column=1, padx=8, pady=4, sticky="ew")

        grid("Maps (space-separated):",
             tk.Entry(dlg, textvariable=maps_var, width=30), 0)
        grid("Pairs:",
             tk.OptionMenu(dlg, pairs_var, "all", "cross"), 1)
        grid("Side swap:",
             tk.Checkbutton(dlg, variable=no_swap_var,
                            text="no swap (skip side-2 rerun)"), 2)
        grid("Parallel games:",
             tk.Spinbox(dlg, from_=1, to=8, textvariable=parallel_var, width=8), 3)
        grid("Max actions/game:",
             tk.Spinbox(dlg, from_=50, to=2000,
                        textvariable=max_actions_var, width=8), 4)

        def on_ok():
            try:
                eval_state = {
                    "maps":        maps_var.get(),
                    "pairs":       pairs_var.get(),
                    "no_swap":     bool(no_swap_var.get()),
                    "parallel":    parallel_var.get(),
                    "max_actions": max_actions_var.get(),
                }
            except (tk.TclError, ValueError) as e:
                messagebox.showerror(
                    "Invalid value",
                    f"One or more fields has an invalid or empty "
                    f"value:\n\n{e}\n\nFix the field(s) and click "
                    f"Run again.")
                return
            # Persist the dialog state so the next open seeds the
            # operator's last-picked values rather than the static
            # defaults. Mirrors the self-play dialog's pattern.
            state = _load_gui_state()
            state["eval_custom"] = eval_state
            _save_gui_state(state)
            argv = [
                PYTHON, str(EVAL_SCRIPT),
                "--checkpoint", path,
                "--pairs", eval_state["pairs"],
                "--parallel", str(eval_state["parallel"]),
                "--max-actions", str(eval_state["max_actions"]),
            ]
            maps = eval_state["maps"].split()
            if maps:
                argv += ["--maps", *maps]
            if eval_state["no_swap"]:
                argv += ["--no-swap"]
            dlg.destroy()
            self._spawn(argv, needs_password=False,
                        label=f"eval: {Path(path).name}")

        btnbar = tk.Frame(dlg); btnbar.grid(row=10, column=0, columnspan=2,
                                            pady=8)
        tk.Button(btnbar, text="Cancel",
                  command=dlg.destroy).pack(side="right", padx=4)
        tk.Button(btnbar, text="Run",
                  command=on_ok).pack(side="right", padx=4)
        dlg.columnconfigure(1, weight=1)

    def _op_diagnose(self) -> None:
        """Open a tiny dialog (games count + max turns), then fire
        `tools/diagnose_selfplay.py` against the freshest local
        checkpoint. The output streams to the GUI panel.

        Why a dialog: the diagnostic's wall time scales linearly
        with --games and quadratically-ish with --max-turns (most
        games hit the cap when the policy doesn't engage). Letting
        the operator pick saves them from the default-vs-custom
        round-trip.
        """
        if self._is_busy():
            messagebox.showinfo("Busy", "Another operation is running.")
            return
        ckpt = self._autoselect_checkpoint()
        if ckpt is None:
            messagebox.showwarning(
                "No checkpoint",
                "No *.pt found under training/checkpoints/. "
                "Pull a checkpoint from the cluster first.")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Diagnose self-play")
        dlg.transient(self.root)
        dlg.grab_set()
        # Persisted defaults via the same gui_state.json mechanism
        # the self-play dialog uses, so last-picked values stick.
        persisted = _load_gui_state().get("diagnose", {})
        games_var     = tk.IntVar(value=int(persisted.get("games", 20)))
        max_turns_var = tk.IntVar(value=int(persisted.get("max_turns", 60)))

        def grid(label: str, widget, row: int, hint: str = ""):
            tk.Label(dlg, text=label, anchor="e").grid(
                row=row, column=0, padx=6, pady=4, sticky="e")
            widget.grid(row=row, column=1, padx=6, pady=4, sticky="w")
            if hint:
                tk.Label(dlg, text=hint, fg="gray").grid(
                    row=row, column=2, padx=4, pady=4, sticky="w")

        grid("Games:",
             tk.Spinbox(dlg, from_=1, to=200, textvariable=games_var,
                        width=8),
             0, "20 is a reasonable sanity-check; 50+ for more signal")
        grid("Max turns:",
             tk.Spinbox(dlg, from_=10, to=200,
                        textvariable=max_turns_var, width=8),
             1, "cap per game (no-kills runs hit this often)")
        tk.Label(dlg, text=f"Checkpoint: {ckpt.name}", fg="gray").grid(
            row=2, column=0, columnspan=3, padx=6, pady=(0, 4),
            sticky="w")

        def on_ok():
            try:
                d_state = {
                    "games":     games_var.get(),
                    "max_turns": max_turns_var.get(),
                }
            except (tk.TclError, ValueError) as e:
                messagebox.showerror(
                    "Invalid value",
                    f"Games or Max turns is empty/invalid:\n\n{e}\n\n"
                    f"Fix and click Run again.")
                return
            state = _load_gui_state()
            state["diagnose"] = d_state
            _save_gui_state(state)
            argv = [
                PYTHON, str(SCRIPT_DIAGNOSE),
                "--checkpoint",  str(ckpt),
                "--games",       str(d_state["games"]),
                "--max-turns",   str(d_state["max_turns"]),
                "--log-level",   "WARNING",
            ]
            dlg.destroy()
            self._spawn(argv, needs_password=False,
                        label=f"diagnose ({d_state['games']} games)")

        btns = tk.Frame(dlg)
        btns.grid(row=10, column=0, columnspan=3, pady=10)
        tk.Button(btns, text="Cancel", width=10,
                  command=dlg.destroy).pack(side="right", padx=4)
        tk.Button(btns, text="Run", width=10,
                  command=on_ok).pack(side="right", padx=4)

    def _op_eval_daily(self) -> None:
        """Fire `tools/eval_daily.py` with the quick preset on the
        freshest local checkpoint. Appends a row to
        `training/eval_history.{csv,md}` -- the trend across days
        is the headline signal we want once self-play training
        starts producing real wins.

        No dialog: this is the "one-click between cluster pulls"
        button. For tuning (full grid, specific maps, side-swap on),
        use the regular "Run eval ..." button instead.
        """
        argv = [PYTHON, str(PROJECT_ROOT / "tools" / "eval_daily.py"),
                "--preset", "quick"]
        self._spawn(argv, needs_password=False,
                    label="daily eval (quick preset)")

    def _op_edit_rewards(self) -> None:
        """Open a small grid editor for `cluster/configs/reward_selfplay.json`.

        Each top-level numeric field becomes a row; the `_about`
        free-text field is shown read-only at the top so the
        editor doesn't lose the context comment. Non-numeric fields
        (`unit_type_bonuses`, `turn_conditional_bonuses`) are shown
        as a count + a hint that those need a text editor for now.

        On Save, the file is rewritten preserving the original key
        order so the JSON diff stays readable across edits. The
        atomic-replace pattern (write to .tmp, then os.replace) avoids
        the half-written-file race if the GUI is killed mid-save.
        """
        if self._is_busy():
            messagebox.showinfo("Busy", "Another operation is running.")
            return
        cfg_path = PROJECT_ROOT / "cluster" / "configs" / "reward_selfplay.json"
        if not cfg_path.exists():
            messagebox.showerror(
                "Missing",
                f"Reward config not found:\n{cfg_path}")
            return
        import json as _json
        try:
            with cfg_path.open(encoding="utf-8") as f:
                cfg = _json.load(f)
        except (OSError, ValueError) as e:
            messagebox.showerror("Parse error", f"Couldn't load: {e}")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title(f"Edit rewards -- {cfg_path.name}")
        dlg.transient(self.root)
        dlg.grab_set()

        # Header / context row
        about = cfg.get("_about", "")
        if about:
            hdr = tk.Label(dlg, text=about, fg="gray",
                           wraplength=560, justify="left", anchor="w")
            hdr.grid(row=0, column=0, columnspan=3,
                     padx=8, pady=(8, 4), sticky="w")

        # Build one StringVar per numeric field, in original key order
        # so the JSON diff stays minimal. Booleans (none currently)
        # would go in a separate branch; we don't need them yet.
        numeric_vars: Dict[str, tk.StringVar] = {}
        skip_keys = {"_about"}
        list_keys: List[Tuple[str, int]] = []  # (key, len)
        row_i = 1
        for k, v in cfg.items():
            if k in skip_keys:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                sv = tk.StringVar(value=str(v))
                numeric_vars[k] = sv
                tk.Label(dlg, text=k + ":", anchor="e").grid(
                    row=row_i, column=0, padx=6, pady=2, sticky="e")
                tk.Entry(dlg, textvariable=sv, width=14).grid(
                    row=row_i, column=1, padx=6, pady=2, sticky="w")
                # Hint: a + or - depending on default sign so it's
                # obvious which fields are penalties vs rewards.
                sign = ("(reward)" if v > 0 else
                        "(penalty)" if v < 0 else "(neutral)")
                tk.Label(dlg, text=sign, fg="gray").grid(
                    row=row_i, column=2, padx=4, pady=2, sticky="w")
                row_i += 1
            elif isinstance(v, list):
                list_keys.append((k, len(v)))

        # Show list-typed fields with their count + note that the GUI
        # can't edit nested structures. Keeps the operator from
        # silently losing them on save.
        if list_keys:
            tk.Label(dlg, text="(read-only; edit JSON directly:)",
                     fg="gray").grid(row=row_i, column=0, columnspan=3,
                                     padx=6, pady=(8, 2), sticky="w")
            row_i += 1
            for k, n in list_keys:
                tk.Label(dlg, text=k + ":", anchor="e").grid(
                    row=row_i, column=0, padx=6, pady=2, sticky="e")
                tk.Label(dlg, text=f"{n} entr{'y' if n == 1 else 'ies'}",
                         fg="gray").grid(
                    row=row_i, column=1, padx=6, pady=2, sticky="w")
                row_i += 1

        def on_save():
            # Re-parse the file (preserving keys we don't show in the
            # editor: _about, lists) and update only the numeric
            # fields. Strict float-parse so a typo doesn't get
            # silently coerced.
            try:
                with cfg_path.open(encoding="utf-8") as f:
                    fresh = _json.load(f)
            except (OSError, ValueError) as e:
                messagebox.showerror("Read error",
                                     f"Couldn't reread: {e}")
                return
            errors: List[str] = []
            for k, sv in numeric_vars.items():
                raw = sv.get().strip()
                try:
                    val = float(raw)
                except ValueError:
                    errors.append(f"  {k}: {raw!r} is not a number")
                    continue
                # Preserve int-ness when the original was an int and
                # the new value is integer-valued, so the JSON diff
                # stays clean (no spurious 0 -> 0.0).
                orig = fresh.get(k)
                if isinstance(orig, int) and not isinstance(orig, bool) \
                        and val == int(val):
                    fresh[k] = int(val)
                else:
                    fresh[k] = val
            if errors:
                messagebox.showerror(
                    "Bad values",
                    "Fix these and save again:\n" + "\n".join(errors))
                return
            # Atomic write: tmp + rename. JSON dumped with the same
            # 2-space indent the original file uses (see
            # `cluster/configs/reward_selfplay.json` for the style),
            # trailing newline so editors don't fight us.
            tmp = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
            try:
                with tmp.open("w", encoding="utf-8") as f:
                    _json.dump(fresh, f, indent=2)
                    f.write("\n")
                os.replace(tmp, cfg_path)
            except OSError as e:
                messagebox.showerror("Write error", str(e))
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                return
            self._log(f"[rewards] saved {cfg_path.relative_to(PROJECT_ROOT)}")
            dlg.destroy()
            # Rewards apply on the NEXT cluster job's startup, not
            # mid-run. After save, offer to push the new config so
            # the next chain link picks it up. The operator can
            # also push later via Sync code -- this is just a
            # one-click shortcut for the common case.
            if messagebox.askyesno(
                    "Sync to cluster?",
                    "Reward config saved locally. The cluster's "
                    "next self-play chain link will only see "
                    "this change after a Sync.\n\n"
                    "Sync code now? (Code-only -- does NOT push "
                    "your local sim_selfplay.pt and does NOT "
                    "restart the cluster job. Use Sync + Continue "
                    "(self-play) for that combined action.)"):
                self._spawn(self._ps(SCRIPT_SYNC),
                            needs_password=True,
                            label="sync code (after rewards edit)")

        btns = tk.Frame(dlg)
        btns.grid(row=row_i + 1, column=0, columnspan=3, pady=10)
        tk.Button(btns, text="Cancel", width=10,
                  command=dlg.destroy).pack(side="right", padx=4)
        tk.Button(btns, text="Save", width=10,
                  command=on_save).pack(side="right", padx=4)
        dlg.columnconfigure(1, weight=1)

    def _os_open(self, path: Path) -> None:
        """Open `path` in the system's default app (Notepad/VS Code
        for .md/.csv, Explorer for directories). Falls back to
        dumping the contents into the GUI panel if the OS handler
        fails or isn't available.

        Used by View profile / View eval history / View train
        history -- all read-only surfacing of files the GUI didn't
        create."""
        try:
            if os.name == "nt":
                os.startfile(str(path))   # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as e:
            self._log(f"[view] OS open failed ({e}); dumping contents")
            if path.is_file():
                try:
                    self._log(path.read_text(encoding="utf-8"))
                except OSError as e2:
                    self._log(f"[view] read also failed: {e2}")

    def _op_view_eval_history(self) -> None:
        """Open `training/eval_history.md` in the default text editor.
        Prefers .md (richer formatting) but falls back to the .csv if
        the .md is missing -- which can happen if the operator deleted
        it to start fresh, or if no daily-eval has run yet."""
        md  = PROJECT_ROOT / "training" / "eval_history.md"
        csv = PROJECT_ROOT / "training" / "eval_history.csv"
        if md.exists():
            self._log(f"[view] {md.relative_to(PROJECT_ROOT)}")
            self._os_open(md)
        elif csv.exists():
            self._log(f"[view] {csv.relative_to(PROJECT_ROOT)} "
                      f"(.md missing, falling back to .csv)")
            self._os_open(csv)
        else:
            messagebox.showinfo(
                "No history",
                "No eval history yet. Run Daily eval at least once "
                "to populate training/eval_history.md.")

    def _op_view_train_history(self) -> None:
        """Open the freshest `trainer_history_*.csv` in the default
        editor. Local Train writes `trainer_history_local.csv`;
        cluster jobs write `trainer_history_<SLURM_JOB_ID>.csv` and
        Pull logs brings them home -- so multiple cluster history
        CSVs may live side by side. Pick freshest by mtime."""
        logs_dir = PROJECT_ROOT / "training" / "logs"
        if not logs_dir.is_dir():
            messagebox.showinfo(
                "No history",
                "No training/logs/ directory yet. Run local Train "
                "or pull a cluster log first.")
            return
        cands = sorted(logs_dir.glob("trainer_history_*.csv"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            messagebox.showinfo(
                "No history",
                "No trainer_history_*.csv yet. Run local Train, or "
                "pull a cluster log (the cluster sbatch writes its "
                "own history CSV per job).")
            return
        target = cands[0]
        self._log(f"[view] {target.relative_to(PROJECT_ROOT)}")
        self._os_open(target)

    def _op_setup_tdr(self) -> None:
        """One-time machine setup: raise the Windows GPU TDR
        threshold so long DML training kernels don't trip the
        device-suspended crash.

        Why a separate elevated window, not _spawn: writing
        HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers
        needs admin. The GUI isn't elevated, and Windows doesn't
        let a non-elevated parent capture stdout from an elevated
        child (security boundary -- a malicious low-priv process
        could otherwise piggyback on admin output). The canonical
        Windows pattern is `Start-Process -Verb RunAs` from an
        intermediate (non-elevated) PowerShell, which fires UAC
        and launches a new elevated console for the actual work.

        UAC behavior: the system shows the consent dialog; user
        clicks Yes -> elevated PowerShell opens, runs the script,
        stays open via -NoExit so the [tdr] BEFORE/AFTER output
        is readable. User clicks No -> nothing happens, no error.

        The script's defaults (TdrDelay=60, TdrDdiDelay=60) are
        applied; subsequent reboot is required. To customize, run
        cluster\\setup_tdr.ps1 manually from an elevated terminal
        with -Delay N or -Restore. See the script's docstring for
        full options.
        """
        if not SCRIPT_TDR.exists():
            messagebox.showerror(
                "Missing", f"{SCRIPT_TDR} not found. Pull or sync "
                f"the project tree first.")
            return
        # Confirm intent. This is a machine-level registry change
        # requiring a reboot; we want the operator to think about
        # it once rather than treating it as routine.
        proceed = self._confirm_destructive(
            title="Setup TDR",
            action=("Adjust the Windows GPU TDR (Timeout Detection "
                    "and Recovery) threshold to 60s?"),
            consequences=(
                "Why: long DML/CUDA training kernels can otherwise "
                "trip the device-suspended crash.\n\n"
                "Requires:\n"
                "  * UAC elevation (admin consent prompt)\n"
                "  * REBOOT for the change to take effect\n\n"
                "Fully reversible: run setup_tdr.ps1 -Restore from "
                "an elevated PowerShell, or click 'Setup TDR' "
                "again and pick a different value via the script."),
            danger="high",
            proceed_text="Setup TDR",
        )
        if not proceed:
            self._log("[tdr] setup cancelled (user clicked No)")
            return
        # Build the elevation command. The intermediate (non-
        # elevated) PowerShell runs Start-Process -Verb RunAs,
        # which triggers UAC and launches the actual elevated
        # PowerShell window. -NoExit keeps the elevated console
        # open after the script finishes so the operator can read
        # the BEFORE/AFTER output before closing.
        #
        # We DON'T -Wait on the elevated process and DON'T pipe
        # its stdout (Windows blocks the latter for the security
        # reasons noted above). Fire-and-forget.
        script_path = str(SCRIPT_TDR)
        # ArgumentList values are passed individually so the
        # -File path can contain spaces without quoting headaches.
        # The leading `'-NoExit',` is the persistence flag for the
        # elevated session.
        intermediate = (
            "Start-Process powershell -Verb RunAs -ArgumentList "
            "'-NoExit','-NoProfile','-ExecutionPolicy','Bypass',"
            f"'-File','{script_path}'"
        )
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW
        try:
            subprocess.Popen(
                [POWERSHELL, "-NoProfile", "-Command", intermediate],
                cwd=str(PROJECT_ROOT),
                creationflags=creationflags,
            )
        except OSError as e:
            self._log(f"[tdr] launch failed: {e}")
            messagebox.showerror("Launch failed", str(e))
            return
        self._log(
            "[tdr] elevated PowerShell launched. Accept the UAC "
            "prompt; the script will show BEFORE / AFTER values "
            "in its own console window. REBOOT after it "
            "completes for the change to take effect.")

    # -- lifecycle ------------------------------------------------------

    def _forget_password(self) -> None:
        self._password.set("")
        self._log("Password cleared.")

    def _on_close(self) -> None:
        self._password.set("")
        self._cancel_op()
        if self._askpass is not None:
            self._askpass.cleanup()
            self._askpass = None
        self.root.destroy()


def main() -> int:
    # Sweep stale askpass dirs from prior GUI runs that crashed or
    # got force-killed before AskpassFiles.cleanup() fired. Each
    # `wai_gui_pw_*` holds (a) the password file (sensitive!) and
    # (b) the askpass.bat helper. Leaving them in %TEMP% is both
    # an inode-pressure issue and a minor security concern, so
    # purge on startup.
    n_purged = _purge_stale_askpass_dirs()
    if n_purged:
        # No logger configured yet; print to stderr so the user sees
        # it in the launching shell when they run pythonw and watch
        # for output, but it won't pop a dialog.
        print(f"[gui] purged {n_purged} stale askpass dir(s) "
              f"from {tempfile.gettempdir()}", file=sys.stderr)
    # Pull the live faction list from scenario_pool (single source
    # of truth) so the dialog's dropdown stays in sync with whatever
    # the trainer actually accepts. Non-fatal on failure -- the
    # baseline default-era list is the fallback.
    App._load_factions_at_startup()
    # Use ThemedTk for nicer ttk widget styling (Notebook tabs,
    # PanedWindow sash, themed buttons). Falls back to plain Tk if
    # ttkthemes isn't installed -- the GUI still works, just with
    # the stock vista theme.
    # Theme preference is persisted in gui_state.json by the
    # Settings dialog. Fall back to the module-level GUI_THEME
    # default if it's unset (first-ever launch) or if ttkthemes
    # rejects the saved value (e.g. operator hand-edited
    # gui_state.json with a typo'd theme name).
    saved_theme = _load_gui_state().get("theme", GUI_THEME)
    if _HAVE_TTKTHEMES:
        try:
            root = ThemedTk(theme=saved_theme)
        except Exception as e:
            print(f"[gui] ThemedTk('{saved_theme}') failed ({e}); "
                  f"trying default '{GUI_THEME}'", file=sys.stderr)
            try:
                root = ThemedTk(theme=GUI_THEME)
            except Exception as e2:
                print(f"[gui] fallback ThemedTk('{GUI_THEME}') also "
                      f"failed ({e2}); using plain tk.Tk",
                      file=sys.stderr)
                root = tk.Tk()
    else:
        root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
