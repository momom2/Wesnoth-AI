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
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import tkinter.scrolledtext as scrolledtext
from pathlib import Path
from typing import Callable, List, Optional


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
# This file lives at <PROJECT>/cluster/gui.pyw -- compute the project
# root so the GUI works no matter what cwd the user launches from.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUSTER_DIR  = PROJECT_ROOT / "cluster"

POWERSHELL = "powershell.exe"
PYTHON     = sys.executable  # the same interpreter running the GUI

SCRIPT_SYNC          = CLUSTER_DIR / "sync.ps1"
SCRIPT_PULL          = CLUSTER_DIR / "pull_checkpoint.ps1"
SCRIPT_SELFPLAY      = PROJECT_ROOT / "run_self_play.ps1"
EVAL_SCRIPT          = PROJECT_ROOT / "tools" / "eval_vs_builtin.py"

REMOTE_HOST = "mesogip_outside"
REMOTE_PATH = "~/wesnoth-ai"


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

    OUTPUT_POLL_MS = 80   # how often the GUI thread drains the stdout queue

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Wesnoth AI -- Cluster GUI")
        # Reasonable default size; user can resize.
        root.geometry("780x620")
        root.minsize(620, 480)

        self._password    = tk.StringVar(value="")
        self._proc:        Optional[subprocess.Popen] = None
        self._proc_lock   = threading.Lock()
        self._askpass:    Optional[AskpassFiles] = None
        self._stdout_q:   queue.Queue            = queue.Queue()
        self._epoch_var   = tk.StringVar(value="latest")  # for pull-checkpoint
        self._build_widgets()
        # Periodic pump from worker threads' queue -> output panel.
        root.after(self.OUTPUT_POLL_MS, self._drain_output)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- widgets --------------------------------------------------------

    def _build_widgets(self):
        pad = {"padx": 8, "pady": 4}

        # Password row
        pw_frame = tk.LabelFrame(self.root, text="ssh password (kept in memory only)")
        pw_frame.pack(fill="x", **pad)
        self._pw_entry = tk.Entry(pw_frame, textvariable=self._password,
                                  show="*", width=32)
        self._pw_entry.pack(side="left", padx=8, pady=6)
        tk.Button(pw_frame, text="Forget",
                  command=self._forget_password).pack(side="left", padx=4)
        tk.Label(pw_frame, text="(deleted on window close, too)",
                 fg="gray").pack(side="left", padx=8)

        # Cluster ops
        cluster_frame = tk.LabelFrame(self.root, text="Cluster (uses password)")
        cluster_frame.pack(fill="x", **pad)
        row1 = tk.Frame(cluster_frame); row1.pack(fill="x", padx=4, pady=4)
        tk.Button(row1, text="Status",
                  width=14, command=self._op_status).pack(side="left", padx=4)
        tk.Button(row1, text="Sync code",
                  width=14, command=self._op_sync).pack(side="left", padx=4)
        tk.Button(row1, text="Sync + Restart",
                  width=14, command=self._op_sync_restart).pack(side="left", padx=4)
        row2 = tk.Frame(cluster_frame); row2.pack(fill="x", padx=4, pady=4)
        tk.Label(row2, text="Pull checkpoint  epoch:").pack(side="left", padx=4)
        tk.Entry(row2, textvariable=self._epoch_var, width=10).pack(side="left", padx=4)
        tk.Label(row2, text="(`latest`, `rolling`, or N)",
                 fg="gray").pack(side="left", padx=4)
        tk.Button(row2, text="Pull",
                  width=14, command=self._op_pull).pack(side="left", padx=4)

        # Local ops
        local_frame = tk.LabelFrame(self.root, text="Local (no password needed)")
        local_frame.pack(fill="x", **pad)
        row3 = tk.Frame(local_frame); row3.pack(fill="x", padx=4, pady=4)
        tk.Button(row3, text="Run self-play",
                  width=14, command=self._op_selfplay).pack(side="left", padx=4)
        tk.Button(row3, text="Run eval ...",
                  width=14, command=self._op_eval_dialog).pack(side="left", padx=4)
        tk.Label(row3, text="(self-play picks newest local *.pt)",
                 fg="gray").pack(side="left", padx=4)

        # Output panel
        out_frame = tk.LabelFrame(self.root, text="Output")
        out_frame.pack(fill="both", expand=True, **pad)
        self._output = scrolledtext.ScrolledText(out_frame, wrap="word",
                                                 height=15, font=("Consolas", 9))
        self._output.pack(fill="both", expand=True, padx=4, pady=4)
        self._output.configure(state="disabled")

        # Bottom bar: cancel + clear + status
        bar = tk.Frame(self.root); bar.pack(fill="x", **pad)
        self._cancel_btn = tk.Button(
            bar, text="Cancel running op",
            command=self._cancel_op, state="disabled",
        )
        self._cancel_btn.pack(side="left", padx=4)
        tk.Button(bar, text="Clear output",
                  command=self._clear_output).pack(side="left", padx=4)
        self._status_var = tk.StringVar(value="Idle.")
        tk.Label(bar, textvariable=self._status_var, anchor="w",
                 fg="gray").pack(side="left", fill="x", expand=True, padx=8)

    # -- output stream --------------------------------------------------

    def _log(self, line: str, tag: str = "info") -> None:
        """Append a line to the output panel from the main thread."""
        self._output.configure(state="normal")
        self._output.insert("end", line if line.endswith("\n") else line + "\n")
        self._output.see("end")
        self._output.configure(state="disabled")

    def _drain_output(self) -> None:
        """Pull queued lines from worker threads onto the output panel.
        Re-arms via tk.after; runs forever."""
        try:
            while True:
                tag, line = self._stdout_q.get_nowait()
                self._log(line.rstrip("\r\n"), tag)
        except queue.Empty:
            pass
        self.root.after(self.OUTPUT_POLL_MS, self._drain_output)

    def _clear_output(self) -> None:
        self._output.configure(state="normal")
        self._output.delete("1.0", "end")
        self._output.configure(state="disabled")

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

    def _spawn(self,
               argv: List[str],
               *,
               needs_password: bool,
               label: str,
               cwd: Optional[Path] = None) -> None:
        """Spawn a subprocess in a background thread, stream output."""
        if not self._begin_op(label):
            return

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
        self._spawn(
            ["ssh", REMOTE_HOST,
             f"cd {REMOTE_PATH} && bash cluster/run.sh status"],
            needs_password=True, label="cluster status",
        )

    def _op_sync(self) -> None:
        self._spawn(self._ps(SCRIPT_SYNC),
                    needs_password=True, label="sync code")

    def _op_sync_restart(self) -> None:
        self._spawn(self._ps(SCRIPT_SYNC, "-Restart"),
                    needs_password=True, label="sync code + restart")

    def _op_pull(self) -> None:
        spec = self._epoch_var.get().strip().lower()
        args: List[str] = []
        if spec in ("", "latest"):
            pass                       # default = latest per-epoch snapshot
        elif spec == "rolling":
            args = ["-Rolling"]
        elif spec == "list":
            args = ["-List"]
        else:
            try:
                n = int(spec)
            except ValueError:
                messagebox.showerror(
                    "Bad epoch",
                    f"`{spec}` is not a valid epoch. Use a number, "
                    f"`latest`, `rolling`, or `list`.")
                return
            args = ["-Epoch", str(n)]
        self._spawn(self._ps(SCRIPT_PULL, *args),
                    needs_password=True, label=f"pull checkpoint ({spec or 'latest'})")

    # -- local operations -----------------------------------------------

    def _op_selfplay(self) -> None:
        self._spawn(self._ps(SCRIPT_SELFPLAY),
                    needs_password=False, label="self-play")

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

        maps_var     = tk.StringVar(value="caves den")
        pairs_var    = tk.StringVar(value="cross")
        no_swap_var  = tk.BooleanVar(value=True)
        parallel_var = tk.IntVar(value=4)
        max_actions_var = tk.IntVar(value=500)

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
            argv = [
                PYTHON, str(EVAL_SCRIPT),
                "--checkpoint", path,
                "--pairs", pairs_var.get(),
                "--parallel", str(parallel_var.get()),
                "--max-actions", str(max_actions_var.get()),
            ]
            maps = maps_var.get().split()
            if maps:
                argv += ["--maps", *maps]
            if no_swap_var.get():
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
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
