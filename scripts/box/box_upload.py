#!/usr/bin/env python3
"""Keep a box run's records in step with the model host (HF).

    python box_upload.py --out DIR --hf-dir PREFIX [--spec FILE] [--final] [--budget-s N]
    python box_upload.py --out DIR --hf-dir PREFIX --clear NAME...
    python box_upload.py --out DIR --hf-dir PREFIX --restore NAME...

An upload round sends, each under its own name in PREFIX:
  1. every regular file directly in --out and every `extra` file of the
     spec, smallest first;
  2. every `dir NAME PATH` of the spec as the tarball NAME.tar.gz, holding
     PATH's files under PATH's own name (files ending in .tmp, still being
     written, stay out);
  3. every `hold NAME DEP...` file of the spec, once each DEP exists and
     has landed in its current version, so that a marker never reaches HF
     before the data it vouches for;
  4. upload.log;
  5. with --final, ALL_DONE, last.
A file or directory whose size and modification time are those of the
version that last landed is not sent again (the landed versions are kept
in --out/tmp/uploaded.json). What goes up is a copy taken at the start of
its upload, so a file rewritten meanwhile goes up whole. Each file gets
three attempts, each under a timeout that grows with its size; a round
stops starting uploads once its --budget-s is spent, keeping time for
upload.log and ALL_DONE. Every outcome is a line in upload.log, and a
failure is named by the exception's type only: messages can quote URLs
and headers.

--clear deletes each NAME from PREFIX when it is there (a previous entry's
ALL_DONE and FAILED) and forgets it as landed. --restore fetches each NAME
absent from --out and records it as landed; a NAME that is not on HF stays
absent, and any other failure exits 1: with HF unreachable, starting over
would overwrite what HF holds.

The spec (written by boxlib.sh) holds one tab-separated entry per line:
    extra PATH   |   dir NAME PATH   |   hold NAME DEP...   |   skip NAME
A DEP is a name as sent: a file's name, an extra's base name, NAME.tar.gz
for a directory. A skipped NAME stays on the box.

Exit status: 0 when everything the call tried landed, 1 otherwise.
Needs huggingface_hub and HF_TOKEN in the environment.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = "momom2/wesnoth-model-checkpoints"
ALL_DONE = "ALL_DONE"
LOG_NAME = "upload.log"
STATE = Path("tmp") / "uploaded.json"
SNAPSHOTS = Path("tmp") / "upload_snapshots"   # one directory per round's process
STALE_SNAPSHOT_S = 3 * 3600                    # older than any round: left by a killed one
ABSENT_ERRORS = {"EntryNotFoundError", "RemoteEntryNotFoundError"}


@dataclass
class Timing:
    """How long a round may wait on the network. An attempt on a file of N
    bytes gets `base_s + N / min_rate` seconds (180 MB: about 14 minutes),
    less when the round's budget runs out."""
    base_s: float = 120.0
    min_rate: float = 250_000.0     # bytes/s: a quarter of the slowest uplink a box may keep
    reserve_s: float = 90.0         # kept at the end of a round for upload.log and ALL_DONE
    min_attempt_s: float = 20.0     # an attempt shorter than this is not started
    retry_pause_s: float = 20.0
    attempts: int = 3


@dataclass
class Spec:
    extras: list[str] = field(default_factory=list)
    dirs: dict[str, str] = field(default_factory=dict)       # name as sent -> directory
    holds: dict[str, list[str]] = field(default_factory=dict)
    skips: set[str] = field(default_factory=set)


def read_spec(path: str | None) -> Spec:
    spec = Spec()
    if not path or not os.path.exists(path):
        return spec
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            kind, *fields = line.rstrip("\n").split("\t")
            if kind == "extra" and len(fields) == 1 and fields[0] not in spec.extras:
                spec.extras.append(fields[0])
            elif kind == "dir" and len(fields) == 2:
                spec.dirs[fields[0] + ".tar.gz"] = fields[1]
            elif kind == "hold" and len(fields) >= 2:
                deps = spec.holds.setdefault(fields[0], [])
                deps.extend(d for d in fields[1:] if d not in deps)
            elif kind == "skip" and len(fields) == 1:
                spec.skips.add(fields[0])
    return spec


def load_state(out: Path, hf_dir: str) -> dict:
    """name -> the key of the version on HF, for this --hf-dir."""
    try:
        state = json.loads((out / STATE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return state.get("files", {}) if state.get("hf_dir") == hf_dir else {}


def save_state(out: Path, hf_dir: str, landed: dict) -> None:
    path = out / STATE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"hf_dir": hf_dir, "files": landed}), encoding="utf-8")
    os.replace(tmp, path)


def file_key(path: str) -> list:
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns]


def dir_files(path: str) -> list[tuple[str, str]]:
    """(absolute path, path relative to `path`) of the finished files under
    `path`, sorted."""
    found = []
    for root, _dirs, names in os.walk(path):
        for name in names:
            if name.endswith(".tmp"):
                continue
            full = os.path.join(root, name)
            found.append((full, os.path.relpath(full, path).replace(os.sep, "/")))
    return sorted(found, key=lambda item: item[1])


def dir_key(path: str) -> list:
    count = total = newest = 0
    for full, _rel in dir_files(path):
        try:
            st = os.stat(full)
        except OSError:
            continue
        count, total, newest = count + 1, total + st.st_size, max(newest, st.st_mtime_ns)
    return [count, total, newest]


def is_absent(exc: BaseException) -> bool:
    """The file is not on HF, as opposed to HF being unreachable. By the
    exact type: huggingface_hub's LocalEntryNotFoundError, raised when the
    Hub cannot be reached and the file is not cached, also derives from
    EntryNotFoundError."""
    return type(exc).__name__ in ABSENT_ERRORS


class Round:
    """One call's work against the model host, logged to --out/upload.log."""

    def __init__(self, out: Path, hf_dir: str, api, *, spec: Spec | None = None,
                 budget_s: float = 1500.0, timing: Timing | None = None, repo: str = REPO,
                 clock=time.monotonic, sleep=time.sleep):
        self.out, self.hf_dir, self.api, self.repo = out, hf_dir, api, repo
        self.spec = spec or Spec()
        self.timing = timing or Timing()
        self.clock, self.sleep = clock, sleep
        self.deadline = clock() + budget_s
        self.snapshots = out / SNAPSHOTS / str(os.getpid())
        self.landed = load_state(out, hf_dir)
        self.current: dict[str, list] = {}
        self.failed: list[str] = []
        self.sent = self.unchanged = 0

    # ---- logging and state
    def log(self, line: str) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.out / LOG_NAME, "a", encoding="utf-8") as fh:
            fh.write(f"{stamp} {line}\n")

    def mark_landed(self, name: str, key: list) -> None:
        self.landed[name] = key
        save_state(self.out, self.hf_dir, self.landed)

    # ---- one file
    def attempt(self, path: str, name: str, timeout_s: float) -> str | None:
        """None when the file landed within `timeout_s`, else what stopped it.
        The upload runs in a daemon thread: one hung on a socket is left
        behind, and the process ends with os._exit."""
        outcome: dict = {}

        def run():
            try:
                self.api.upload_file(path_or_fileobj=path, path_in_repo=f"{self.hf_dir}/{name}",
                                     repo_id=self.repo)
                outcome["error"] = None
            except BaseException as exc:  # noqa: BLE001 -- the type name only
                outcome["error"] = type(exc).__name__

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(timeout_s)
        return outcome.get("error", f"no answer in {timeout_s:.0f} s")

    def send(self, path: str, name: str, reserve_s: float) -> bool:
        """Up to `timing.attempts` attempts; True when `path` landed as `name`."""
        t = self.timing
        size = os.path.getsize(path)
        wanted = t.base_s + size / t.min_rate
        for i in range(1, t.attempts + 1):
            left = self.deadline - reserve_s - self.clock()
            if left < t.min_attempt_s:
                self.log(f"{name} {size} not sent: the round's time is spent")
                return False
            start = self.clock()
            error = self.attempt(path, name, min(wanted, left))
            if error is None:
                took = max(self.clock() - start, 1e-3)
                self.log(f"{name} {size} ok {took:.0f} s {size / took / 1e3:.0f} kB/s")
                return True
            self.log(f"{name} {size} attempt {i} failed: {error}")
            if error.startswith("no answer"):
                return False          # a hung connection: this round leaves the file
            if i < t.attempts:
                self.sleep(t.retry_pause_s)
        return False

    def prepare_snapshots(self) -> None:
        """This round's copy directory, and none left by killed rounds."""
        parent = self.out / SNAPSHOTS
        parent.mkdir(parents=True, exist_ok=True)
        for old in parent.iterdir():
            try:
                if time.time() - old.stat().st_mtime > STALE_SNAPSHOT_S:
                    shutil.rmtree(old, ignore_errors=True)
            except OSError:
                pass
        self.snapshots.mkdir(parents=True, exist_ok=True)

    def upload_copy(self, name: str, make_copy, reserve_s: float) -> bool:
        """Send what `make_copy(destination)` writes as `name`; True when it landed."""
        copy = self.snapshots / name
        try:
            make_copy(copy)
        except OSError as exc:
            self.log(f"{name} not sent: its copy failed ({type(exc).__name__})")
            return False
        try:
            return self.send(str(copy), name, reserve_s)
        finally:
            try:
                copy.unlink()
            except OSError:
                pass                  # still open in an abandoned upload thread

    def send_if_changed(self, name: str, key: list, make_copy) -> None:
        """Send what `make_copy(destination)` writes, unless `key` landed already."""
        self.current[name] = key
        if self.landed.get(name) == key:
            self.unchanged += 1
            return
        if self.upload_copy(name, make_copy, self.timing.reserve_s):
            self.sent += 1
            self.mark_landed(name, key)
        else:
            self.failed.append(name)

    # ---- what a round holds
    def files(self) -> list[tuple[str, str, list]]:
        """(name, path, key) of the plain files to send, smallest first."""
        held_back = {ALL_DONE, LOG_NAME} | set(self.spec.holds) | self.spec.skips
        found = {}
        for entry in os.scandir(self.out):
            if entry.is_file() and entry.name not in held_back and not entry.name.endswith(".tmp"):
                found[entry.name] = entry.path
        for extra in self.spec.extras:
            name = os.path.basename(extra)
            if name in found or name in held_back or not os.path.isfile(extra):
                continue
            found[name] = extra
        items = []
        for name, path in found.items():
            try:
                items.append((name, path, file_key(path)))
            except OSError:
                continue              # removed since the listing
        return sorted(items, key=lambda item: (item[2][0], item[0]))

    def send_file(self, name: str, path: str, key: list) -> None:
        self.send_if_changed(name, key, lambda copy: shutil.copyfile(path, copy))

    def send_dir(self, name: str, path: str) -> None:
        if name in self.spec.skips or not os.path.isdir(path):
            return
        key = dir_key(path)
        if key[0] == 0:
            return                    # nothing finished in it yet
        top = os.path.basename(os.path.normpath(path))

        def tar(copy: Path) -> None:
            with tarfile.open(copy, "w:gz", compresslevel=6) as tf:
                for full, rel in dir_files(path):
                    tf.add(full, arcname=f"{top}/{rel}")

        self.send_if_changed(name, key, tar)

    def vouched(self, deps: list[str]) -> str | None:
        """None when every dep exists and landed in its current version,
        else the first that did not."""
        for dep in deps:
            if dep not in self.current or self.landed.get(dep) != self.current[dep]:
                return dep
        return None

    def send_holds(self) -> None:
        waiting = {name: deps for name, deps in self.spec.holds.items()
                   if name not in self.spec.skips and os.path.isfile(self.out / name)}
        progress = True
        while waiting and progress:   # a hold may vouch for another hold
            progress = False
            for name, deps in list(waiting.items()):
                if self.vouched(deps) is None:
                    del waiting[name]
                    progress = True
                    self.send_file(name, str(self.out / name), file_key(self.out / name))
        for name, deps in waiting.items():
            self.log(f"{name} held back: {self.vouched(deps)} has not landed in its current version")

    def send_last(self, name: str) -> None:
        """upload.log, ALL_DONE: sent whatever their state, in the reserved time."""
        path = self.out / name
        if not path.is_file():
            return
        if self.upload_copy(name, lambda copy: shutil.copyfile(path, copy), 0.0):
            self.sent += 1
        else:
            self.failed.append(name)

    def run(self, final: bool = False) -> bool:
        """One round; True when everything it tried landed."""
        self.prepare_snapshots()
        try:
            for name, path, key in self.files():
                self.send_file(name, path, key)
            for name, path in sorted(self.spec.dirs.items()):
                self.send_dir(name, path)
            self.send_holds()
            self.log(f"{'final' if final else 'periodic'} round: {self.sent} sent, "
                     f"{self.unchanged} unchanged, {len(self.failed)} failed"
                     + (f" ({' '.join(self.failed)})" if self.failed else ""))
            self.send_last(LOG_NAME)
            if final:
                self.send_last(ALL_DONE)
        finally:
            shutil.rmtree(self.snapshots, ignore_errors=True)
        return not self.failed

    # ---- the previous entry's state
    def clear(self, names: list[str]) -> bool:
        """Delete each name from PREFIX when it is there; forget it as landed."""
        ok = True
        for name in names:
            path = f"{self.hf_dir}/{name}"
            try:
                if self.api.file_exists(self.repo, path):
                    self.api.delete_file(path, repo_id=self.repo)
                    self.log(f"{name} cleared from HF")
            except Exception as exc:  # noqa: BLE001 -- the type name only
                self.log(f"{name} not cleared from HF: {type(exc).__name__}")
                ok = False
            self.landed.pop(name, None)
        save_state(self.out, self.hf_dir, self.landed)
        return ok

    def restore(self, names: list[str]) -> bool:
        """Fetch each name absent from --out; False when HF could not answer
        for one of them."""
        failed = []
        for name in names:
            target = self.out / name
            if target.exists():
                continue
            for i in range(1, self.timing.attempts + 1):
                try:
                    source = self.api.hf_hub_download(self.repo, f"{self.hf_dir}/{name}")
                except Exception as exc:  # noqa: BLE001 -- the type name only
                    if is_absent(exc):
                        self.log(f"{name} not on HF: nothing to restore")
                        break
                    self.log(f"{name} restore attempt {i} failed: {type(exc).__name__}")
                    if i < self.timing.attempts:
                        self.sleep(self.timing.retry_pause_s)
                    continue
                tmp = target.with_name(target.name + ".tmp")
                shutil.copyfile(source, tmp)
                os.replace(tmp, target)
                self.mark_landed(name, file_key(target))
                self.log(f"{name} restored from HF: {target.stat().st_size} bytes")
                break
            else:
                failed.append(name)
        return not failed


def make_api(token: str):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sync a box run's records with HF (see the module docstring).")
    ap.add_argument("--out", required=True, type=Path, help="the run's records directory")
    ap.add_argument("--hf-dir", required=True, help="their folder on the model host")
    ap.add_argument("--spec", help="the spec file boxlib.sh writes")
    ap.add_argument("--final", action="store_true", help="send ALL_DONE last")
    ap.add_argument("--budget-s", type=float, default=1500.0, help="the round's time")
    ap.add_argument("--clear", nargs="+", metavar="NAME", help="delete these from HF")
    ap.add_argument("--restore", nargs="+", metavar="NAME", help="fetch these when absent here")
    args = ap.parse_args(argv)
    hf_dir = args.hf_dir.strip("/")
    args.out.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN", "").strip()
    probe = Round(args.out, hf_dir or "-", None)
    if not hf_dir or not token:
        probe.log("no upload: " + ("--hf-dir is empty" if not hf_dir else "HF_TOKEN is not set"))
        return 1
    try:
        api = make_api(token)
        work = Round(args.out, hf_dir, api, spec=read_spec(args.spec), budget_s=args.budget_s)
        if args.clear:
            return 0 if work.clear(args.clear) else 1
        if args.restore:
            return 0 if work.restore(args.restore) else 1
        return 0 if work.run(final=args.final) else 1
    except Exception as exc:  # noqa: BLE001 -- the type name only
        probe.log(f"stopped by an unexpected {type(exc).__name__}")
        return 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)                   # an upload thread hung on a socket must not hold the exit
