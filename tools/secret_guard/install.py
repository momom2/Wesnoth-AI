#!/usr/bin/env python3
"""Install secret_guard for Claude Code (see README.md).

    python tools/secret_guard/install.py                   # copy, check, print the settings
    python tools/secret_guard/install.py --write-settings  # also add them to ~/.claude/settings.json
    python tools/secret_guard/install.py --remove-settings # take them out again

Copies secret_guard.py and guard_shell.sh to ~/.claude/secret_guard/, runs
the installed copy on a random sentinel credential (the shell prefix and the
hook), and prints the user settings that enable it. --write-settings merges
them into ~/.claude/settings.json after saving a backup next to it;
--remove-settings removes exactly what it added.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

SOURCE = Path(__file__).resolve().parent
FILES = ("secret_guard.py", "guard_shell.sh")
# The tools whose results the hook rewrites. Bash output is redacted by the
# shell prefix before Claude Code reads it.
HOOK_TOOLS = "^(Read|Grep|PowerShell|WebFetch|WebSearch|Monitor|BashOutput|TaskOutput|mcp__.*)$"
MARK = "secret_guard.py"          # in the hook's arguments
PREFIX_NAME = "guard_shell.sh"    # at the end of the shell prefix


def target_dir() -> Path:
    return Path(os.environ.get("SECRET_GUARD_HOME") or Path.home()) / ".claude" / "secret_guard"


def settings_path() -> Path:
    return Path(os.environ.get("SECRET_GUARD_HOME") or Path.home()) / ".claude" / "settings.json"


def install(target: Path) -> None:
    """Copy the guard, the shell script with LF line endings."""
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        (target / name).write_bytes((SOURCE / name).read_bytes().replace(b"\r\n", b"\n"))


def guard_settings(target: Path) -> dict:
    hook = {"type": "command", "command": "python",
            "args": ["-I", "-S", (target / "secret_guard.py").as_posix(), "hook"], "timeout": 30}
    return {
        "env": {"CLAUDE_CODE_SHELL_PREFIX": (target / "guard_shell.sh").as_posix()},
        "hooks": {
            "PostToolUse": [{"matcher": HOOK_TOOLS, "hooks": [hook]}],
            "PostToolUseFailure": [{"matcher": "*", "hooks": [hook]}],
        },
    }


def check(target: Path) -> List[str]:
    """What went wrong running the installed copy on a sentinel credential
    (empty when the prefix and the hook both redact it)."""
    sentinel = "SGcheck" + os.urandom(12).hex()
    env = dict(os.environ, SECRET_GUARD_CHECK_TOKEN=sentinel)
    problems = []
    bash = shutil.which("bash")
    if bash is None:
        return ["no bash on PATH"]
    run = subprocess.run([bash, (target / "guard_shell.sh").as_posix(),
                          f"echo out {sentinel}; echo err {sentinel} >&2; exit 7"],
                         env=env, capture_output=True, timeout=60)
    out, err = run.stdout.decode(errors="replace"), run.stderr.decode(errors="replace")
    if sentinel in out or sentinel in err:
        problems.append("the shell prefix let the sentinel through")
    if "out <redacted>" not in out or "err <redacted>" not in err:
        problems.append(f"the shell prefix's output is not the expected one: {out.strip()!r} / {err.strip()!r}")
    if run.returncode != 7:
        problems.append(f"the shell prefix returned {run.returncode}, not the command's 7")
    event = {"hook_event_name": "PostToolUse", "tool_name": "Read",
             "tool_response": {"file": {"content": f"a {sentinel} b"}}}
    hook = subprocess.run([sys.executable, "-I", "-S", str(target / "secret_guard.py"), "hook"],
                          input=json.dumps(event).encode(), env=env, capture_output=True, timeout=60)
    text = hook.stdout.decode(errors="replace")
    if sentinel in text or "<redacted>" not in text:
        problems.append("the hook did not redact the sentinel")
    return problems


def _is_ours(group: dict) -> bool:
    return any(MARK in " ".join([h.get("command", "")] + list(h.get("args", []))) for h in group.get("hooks", []))


def write_settings(path: Path, target: Path) -> Optional[Path]:
    """Merge the guard's settings into `path`; returns the backup of the
    previous file, if there was one."""
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    backup = None
    if path.exists():
        backup = path.with_name(f"settings.backup-{time.strftime('%Y%m%d-%H%M%S')}.json")
        shutil.copy2(path, backup)
    wanted = guard_settings(target)
    merged = remove_from(current)
    merged.setdefault("env", {}).update(wanted["env"])
    hooks = merged.setdefault("hooks", {})
    for event, groups in wanted["hooks"].items():
        hooks.setdefault(event, []).extend(groups)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return backup


def remove_from(settings: dict) -> dict:
    """`settings` without the guard's prefix and hook groups."""
    out = json.loads(json.dumps(settings))
    env = out.get("env", {})
    if env.get("CLAUDE_CODE_SHELL_PREFIX", "").endswith(PREFIX_NAME):
        env.pop("CLAUDE_CODE_SHELL_PREFIX")
    if "env" in out and not env:
        out.pop("env")
    for event in list(out.get("hooks", {})):
        kept = [g for g in out["hooks"][event] if not _is_ours(g)]
        if kept:
            out["hooks"][event] = kept
        else:
            out["hooks"].pop(event)
    if "hooks" in out and not out["hooks"]:
        out.pop("hooks")
    return out


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--write-settings", action="store_true")
    group.add_argument("--remove-settings", action="store_true")
    args = ap.parse_args(argv)
    path, target = settings_path(), target_dir()
    if args.remove_settings:
        if path.exists():
            path.write_text(json.dumps(remove_from(json.loads(path.read_text(encoding="utf-8"))), indent=2) + "\n",
                            encoding="utf-8")
        print(f"secret_guard: removed from {path}")
        return 0
    install(target)
    problems = check(target)
    if problems:
        print("secret_guard: the installed copy failed its check; settings unchanged:")
        for p in problems:
            print("  " + p)
        return 1
    print(f"secret_guard: installed in {target} and checked")
    if args.write_settings:
        backup = write_settings(path, target)
        saved = f" (previous file saved as {backup.name})" if backup else ""
        print(f"secret_guard: added to {path}{saved}")
    else:
        print(f"Add to {path}:")
        print(json.dumps(guard_settings(target), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
