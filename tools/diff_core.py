#!/usr/bin/env python3
"""The Rust-owned game state against the Python applier, over replays.

Every command of each replay goes to the Python state through
`replay_dataset._apply_command` and to a `wesnoth_ai.game_core.CoreState`
built from a copy of the initial state; after every command (or every
`--every` commands) the two states are compared over their modeled
content (`game_core.state_differences`). A difference is a divergence,
reported with the command's index and kind and the path the core took
("rust" or the Python fallback). A replay on which the core raises or
panics (a Rust panic reaches Python as pyo3's PanicException, a
BaseException) is listed as that replay's divergence and the sweep goes
on. Certification of the port's step kernels (docs/rust_port_plan.md
phase 4).

    python tools/diff_core.py replays_dataset_imitation/*.json.gz --limit 200
    python tools/diff_core.py DIR --every 1 --stop-on-first

Prints one summary line: `diff_core: N replays, M clean, K with
divergences; commands rust=A python=B`, then the divergences.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("diff_core")


def is_rust_panic(exc: BaseException) -> bool:
    """A Rust panic reaches Python as pyo3's `PanicException`, which
    derives from BaseException precisely so that `except Exception`
    does not catch it. It has no importable home (`pyo3_runtime` is
    not a module), so it is recognised by its name."""
    return type(exc).__name__ == "PanicException"


def _describe_failure(exc: BaseException) -> str:
    """A replay's failure as one listed line, or the exception re-raised
    when it is not the replay's (KeyboardInterrupt, SystemExit)."""
    if is_rust_panic(exc):
        return f"panicked {exc!r}"
    if isinstance(exc, Exception):
        return f"raised {exc!r}"
    raise exc


def diff_core(gz_path: Path, *, every: int = 1, stop_on_first: bool = True,
              counts: Optional[Counter] = None) -> List[str]:
    from tools.replay_dataset import _apply_command, _build_initial_gamestate, _setup_scenario_events
    from wesnoth_ai.game_core import CoreState, state_differences
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))
    cs = CoreState.from_state(copy.deepcopy(gs))
    out: List[str] = []
    for idx, cmd in enumerate(data.get("commands", [])):
        kind = cmd[0] if cmd else "?"
        _apply_command(gs, list(cmd))
        try:
            path = cs.apply_command(list(cmd))
        except BaseException as e:  # noqa: BLE001 - a panic is a divergence too
            out.append(f"{gz_path.name}#{idx} {kind}: core {_describe_failure(e)}")
            break
        if counts is not None:
            counts[(kind, path)] += 1
        if idx % every == 0 or kind in ("init_side", "attack"):
            diffs = state_differences(gs, cs.to_state(), stash=False)
            if diffs:
                out.append(f"{gz_path.name}#{idx} {kind} via {path}: " + " | ".join(diffs[:4]))
                if stop_on_first:
                    break
    return out


def _walk(inputs: List[Path]) -> List[Path]:
    files: List[Path] = []
    for p in inputs:
        if p.is_dir():
            files.extend(sorted(p.glob("*.json.gz")))
        else:
            files.append(p)
    return files


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--every", type=int, default=1, help="compare every N commands (init_side and attack always)")
    ap.add_argument("--stop-on-first", action="store_true")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING),
                        format="%(levelname)s %(name)s: %(message)s")
    files = _walk(args.inputs)
    if args.limit:
        files = files[:args.limit]
    counts: Counter = Counter()
    clean = 0
    divergences: List[str] = []
    for gz in files:
        try:
            d = diff_core(gz, every=args.every, stop_on_first=args.stop_on_first, counts=counts)
        except BaseException as e:  # noqa: BLE001 - one bad file, panic included, must not end the sweep
            d = [f"{gz.name}: harness {_describe_failure(e)}"]
        if d:
            divergences.extend(d)
        else:
            clean += 1
    rust = sum(v for (k, p), v in counts.items() if p == "rust")
    py = sum(v for (k, p), v in counts.items() if p == "python")
    print(f"diff_core: {len(files)} replays, {clean} clean, {len(files) - clean} with divergences; "
          f"commands rust={rust} python={py}")
    by_kind = Counter()
    for (k, p), v in counts.items():
        by_kind[f"{k}:{p}"] += v
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))
    for line in divergences[:200]:
        print("  " + line)
    return 0 if len(files) == clean else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
