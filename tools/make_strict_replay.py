"""Generate a fresh strict-sync save Wesnoth can play out as AI vs AI.

Strategy: take an existing strict-sync replay (which already has
oos_debug=yes set at the top level), strip its [replay] command
stream, and save as a new .gz. Wesnoth's --load will treat this as
a fresh game, run the AIs through it, and produce a new replay
enriched with [mp_checkup] data.

Optionally swap the scenario_id and the [side] controller blocks.

Usage:
    python tools/make_strict_replay.py <template.gz> <out.gz> [scenario_id]

The output goes to ~/Documents/My Games/Wesnoth1.18/saves/ (so
`wesnoth.exe --load <out_basename>` finds it).

Dependencies: stdlib (gzip, re).
Dependents: standalone CLI; plus you launch wesnoth.exe afterward.
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
from pathlib import Path
from typing import List


def _strip_replay(text: str) -> str:
    """Replace `[replay]...[/replay]` with empty `[replay]\\n[/replay]`."""
    return re.sub(
        r"\[replay\][\s\S]*?\[/replay\]",
        "[replay]\n[/replay]",
        text,
        count=1,
    )


def _force_ai_controllers(text: str) -> str:
    """Set every [side]'s controller to 'ai' so a CLI --load runs
    autonomously."""
    return re.sub(
        r'controller="(?:human|network|empty|null)"',
        'controller="ai"',
        text,
    )


def _set_oos_debug(text: str) -> str:
    """Ensure top-level `oos_debug=yes` (idempotent)."""
    if re.search(r'^\s*oos_debug\s*=\s*"?yes"?\s*$', text, re.M):
        return text
    if re.search(r'^\s*oos_debug\s*=', text, re.M):
        return re.sub(
            r'^(\s*oos_debug\s*=\s*)"?[^"\n]*"?$',
            r'\1yes',
            text,
            count=1,
            flags=re.M,
        )
    # Inject after `version=` near the top.
    return re.sub(
        r'^(version="[^"]+"\s*)$',
        r'\1\noos_debug=yes',
        text,
        count=1,
        flags=re.M,
    )


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("template", type=Path,
                    help="Existing strict-sync .gz replay to use as template.")
    ap.add_argument("out_name", type=str,
                    help="Output filename (without .gz). Goes to "
                         "~/Documents/My Games/Wesnoth1.18/saves/.")
    args = ap.parse_args(argv[1:])

    if not args.template.exists():
        print(f"template not found: {args.template}", file=sys.stderr)
        return 1

    with gzip.open(args.template, "rt", encoding="utf-8",
                   errors="replace") as f:
        text = f.read()

    text = _set_oos_debug(text)
    text = _force_ai_controllers(text)
    text = _strip_replay(text)

    saves_dir = Path(os.path.expanduser(
        "~/Documents/My Games/Wesnoth1.18/saves"))
    out_path = saves_dir / f"{args.out_name}.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
    print(f"now run: wesnoth.exe --load {args.out_name}.gz --exit-at-end")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
