"""The window's results in one table, for the user's return: every match's
reading against the reference, and every round's adoption.

Read-only on C:/Users/momom2/wesnoth_runs (the matches' result.json and
fit.json files, rounds_local/runner.log); Idle priority, no GPU. Prints
Markdown, or writes it with --out.

    python scripts/window_20261029/return_table.py [--out <md>]

Sections:
1. Argmax (raw:t0+eo-1.5 on both sides): the final matches (seed base
   900,000; paired on their seeds), the confirmations (980,000) and the
   extras, from each match_local result.json.
2. T=1 against the reference at argmax: the tails (950,000), the claims
   (rounds_local/round_KK/claim.fit.json) and D0 (own2 at T=1).
3. The rounds: each round's adoption and its confirmed reading, the last
   line runner.log logged for it.

Every reading is the PURE Elo of the candidate with the reference at 0 and
its standard error. Matches served at fp16 say so: the reference player is
pinned at fp32 serving, so a strength claim on them is not yet the
standard one.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception:
    pass

RUNS = Path("C:/Users/momom2/wesnoth_runs")
ROUNDS = RUNS / "rounds_local"
PURE = "PURE (decisive only, primary)"
ROUND_LINE = re.compile(r"^(\S+ \S+) round (\d+): adopt=(\w+) next_eps=\S+?(?: along (\w+))?: (.*)$")


def round_of(path: str) -> str:
    """'round K' from a checkpoint path under rounds_local, else its name."""
    m = re.search(r"round_(\d+)", path or "")
    return f"round {int(m.group(1))}" if m else Path(path or "?").name


def matches(globs: List[str]) -> List[Dict]:
    """Each match_local result.json under RUNS matching `globs`, oldest first."""
    rows = []
    for g in globs:
        for res in sorted(RUNS.glob(f"{g}/result.json")):
            try:
                r = json.loads(res.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                rows.append({"dir": res.parent.name, "error": str(e)})
                continue
            rows.append({"dir": res.parent.name, "policy": round_of(r.get("checkpoint", "")),
                         "label": r.get("candidate"), "elo": r.get("elo"), "se": r.get("se"),
                         "decisive": r.get("decisive"), "wanted": r.get("games_wanted"),
                         "complete": r.get("complete"), "seed": r.get("seed_base"),
                         "fp16": r.get("infer_fp16"), "time": r.get("time"),
                         "procedures": r.get("procedures")})
    return sorted(rows, key=lambda x: x.get("time") or "")


def fit(path: Path, label: str) -> Optional[Dict]:
    """A label's PURE reading from an elo_collect --save-json file."""
    try:
        t = json.loads(path.read_text(encoding="utf-8"))
        cell = t["tables"][PURE][label]
        return {"elo": cell["elo"], "se": cell["se"], "n": t.get("n_games")}
    except (OSError, ValueError, KeyError):
        return None


def match_table(rows: List[Dict]) -> List[str]:
    out = ["| Match | Policy | Elo | Decisive | Seed base | Served | Ended |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            out.append(f"| {r['dir']} | unreadable: {r['error']} | | | | | |")
            continue
        state = "" if r["complete"] else " (incomplete)"
        out.append(f"| {r['dir']} | {r['policy']} | {r['elo']:+.1f} ± {r['se']:.1f}{state} | "
                   f"{r['decisive']} of {r['wanted']} | {r['seed']:,} | {'fp16' if r['fp16'] else 'fp32'} | "
                   f"{r['time']} |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    lines = ["# The window's results against the reference (`own2` at argmax, end_turn offset -1.5)", ""]

    lines += ["## Argmax: both sides raw:t0+eo-1.5", ""]
    lines += match_table(matches(["final_match*", "confirm_match*"]))
    lines += ["", "The final matches share seed base 900,000, so they pair game for game; the confirmations "
              "use fresh seeds from 980,000.", ""]

    lines += ["## T=1 against the reference at argmax", ""]
    lines += match_table(matches(["tail_match*"]))
    lines += ["", "| Claim | Policy | Elo | Decisive |", "|---|---|---|---|"]
    d0 = fit(RUNS / "selfplay_t1_local" / "d0.fit.json", "own2t1")
    if d0:
        lines.append(f"| D0 | `own2` at T=1 | {d0['elo']:+.1f} ± {d0['se']:.1f} | {d0['n']} |")
    for f in sorted(ROUNDS.glob("round_*/claim.fit.json")):
        k = int(f.parent.name.split("_")[1])
        c = fit(f, f"pi{k}t1")
        if c:
            lines.append(f"| round {k}'s claim | round {k} at T=1 | {c['elo']:+.1f} ± {c['se']:.1f} | {c['n']} |")
    lines.append("")

    # each start of rounds_local logs every earlier round's decision again, so the first line is the adoption's
    lines += ["## The rounds", "", "| Round | Adopted | Along | Reading | First logged |", "|---|---|---|---|---|"]
    first: Dict[int, re.Match] = {}
    for line in (ROUNDS / "runner.log").read_text(encoding="utf-8").splitlines():
        m = ROUND_LINE.match(line)
        if m:
            first.setdefault(int(m.group(2)), m)
    for k in sorted(first):
        m = first[k]
        lines.append(f"| {k} | {m.group(3)} | {m.group(4) or ''} | {m.group(5)} | {m.group(1)} |")
    lines.append("")

    text = "\n".join(lines) + "\n"
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(text, end="")


if __name__ == "__main__":
    main()
