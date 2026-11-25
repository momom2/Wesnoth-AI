"""How often a recruit bounces in the simple loop's game records.

Reads game records (gzip JSON, one object a game), changes nothing and
imports nothing of the project. A record's `rejections` are (command
index, x, y): a recruit ordered onto a hex a hidden unit holds, whether
the engine then put it elsewhere (a `recruit` command follows at that
index, on another hex) or refused it (a decision marked `refused`).
"""
import collections
import gzip
import json
import sys
from pathlib import Path


def main() -> None:
    games = decisions = refused = rejections = recruits = 0
    games_with = 0
    redirected = 0
    cmd_kinds = collections.Counter()
    sample = []
    for rd in (Path(p) for p in sys.argv[1:]):
        for sub in ("games_self", "games_past"):
            for path in sorted((rd / sub).glob("*.game.jsonl.gz")):
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    rec = json.loads(f.readline())
                games += 1
                cmds = rec["commands"]
                for c in cmds:
                    cmd_kinds[c[0]] += 1
                recruits += sum(1 for c in cmds if c[0] == "recruit")
                decisions += len(rec["decisions"])
                n_ref = sum(1 for d in rec["decisions"] if d.get("refused"))
                refused += n_ref
                rej = rec.get("rejections", [])
                rejections += len(rej)
                games_with += bool(rej)
                for k, x, y in rej:
                    k = int(k)
                    if k < len(cmds) and cmds[k][0] == "recruit":
                        redirected += 1
                    if len(sample) < 6:
                        sample.append((path.name[:40], k, x, y, cmds[k] if k < len(cmds) else None))
    print(f"games {games}, decisions {decisions}, recruit commands {recruits}")
    print(f"rejections {rejections} in {games_with} games; decisions marked refused {refused}; "
          f"rejections followed at once by a recruit command {redirected}")
    print("command kinds", dict(cmd_kinds))
    for s in sample:
        print("  ", s)


if __name__ == "__main__":
    main()
