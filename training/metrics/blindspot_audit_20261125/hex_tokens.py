"""How much of the board the network has a hex token for
(docs/blindspot_audit_20261125.md, the map part).

Replays recorded games command by command on the CPU, changes nothing.
At every command a player decided (all but init_side) it encodes the
position as the simple loop's network reads it and counts: the board's
hexes, the hex tokens, the hexes the side sees now, and how many of those
have a token (the tokens' "seen now" column).

    python hex_tokens.py --checkpoint round_0576/pi.pt --out hex_tokens.json <game.jsonl.gz> ...
"""
import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

SEEN_COL = 3          # encoder.py, hex dynamic flags: "the side sees the hex now"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("games", type=Path, nargs="*")
    ap.add_argument("--list", type=Path, default=None, help="a file of record names, one a line, under --dir")
    ap.add_argument("--dir", type=Path, default=Path("."))
    ap.add_argument("--checkpoint", type=Path, required=True, help="the network whose vocabularies encode")
    ap.add_argument("--every", type=int, default=5, help="encode one decision in this many")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.list:
        args.games += [args.dir / name for name in args.list.read_text(encoding="utf-8").split()]
    import torch
    torch.set_num_threads(2)
    from tools.game_record import walk_core
    from tools.preencode_corpus import vocab_from_checkpoint
    from tools.preencode_sequences import ENCODING
    type_to_id, faction_to_id = vocab_from_checkpoint(args.checkpoint)
    by_map = collections.defaultdict(lambda: collections.Counter())
    by_turn = collections.defaultdict(lambda: collections.Counter())
    for path in args.games:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            rec = json.loads(f.readline())
        scenario = rec["scenario_id"].replace("multiplayer_", "")
        n = 0
        for _k, cs, cmd in walk_core(rec, verify=False):
            if cmd[0] == "init_side":
                continue
            n += 1
            if n % args.every:
                continue
            side = int(cs.core.current_side)
            raw = cs.encode_raw(type_to_id=type_to_id, faction_to_id=faction_to_id, masks=False,
                                unit_row_version=2, **ENCODING)
            seen = np.asarray(cs.core.seen_export(side))
            tokens = len(raw.hex_xs)
            seen_tokens = int(np.asarray(raw.hex_dynamic_flags)[:, SEEN_COL].sum())
            row = {"decisions": 1, "board": int(seen.size), "tokens": tokens, "seen": int(seen.sum()),
                   "seen_with_token": seen_tokens, "units": len(raw.unit_xs)}
            by_map[scenario].update(row)
            turn = int(cs.core.turn_number)
            by_turn[min(turn, 30) // 5 * 5].update(row)
    total = collections.Counter()
    for c in by_map.values():
        total.update(c)

    def line(name, c):
        d = c["decisions"]
        return (f"{name}: {d} decisions, board {c['board'] / d:.0f} hexes, tokens {c['tokens'] / d:.0f} "
                f"({100 * c['tokens'] / c['board']:.1f}% of the board), seen {c['seen'] / d:.0f} "
                f"({100 * c['seen'] / c['board']:.1f}%), seen without a token "
                f"{100 * (c['seen'] - c['seen_with_token']) / max(c['seen'], 1):.1f}% of the seen")

    print(line("all", total))
    for name in sorted(by_map):
        print("  " + line(name, by_map[name]))
    for t in sorted(by_turn):
        print("  " + line(f"turns {t} to {t + 4}" if t < 30 else "turns 30 on", by_turn[t]))
    if args.out:
        args.out.write_text(json.dumps({"all": dict(total), "by_map": {k: dict(v) for k, v in by_map.items()},
                                        "by_turn": {str(k): dict(v) for k, v in by_turn.items()},
                                        "games": len(args.games), "every": args.every}, indent=1) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
