"""Counts on recorded games for the blind-spot audit
(docs/blindspot_audit_20261125.md).

Replays recorded games command by command on the CPU, changes nothing.
- Advancements: a unit whose type changes over an attack. How many came
  from a type with two or more `advances_to` (a real choice), on whose
  turn (the attacker's side chooses in Wesnoth, the defender gets the
  first option), and which option was taken.
- Moves that bring an enemy into view: a move after which the side sees an
  enemy unit it did not see before it. In Wesnoth such a move stops where
  the enemy is sighted; here it runs to its end. Counted with its number
  of steps (a one-step move cannot differ).
- The enemy-faction input at each side's first decision: whether it is
  one-hot.

    python audit_counts.py --checkpoint round_0576/pi.pt --list games.txt --dir round_0576/games_self
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


def units_of(cs) -> dict:
    return {u.id: (u.name, int(u.side)) for u in cs.to_state().map.units}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("games", type=Path, nargs="*")
    ap.add_argument("--list", type=Path, default=None)
    ap.add_argument("--dir", type=Path, default=Path("."))
    ap.add_argument("--checkpoint", type=Path, required=True)
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
    stats = json.loads((REPO / "unit_stats.json").read_text(encoding="utf-8"))["units"]
    c = collections.Counter()
    taken = collections.Counter()
    steps_hist = collections.Counter()
    posterior = []

    def enemies_seen(cs) -> set:
        raw = cs.encode_raw(type_to_id=type_to_id, faction_to_id=faction_to_id, masks=False,
                            unit_row_version=2, **ENCODING)
        return raw, {i for i, s in zip(raw.unit_ids, np.asarray(raw.unit_side_ids)) if int(s) == 1}

    for path in args.games:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            rec = json.loads(f.readline())
        c["games"] += 1
        prev = None                    # (command, side to move, units before, enemies seen before)
        first_seen = set()
        for _k, cs, cmd in walk_core(rec, verify=False):
            side = int(cs.core.current_side)
            kind = cmd[0]
            if prev is not None and prev[1] == side:
                p_cmd, _s, p_units, p_seen = prev
                if p_cmd[0] == "attack":
                    now = units_of(cs)
                    for uid, (name, uside) in p_units.items():
                        if uid in now and now[uid][0] != name:
                            options = list(stats.get(name, {}).get("advances_to") or [])
                            c["advancements"] += 1
                            if len(options) >= 2:
                                c["advancements_with_a_choice"] += 1
                                own_turn = uside == side
                                c["choice_on_own_turn" if own_turn else "choice_on_enemy_turn"] += 1
                                idx = options.index(now[uid][0]) if now[uid][0] in options else -1
                                taken[idx] += 1
                elif p_cmd[0] == "move":
                    _raw, seen_now = enemies_seen(cs)
                    steps = len(p_cmd[1]) - 1
                    c["moves"] += 1
                    if seen_now - p_seen:
                        c["moves_revealing_an_enemy"] += 1
                        steps_hist[min(steps, 6)] += 1
                        if steps >= 2:
                            c["of_those_two_steps_or_more"] += 1
            if kind == "init_side":
                prev = None
                continue
            p_units = units_of(cs) if kind == "attack" else None
            p_seen = None
            if kind == "move" or side not in first_seen:
                raw, p_seen = enemies_seen(cs)
                if side not in first_seen:
                    first_seen.add(side)
                    probs = np.asarray(raw.their_faction_probs, dtype=np.float64)
                    posterior.append(float(probs.max()))
            prev = (cmd, side, p_units, p_seen)
    out = {"counts": dict(c), "option_taken": {str(k): v for k, v in sorted(taken.items())},
           "revealing_moves_by_steps": {str(k): v for k, v in sorted(steps_hist.items())},
           "first_decision_faction_input_max": {"n": len(posterior),
                                                "one_hot": int(sum(p > 0.999 for p in posterior)),
                                                "smallest_max": min(posterior) if posterior else None}}
    print(json.dumps(out, indent=1))
    if args.out:
        args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
