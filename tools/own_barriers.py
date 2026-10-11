"""The crash barriers of the fine-tune on the memory player's own
trajectories (docs/memory_own_trajectories_prereg_20261011.md
"Measurements", revised before any run), read before any match on this
run's records:

- the holdout imitation loss, `parity3`'s from the trainer's probe of the
  starting weights (`--probe-at-start`, the first row of its
  <out>.probe.jsonl) and the student's from its final probe, on the same
  holdout game-sides: the student's at 0 slots within `--tolerance` (0.05)
  of `parity3`'s at 0 slots (the base policy intact), and the student's at
  64 slots no higher than `parity3`'s at 0 slots plus the tolerance (the
  memory may give up what it predicted of humans, not break the policy);
- on the held-out own game-sides (tools/own_trajectories.py holds out 5% of
  the own games), the student's agreement with its target above
  `parity3`'s at 64 slots, both in turns 6 and later: each network reads
  every decision of a game-side at 64 slots, its memory carried along the
  game-side as in play (tools/own_targets.walk_sides, the match's forward),
  and agrees when its choice at the reference decode (the joint argmax after
  the end_turn offset of configs/reference_player.json) is the action the
  target's distribution decodes to at that offset (`own_loss.target_key`).
  The raw argmax's agreement (no offset) is reported beside it.

The fourth barrier, the replay of the own trajectories (`replay_barrier`:
at least 95% of the recorded decisions in turns 6 and later reproduced by
the teacher at the player's slots, tools/own_targets.py's `repro_late`), is
read on the targets step's counts before training, by the box script.

Writes `--out` (JSON) and exits 0 when every barrier passes, 8 when one
fails.

    python tools/own_barriers.py --student own.pt --teacher parity3.pt --own-sequences own_sequences \\
        --probes own.probe.jsonl --out barriers.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from tools.own_targets import LATE_TURN, SideWalk, decode_of, end_actor_of, teacher_server, walk_sides  # noqa: E402
from tools.preencode_sequences import load_manifest, read_record, record_path  # noqa: E402
from wesnoth_ai.own_loss import target_key  # noqa: E402

log = logging.getLogger("own_barriers")

EXIT_BARRIER = 8
TOLERANCE = 0.05
SLOTS = 64
REPLAY_FLOOR = 0.95


def holdout_ce_barrier(probe_rows: List[Dict], tolerance: float) -> Dict:
    """The holdout barriers (module docstring) on the trainer's probe rows:
    the starting weights' (`parity3`'s) and the final ones' (the student's)
    policy cross-entropy at 0 and 64 slots."""
    start = next((r for r in probe_rows if int(r.get("positions", -1)) == 0 and not r.get("final")), None)
    final = next((r for r in reversed(probe_rows) if r.get("final")), None)

    def ce(row, k):
        return None if row is None else row.get(f"k{k}", {}).get("ce_all")

    p0, p64, s0, s64 = ce(start, 0), ce(start, 64), ce(final, 0), ce(final, 64)
    base_ok = p0 is not None and s0 is not None and abs(s0 - p0) <= tolerance
    memory_ok = p0 is not None and s64 is not None and s64 <= p0 + tolerance
    return {"tolerance": tolerance, "parity3": {"k0": p0, "k64": p64}, "student": {"k0": s0, "k64": s64},
            "base_policy": {"diff": None if s0 is None or p0 is None else s0 - p0, "pass": base_ok},
            "memory_policy": {"student_k64_less_parity3_k0": None if s64 is None or p0 is None else s64 - p0,
                              "pass": memory_ok},
            "pass": bool(base_ok and memory_ok)}


def replay_barrier(totals: Dict[str, int], floor: float = REPLAY_FLOOR) -> Dict:
    """The replay barrier on the targets step's counts: the share of the
    recorded decisions in turns 6 and later that the teacher at the
    player's slots, its memory carried, reproduces at the player's decode."""
    n_late = totals.get("decisions_late", 0)
    late = totals.get("repro_late", 0) / max(1, n_late)
    return {"repro_late": late, "repro": totals.get("repro", 0) / max(1, totals.get("decisions", 0)),
            "decisions_late": n_late, "floor": floor, "pass": n_late > 0 and late >= floor}


def own_holdout_sides(directory: Path) -> List[SideWalk]:
    """The held-out own game-sides as walks over their positions."""
    man = load_manifest(directory)
    walks = []
    for name in sorted(n for n, r in man["games"].items() if r["holdout"]):
        rec = read_record(record_path(directory, name))
        for side in sorted(rec.sides):
            walks.append(SideWalk((name, side), rec.sides[side]))
    return walks


def agreement(server, walks: List[SideWalk], offset: float) -> Dict:
    """How often the network's choice at 64 slots, its memory carried,
    is its target's, at `offset` and at no offset; overall and from turn
    LATE_TURN on."""
    counts: Counter = Counter()

    def on_position(w: SideWalk, out, _zero) -> int:
        if out is None:
            return 0
        pos = w.positions[w.t]
        late = pos.turn >= LATE_TURN
        end = end_actor_of(pos.raw)
        for tag, at in (("decode", offset), ("raw", 0.0)):
            hit = decode_of(out.legal_compact, at) == target_key(pos.own_target, at, end)
            counts[tag] += int(hit)
            counts[f"{tag}_late"] += int(hit and late)
        counts["n"] += 1
        counts["n_late"] += int(late)
        return 1

    for w in walks:
        w.t, w.state = 0, None
    walk_sides(server, walks, SLOTS, on_position, zero_slots=False)
    return {"agreement": counts["decode"] / max(1, counts["n"]),
            "agreement_late": counts["decode_late"] / max(1, counts["n_late"]),
            "raw_agreement": counts["raw"] / max(1, counts["n"]),
            "raw_agreement_late": counts["raw_late"] / max(1, counts["n_late"]),
            "n": counts["n"], "n_late": counts["n_late"], "n_game_sides": len(walks)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--student", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--own-sequences", type=Path, required=True)
    ap.add_argument("--probes", type=Path, required=True, help="the fine-tune's <out>.probe.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    ap.add_argument("--offset", type=float, default=None,
                    help="the decode's end_turn offset (default: configs/reference_player.json's)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    offset = args.offset
    if offset is None:
        offset = float(json.loads(Path("configs/reference_player.json").read_text(encoding="utf-8"))
                       ["decode"]["raw_end_turn_offset"])
    rows = [json.loads(line) for line in args.probes.read_text(encoding="utf-8").splitlines() if line.strip()]
    result: Dict = {"holdout_ce": holdout_ce_barrier(rows, args.tolerance), "offset": offset}
    walks = own_holdout_sides(args.own_sequences)
    device = torch.device(args.device)
    reads = {}
    with torch.no_grad():
        for name, spec in (("student", args.student), ("parity3", args.teacher)):
            server, slots = teacher_server(spec, device)
            if slots != SLOTS:
                raise SystemExit(f"{spec} has {slots} memory slots, not {SLOTS}")
            reads[name] = agreement(server, walks, offset)
            log.info("%s on the held-out own game-sides: %s", name, json.dumps(reads[name]))
            del server
    reads["pass"] = reads["student"]["agreement_late"] > reads["parity3"]["agreement_late"]
    result["agreement"] = reads
    result["pass"] = bool(result["holdout_ce"]["pass"] and reads["pass"])
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
    tmp.replace(args.out)
    log.info("BARRIERS %s %s", "PASSED" if result["pass"] else "FAILED", json.dumps(result))
    return 0 if result["pass"] else EXIT_BARRIER


if __name__ == "__main__":
    raise SystemExit(main())
