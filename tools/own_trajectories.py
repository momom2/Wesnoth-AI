"""The memory player's own trajectories for the sequence trainer
(docs/memory_own_trajectories_prereg_20261011.md), step one of two: which
game-sides of the recorded matches are own, and the position before each of
their recorded commands. Step two (tools/own_targets.py) decides which of
those positions were decisions and labels them with the teacher's
distribution.

An own game-side is one a match played with the teacher's checkpoint (the
result's `checkpoint_sha256_x`, against the SHA-256 of `--teacher`) at
`--slots` slots (`memory_x`) without a per-turn memory reset
(`memory_reset_x` false, or absent: matches before the reset existed,
2026-10-09, had none). Labels are never read.

A candidate is the position before one of the game-side's recorded commands
(move, attack, recruit, end_turn), encoded by the pre-encoder's code
(`preencode_sequences.encode_position`: the parity observation, the
relevant set, the label in the encoding's basis, the belief targets from the
true state), with the legality masks the player decided under, packed as
the match's actor packed them (`server_priors.pack_masks`). A move followed
by an attack from its end on a unit the mover did not stand next to, whose
approach the simulator would have chosen for that attack
(`wesnoth_sim.attack_hex`), may be one attack decision: its candidate keeps
that attack's label beside the move's.

Every game-side not kept is counted by reason (`counts`), and a game whose
record does not rebuild or holds a position the encoder refuses is dropped
whole with its reason (`errors`).

    OUT/<name>.cand.zpkl          one OwnCandidateGame (zlib pickle)
    OUT/candidates_manifest.json  the selection, the encoding and the counts

    python tools/own_trajectories.py --games DIR [DIR ...] --teacher CKPT --slots 64 \\
        --out own_candidates --workers 30
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from tools.preencode_sequences import (ENCODING, encoding_fingerprint, read_record,  # noqa: E402
                                       write_record)
from wesnoth_ai.constants import OBSERVATION_EPOCH  # noqa: E402
from wesnoth_ai.sequence_records import OwnCandidate, OwnCandidateGame, OwnCandidateSide  # noqa: E402
from wesnoth_ai.sequence_streams import unit_draw  # noqa: E402

log = logging.getLogger("own_trajectories")

CANDIDATE_SUFFIX = ".cand.zpkl"
CANDIDATES_MANIFEST = "candidates_manifest.json"
PLAYER_COMMANDS = frozenset({"move", "attack", "recruit", "end_turn"})
# Own games held out for the agreement barrier, drawn by game.
OWN_HOLDOUT = 0.05


class OwnSideError(ValueError):
    """A recorded command of an own game-side yields no position to train on."""


@dataclass(frozen=True)
class OwnSide:
    """One own game-side, as its match's result names it."""
    name: str                       # the game's own record name: <match dir>__<result stem>
    record: str                     # the game record (<stem>.game.jsonl.gz)
    side: int
    forwards: Optional[int]         # the player's forwards (forwards_x)
    offset: float                   # its decode's end_turn offset
    winner: int                     # 0 for a game without one


def own_holdout(name: str) -> bool:
    """Whether the own game `name` is held out (both its sides)."""
    return unit_draw("own_holdout", name) < OWN_HOLDOUT


def select_own_sides(match_dirs: List[Path], teacher_sha256: str, slots: int
                     ) -> Tuple[Dict[str, List[OwnSide]], Counter]:
    """The own game-sides of the match directories, by game, and the count
    of every game-side read by what became of it."""
    games: Dict[str, List[OwnSide]] = {}
    counts: Counter = Counter()
    for d in match_dirs:
        for path in sorted(Path(d).glob("game_*.json")):
            try:
                res = json.loads(path.read_text(encoding="utf-8"))
                side_a = int(res["side_a"])
            except (OSError, ValueError, KeyError, TypeError):
                counts["unreadable_result"] += 1
                continue
            counts["results"] += 1
            record = path.with_name(path.stem + ".game.jsonl.gz")
            outcome = res.get("outcome_a")
            winner = side_a if outcome == "win" else 3 - side_a if outcome == "loss" else 0
            for x in ("a", "b"):
                counts["game_sides"] += 1
                if res.get(f"checkpoint_sha256_{x}") != teacher_sha256:
                    counts["not_own: another checkpoint"] += 1
                elif res.get(f"memory_{x}") != slots:
                    counts["not_own: another memory size"] += 1
                elif res.get(f"memory_reset_{x}"):
                    counts["not_own: memory reset each turn"] += 1
                elif not record.is_file():
                    counts["dropped: no game record"] += 1
                else:
                    counts["own"] += 1
                    forwards = res.get(f"forwards_{x}")
                    name = f"{Path(d).name}__{path.stem}"
                    games.setdefault(name, []).append(OwnSide(
                        name=name, record=str(record), side=side_a if x == "a" else 3 - side_a,
                        forwards=None if forwards is None else int(forwards),
                        offset=float(res.get(f"raw_end_turn_offset_{x}") or 0.0), winner=winner))
    return games, counts


def read_game_record(path: Path) -> dict:
    from tools.game_record import read_records
    records = list(read_records(path))
    if len(records) != 1:
        raise OwnSideError(f"{path} holds {len(records)} records, not one")
    return records[0]


def approach_attack(gs, commands: List[list], k: int, side: int):
    """When command k, a move of `side`, is followed by an attack from the
    move's end on a hex not next to the move's start, and the simulator's
    approach for that attack from the start is the move's end: the label
    (full-board basis) of the attack decided before the move. Else None."""
    from tools.abilities import hex_neighbors
    from tools.replay_dataset import _action_indices
    from tools.wesnoth_sim import attack_hex
    from wesnoth_ai.classes import Position
    cmd = commands[k]
    nxt = commands[k + 1] if k + 1 < len(commands) else None
    if cmd[0] != "move" or nxt is None or nxt[0] != "attack":
        return None
    start = (int(cmd[1][0]), int(cmd[2][0]))
    end = (int(cmd[1][-1]), int(cmd[2][-1]))
    target = (int(nxt[3]), int(nxt[4]))
    if (int(nxt[1]), int(nxt[2])) != end or target in hex_neighbors(*start):
        return None
    units = {(u.position.x, u.position.y): u for u in gs.map.units}
    attacker = units.get(start)
    if attacker is None or attacker.side != side or target not in units:
        return None
    hex_ = attack_hex(gs, attacker, Position(*target))
    if hex_ is None or (hex_.x, hex_.y) != end:
        return None
    return _action_indices(gs, ["attack", start[0], start[1], target[0], target[1], int(nxt[5])])


def candidates_of(rec: dict, sides: List[OwnSide], type_to_id: Dict[str, int],
                  faction_to_id: Dict[str, int]) -> OwnCandidateGame:
    """The candidates of a game's own sides (module docstring). Raises
    `game_record.RecordMismatch` when the record does not rebuild its game,
    `OwnSideError` or `LabelSlotMismatch` when a command yields no position."""
    import torch
    from tools.game_record import _walk_core
    from tools.inference_seam import build_light_encoded
    from tools.preencode_sequences import checked_label, encode_position
    from tools.replay_dataset import _action_indices
    from wesnoth_ai.game_core import bind_view
    from wesnoth_ai.server_priors import pack_masks
    name = sides[0].name
    by_side = {s.side: s for s in sides}
    stats: Counter = Counter()
    found: Dict[int, List[OwnCandidate]] = {s: [] for s in by_side}
    rejected: Counter = Counter()
    refused = Counter(int(k) for k, _x, _y in rec.get("rejections", ()))
    commands = rec["commands"]
    cpu = torch.device("cpu")
    for k, cs, cmd in _walk_core(rec, verify=True):
        side = int(cs.core.current_side)
        rejected[side] += refused.get(k, 0)
        if side not in by_side or cmd[0] not in PLAYER_COMMANDS:
            continue
        snapshot = cs.fork()
        gs = snapshot.to_state()
        bind_view(gs, snapshot)
        where = f"{name}, side {side}, command {k}"
        ai = _action_indices(gs, cmd, relevant_set=False, stats=stats)
        if ai is None:
            raise OwnSideError(f"{where} ({cmd[0]}) names no action of the side")
        pos = encode_position(gs, ai, where, type_to_id, faction_to_id, stats)
        pos.masks = pack_masks(build_light_encoded(pos.raw, cpu), gs)
        attack = approach_attack(gs, commands, k, side)
        if attack is not None:
            attack = checked_label(attack, pos.raw, where + " (its approach's attack)", stats)
            stats["approach_pairs"] += 1
        found[side].append(OwnCandidate(position=pos, attack_label=attack))
    rejected[int(rec.get("final_side", 0))] += refused.get(len(commands), 0)
    game = OwnCandidateGame(file=name, winner=int(rec.get("winner", 0)), n_commands=len(commands))
    for side, s in by_side.items():
        game.sides[side] = OwnCandidateSide(side=side, candidates=found[side], forwards=s.forwards,
                                            rejections=int(rejected[side]), offset=s.offset)
        stats[f"candidates_side{side}"] = len(found[side])
    stats["candidates"] = sum(len(v) for v in found.values())
    game.counts = {k: int(v) for k, v in stats.items()}
    return game


def _memory_of(rec: dict, side: int) -> Optional[int]:
    """The memory size the record says `side` played with."""
    for p in rec.get("players", {}).values():
        if int(p.get("side", 0)) == side:
            return p.get("memory")
    return None


def candidate_path(out_dir: Path, name: str) -> Path:
    return Path(out_dir) / (name + CANDIDATE_SUFFIX)


_W: Dict = {}


def _worker_init(type_to_id, faction_to_id, out_dir, slots) -> None:
    _W.update(type_to_id=type_to_id, faction_to_id=faction_to_id, out_dir=Path(out_dir), slots=int(slots))


def _worker_game(job: Tuple[str, List[dict]]) -> Tuple[str, str, Dict[str, int]]:
    """(name, status, the game's counts) for one game's own sides."""
    name, side_dicts = job
    sides = [OwnSide(**d) for d in side_dicts]
    dst = candidate_path(_W["out_dir"], name)
    if dst.exists():
        return name, "exists", {}
    try:
        rec = read_game_record(Path(sides[0].record))
        if any(_memory_of(rec, s.side) != _W["slots"] for s in sides):
            raise OwnSideError(f"{name}: the record's players do not say the result's memory size")
        game = candidates_of(rec, sides, _W["type_to_id"], _W["faction_to_id"])
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:  # noqa: BLE001 - one bad game, a Rust panic included, must not stop the pass
        return name, f"error: {type(e).__name__}: {e}"[:300], {}
    write_record(dst, game)
    return name, "ok", game.counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", type=Path, nargs="+", required=True, help="match directories")
    ap.add_argument("--teacher", type=Path, required=True, help="the checkpoint whose game-sides are own")
    ap.add_argument("--slots", type=int, required=True, help="the memory size own game-sides played at")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=None, help="the first N own games")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import wesnoth_core
    from tools.eval_provenance import file_sha256
    from tools.preencode_corpus import vocab_from_checkpoint
    teacher_sha = file_sha256(args.teacher)
    type_to_id, faction_to_id = vocab_from_checkpoint(args.teacher)
    phase = int(wesnoth_core.__phase__)
    games, counts = select_own_sides(args.games, teacher_sha, args.slots)
    names = sorted(games)[:args.limit]
    log.info("%d own game-sides in %d games of %d results (%s)", counts["own"], len(games), counts["results"],
             json.dumps(dict(counts)))
    args.out.mkdir(parents=True, exist_ok=True)
    done: Dict[str, Dict[str, int]] = {}
    errors: Dict[str, str] = {}
    t0 = time.time()

    def flush() -> None:
        totals: Counter = Counter()
        for c in done.values():
            totals.update(c)
        manifest = {
            "teacher": str(args.teacher), "teacher_sha256": teacher_sha, "slots": args.slots,
            "fingerprint": encoding_fingerprint(type_to_id, faction_to_id, 0, phase),
            "encoding": ENCODING, "observation_epoch": int(OBSERVATION_EPOCH), "core_phase": phase,
            "unit_type_to_id": type_to_id, "faction_to_id": faction_to_id,
            "match_dirs": [str(d) for d in args.games], "selection": dict(counts),
            "n_games": len(names), "n_done": len(done), "totals": dict(totals), "errors": errors,
            "games": {n: {"sides": [asdict(s) for s in games[n]], "holdout": own_holdout(n),
                          "counts": done.get(n)} for n in names},
        }
        tmp = args.out / (CANDIDATES_MANIFEST + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=0), encoding="utf-8")
        os.replace(tmp, args.out / CANDIDATES_MANIFEST)

    jobs = [(n, [asdict(s) for s in games[n]]) for n in names]
    with mp.get_context("spawn").Pool(args.workers, initializer=_worker_init,
                                      initargs=(type_to_id, faction_to_id, str(args.out), args.slots)) as pool:
        for i, (name, status, game_counts) in enumerate(pool.imap_unordered(_worker_game, jobs, chunksize=1), 1):
            if status == "ok":
                done[name] = game_counts
            elif status == "exists":
                done[name] = read_record(candidate_path(args.out, name)).counts
            else:
                errors[name] = status
                log.warning("%s: %s", name, status)
            if i % 100 == 0 or i == len(jobs):
                n = sum(c.get("candidates", 0) for c in done.values())
                el = time.time() - t0
                log.info("%d/%d games, %d candidates, %d errors, %.0f s, %.1f candidates/s",
                         i, len(jobs), n, len(errors), el, n / max(el, 1e-9))
                flush()
    flush()
    n = sum(c.get("candidates", 0) for c in done.values())
    log.info("CANDIDATES_DONE %d games, %d candidates, %d errors in %.0f s", len(done), n, len(errors),
             time.time() - t0)
    return 0


def read_candidates_manifest(out_dir: Path) -> dict:
    return json.loads((Path(out_dir) / CANDIDATES_MANIFEST).read_text(encoding="utf-8"))



if __name__ == "__main__":
    raise SystemExit(main())
