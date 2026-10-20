"""The coin rows: one row per game of a coin match, read off its result
file and game record (the label code's input, tools/coin_labels.py). numpy
free, standard library only.

A match's rows end with its manifest, {"match": NAME, "manifest":
{"results": N}}, N the result files the extraction saw, so the label code
can count what it did not read. A game whose record is missing or cannot be
read has a row with "error" in place of its coins.

I-log. Each band decision's logged choice (a1 where its coin landed 1, a0
otherwise) is checked against the command the record holds at its index:
"matched"; "relocated", a recruit of the logged type that the engine put on
another castle hex because a hidden unit held the ordered one (the record's
rejection of the ordered hex at that index says so,
WesnothSim._action_to_command); "unresolved", nothing played (a refused
recruit, an attack whose approach was cut short) or no index; "mismatched"
otherwise.

The plan. A box script's matches.txt names each match's games and seed
base ("NAME: games N from seed B, ..."), from which the readout counts the
slots without a result (`parse_plan`).
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.coin_decode import KIND_CODES  # noqa: E402
from tools.match_luck import match_items, parse_record, result_z, trim_partial_row  # noqa: E402

log = logging.getLogger("coin_rows")

STATUSES = ("matched", "relocated", "mismatched", "unresolved")


def _hex(xs, ys, at: int) -> list:
    return [int(xs[at]), int(ys[at])]


def _attack_matches(cmd: list, want: dict) -> bool:
    return [int(cmd[3]), int(cmd[4])] == want["to"] and int(cmd[5]) == int(want["weapon"])


def _recruit_status(cmd: list, want: dict, k: int, rejected: Set[Tuple[int, int, int]]) -> str:
    if cmd[0] != "recruit" or cmd[1] != want["unit"]:
        return "mismatched"
    if [int(cmd[2]), int(cmd[3])] == want["to"]:
        return "matched"
    return "relocated" if (k, *want["to"]) in rejected else "mismatched"


def _attack_status(commands: Sequence[list], k: int, want: dict) -> str:
    cmd = commands[k]
    if cmd[0] == "attack":
        ok = [int(cmd[1]), int(cmd[2])] == want["from"] and _attack_matches(cmd, want)
        return "matched" if ok else "mismatched"
    if cmd[0] != "move" or _hex(cmd[1], cmd[2], 0) != want["from"]:
        return "mismatched"
    # The simulator's approach move, then the attack from the hex it ended on.
    nxt = commands[k + 1] if k + 1 < len(commands) else None
    if nxt is None or nxt[0] != "attack":
        return "unresolved"
    ok = [int(nxt[1]), int(nxt[2])] == _hex(cmd[1], cmd[2], -1) and _attack_matches(nxt, want)
    return "matched" if ok else "mismatched"


def played_status(entry: dict, commands: Sequence[list], rejected: Set[Tuple[int, int, int]] = frozenset()) -> str:
    """The I-log status of one band decision (module docstring); `rejected`
    holds the record's recruit rejections as (command index, x, y)."""
    k = entry.get("cmd")
    if entry.get("refused") or k is None or not 0 <= int(k) < len(commands):
        return "unresolved"
    k = int(k)
    want = entry["a1"] if int(entry["d"]) else entry["a0"]
    cmd = commands[k]
    kind = want["kind"]
    if kind == "end_turn":
        return "matched" if cmd[0] == "end_turn" else "mismatched"
    if kind == "recruit":
        return _recruit_status(cmd, want, k, rejected)
    if kind == "move":
        ok = (cmd[0] == "move" and _hex(cmd[1], cmd[2], 0) == want["from"]
              and _hex(cmd[1], cmd[2], -1) == want["to"])
        return "matched" if ok else "mismatched"
    return _attack_status(commands, k, want)


def extract_game(match: str, game: str, result: dict, rec: dict) -> dict:
    """One game's row: its result, its attack coin, its band decisions as
    columns and the I-log count."""
    side_a = int(rec.get("players", {}).get("a", {}).get("side", result.get("side_a")))
    z = result_z(result.get("outcome_a"))
    if z and (int(rec["winner"]) == side_a) != (z == 1):
        raise ValueError(f"{game}: the record's winner {rec['winner']} contradicts outcome_a {result.get('outcome_a')}")
    coins = rec.get("coins") or {}
    entries = coins.get("decisions", [])
    rejected = {(int(k), int(x), int(y)) for k, x, y in rec.get("rejections", ())}
    status = dict.fromkeys(STATUSES, 0)
    for e in entries:
        status[played_status(e, rec["commands"], rejected)] += 1
    band = {"side": [int(e["side"]) for e in entries], "i": [int(e["i"]) for e in entries],
            "gap": [float(e["gap"]) for e in entries], "p": [float(e["p"]) for e in entries],
            "d": [int(e["d"]) for e in entries],
            "k0": [KIND_CODES[e["a0"]["kind"]] for e in entries],
            "k1": [KIND_CODES[e["a1"]["kind"]] for e in entries],
            "refused": [int(bool(e.get("refused"))) for e in entries]}
    attack = coins.get("attack")
    return {"match": match, "game": game, "seed": result.get("seed"), "side_a": side_a, "z": z,
            "decisive": z != 0, "ended_by": result.get("ended_by"),
            "attack": None if attack is None else {"d": int(attack["d"]), "p": float(attack["p"])},
            "bands": coins.get("bands", {}), "band": band, "i_log": status}


def extract_rows(match: str, path: Path, skip: Set[Tuple[str, str]] = frozenset()) -> Iterator[dict]:
    """The rows of one match's games directory or tarball, then its manifest:
    a game's row, or an error row when its record is missing or unreadable.
    Games in `skip` are counted in the manifest but not read again."""
    items = match_items(match, path)
    for item in items:
        if (match, item.game) in skip:
            continue
        if item.record is None:
            yield {"match": match, "game": item.game, "seed": item.result.get("seed"), "error": "no record"}
            continue
        try:
            yield extract_game(match, item.game, item.result, parse_record(item.record, item.game))
        except Exception as e:                            # noqa: BLE001 -- the row names it
            log.warning("%s %s: %s: %s", match, item.game, type(e).__name__, e)
            yield {"match": match, "game": item.game, "seed": item.result.get("seed"),
                   "error": f"{type(e).__name__}: {e}"}
    yield {"match": match, "manifest": {"results": len(items)}}


def extract_matches(sources: Sequence[Tuple[str, Path]], out: Path) -> int:
    """Append the rows of each match to `out`, the games already there
    skipped (a rerun reads only what it lacks, and writes a new manifest);
    returns the number of rows written."""
    trim_partial_row(out)
    done = _done_games(out)
    written = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        for match, path in sources:
            for row in extract_rows(match, path, done):
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
                written += 1
    return written


def _done_games(out: Path) -> Set[Tuple[str, str]]:
    """The (match, game) rows already in `out`, manifests left aside."""
    done: Set[Tuple[str, str]] = set()
    if out.exists():
        with open(out, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if "game" in r:
                        done.add((r["match"], r["game"]))
    return done


# ---------------------------------------------------------------------
# A pulled records directory, and the plan
# ---------------------------------------------------------------------
_PLAN = re.compile(r"\s(\S+): games (\d+) from seed (\d+),")


def parse_plan(path: Path) -> Dict[str, Tuple[int, int]]:
    """{match: (games, seed base)} from a box script's matches.txt, the last
    line of each match."""
    plan = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = _PLAN.search(" " + line)
        if m:
            plan[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return plan


def slot_report(results: Sequence[Tuple[Optional[int], bool]], planned: Tuple[int, int]) -> dict:
    """A match's slots against its plan (games, seed base): slot i plays
    seed base + i, its replacements base + i + k x REPL_SEED_OFFSET
    (run_elo_batch). `results` holds (seed, decisive) per result file. A
    missing slot has no result file at all: a match cut short has a
    complete prefix of its slots and the rest missing."""
    from tools.run_elo_batch import REPL_SEED_OFFSET
    games, base = planned
    have, decided, outside = set(), set(), 0
    for seed, decisive in results:
        off = None if seed is None else int(seed) - base
        if off is None or off < 0 or off % REPL_SEED_OFFSET >= games:
            outside += 1
            continue
        have.add(off % REPL_SEED_OFFSET)
        if decisive:
            decided.add(off % REPL_SEED_OFFSET)
    missing = sorted(set(range(games)) - have)
    return {"planned": games, "seed_base": base, "with_result": len(have), "decisive": len(decided),
            "missing": len(missing), "first_missing": missing[:10],
            "complete_prefix": missing[0] if missing else games, "outside_plan": outside, "cut": bool(missing)}


def record_sources(directory: Path, names: Optional[Sequence[str]] = None) -> List[Tuple[str, Path]]:
    """(match, games tarball or directory) in a pulled records directory:
    games_NAME.tar.gz or games_NAME/, those of `names` when given."""
    found: Dict[str, Path] = {}
    for p in sorted(Path(directory).iterdir()):
        name = p.name
        if p.is_dir() and name.startswith("games_"):
            found.setdefault(name[len("games_"):], p)
        elif p.is_file() and name.startswith("games_") and name.endswith(".tar.gz"):
            found[name[len("games_"):-len(".tar.gz")]] = p
    if names is not None:
        missing = sorted(set(names) - set(found))
        if missing:
            raise FileNotFoundError(f"{directory}: no games for {missing}")
        found = {n: found[n] for n in names}
    return sorted(found.items())
