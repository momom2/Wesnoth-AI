"""Match luck: how much of a match's result the dice decide, read off the
recorded games with an AIVAT-style correction (Burch, Schmid, Moravcik,
Morrill and Bowling, "AIVAT: A New Variance Reduction Technique for Agent
Evaluation in Imperfect Information Games", AAAI 2018, arXiv 1612.06915).

At every attack of a recorded game, of either side, the core enumerates
the fight's exact outcomes (`GameCore.attack_outcomes`, the distribution
`combat_outcomes.enumerate_attack_outcomes` returns), and the luck of the
attack under a control variate Q is

    L = Q(the realized outcome) - sum over outcomes o of p(o) Q(o).

Given the position and the attack chosen, L has mean zero, so a game's
luck S (the sum of its L) has mean zero whatever the players do, and the
corrected result G = z - beta S estimates the same mean as the result z
for any beta fixed before the game is read (tools/match_luck_readout.py
fits beta with two-fold cross-fitting). Where L is not computed (below)
it is 0, decided on the position before the attack, which keeps S's mean
zero.

Q reads the two fighters only (no fork per outcome), from side A's view
(+ for side A's units, - for side B's, other sides' units count nothing).
Each outcome is reduced to each fighter's state: dead, advanced, or alive
with its hit points. Advanced is any change of type, or hit points past
the pre-fight maximum by more than a feeding kill's +1 (an after-max-level
advancement). The reduction does not name the type a unit advances to, so
its probabilities are the same whether the advancement is drawn uniformly
(self-play) or takes the first type (the matches): the core's uniform
expansion is merged back. The columns (`COLUMNS`):

    hp        hit points (an advanced unit counts its pre-fight maximum):
              the HP margin of side A over side B, as
              tools/turn_gap._hp_margin defines it (and
              critic_data.player_hp_margin on feature/critic-step1);
    material  cost x hit points / maximum hit points (an advanced unit
              counts its pre-fight cost): the cost-weighted material of
              the exposure evaluator (exp/exposure-evaluator,
              wesnoth_ai/lookahead_exposure.material), whose gold term no
              fight changes;
    material_nonterminal, terminal
              the material with a terminal outcome scored at its result:
              a side left without a leader has lost (the simulator's
              `_check_game_over`), so terminal is +1 when side B's last
              leader dies, -1 when side A's does, and the board's whole
              material counts only when neither dies. The readout fits
              both columns, so the scale of a won game in gold is fitted,
              not chosen;
    value_50, value_100, value_200
              a saturating value: tanh(the board's material / scale) with
              the scale in gold, a terminal outcome scored +1 or -1.

Not computed (L = 0, counted per game): a fight the core does not
enumerate (berserk past the DP's caps, too complex), one whose recorded
counter weapon is not the one the core chooses now, a hex without a unit;
recruits' trait rolls (counted as `recruits`). An advancement's choice
among types carries no luck under the reduction (realized advancements are
counted as `advancements`). A corpse a plague attack raises is not
counted. A realized outcome outside the enumerated support is counted as
`outside_support` and keeps its L.

Rows (one JSON line per game, written as each game finishes; a rerun
skips the games already written and walks again a last row cut short):

    python tools/match_luck.py fetch --data DIR --manifest data.txt NAME=HF_PATH=SHA256 ...
    python tools/match_luck.py walk --match NAME=PATH [--match ...] --out rows.jsonl
        [--jobs N] [--limit N] [--no-verify]
    python tools/match_luck.py readout rows.jsonl [--matches NAME ...] [--save-json PATH]

fetch downloads each match's tarball from the model host, checks it
against its pinned SHA-256 and lists the ones that land in the manifest
(name, local path, bytes, HF path; tab-separated), the others logged and
left out. PATH is a match's games directory or its tarball
(games_<name>.tar.gz as the box scripts upload it): run_elo_batch's result
files `game_*.json` beside their records `game_*.game.jsonl.gz`
(tools/game_record.py).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import math
import multiprocessing as mp
import sys
import tarfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("match_luck")

MODEL_HOST = "momom2/wesnoth-model-checkpoints"

# The value columns' scales, in gold: tanh(material / scale) reads a margin
# of one scale as three quarters of a won game.
VALUE_SCALES = (50, 100, 200)
COLUMNS = ("hp", "material", "material_nonterminal", "terminal") + tuple(f"value_{s}" for s in VALUE_SCALES)

DEAD, ADVANCED, ALIVE = "dead", "advanced", "alive"
# A feeding kill raises the feeder's maximum and its hit points by 1.
FEEDING_HP = 1

RECORD_SUFFIX = ".game.jsonl.gz"
RESULT_SUFFIX = ".json"

FightState = Tuple[str, int]


# ---------------------------------------------------------------------
# The luck of one fight
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Fighter:
    """A unit in a fight as the luck terms read it, before the fight."""
    id: str
    side: int
    sign: int          # +1: side A's, -1: side B's, 0: another side's
    ends_game: bool    # its side's last leader: its death ends the game
    name: str
    hp: int
    max_hp: int
    cost: float


def side_sign(side: int, side_a: int) -> int:
    from wesnoth_ai.classes import opponent_of
    if side == side_a:
        return 1
    return -1 if side == opponent_of(side_a) else 0


def leader_counts(units: Sequence[Mapping]) -> Counter:
    """{side: its leaders on the board} over core unit dicts."""
    return Counter(int(u["side"]) for u in units if u["is_leader"])


def fighter_of(unit: Mapping, side_a: int, leaders: Mapping[int, int]) -> Fighter:
    """A core unit dict (`GameCore.unit_export`) as a Fighter; `leaders`
    counts each side's leaders on the board (`leader_counts`)."""
    side = int(unit["side"])
    return Fighter(id=str(unit["id"]), side=side, sign=side_sign(side, side_a),
                   ends_game=bool(unit["is_leader"]) and leaders.get(side, 0) == 1, name=str(unit["name"]),
                   hp=int(unit["current_hp"]), max_hp=int(unit["max_hp"]), cost=float(unit["cost"]))


def fight_state(f: Fighter, hp: int, name: str) -> FightState:
    """The fighter's state after the fight from its hit points and type
    name: dead, advanced, or alive with its hit points."""
    if hp <= 0:
        return (DEAD, 0)
    if name != f.name or hp > f.max_hp + FEEDING_HP:
        return (ADVANCED, 0)
    return (ALIVE, int(hp))


def coarse_distribution(probs: Mapping[tuple, float], fighters: Sequence[Fighter]
                        ) -> Dict[Tuple[FightState, FightState], float]:
    """The core's outcome distribution ({outcome key: p}, the key's hit
    points at 0 and 1 and type names at 8 and 9) over the fighters'
    states, normalized."""
    out: Dict[Tuple[FightState, FightState], float] = {}
    for key, p in probs.items():
        states = (fight_state(fighters[0], key[0], key[8]), fight_state(fighters[1], key[1], key[9]))
        out[states] = out.get(states, 0.0) + float(p)
    total = sum(out.values())
    return {s: p / total for s, p in out.items()}


def unit_material(cost: float, hp: int, max_hp: int) -> float:
    return cost * min(1.0, hp / max(1, max_hp))


def column_values(fighters: Sequence[Fighter], states: Sequence[FightState],
                  rest_material: float) -> Dict[str, float]:
    """Each column's Q for the fighters in `states`; `rest_material` is
    the material of every other unit of the two players (hp and material
    leave it out: it is the same in every outcome)."""
    hp = material = 0.0
    terminal = 0
    for f, (kind, h) in zip(fighters, states):
        if kind == DEAD:
            if f.ends_game and f.sign:
                terminal = -f.sign
            continue
        hp += f.sign * (f.max_hp if kind == ADVANCED else h)
        material += f.sign * (f.cost if kind == ADVANCED else unit_material(f.cost, h, f.max_hp))
    board = rest_material + material
    out = {"hp": hp, "material": material,
           "material_nonterminal": 0.0 if terminal else board, "terminal": float(terminal)}
    for scale in VALUE_SCALES:
        out[f"value_{scale}"] = float(terminal) if terminal else math.tanh(board / scale)
    return out


def luck_terms(dist: Mapping[Tuple[FightState, FightState], float],
               realized: Tuple[FightState, FightState], fighters: Sequence[Fighter],
               rest_material: float) -> Dict[str, float]:
    """L for each column: Q of the realized states less Q's expectation
    over `dist`."""
    expected = dict.fromkeys(COLUMNS, 0.0)
    for states, p in dist.items():
        for c, v in column_values(fighters, states, rest_material).items():
            expected[c] += p * v
    got = column_values(fighters, realized, rest_material)
    return {c: got[c] - expected[c] for c in COLUMNS}


def rest_material(units: Sequence[Mapping], side_a: int, exclude: Set[str]) -> float:
    """Side A's material less side B's over the units not in `exclude`."""
    total = 0.0
    for u in units:
        if str(u["id"]) in exclude:
            continue
        sign = side_sign(int(u["side"]), side_a)
        if sign:
            total += sign * unit_material(float(u["cost"]), int(u["current_hp"]), int(u["max_hp"]))
    return total


def realized_state(core, f: Fighter) -> FightState:
    """The fighter's state on `core` after the fight. A unit a plague
    attack killed is dead: the corpse that rises in its place, under its
    id, is the killer's side's."""
    try:
        u = core.unit_export(f.id)
    except KeyError:
        return (DEAD, 0)
    if int(u["side"]) != f.side:
        return (DEAD, 0)
    return fight_state(f, int(u["current_hp"]), str(u["name"]))


@dataclass
class AttackLuck:
    terms: Optional[Dict[str, float]]   # None: not computed (L = 0)
    reason: str                         # enumerated, outside_support, or why not computed
    advanced: bool = False              # a fighter advanced in the realized outcome


def attack_luck(cs, cmd: list, side_a: int) -> AttackLuck:
    """The luck of the attack command `cmd` (["attack", ax, ay, dx, dy,
    a_weapon, d_weapon, seed, ...]) played on the CoreState `cs`, which
    is not changed: the realized outcome is read on a fork."""
    ax, ay, dx, dy, a_weapon = (int(v) for v in cmd[1:6])
    d_weapon = int(cmd[6]) if len(cmd) > 6 else -1
    core = cs.core
    choice = core.counter_weapon_choice(ax, ay, dx, dy, a_weapon)
    if choice is None:
        return AttackLuck(None, "no_units")
    if int(choice[0]) != d_weapon:
        return AttackLuck(None, "counter_weapon_differs")
    res = core.attack_outcomes(ax, ay, dx, dy, a_weapon, True)
    if res is None:
        return AttackLuck(None, "not_enumerable")
    probs, att_id, dfd_id = res
    units = core.units_export()
    leaders = leader_counts(units)
    fighters = (fighter_of(core.unit_export(att_id), side_a, leaders),
                fighter_of(core.unit_export(dfd_id), side_a, leaders))
    rest = rest_material(units, side_a, {att_id, dfd_id})
    dist = coarse_distribution(probs, fighters)
    after = cs.fork()
    after.apply_command(list(cmd))
    realized = (realized_state(after.core, fighters[0]), realized_state(after.core, fighters[1]))
    reason = "enumerated"
    if realized not in dist:
        reason = "outside_support"
        log.warning("attack %s: the realized outcome %s is outside the enumerated support %s",
                    cmd, realized, sorted(dist))
    return AttackLuck(luck_terms(dist, realized, fighters, rest), reason,
                      advanced=any(s[0] == ADVANCED for s in realized))


# ---------------------------------------------------------------------
# The luck of one game
# ---------------------------------------------------------------------
def game_luck(rec: dict, side_a: int, *, verify: bool = True) -> dict:
    """{"S": {column: the game's luck}, "SS": {column: sum of L squared},
    "counts": {...}} over every attack of the record `rec`, from side A's
    view. With `verify`, a record that does not rebuild its game raises
    `game_record.RecordMismatch`."""
    from tools import game_record
    luck = dict.fromkeys(COLUMNS, 0.0)
    squares = dict.fromkeys(COLUMNS, 0.0)
    counts: Counter = Counter()
    for _k, cs, cmd in game_record.walk_core(rec, verify=verify):
        if cmd[0] == "recruit":
            counts["recruits"] += 1
        if cmd[0] != "attack":
            continue
        counts["attacks"] += 1
        a = attack_luck(cs, cmd, side_a)
        counts[a.reason] += 1
        counts["advancements"] += int(a.advanced)
        if a.terms is None:
            continue
        for c, v in a.terms.items():
            luck[c] += v
            squares[c] += v * v
    return {"S": luck, "SS": squares, "counts": dict(counts)}


def result_z(outcome_a: Optional[str]) -> int:
    """Side A's result: +1 a win, -1 a loss, 0 anything else (no result)."""
    return {"win": 1, "loss": -1}.get(outcome_a or "", 0)


# Result-file fields copied onto each row.
RESULT_FIELDS = ("seed", "scenario_id", "turns", "side_a", "label_a", "label_b", "procedure_a",
                 "procedure_b", "raw_temperature_a", "raw_temperature_b", "faction_a", "faction_b",
                 "ended_by", "outcome_a", "code_version")


@dataclass
class GameItem:
    match: str
    game: str
    result: dict
    record: Optional[bytes]             # the gzip bytes of the game's record file


def parse_record(data: bytes, game: str) -> dict:
    """The record of `game` in a record file's bytes (gzip members of
    JSON lines)."""
    recs = [json.loads(line) for line in gzip.decompress(data).decode("utf-8").splitlines()
            if line.strip()]
    for rec in recs:
        if rec.get("game_label") == game:
            return rec
    if len(recs) == 1:
        return recs[0]
    raise ValueError(f"{game}: {len(recs)} records, none labelled {game}")


def walk_item(item: GameItem, verify: bool = True) -> dict:
    """A row: the result's fields, z, and the game's luck (or an error)."""
    res = item.result
    row = {"match": item.match, "game": item.game, **{k: res.get(k) for k in RESULT_FIELDS}}
    z = result_z(res.get("outcome_a"))
    row.update(z=z, decisive=z != 0)
    if item.record is None:
        row["error"] = "no record"
        return row
    t0 = time.perf_counter()
    try:
        rec = parse_record(item.record, item.game)
        side_a = int(rec.get("players", {}).get("a", {}).get("side", res.get("side_a")))
        if res.get("side_a") is not None and int(res["side_a"]) != side_a:
            raise ValueError(f"side A is {res['side_a']} in the result and {side_a} in the record")
        if z and (int(rec["winner"]) == side_a) != (z == 1):
            raise ValueError(f"the record's winner {rec['winner']} contradicts outcome_a {res.get('outcome_a')}")
        row.update(side_a=side_a, record_code_version=rec.get("code_version"),
                   uniform_advancement=bool(rec.get("uniform_advancement")))
        row.update(game_luck(rec, side_a, verify=verify))
    except Exception as e:                            # noqa: BLE001 -- the row records it
        row["error"] = f"{type(e).__name__}: {e}"
    row["secs"] = round(time.perf_counter() - t0, 3)
    return row


# ---------------------------------------------------------------------
# Reading matches
# ---------------------------------------------------------------------
def _stem(name: str) -> Optional[Tuple[str, str]]:
    """(game, "record" or "result") for a match file's base name."""
    if name.startswith(".") or not name.startswith("game_"):
        return None
    if name.endswith(RECORD_SUFFIX):
        return name[:-len(RECORD_SUFFIX)], "record"
    if name.endswith(RESULT_SUFFIX):
        return name[:-len(RESULT_SUFFIX)], "result"
    return None


def _match_files(path: Path) -> Iterator[Tuple[str, bytes]]:
    """(base name, content) of every file of a games directory or tarball
    (read in place, nothing extracted)."""
    if path.is_dir():
        for p in sorted(path.iterdir()):
            if p.is_file():
                yield p.name, p.read_bytes()
        return
    with tarfile.open(path, "r:*") as tf:
        for m in tf:
            if m.isfile():
                fh = tf.extractfile(m)
                if fh is not None:
                    yield m.name.rsplit("/", 1)[-1], fh.read()


def match_items(match: str, path: Path) -> List[GameItem]:
    """The games of one match: every result file, with its record when
    there is one."""
    results: Dict[str, dict] = {}
    records: Dict[str, bytes] = {}
    for name, data in _match_files(path):
        s = _stem(name)
        if s is None:
            continue
        game, kind = s
        if kind == "record":
            records[game] = data
        else:
            results[game] = json.loads(data.decode("utf-8"))
    return [GameItem(match, g, results[g], records.get(g)) for g in sorted(results)]


def trim_partial_row(out: Path) -> None:
    """Drop a last row cut short (a walk ended mid-write, or a copy of the
    file taken while a row was written), so that the rows read and the
    rows appended stay whole lines; its game is walked again."""
    if not out.exists():
        return
    data = out.read_bytes()
    keep = data.rfind(b"\n") + 1
    if keep < len(data):
        log.warning("%s: dropping a last row cut short (%d bytes)", out, len(data) - keep)
        with open(out, "r+b") as fh:
            fh.truncate(keep)


def done_keys(out: Path) -> Set[Tuple[str, str]]:
    if not out.exists():
        return set()
    keys = set()
    with open(out, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                keys.add((r["match"], r["game"]))
    return keys


def _walk_task(args: Tuple[GameItem, bool]) -> dict:
    return walk_item(*args)


def _init_worker() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def walk_matches(sources: Sequence[Tuple[str, Path]], out: Path, *, jobs: int = 1,
                 limit: Optional[int] = None, verify: bool = True) -> int:
    """Append a row per game of each match to `out`, the games already
    there skipped; returns the number of rows written."""
    trim_partial_row(out)
    done = done_keys(out)
    written = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    pool = mp.get_context("spawn").Pool(jobs, initializer=_init_worker) if jobs > 1 else None
    try:
        with open(out, "a", encoding="utf-8") as fh:
            for match, path in sources:
                items = match_items(match, path)
                todo = [it for it in items if (match, it.game) not in done][:limit]
                log.info("%s: %d games, %d with records, %d to walk", match, len(items),
                         sum(it.record is not None for it in items), len(todo))
                tasks = [(it, verify) for it in todo]
                rows = pool.imap_unordered(_walk_task, tasks) if pool else map(_walk_task, tasks)
                for row in rows:
                    fh.write(json.dumps(row, separators=(",", ":")) + "\n")
                    fh.flush()
                    written += 1
                    if "error" in row:
                        log.warning("%s %s: %s", match, row["game"], row["error"])
                log.info("%s: walked", match)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return written


# ---------------------------------------------------------------------
# Fetching matches
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class MatchSpec:
    name: str
    path: str          # the tarball on the model host
    sha256: str        # its pinned content


def parse_spec(text: str) -> MatchSpec:
    """NAME=HF_PATH=SHA256 as a MatchSpec."""
    parts = text.split("=")
    if (len(parts) != 3 or not parts[0] or not parts[1].endswith(".tar.gz") or len(parts[2]) != 64
            or any(c not in "0123456789abcdef" for c in parts[2])):
        raise argparse.ArgumentTypeError(f"a match is NAME=HF_PATH.tar.gz=SHA256, not {text!r}")
    return MatchSpec(*parts)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hf_download(path: str, data: Path) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(MODEL_HOST, path, cache_dir=str(data)))


def fetch_matches(specs: Sequence[MatchSpec], data: Path, manifest: Path, *,
                  download: Callable[[str, Path], Path] = hf_download) -> int:
    """Each match's tarball into `data`, checked against its pinned
    SHA-256; a manifest line (name, local path, bytes, HF path) is appended
    as each one lands. A match that cannot be fetched, or whose content is
    not the pinned one, is logged and left out. Returns the number
    fetched."""
    data.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for spec in specs:
        try:
            local = download(spec.path, data)
            digest = sha256_of(local)
        except Exception as e:                        # noqa: BLE001 -- this match is left out, the others go on
            log.error("%s: %s not fetched: %s: %s", spec.name, spec.path, type(e).__name__, e)
            continue
        if digest != spec.sha256:
            log.error("%s: %s is not the pinned content (sha256 %s, pinned %s): left out",
                      spec.name, spec.path, digest, spec.sha256)
            continue
        with open(manifest, "a", encoding="utf-8") as fh:
            fh.write(f"{spec.name}\t{local}\t{local.stat().st_size}\t{spec.path}\n")
        fetched += 1
        log.info("%s: %s fetched, %d bytes", spec.name, spec.path, local.stat().st_size)
    return fetched


def _source(text: str) -> Tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"--match takes NAME=PATH, not {text!r}")
    return name, Path(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="the matches' tarballs from the model host, checked against their pins")
    f.add_argument("specs", type=parse_spec, nargs="+", metavar="NAME=HF_PATH=SHA256")
    f.add_argument("--data", type=Path, required=True)
    f.add_argument("--manifest", type=Path, required=True)
    w = sub.add_parser("walk", help="the luck terms of every recorded game, one row per game")
    w.add_argument("--match", type=_source, action="append", required=True, metavar="NAME=PATH")
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--jobs", type=int, default=1)
    w.add_argument("--limit", type=int, default=None, help="at most this many games per match")
    w.add_argument("--no-verify", action="store_true",
                   help="skip the records' fingerprint checks (game_record.walk)")
    r = sub.add_parser("readout", help="the variance the luck correction removes")
    r.add_argument("rows", type=Path)
    r.add_argument("--matches", nargs="+", default=None, metavar="NAME",
                   help="read only these matches' rows (default: every match in the rows)")
    r.add_argument("--save-json", type=Path, default=None)
    r.add_argument("--seed", type=int, default=0, help="the cross-fitting split and the bootstrap")
    r.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "fetch":
        names = [s.name for s in args.specs]
        if len(set(names)) != len(names):
            ap.error(f"a match name is given twice: {sorted(n for n in set(names) if names.count(n) > 1)}")
        n = fetch_matches(args.specs, args.data, args.manifest)
        log.info("%d of %d matches fetched", n, len(args.specs))
        return 0 if n == len(args.specs) else 1
    if args.command == "walk":
        n = walk_matches(args.match, args.out, jobs=args.jobs, limit=args.limit, verify=not args.no_verify)
        log.info("%d rows written to %s", n, args.out)
        return 0
    from tools.match_luck_readout import load_rows, readout, report_text
    rows = load_rows(args.rows)
    if args.matches is not None:
        rows = [row for row in rows if row["match"] in set(args.matches)]
    result = readout(rows, seed=args.seed, n_boot=args.bootstrap)
    print(report_text(result))
    if args.save_json is not None:
        args.save_json.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
