"""Recruit-rule files, and synthetic games of match L in the forms its
readout reads: a result file's fields and a game record's commands, seat
coin and rule decisions (tools/recruit_rule.py,
tools/recruit_rule_readout.py)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from tools.recruit_rule import SHORT_HASH, RecruitRule, rule_hash, rule_seat_coin

# A few recruits of each default-era faction, enough for synthetic games.
RECRUITS = {"Drakes": ["Drake Burner", "Drake Clasher", "Saurian Skirmisher"],
            "Knalgan Alliance": ["Dwarvish Fighter", "Thief", "Footpad"],
            "Loyalists": ["Spearman", "Cavalryman", "Mage"],
            "Northerners": ["Orcish Grunt", "Troll Whelp", "Wolf Rider"],
            "Rebels": ["Elvish Fighter", "Elvish Archer", "Wose"],
            "Undead": ["Skeleton", "Dark Adept", "Ghoul"]}


def _spread(key: str, scale: float) -> float:
    """A value in [-scale, scale) that is a pure function of `key`."""
    u = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") / 2.0 ** 64
    return scale * (2.0 * u - 1.0)


def rule_file(units: Dict[str, float], matchups: Dict[str, float], count: float, band: float = 0.35,
              cap: int = 2) -> dict:
    """A rule file's content, hashed as tools/recruit_rule_fit.py writes it."""
    rule = {"name": "test", "format": 1, "band": band, "cap": cap, "ridge": 5000.0, "count_ridge": 1.0,
            "units": units, "matchups": matchups, "count": count}
    return {"rule": rule, "hash": rule_hash(rule), "provenance": {"tool": "tests/helpers/rule_games.py"}}


def spread_rule(recruits: Dict[str, List[str]], band: float = 0.35, cap: int = 2, scale: float = 0.05,
                count: float = 0.02) -> dict:
    """A rule over `recruits` (faction: its units) whose unit and matchup
    values are spread by a hash of their keys, so it switches at about half
    the decisions."""
    units = {f"{fac}|{u}": _spread(f"u:{fac}|{u}", scale) for fac, us in recruits.items() for u in us}
    matchups = {f"{k}@{e}": _spread(f"m:{k}@{e}", scale / 2) for k in units for e in recruits}
    return rule_file(units, matchups, count, band, cap)


def write_rule(directory: Path, data: dict, seal=(20.0, 3.0)) -> Path:
    """The rule file and its readout's seal beside it, named as R0 names them."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"recruit_rule_bd_{data['hash'][:SHORT_HASH]}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    path.with_name(path.stem + "_readout.json").write_text(
        json.dumps({"hash": data["hash"], "seal": {"elo": seal[0], "se": seal[1]}}), encoding="utf-8")
    return path


def rule_game(rng, seed: int, side_a: int, data: dict, effect: float, game: Optional[str] = None,
              variant: str = "ok"):
    """(result, record) of one synthetic game of match L: the seat coin of
    `seed` puts the rule on player A or B; three turns of recruits a side,
    each of the hooked side's a rule decision between two of its faction's
    types, played as the rule and its cap say; the hooked seat wins with
    probability (1 + effect) / 2. `variant`: "overcap" switches past the
    cap, "misplayed" plays a0 at a switch."""
    rule = RecruitRule.from_data(data)
    seat = "a" if rule_seat_coin(seed) else "b"
    hooked = side_a if seat == "a" else 3 - side_a
    factions = sorted(RECRUITS)
    fac = {1: factions[int(rng.integers(6))], 2: factions[int(rng.integers(6))]}
    commands, decisions, index = [], [], {1: 0, 2: 0}
    made = {1: [], 2: []}
    fires = 0
    for turn in (1, 2, 3):
        for side in (1, 2):
            commands.append(["init_side", side])
            for slot in range(3 if turn == 1 else 1):
                units = RECRUITS[fac[side]]
                pick = rng.permutation(len(units))
                u0, u1 = units[int(pick[0])], units[int(pick[1])]
                unit = u0
                if side == hooked:
                    n0, n1 = made[side].count(u0), made[side].count(u1)
                    f = rule.value(fac[side], fac[3 - side], u0, u1, n0, n1)
                    switched = f > 0.0 and (fires < rule.cap or variant == "overcap")
                    decisions.append({"side": side, "fac": fac[side], "enemy": fac[3 - side], "u0": u0, "u1": u1,
                                      "n0": n0, "n1": n1, "i": index[side], "cmd": len(commands),
                                      "gap": float(rng.uniform(0, rule.band)), "f": f, "switched": switched,
                                      "switches_before": fires,
                                      "a0": {"kind": "recruit", "unit": u0, "to": [slot + 1, 1]},
                                      "a1": {"kind": "recruit", "unit": u1, "to": [slot + 1, 2]}})
                    fires += switched
                    unit = u1 if switched and variant != "misplayed" else u0
                    to = [slot + 1, 2 if unit == u1 and switched else 1]
                else:
                    to = [slot + 1, 1]
                commands.append(["recruit", unit, to[0], to[1], f"{seed:x}{len(commands)}"])
                made[side].append(unit)
                index[side] += 1
            commands.append(["end_turn"])
            index[side] += 1
    win_hooked = bool(rng.random() < (1 + effect) / 2)
    win_a = win_hooked == (seat == "a")
    game = game or f"game_own1a_own1b_s{side_a}_{seed}"
    short = data["hash"][:SHORT_HASH]
    procedure = f"raw:t0+eo-1.5+rr{short}"
    result = {"outcome_a": "win" if win_a else "loss", "seed": seed, "side_a": side_a, "label_a": "own1a",
              "label_b": "own1b", "procedure_a": procedure, "procedure_b": procedure, "recruit_rule": short,
              "recruit_rule_seat": seat, "margin_a": 0.0, "ended_by": "leader_killed",
              "rule_decisions": len(decisions), "rule_switches": sum(e["switched"] for e in decisions)}
    record = {"game_label": game, "setup": {"faction1": fac[1], "faction2": fac[2]},
              "players": {"a": {"side": side_a}, "b": {"side": 3 - side_a}},
              "winner": side_a if win_a else 3 - side_a, "commands": commands, "rejections": [],
              "coins": {"bands": {}, "decisions": [], "rule": {"hash": data["hash"], "p": 0.5, "d": int(seat == "a"),
                                                               "seat": seat}},
              "recruit_rule": {"hash": data["hash"], "band": rule.band, "cap": rule.cap, "seat": seat,
                               "decisions": decisions}}
    return result, record


def write_games(outdir: Path, rng, data: dict, games: int, base: int, effect: float, variant: str = "ok") -> None:
    """A match directory of synthetic games, seeds base..base + games - 1,
    sides alternating; `variant` applies to every tenth game."""
    import gzip
    outdir.mkdir(parents=True, exist_ok=True)
    for k in range(games):
        seed, side_a = base + k, 1 if k % 2 == 0 else 2
        res, rec = rule_game(rng, seed, side_a, data, effect, variant=variant if k % 10 == 0 else "ok")
        stem = rec["game_label"]
        (outdir / f"{stem}.json").write_text(json.dumps(res), encoding="utf-8")
        (outdir / f"{stem}.game.jsonl.gz").write_bytes(gzip.compress((json.dumps(rec) + "\n").encode("utf-8")))
