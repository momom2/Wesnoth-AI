#!/usr/bin/env python3
"""The scenario-init oracle: the starting state we build for each pool
scenario against the one real Wesnoth builds
(docs/scenario_build_plan_20260922.md, W4).

For each scenario it launches a real multiplayer game -- the scenario's
own [side] blocks, the default era, both factions named, every side
played by the AI -- with `init_oracle_ai.cfg` on side 1, whose Lua
(`lua/init_oracle.lua`) reports the whole board at side 1's first turn.
It then builds the same game the way self-play does
(`build_scenario_gamestate`, then `WesnothSim`, which fires the
scenario's events and side 1's first init_side) with the leaders the
engine drew, and compares field by field (`compare`).

**The launch stands in for a lobby.** A command-line start skips
`configure_engine::write_parameters` (src/game_initialization/
configure_engine.cpp:168-200, 1.18.4), which in a hosted game writes
the host's fog, shroud, village gold and village support into every
[side] that does not declare them; without it a side's villages pay the
engine's base 1 gold (src/team.cpp:236). The harness writes those same
values with `--parm`, and decides which sides lack them from Wesnoth's
own preprocessing of the scenario (`engine_declarations`), never from
our reader, so a reader that misses a declaration cannot hide behind the
harness.

    python tools/scenario_init_oracle.py [--only SUBSTR ...] [--out FILE.json]
                                         [--frames DIR [--from-frames]]

Runs Wesnoth minimized on this machine, one process per scenario (about
50 s each); the box rules do not apply because nothing here is compute.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from wesnoth_ai.rules.build_scenario_templates import (LADDER_SRC, MINI_SRC,  # noqa: E402
                                                       index_preprocessed, run_preprocessor)
from tools.replay_dataset import _lawful_bonus_at, side_income  # noqa: E402
from tools.replay_extract import parse_wml  # noqa: E402
from wesnoth_ai.rules.scenario_pool import (LADDER_SCENARIO_IDS, MINI_MAP_SCENARIO_IDS,  # noqa: E402
                                            ScenarioSetup, _scenario_tod_info,
                                            build_scenario_gamestate, classify_scenario)
from wesnoth_ai.sim.traits import TRAITS  # noqa: E402
from tools.wesnoth_sim import WesnothSim  # noqa: E402
from wesnoth_ai.sim.classes import PLAYER_SIDES  # noqa: E402

log = logging.getLogger("scenario_init_oracle")

POOL = list(LADDER_SCENARIO_IDS) + list(MINI_MAP_SCENARIO_IDS)
FACTIONS = ("Drakes", "Knalgan Alliance", "Loyalists", "Northerners", "Rebels", "Undead")
AI_CONFIG = "~add-ons/wesnoth_ai/init_oracle_ai.cfg"
TOD_IDS = ("dawn", "morning", "afternoon", "dusk", "first_watch", "second_watch")
SIDE_FIELDS = ("gold", "base_income", "total_income", "net_income", "village_gold",
               "village_support", "fog", "recruit", "faction")
UNIT_FIELDS = ("hitpoints", "max_hitpoints", "moves", "max_moves", "experience", "max_experience")

# write_parameters with map settings on, per [side] lacking the
# attribute: (the scenario attribute it takes, the engine default).
# Defaults: configure_engine.cpp:72-140 and src/map_settings.cpp:40-53
# (a normal MP game: fog on, 2 village gold, support 1, no shroud).
LOBBY_SIDE_WRITES = {
    "fog": ("mp_fog", "yes"),
    "shroud": ("mp_shroud", "no"),
    "village_gold": ("mp_village_gold", "2"),
    "village_support": ("mp_village_support", "1"),
}


@dataclass
class Declared:
    """A scenario's attributes as Wesnoth's preprocessor expands it."""
    scenario: Dict[str, str]
    sides: Dict[int, Dict[str, str]]


def engine_declarations(scenario_ids: Sequence[str]) -> Dict[str, Declared]:
    """Expand the pool's sources with the game's own preprocessor and
    read each scenario's [multiplayer] and [side] attributes."""
    wanted = set(scenario_ids)
    blocks: Dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="init_oracle_pp_") as td:
        for name, src, ids in (("ladder", LADDER_SRC, LADDER_SCENARIO_IDS),
                               ("mini", MINI_SRC, MINI_MAP_SCENARIO_IDS)):
            if wanted & set(ids):
                run_preprocessor(src, Path(td) / name)
                blocks.update(index_preprocessed(Path(td) / name))
    out = {}
    for sid in scenario_ids:
        if sid not in blocks:
            raise RuntimeError(f"Wesnoth's preprocessor did not produce scenario {sid}")
        block = parse_wml(blocks[sid]).first("multiplayer")
        sides = {int(s.attrs["side"]): dict(s.attrs) for s in block.all("side") if s.attrs.get("side")}
        out[sid] = Declared(scenario=dict(block.attrs), sides=sides)
    return out


def _wml_bool(value: str) -> str:
    return "yes" if str(value).strip().lower() in ("yes", "true", "1") else "no"


def lobby_parms(decl: Declared) -> List[Tuple[int, str, str]]:
    """(side, attribute, value) that a hosted game's write_parameters
    would add to the scenario's sides."""
    parms = []
    for side, attrs in sorted(decl.sides.items()):
        for attr, (scenario_attr, default) in LOBBY_SIDE_WRITES.items():
            if str(attrs.get(attr, "")).strip():
                continue
            value = str(decl.scenario.get(scenario_attr, "")).strip() or default
            parms.append((side, attr, _wml_bool(value) if attr in ("fog", "shroud") else value))
    return parms


LOBBY_XP_MODIFIER = 70   # settings::get_xp_modifier(""): the default when a scenario sets none




def launch_args(scenario_id: str, factions: Tuple[str, str], decl: Declared) -> List[str]:
    args = ["--multiplayer", f"--scenario={scenario_id}", "--era=era_default",
            "--side", f"1:{factions[0]}", "--side", f"2:{factions[1]}",
            "--controller", "1:ai", "--controller", "2:ai", "--ai-config", f"1:{AI_CONFIG}"]
    for side, attr, value in lobby_parms(decl):
        args += ["--parm", f"{side}:{attr}:{value}"]
    return args


def run_engine(args: List[str], label: str, timeout: float = 300.0) -> dict:
    from wesnoth_ai.wesnoth_interface import WesnothGame
    game = WesnothGame(label=label, launch_args=args)
    try:
        game.start_wesnoth()
        frame = game.read_state(timeout=timeout)
        if frame is None:
            raise RuntimeError(f"no frame from Wesnoth (log {game._log_path})")
        return json.loads(frame)
    finally:
        game.terminate()


# ---------------------------------------------------------------------
# Our side, in the engine record's shape (Wesnoth's 1-based coordinates)
# ---------------------------------------------------------------------
def applied_experience_modifier(engine: dict) -> int:
    """The modifier the engine's game applies, read from a unit type whose
    base experience is 100 (lua/init_oracle.lua)."""
    probe = engine["settings"]["experience_probe"]
    if int(probe["base"]) != 100:
        raise RuntimeError(f"experience probe {probe['type']} has base {probe['base']}, not 100")
    return int(probe["applied"])


def lobby_experience_modifier(decl: Declared) -> int:
    """What a lobby with map settings plays: the scenario's value through
    `lexical_cast_default<int>(value, 70)` (src/map_settings.cpp:50-53)."""
    value = str(decl.scenario.get("experience_modifier", "")).strip()
    return int(value) if value.isdigit() else LOBBY_XP_MODIFIER


def our_setup(scenario_id: str, factions: Tuple[str, str], engine: dict) -> ScenarioSetup:
    """The game the engine played: same factions, the leaders it drew, and
    its start time when the scenario draws one."""
    leaders = {u["side"]: u["type"] for u in engine["units"] if u["canrecruit"] and u["side"] in PLAYER_SIDES}
    _, random_start, _ = _scenario_tod_info(scenario_id)
    return ScenarioSetup(scenario_id, factions[0], leaders[1], factions[1], leaders[2],
                         category=classify_scenario(scenario_id),
                         tod_start=TOD_IDS.index(engine["time_of_day"]) if random_start else None)


def our_state(setup: ScenarioSetup, experience_modifier: int):
    """Our starting state at side 1's first turn, built at the engine's
    applied experience modifier (the command line plays 100 where a lobby
    plays 70; the lobby's value is its own field, lobby.experience_modifier)."""
    gs = build_scenario_gamestate(setup, experience_modifier=experience_modifier)
    return WesnothSim(gs, scenario_id=setup.scenario_id).gs


def our_statuses(unit) -> List[str]:
    """A unit's statuses as the engine names them. `ai_special=guardian`
    is the engine's STATE_GUARDIAN (src/units/unit.cpp:659); we keep it as
    the unit's `_ai_guardian` flag, which the neutral AI reads."""
    found = {str(t) for t in unit.statuses}
    if getattr(unit, "_ai_guardian", False):
        found.add("guardian")
    return sorted(found)


def our_record(gs) -> dict:
    gi = gs.global_info
    sides = []
    for side, s in enumerate(gs.sides, start=1):
        income, net_upkeep = side_income(gs, side)
        sides.append({"side": side, "gold": s.current_gold, "base_income": s.base_income,
                      "total_income": income, "net_income": income - net_upkeep,
                      "village_gold": gi.village_gold, "village_support": gi.village_upkeep,
                      "fog": bool(getattr(gi, "_fog", True)), "recruit": list(s.recruits),
                      "faction": s.faction})
    units = [{"type": u.name, "side": u.side, "x": u.position.x + 1, "y": u.position.y + 1,
              "canrecruit": u.is_leader, "hitpoints": u.current_hp, "max_hitpoints": u.max_hp,
              "moves": u.current_moves, "max_moves": u.max_moves, "experience": u.current_exp,
              "max_experience": u.max_exp, "traits": sorted(str(t) for t in u.traits),
              "status": our_statuses(u)} for u in gs.map.units]
    owners = [{"x": x + 1, "y": y + 1, "side": side}
              for (x, y), side in (getattr(gi, "_village_owner", None) or {}).items() if side]
    codes = getattr(gi, "_terrain_codes", {})
    return {"time_of_day": gi.time_of_day, "sides": sides, "units": units, "village_owners": owners,
            "terrain": {(x + 1, y + 1): code for (x, y), code in codes.items()},
            "lawful_bonus": {(x + 1, y + 1): _lawful_bonus_at(gs, x, y, gi.turn_number) for (x, y) in codes},
            "empty_sides": set(getattr(gi, "_null_controller_sides", ()) or ()),
            "acting_sides": set(getattr(gi, "_neutral_actor_sides", ()) or ()) | set(PLAYER_SIDES)}


# ---------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------
class Field:
    """One compared field: how many items agree, and the first few that do not."""

    def __init__(self):
        self.agree, self.total, self.diffs = 0, 0, []

    def check(self, item, engine_value, our_value) -> None:
        self.total += 1
        if engine_value == our_value:
            self.agree += 1
        elif len(self.diffs) < 8:
            self.diffs.append({"item": item, "engine": engine_value, "ours": our_value})

    def as_dict(self) -> dict:
        return {"agree": self.agree, "total": self.total, "diffs": self.diffs}


def _normalised(field: str, value):
    if field == "recruit":
        return sorted(value or [])
    if field == "fog":
        return bool(value)
    return value


NOT_LIVING_PARTS = {"undrainable", "unpoisonable", "unplagueable"}


def engine_statuses(statuses: List[str]) -> List[str]:
    """The engine's status list without `not_living` where it is only the
    name for all three of its parts: `unit::get_states` adds it whenever
    they are all set (src/units/unit.cpp:1334-1337), so it is not a
    status of its own."""
    found = set(statuses)
    if NOT_LIVING_PARTS <= found:
        found.discard("not_living")
    return sorted(found)


def engine_named_traits(traits: List[str]) -> List[str]:
    """The engine's traits that our units carry by name (wesnoth_ai/sim/traits.TRAITS).
    A custom [trait], such as the statues' remove_hp, exists in our state
    only through its effects, which the unit's numbers compare."""
    return sorted(t for t in traits if t in TRAITS)


def compare(engine: dict, ours: dict) -> Dict[str, dict]:
    fields: Dict[str, Field] = {}

    def f(name: str) -> Field:
        return fields.setdefault(name, Field())

    e_sides = {s["side"]: s for s in engine["sides"]}
    o_sides = {s["side"]: s for s in ours["sides"]}
    for side in PLAYER_SIDES:
        for name in SIDE_FIELDS:
            f(f"side.{name}").check(side, _normalised(name, e_sides[side].get(name)),
                                    _normalised(name, (o_sides.get(side) or {}).get(name)))
    for side, s in e_sides.items():
        our_turn = ("acting" if side in ours["acting_sides"]
                    else "empty" if side in ours["empty_sides"] else "absent")
        f("side.turn").check(side, "empty" if s["controller"] == "null" else "acting", our_turn)

    e_units = {(u["side"], u["x"], u["y"]): u for u in engine["units"]}
    o_units = {(u["side"], u["x"], u["y"]): u for u in ours["units"]}
    for key in sorted(set(e_units) | set(o_units)):
        e, o = e_units.get(key), o_units.get(key)
        f("unit.present").check(key, e is not None, o is not None)
        if e is None or o is None:
            continue
        f("unit.type").check(key, e["type"], o["type"])
        f("unit.leader").check(key, bool(e["canrecruit"]), bool(o["canrecruit"]))
        traits = engine_named_traits(e["traits"])
        f("unit.traits").check(key, traits, o["traits"])
        f("unit.status").check(key, engine_statuses(e["status"]), o["status"])
        # Where the traits differ the numbers may differ for that reason
        # alone, so their diffs say so.
        item = key if traits == o["traits"] else (*key, "traits differ")
        for name in UNIT_FIELDS:
            f(f"unit.{name}").check(item, e[name], o[name])

    e_owner = {(v["x"], v["y"]): v["side"] for v in engine["village_owners"]}
    o_owner = {(v["x"], v["y"]): v["side"] for v in ours["village_owners"]}
    for hex_ in sorted(set(e_owner) | set(o_owner)):
        f("village.owner").check(hex_, e_owner.get(hex_, 0), o_owner.get(hex_, 0))

    exceptions = {(b["x"], b["y"]): b["lawful_bonus"] for b in engine["lawful_bonus_exceptions"]}
    for y, row in enumerate(engine["terrain_rows"], start=1):
        for x, code in enumerate(row, start=1):
            f("terrain").check((x, y), code, ours["terrain"].get((x, y)))
            f("lawful_bonus").check((x, y), exceptions.get((x, y), engine["lawful_bonus"]),
                                    ours["lawful_bonus"].get((x, y)))
    f("time_of_day").check("turn 1", engine["time_of_day"], ours["time_of_day"])
    return {name: field.as_dict() for name, field in sorted(fields.items())}


def factions_for(index: int) -> Tuple[str, str]:
    """A rotation that puts every faction on each side about equally often."""
    return FACTIONS[index % len(FACTIONS)], FACTIONS[(index + 2) % len(FACTIONS)]


def engine_frame(index: int, scenario_id: str, decl: Declared,
                 frames: Optional[Path], from_frames: bool) -> dict:
    """The engine's report for this scenario: launched, or read back from
    `frames` (one JSON per scenario) when `from_frames` is set."""
    path = frames / f"{scenario_id}.json" if frames else None
    if from_frames:
        return json.loads(path.read_text(encoding="utf-8"))
    engine = run_engine(launch_args(scenario_id, factions_for(index), decl), label=f"init_{index}")
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(engine), encoding="utf-8")
    return engine


def run_scenario(index: int, scenario_id: str, decl: Declared,
                 frames: Optional[Path] = None, from_frames: bool = False) -> dict:
    factions = factions_for(index)
    record = {"scenario_id": scenario_id, "factions": list(factions),
              "parms": [list(p) for p in lobby_parms(decl)]}
    try:
        engine = engine_frame(index, scenario_id, decl, frames, from_frames)
    except Exception as exc:  # noqa: BLE001 -- recorded, the sweep goes on
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    record["leaders"] = sorted(u["type"] for u in engine["units"] if u["canrecruit"])
    record["time_of_day"] = engine["time_of_day"]
    setup = our_setup(scenario_id, factions, engine)
    applied = applied_experience_modifier(engine)
    record["experience_modifier_played"] = applied
    fields = compare(engine, our_record(our_state(setup, applied)))
    lobby = Field()
    lobby.check("experience_modifier", lobby_experience_modifier(decl),
                build_scenario_gamestate(setup).global_info._experience_modifier)
    fields["lobby.experience_modifier"] = lobby.as_dict()
    record["fields"] = dict(sorted(fields.items()))
    return record


def summarize(records: List[dict]) -> dict:
    per_field: Dict[str, Counter] = {}
    for rec in records:
        for name, res in (rec.get("fields") or {}).items():
            c = per_field.setdefault(name, Counter())
            c["agree"] += res["agree"]
            c["total"] += res["total"]
            c["scenarios_all_agree"] += res["agree"] == res["total"]
    return {"scenarios": len(records), "errors": sum("error" in r for r in records),
            "scenarios_all_agree": sum(bool(r.get("fields")) and all(
                v["agree"] == v["total"] for v in r["fields"].values()) for r in records),
            "fields": {k: dict(v) for k, v in sorted(per_field.items())}}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None,
                    help="run the scenarios whose id contains any of these")
    ap.add_argument("--out", type=Path, default=None,
                    help="the record (JSON); scenarios run now replace their entries, others are kept")
    ap.add_argument("--frames", type=Path, default=None,
                    help="save each engine report in this directory (one JSON per scenario)")
    ap.add_argument("--from-frames", action="store_true",
                    help="compare against the reports saved in --frames instead of launching Wesnoth")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    if args.from_frames and not args.frames:
        ap.error("--from-frames reads the directory named by --frames")
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")

    chosen = [(i, sid) for i, sid in enumerate(POOL)
              if not args.only or any(part in sid for part in args.only)]
    decls = engine_declarations([sid for _, sid in chosen])
    kept = {}
    if args.out and args.out.exists():
        kept = {r["scenario_id"]: r for r in json.loads(args.out.read_text(encoding="utf-8"))["records"]}
    records = []
    for i, sid in chosen:
        rec = run_scenario(i, sid, decls[sid], args.frames, args.from_frames)
        records.append(rec)
        kept[sid] = rec
        bad = [k for k, v in (rec.get("fields") or {}).items() if v["agree"] != v["total"]]
        log.info(f"{sid}: {rec.get('error') or ('all fields agree' if not bad else 'differs: ' + ', '.join(bad))}")
        if args.out:                        # written as it goes, not as one dump at the end
            merged = [kept[sid] for sid in POOL if sid in kept]
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({"summary": summarize(merged), "records": merged},
                                           indent=1, default=str), encoding="utf-8")
    summary = summarize(records)
    print(json.dumps(summary, indent=1))
    return 0 if summary["scenarios_all_agree"] == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
