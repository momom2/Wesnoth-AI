#!/usr/bin/env python3
"""The hidden-units oracle: the simulator's hiding rules against real
Wesnoth, position by position.

The 17,039-replay sweep cannot certify `ambush` / `concealment` /
`submerge` / `nightstalk` (docs/wesnoth_rules.md, BACKLOG "Open after
the hide-cover review"): a replay carries the path the engine already
truncated, so nothing in it reads where a move STOPPED or which units
a side could SEE. This tool asks the engine directly. For each scripted
position it launches Wesnoth on the `ai_oracle` test board (a 14x14
grass map whose terrain, units, fog and time of day the scenario sets
at prestart from `games/oracle/setup.lua`), reads the engine's answer
to "which units does side 1 see" (`[filter_vision]`, the engine's own
predicate), orders a move through the AI stage and reads the route the
engine chose, where the unit ended and what it can see afterwards.
Then it builds the same position in the simulator, walks the SAME
route through `replay_dataset._apply_command` (the reconstruction path,
`pathfind_sim.walk_move_path`) and compares: landing hex, movement
left, the visible set before and after.

    python tools/hidden_units_oracle.py [--only SUBSTR] [--limit N] [--out FILE.json] [--sim-only]

Every case writes a record; the summary is the count of cases whose
every check agrees. `--sim-only` prints the simulator's predictions
without launching Wesnoth. Runs Wesnoth minimized on this machine, one
process per case (about 15 s each with a warm WML cache); the box
rules do not apply because nothing here is compute.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.replay_dataset import _apply_command, _build_initial_gamestate  # noqa: E402
from tools.terrain_resolver import hides_cover  # noqa: E402
from wesnoth_ai.constants import GAMES_PATH  # noqa: E402
from wesnoth_ai.visibility import units_visible_to  # noqa: E402

log = logging.getLogger("hidden_units_oracle")

BOARD = 14                                   # playable hexes per side, Wesnoth coords 1..14
SETUP_PATH = GAMES_PATH / "oracle" / "setup.lua"
SCENARIO_ID = "ai_oracle"
TOD_IDS = ("dawn", "morning", "afternoon", "dusk", "first_watch", "second_watch")

# The hider hex and the probe geometry: the mover walks straight down
# column 8 from (8,3) to (8,9), six hexes, its whole movement (a
# target past the unit's movement makes the engine's executor pick a
# nearer hex while the reconstruction walk clamps, so targets stay
# within reach); the hexes of that column adjacent to (9,7) are (8,6)
# and (8,7), so a hidden unit there stops the mover at (8,6) and a
# visible one only exerts a zone of control, which the skirmisher
# mover ignores.
HIDER = (9, 7)
MOVER_START = (8, 3)
MOVER_END = (8, 9)
FAR_CORNER = (14, 14)


@dataclass
class Case:
    name: str
    terrain: List[Tuple[int, int, str]] = field(default_factory=list)   # Wesnoth coords
    units: List[dict] = field(default_factory=list)                     # id, type, side, x, y, canrecruit
    fog: Dict[int, bool] = field(default_factory=lambda: {1: True, 2: True})
    tod_index: int = 1                                                   # morning: no nightstalk confound
    moves: List[Tuple[str, int, int]] = field(default_factory=list)      # unit id, target x, y (Wesnoth coords)
    note: str = ""


def _unit(uid: int, type_: str, side: int, x: int, y: int, leader: bool = False) -> dict:
    return {"id": f"u{uid}", "type": type_, "side": side, "x": x, "y": y, "canrecruit": leader}


def _probe(name: str, hider_type: str, code: str, *, mover_type: str = "Fencer",
           tod: int = 1, fog: bool = True, hider_at=HIDER, mover_start=MOVER_START,
           mover_end=MOVER_END, extra_units=(), note: str = "") -> Case:
    units = [_unit(1, mover_type, 1, *mover_start, leader=True),
             _unit(2, hider_type, 2, *hider_at),
             _unit(3, "Spearman", 2, *FAR_CORNER, leader=True)]
    units += list(extra_units)
    # Fog is one setting for the game in the simulator (a fog game when
    # either side has it, the corpus convention), so both sides share it.
    return Case(name=name, terrain=[(hider_at[0], hider_at[1], code)] if code else [],
                units=units, fog={1: fog, 2: fog}, tod_index=tod,
                moves=[("u1", mover_end[0], mover_end[1])], note=note)


def build_cases() -> List[Case]:
    cases: List[Case] = []
    for code in ("Gg^Fp", "Hh^Fp", "Aa^Fpa", "Gg^Fet", "Gg^Fds", "Gg^Fms", "Ss^Fp",
                 "Uu^Fp", "Mm^Fp", "Gg^Qhhf", "Gg^Ftd"):
        cases.append(_probe(f"ambush+{code}", "Elvish Ranger", code, note="forest cover"))
    for code in ("Gg", "Hh", "Gg^Efm", "Gg^Vh", "Gg^Gvs", "Gg^Es"):
        cases.append(_probe(f"ambush-{code}", "Elvish Ranger", code, note="no forest"))
    for code in ("Gg^Vh", "Hh^Vhh", "Gg^Ve", "Gg^Vc", "Ww^Vm", "Ss^Vhs", "Dd^Vda", "Gg^Vl", "Gg^Vht"):
        cases.append(_probe(f"conceal+{code}", "Fugitive", code, note="village cover"))
    for code in ("Gg^Gvs", "Gg", "Gg^Fp", "Hh"):
        cases.append(_probe(f"conceal-{code}", "Fugitive", code, note="no village (farmland is not one)"))
    for code in ("Wo", "Wot", "Wog", "Wo^Bsb|"):
        cases.append(_probe(f"submerge+{code}", "Bone Shooter", code, note="deep water cover"))
    for code in ("Ww", "Wwf", "Wwt", "Wwg", "Wwr", "Ss"):
        cases.append(_probe(f"submerge-{code}", "Bone Shooter", code, note="not deep water"))
    for tod, tag in ((4, "first_watch"), (5, "second_watch"), (1, "morning"), (3, "dusk"), (0, "dawn")):
        cases.append(_probe(f"nightstalk@{tag}", "Shadow", "", tod=tod, note="time of day only"))
    cases.append(_probe("nightstalk@first_watch+illuminated", "Shadow", "", tod=4,
                        extra_units=[_unit(4, "Mage of Light", 2, 10, 7)],
                        note="an ally's illuminates next to the hider lifts the cover"))
    cases.append(_probe("nightstalk@first_watch+mage_two_away", "Shadow", "", tod=4,
                        extra_units=[_unit(4, "Mage of Light", 2, 11, 7)],
                        note="illuminates does not reach two hexes"))
    cases.append(_probe("geometry:odd_column", "Elvish Ranger", "Gg^Fp", hider_at=(8, 7),
                        mover_start=(9, 3), mover_end=(9, 9), note="hider left of the path"))
    cases.append(_probe("geometry:on_path", "Elvish Ranger", "Gg^Fp", hider_at=(8, 7),
                        note="hidden unit on the path: blocked before it, movement kept"))
    cases.append(_probe("zoc:visible_ranger_vs_spearman", "Elvish Ranger", "Gg", mover_type="Spearman",
                        mover_end=(8, 7), note="a visible enemy's zone of control: the engine routes the "
                                               "non-skirmisher around (8,6) and ends on (8,7) with its last point"))
    cases.append(_probe("fog_off:ambush", "Elvish Ranger", "Gg^Fp", fog=False,
                        note="hides works without fog"))
    cases.append(_probe("adjacent_start:ambush", "Elvish Ranger", "Gg^Fp", mover_start=(8, 7),
                        note="an adjacent enemy sees the hider from the start"))
    cases.append(_probe("two_away_start:ambush", "Elvish Ranger", "Gg^Fp", mover_start=(8, 8),
                        mover_end=(8, 4), note="two hexes away is not adjacent: hidden until the walk up passes it"))
    two = _probe("two_moves:short_then_ambush", "Elvish Ranger", "Gg^Fp", note="a frame per move")
    two.moves = [("u1", 8, 5), ("u1", 8, 9)]
    cases.append(two)
    cases += vision_cases()
    return cases


def vision_cases() -> List[Case]:
    """What side 1 sees, not who hides (docs/wesnoth_rules.md "Vision
    and fog"): the Fencer (6 MP, 1 per grass hex) sees what it could
    reach plus the ring around it, terrain it cannot cross blocks its
    view, and what the side cleared at turn start stays clear after it
    walks away. The disc of radius max_moves the simulator drew until
    2026-09-24 answers each of the three the other way."""
    def units(enemy_at):
        return [_unit(1, "Fencer", 1, *MOVER_START, leader=True),
                _unit(2, "Spearman", 2, *enemy_at),
                _unit(3, "Spearman", 2, *FAR_CORNER, leader=True)]
    return [
        Case(name="vision:ring", units=units((8, 10)),
             note="seven hexes down on grass, one beyond the Fencer's reach: seen"),
        Case(name="vision:wall", units=units((8, 7)),
             terrain=[(x, 5, "Xu") for x in range(1, BOARD + 1)],
             note="a cave wall across the board: the Spearman four hexes away is behind it"),
        Case(name="vision:kept_after_walking_away", units=units((8, 10)), moves=[("u1", 8, 1)],
             note="cleared at turn start, still seen after the Fencer walks away from it"),
    ]


# ---------------------------------------------------------------------
# The board, for the simulator and for the setup file
# ---------------------------------------------------------------------
def _grid() -> List[List[str]]:
    n = BOARD + 2
    return [["_off^_usr" if x in (0, n - 1) or y in (0, n - 1) else "Gg" for x in range(n)]
            for y in range(n)]


def map_data(case: Case) -> str:
    grid = _grid()
    for x, y, code in case.terrain:
        grid[y][x] = code                       # border row/col 0: Wesnoth coords index the grid directly
    return "border_size=1\nusage=map\n\n" + "\n".join(", ".join(row) for row in grid)


def setup_lua(case: Case) -> str:
    def q(s: str) -> str:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    terrain = ", ".join(f"{{x={x}, y={y}, code={q(code)}}}" for x, y, code in case.terrain)
    units = ", ".join(
        f"{{id={q(u['id'])}, type={q(u['type'])}, side={u['side']}, x={u['x']}, y={u['y']}, "
        f"canrecruit={'true' if u.get('canrecruit') else 'false'}}}" for u in case.units)
    fog = ", ".join(f"[{s}]={'true' if on else 'false'}" for s, on in sorted(case.fog.items()))
    return (f"return {{\n  terrain = {{{terrain}}},\n  units = {{{units}}},\n"
            f"  fog = {{{fog}}},\n  tod_index = {case.tod_index},\n}}\n")


# ---------------------------------------------------------------------
# The simulator's side
# ---------------------------------------------------------------------
def sim_state(case: Case):
    data = {
        "game_id": f"oracle_{case.name}",
        "scenario_id": "ai_oracle",
        "map_data": map_data(case),
        "experience_modifier": 100,
        "tod_start_index": case.tod_index,
        "starting_units": [
            {"uid": int(u["id"][1:]), "type": u["type"], "side": u["side"],
             "x": u["x"] - 1, "y": u["y"] - 1, "is_leader": bool(u.get("canrecruit"))}
            for u in case.units],
        "starting_sides": [
            {"side": s, "fog": bool(case.fog.get(s, True)), "shroud": False, "recruit": [],
             "gold": 0, "base_income": 2, "village_income": 2, "village_support": 1, "faction": ""}
            for s in (1, 2)],
    }
    gs = _build_initial_gamestate(data)
    _apply_command(gs, ["init_side", 1])
    return gs


def _sim_visible(gs, side: int = 1) -> List[str]:
    return sorted(u.id for u in units_visible_to(gs, side))


def _sim_unit(gs, uid: str):
    return next(u for u in gs.map.units if u.id == uid)


def sim_predictions(case: Case, engine_paths: Optional[List[List[Tuple[int, int]]]] = None) -> dict:
    """The simulator's answers: visible set at the start, then per move
    the landing hex, movement left, stop reason and the visible set
    after. `engine_paths` (Wesnoth coords) are the routes the engine
    chose; without them a straight line to the target is walked, which
    is what --sim-only prints."""
    gs = sim_state(case)
    out = {"visible_start": _sim_visible(gs), "moves": []}
    for k, (uid, tx, ty) in enumerate(case.moves):
        u = _sim_unit(gs, uid)
        if engine_paths is not None and k < len(engine_paths) and engine_paths[k]:
            path = engine_paths[k]
        else:
            path = [(u.position.x + 1, u.position.y + 1)]
            x, y = path[0]
            while (x, y) != (tx, ty):                     # a column walk, the probe geometry
                y += 1 if ty > y else (-1 if ty < y else 0)
                x += 1 if tx > x else (-1 if tx < x else 0)
                path.append((x, y))
        xs = [p[0] - 1 for p in path]
        ys = [p[1] - 1 for p in path]
        _apply_command(gs, ["move", xs, ys, u.side])
        walk = getattr(gs.global_info, "_last_move_walk", {}) or {}
        u2 = _sim_unit(gs, uid)
        out["moves"].append({
            "path": [list(p) for p in path],
            "landed": [u2.position.x + 1, u2.position.y + 1],
            "moves_left": int(u2.current_moves),
            "stop_reason": walk.get("stop_reason"),
            "visible_after": _sim_visible(gs),
        })
    return out


# ---------------------------------------------------------------------
# The engine's side
# ---------------------------------------------------------------------
def _frame(game, timeout: float) -> Optional[dict]:
    payload = game.read_state(timeout=timeout)
    if payload is None:
        return None
    return json.loads(payload)


def _engine_visible(frame: dict, side: int = 1) -> List[str]:
    return sorted(u["id"] for u in (frame.get("oracle_units") or []) if u.get(f"visible_to_{side}"))


def run_engine(case: Case, *, first_timeout: float = 240.0, timeout: float = 90.0) -> dict:
    from wesnoth_ai.wesnoth_interface import WesnothGame
    SETUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETUP_PATH.write_text(setup_lua(case), encoding="utf-8", newline="\n")
    game = WesnothGame(label=f"oracle_{case.name}", scenario_id=SCENARIO_ID)
    out: dict = {"ok": False, "moves": []}
    try:
        game.start_wesnoth()
        frame = _frame(game, first_timeout)
        if frame is None:
            out["error"] = "no first frame"
            return out
        game.adopt_game_id(frame["game_id"])
        out["time_of_day"] = frame.get("time_of_day")
        out["units_start"] = {u["id"]: [u["x"], u["y"]] for u in (frame.get("oracle_units") or [])}
        out["visible_start"] = _engine_visible(frame)
        for uid, tx, ty in case.moves:
            pos = out["units_start"].get(uid)
            last = out["moves"][-1] if out["moves"] else None
            if last and last.get("final"):
                pos = last["final"][:2]
            if pos is None:
                out["error"] = f"unit {uid} not on the board"
                return out
            game.send_action({"type": "move", "start_x": pos[0], "start_y": pos[1],
                              "target_x": tx, "target_y": ty})
            # The stage emits one frame per loop turn; the frame that
            # answers this move carries `last_action`. A frame without it
            # is one the stage emitted before it read the action (seen
            # once in 54 cases): read on.
            frame = None
            for _attempt in range(3):
                frame = _frame(game, timeout)
                if frame is None or frame.get("last_action"):
                    break
            if frame is None:
                out["error"] = "no frame after the move"
                return out
            la = frame.get("last_action") or {}
            oracle = la.get("oracle") or {}
            final = oracle.get("final") or {}
            unit_start = out["units_start"].get(uid)
            out["moves"].append({
                "path": [[p["x"], p["y"]] for p in (oracle.get("path") or [])],
                "cost": oracle.get("cost"),
                # The engine walks a route past the unit's movement as far
                # as the points go; the reconstruction walk clamps instead.
                # A probe whose route costs more than the mover has, and
                # that nothing else stopped (status 0: no ambush, no
                # block), is a harness defect, flagged rather than read
                # as a rule.
                "past_movement": bool(oracle.get("cost") is not None and unit_start is not None
                                      and str(oracle.get("status")) == "0"
                                      and oracle.get("cost") > (next((u.get("max_moves", 0) for u in
                                          (frame.get("oracle_units") or []) if u["id"] == uid), 0) or 0)),
                "status": oracle.get("status"),
                "success": la.get("success"),
                "final": [final.get("x"), final.get("y"), final.get("moves")] if final else None,
                "visible_after": _engine_visible(frame),
            })
        out["ok"] = True
        return out
    finally:
        game.terminate()


# ---------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------
def compare(case: Case, engine: dict, sim: dict) -> dict:
    checks = []

    def check(name, e, s):
        checks.append({"check": name, "engine": e, "sim": s, "agree": e == s})

    check("visible at start", engine.get("visible_start"), sim.get("visible_start"))
    if engine.get("time_of_day") is not None:
        check("time of day", engine["time_of_day"], TOD_IDS[case.tod_index])
    for k, em in enumerate(engine.get("moves", [])):
        sm = sim["moves"][k] if k < len(sim["moves"]) else {}
        final = em.get("final") or [None, None, None]
        if em.get("past_movement"):
            checks.append({"check": f"move {k} route within movement", "engine": em.get("cost"),
                           "sim": "harness: route costs more than the mover has", "agree": False})
        check(f"move {k} landing hex", final[:2], sm.get("landed"))
        check(f"move {k} movement left", final[2], sm.get("moves_left"))
        check(f"move {k} visible after", em.get("visible_after"), sm.get("visible_after"))
    return {"case": case.name, "note": case.note, "terrain": case.terrain, "tod": TOD_IDS[case.tod_index],
            "fog": case.fog, "agree": all(c["agree"] for c in checks) and bool(engine.get("ok")),
            "engine_ok": bool(engine.get("ok")), "engine_error": engine.get("error"),
            "checks": checks, "engine": engine, "sim": sim,
            "rule_says_cover": [bool(hides_cover(code, ab)) for (_x, _y, code) in case.terrain
                                for ab in ("ambush", "concealment", "submerge")]}


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default=None, help="run the cases whose name contains this")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sim-only", action="store_true", help="print the simulator's predictions, no Wesnoth")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    cases = [c for c in build_cases() if not args.only or args.only in c.name]
    if args.limit:
        cases = cases[:args.limit]
    records = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        if args.sim_only:
            sim = sim_predictions(case)
            print(f"{case.name}: visible {sim['visible_start']} | " +
                  " ; ".join(f"landed {m['landed']} mp {m['moves_left']} ({m['stop_reason']}) "
                             f"visible {m['visible_after']}" for m in sim["moves"]))
            continue
        t1 = time.time()
        engine = run_engine(case)
        paths = [[tuple(p) for p in m.get("path") or []] for m in engine.get("moves", [])]
        sim = sim_predictions(case, engine_paths=paths) if engine.get("ok") else sim_predictions(case)
        rec = compare(case, engine, sim)
        rec["wall_s"] = round(time.time() - t1, 1)
        records.append(rec)
        bad = "; ".join(f"{c['check']}: engine {c['engine']} sim {c['sim']}"
                        for c in rec["checks"] if not c["agree"])
        verdict = "AGREE" if rec["agree"] else "DIFFER"
        tail = (" -- " + bad if bad else "") + ("" if engine.get("ok") else f" -- engine: {engine.get('error')}")
        print(f"[{i}/{len(cases)}] {case.name}: {verdict} ({rec['wall_s']} s){tail}", flush=True)
    if args.sim_only:
        return 0
    n_ok = sum(1 for r in records if r["agree"])
    summary = {"cases": len(records), "agree": n_ok, "differ": len(records) - n_ok,
               "engine_failed": sum(1 for r in records if not r["engine_ok"]),
               "wall_s": round(time.time() - t0, 1)}
    print(json.dumps(summary))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "records": records}, indent=1), encoding="utf-8")
        print("wrote", args.out)
    return 0 if n_ok == len(records) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
