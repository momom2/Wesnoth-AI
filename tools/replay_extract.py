"""Extract (state, action) training pairs from a Wesnoth 1.18 replay.

Input:  a .bz2 save-with-replay (the kind replays.wesnoth.org archives —
        contains [replay_start] with starting state + [replay] with a
        sequence of [command] tags).
Output: JSONL records, one per observed player decision:
        { "game_id": str,
          "scenario_id": str,
          "factions": [str, str],
          "acting_side": 1|2,
          "turn": int,
          "action": { "type": "recruit"|"move"|"attack"|"recall"|"end_turn",
                      ... },
          "state_before": { "units": [...], "gold": [..],
                            "villages_owned": {...}, ... } }

Scope for this first pass: positions, unit types, gold, side turn.
HP is tracked approximately (expected-value combat, full for non-combat
transitions). Exact combat resolution requires replaying the stored
RNG seeds through Wesnoth's damage formula — a larger task we defer.
For behavior cloning the position/recruit/movement signal is the main
driver anyway.

Runs as a CLI: `python tools/replay_extract.py PATH_TO_REPLAY.bz2`
(emits JSONL to stdout) or `python tools/replay_extract.py DIR OUT.jsonl`
(walks DIR, writes one line per action across all files).
"""

from __future__ import annotations

import bz2
import gzip
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


log = logging.getLogger("replay_extract")


# --------------------------------------------------------------------
# Minimal WML parser — structured dicts, in-memory, streaming-line.
# --------------------------------------------------------------------

TAG_OPEN_RE  = re.compile(r'^\s*\[([a-zA-Z_][a-zA-Z0-9_]*)\]\s*$')
TAG_CLOSE_RE = re.compile(r'^\s*\[/([a-zA-Z_][a-zA-Z0-9_]*)\]\s*$')
# Values may be quoted "..." or bare 123 / true / false. We handle both.
KEY_RE = re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*?)\s*$')
# Multi-line string values in WML are quoted and may span lines, closing
# with a standalone `"`. We detect the open-quote here, then glue lines
# until we see the closing quote.


class WMLNode:
    """A WML tag instance: attributes + ordered children (also WMLNodes)."""
    __slots__ = ("tag", "attrs", "children")

    def __init__(self, tag: str):
        self.tag: str = tag
        self.attrs: Dict[str, str] = {}
        self.children: List["WMLNode"] = []

    def first(self, tag: str) -> Optional["WMLNode"]:
        for c in self.children:
            if c.tag == tag:
                return c
        return None

    def all(self, tag: str) -> List["WMLNode"]:
        return [c for c in self.children if c.tag == tag]


def _strip_quotes(v: str) -> str:
    if len(v) >= 2 and v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    return v


def _safe_int(v, default: int = 0) -> int:
    """Robust `int()` for WML attribute values. Some replays in the
    wild carry malformed quoted values where multiple attrs got
    concatenated (e.g. `village_gold="1 controller=human"` in a
    handful of 2p Evil Factory saves). We salvage the leading integer
    if present, else return `default`. None / empty / whitespace also
    map to `default`."""
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default
    try:
        return int(s)
    except (ValueError, TypeError):
        # Pull leading integer literal if any. Wesnoth doesn't emit
        # negative ints in side-summary or per-unit health attrs we
        # care about, so a lenient `\d+` prefix is enough.
        import re as _re
        m = _re.match(r"-?\d+", s)
        if m:
            try:
                return int(m.group(0))
            except (ValueError, TypeError):
                pass
        return default


def parse_wml(text: str) -> WMLNode:
    """Parse a WML blob into a root node. Very small, not spec-complete,
    but handles the shape replay files use.

    Known simplifications:
      - Multiline string values: we concatenate lines until a standalone
        closing quote. Escape handling is minimal.
      - `+=` concatenation (seen in some WML) is treated as `=`.
      - Preprocessor directives (`#textdomain`, `#ifdef`) are skipped.
    """
    root = WMLNode("__root__")
    stack: List[WMLNode] = [root]
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1; continue

        m = TAG_OPEN_RE.match(line)
        if m:
            node = WMLNode(m.group(1))
            stack[-1].children.append(node)
            stack.append(node)
            i += 1; continue

        m = TAG_CLOSE_RE.match(line)
        if m:
            if stack and stack[-1].tag == m.group(1):
                stack.pop()
            i += 1; continue

        m = KEY_RE.match(line)
        if m:
            key, val = m.group(1), m.group(2)
            # Multiline quoted string? Starts with `"`, doesn't end with `"`.
            if val.startswith('"') and not (len(val) >= 2 and val.endswith('"')):
                parts = [val]
                i += 1
                while i < len(lines):
                    parts.append(lines[i])
                    if lines[i].rstrip().endswith('"'):
                        break
                    i += 1
                val = "\n".join(parts)
            stack[-1].attrs[key] = _strip_quotes(val)
        i += 1
    return root


def parse_replay_file(path: Path) -> WMLNode:
    with bz2.open(path, "rb") as f:
        text = f.read().decode("utf-8", errors="replace")
    return parse_wml(text)


# --------------------------------------------------------------------
# Game-state reconstruction
# --------------------------------------------------------------------

@dataclass
class Unit:
    uid: int
    unit_type: str
    side: int
    x: int
    y: int
    hp: int               # approximate
    max_hp: int
    moves_left: int
    max_moves: int
    has_attacked: bool = False
    is_leader: bool = False
    petrified: bool = False  # Thousand Stings statues / Cockatrice victims


@dataclass
class SideState:
    side_num: int
    faction: str = ""
    gold: int = 0
    village_income: int = 2
    base_income: int = 2
    recruit_list: List[str] = field(default_factory=list)
    # Leader unit-type from the [side] block (Wesnoth's `type=` attr).
    # When the replay's initial [side] has no [unit] children (a fresh
    # scenario template), this is what the engine spawns at the side's
    # starting position. Persisted so the save-state dumper can recover
    # the leader type even if our reconstruction lost it.
    leader_type: str = ""
    color: str = ""


@dataclass
class GameState:
    # Static-ish
    scenario_id: str
    map_size: Tuple[int, int] = (0, 0)

    # Dynamic
    turn: int = 1
    current_side: int = 1
    units: Dict[int, Unit] = field(default_factory=dict)
    sides: Dict[int, SideState] = field(default_factory=dict)
    villages_owned: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def snapshot(self) -> dict:
        """Minimal JSON-friendly snapshot. We deliberately don't include
        the full hex map (it's static per scenario — a single lookup on
        the Python side given scenario_id). Only what changes."""
        return {
            "turn": self.turn,
            "current_side": self.current_side,
            "units": [
                {
                    "uid": u.uid, "type": u.unit_type, "side": u.side,
                    "x": u.x, "y": u.y, "hp": u.hp, "max_hp": u.max_hp,
                    "moves_left": u.moves_left, "has_attacked": u.has_attacked,
                    "is_leader": u.is_leader,
                }
                for u in self.units.values()
            ],
            "sides": [
                {
                    "side": s.side_num, "faction": s.faction,
                    "gold": s.gold, "recruit": list(s.recruit_list),
                }
                for s in sorted(self.sides.values(), key=lambda s: s.side_num)
            ],
            "villages_owned": {
                f"{x},{y}": side for (x, y), side in self.villages_owned.items()
            },
        }


# Approximate unit stats. We don't ship Wesnoth's full unit DB, so for
# replay extraction we default to typical values. The model will still
# learn useful positional/recruit patterns; HP-sensitive tactics are
# the sacrifice.
_DEFAULT_UNIT = {"max_hp": 33, "max_moves": 5, "is_leader": False}

# Lazily-populated from unit_stats.json (scraped from Wesnoth source).
_UNIT_DB_CACHE: Dict[str, dict] = {}


def _unit_stats(unit_type: str) -> dict:
    """Return {max_hp, max_moves, is_leader} for a unit type. Loads from
    unit_stats.json (scraped from Wesnoth source) on first call; falls
    back to _DEFAULT_UNIT for unknown types (e.g. Ladder Era variants)."""
    global _UNIT_DB_CACHE
    if not _UNIT_DB_CACHE:
        try:
            db_path = Path(__file__).resolve().parent.parent / "unit_stats.json"
            with db_path.open(encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.get("units", {}).items():
                _UNIT_DB_CACHE[k] = {
                    "max_hp":    int(v.get("hitpoints", 33)),
                    "max_moves": int(v.get("moves", 5)),
                    "is_leader": False,
                }
        except FileNotFoundError:
            pass  # silently fall back
    return _UNIT_DB_CACHE.get(unit_type, dict(_DEFAULT_UNIT))


def _parse_map_starting_positions(map_data: str) -> Dict[int, Tuple[int, int]]:
    """Scan map_data for cells with a leading '<digit> ' starting-pos
    marker (e.g. '1 Gg^Vh' = side 1 starts at this hex). Return
    {side_number: (x, y)} in Python 0-indexed coords (after border
    stripping — see parse_map_data)."""
    out: Dict[int, Tuple[int, int]] = {}
    rows = [r for r in map_data.splitlines() if r.strip()]
    border = 1
    for y, row in enumerate(rows):
        cells = [c.strip() for c in row.split(",")]
        for x, cell in enumerate(cells):
            if len(cell) >= 2 and cell[0].isdigit() and cell[1] == " ":
                # Subtract the border offset to align with Python
                # 0-indexed playable-hex coords.
                out[int(cell[0])] = (x - border, y - border)
    return out


def build_initial_state(root: WMLNode) -> GameState:
    """Populate GameState from [replay_start] (or [snapshot]) block."""
    snap = root.first("replay_start") or root.first("snapshot")
    if snap is None:
        # Some saves wrap the start state inside [scenario] directly.
        snap = root.first("scenario")
    if snap is None:
        raise ValueError("no [replay_start] / [snapshot] / [scenario] block found")

    scenario_id = snap.attrs.get("id", "unknown")
    gs = GameState(scenario_id=scenario_id)

    # Map size: parse map_data to count rows/cols (rough).
    md = snap.attrs.get("map_data", "")
    if md:
        rows = [r for r in md.splitlines() if r.strip()]
        if rows:
            gs.map_size = (len(rows[0].split(",")), len(rows))
    starting_positions = _parse_map_starting_positions(md)

    next_uid = 0
    for side_node in snap.all("side"):
        side_num = int(side_node.attrs.get("side", 0) or 0)
        if side_num == 0: continue
        # WML `income=` is an OFFSET added to `game_config::base_income`
        # (=2 by default; see Wesnoth's team.hpp: `base_income() const
        # { return info_.income + game_config::base_income; }`). When
        # absent, info_.income is 0 → effective base_income = 2. When
        # explicitly `income="0"` (common in saves), still +2 → 2.
        # When `income="-1"`, effective = 1.
        # Some replays in the wild carry malformed quoted values where
        # multiple WML attrs got concatenated into one (e.g.,
        # `village_gold="1 controller=human"` in 2p Evil Factory). We
        # use `_safe_int` so a parse failure on a side-summary attr
        # doesn't poison the whole replay — it falls back to the WML
        # default for that field.
        income_offset = _safe_int(side_node.attrs.get("income", 0), 0)
        ss = SideState(
            side_num=side_num,
            faction=side_node.attrs.get("faction", ""),
            gold=_safe_int(side_node.attrs.get("gold", 100), 100),
            village_income=_safe_int(side_node.attrs.get("village_gold", 2), 2),
            base_income=income_offset + 2,   # vanilla game_config::base_income
            recruit_list=[r.strip() for r in
                          (side_node.attrs.get("recruit", "") or "").split(",")
                          if r.strip()],
            leader_type=side_node.attrs.get("type", "").strip(),
            color=side_node.attrs.get("color", "").strip(),
        )
        gs.sides[side_num] = ss
        # Pre-owned [village] children. The replay's [scenario] /
        # [snapshot] / [replay_start] block embeds villages this
        # side owned at the snapshot point; these contribute to
        # turn-1 income and turn-2 upkeep support before any
        # capture-during-play would. Without this, our reconstructor
        # underpowers side-2 income on Clearing Gushes and
        # similar maps where a village is bundled with the keep.
        # Source: Wesnoth's `team::team(const config&)` reads
        # `[village]` children and inserts each (x, y) into
        # `villages_` -- see wesnoth_src/src/team.cpp:208-217.
        for v_node in side_node.all("village"):
            try:
                vx = int(v_node.attrs.get("x", "0") or "0")
                vy = int(v_node.attrs.get("y", "0") or "0")
            except (ValueError, TypeError):
                continue
            if vx <= 0 or vy <= 0:
                continue
            # WML 1-indexed -> Python 0-indexed.
            gs.villages_owned[(vx - 1, vy - 1)] = side_num

        # Leader unit(s) embedded in the side.
        nested_units = side_node.all("unit")
        for u_node in nested_units:
            unit_type = u_node.attrs.get("type", "")
            stats = _unit_stats(unit_type)
            # Convert WML 1-indexed → Python 0-indexed at extract time
            # (CLAUDE.md: "Python uses 0-indexed hex coordinates everywhere
            # internally"). dump_savestate adds +1 when emitting back to WML.
            #
            # Wesnoth campaign saves embed recall-list units inside
            # [side] with `x="recall" y="recall"` to mark "stored, not
            # on the map". Skip those — they'd never show up in the
            # battlefield grid anyway, and our PvP filter strips
            # campaign replays at dataset load time. Without this skip
            # the int() conversion raises and the whole replay is
            # dropped from the corpus on a parser error.
            x_raw = u_node.attrs.get("x", "0") or "0"
            y_raw = u_node.attrs.get("y", "0") or "0"
            try:
                wml_x = int(x_raw)
                wml_y = int(y_raw)
            except (ValueError, TypeError):
                continue
            next_uid += 1
            # Pull [status] petrified=yes if present — Thousand Stings
            # Garrison embeds 17+ side-3 Giant Scorpions as petrified
            # statues that block movement but can't fight. Without the
            # flag, our reconstructor would happily counter-attack
            # with the statue's full Scorpion stats.
            status_node = u_node.first("status")
            petrified = False
            if status_node is not None:
                v = (status_node.attrs.get("petrified", "no") or "").lower()
                petrified = v in ("yes", "true", "1")
            u = Unit(
                uid=next_uid, unit_type=unit_type, side=side_num,
                x=max(0, wml_x - 1),
                y=max(0, wml_y - 1),
                hp=_safe_int(u_node.attrs.get("hitpoints"), stats["max_hp"]),
                max_hp=_safe_int(u_node.attrs.get("max_hitpoints"), stats["max_hp"]),
                moves_left=_safe_int(u_node.attrs.get("moves"), stats["max_moves"]),
                max_moves=_safe_int(u_node.attrs.get("max_moves"), stats["max_moves"]),
                is_leader=(u_node.attrs.get("canrecruit", "no") == "yes"),
                petrified=petrified,
            )
            gs.units[u.uid] = u

        # Auto-spawn leader from side.type at map starting position when
        # no [unit] block exists in the side. Wesnoth normally does this
        # at scenario-start time; replays which begin from the scratch
        # template (no [snapshot]) don't carry the materialized leader.
        has_leader = any(u.is_leader and u.side == side_num
                         for u in gs.units.values())
        side_type = side_node.attrs.get("type", "").strip()
        if not has_leader and side_type and side_num in starting_positions:
            sx, sy = starting_positions[side_num]
            stats = _unit_stats(side_type)
            next_uid += 1
            gs.units[next_uid] = Unit(
                uid=next_uid, unit_type=side_type, side=side_num,
                x=sx, y=sy,
                hp=stats["max_hp"], max_hp=stats["max_hp"],
                moves_left=stats["max_moves"], max_moves=stats["max_moves"],
                is_leader=True,
            )

    return gs


# --------------------------------------------------------------------
# Apply commands, emit (state_before, action) pairs
# --------------------------------------------------------------------

def _find_unit_at(gs: GameState, x: int, y: int) -> Optional[Unit]:
    for u in gs.units.values():
        if u.x == x and u.y == y:
            return u
    return None


def _parse_int_list(s: str) -> List[int]:
    if not s: return []
    return [int(t) for t in s.split(",") if t.strip()]


def _next_uid(gs: GameState) -> int:
    return (max(gs.units.keys()) + 1) if gs.units else 1


def apply_command(gs: GameState, cmd: WMLNode) -> Optional[dict]:
    """Apply a single [command] to `gs`, return the action dict we
    captured (or None if the command wasn't a player action).
    """
    from_side = cmd.attrs.get("from_side")
    # Decision side is usually in from_side; init_side sets current_side.
    action_dict: Optional[dict] = None

    for sub in cmd.children:
        tag = sub.tag
        if tag in ("init_side", "start"):
            side = int(sub.attrs.get("side_number", 0) or 0)
            if side:
                gs.current_side = side
                # Reset moves + has_attacked on new side turn.
                for u in gs.units.values():
                    if u.side == side:
                        u.moves_left = u.max_moves
                        u.has_attacked = False
                # Village income: credit owned villages.
                if side in gs.sides:
                    owned = sum(1 for s in gs.villages_owned.values() if s == side)
                    gs.sides[side].gold += (gs.sides[side].base_income
                                            + owned * gs.sides[side].village_income)
                # Turn advances when side 1 starts again (heuristic).
            return None

        if tag == "end_turn":
            action_dict = {"type": "end_turn"}
            # Turn rollover: if side_number given, use it; else just bump.
            gs.turn += 1
            return action_dict

        if tag == "move":
            xs = _parse_int_list(sub.attrs.get("x", ""))
            ys = _parse_int_list(sub.attrs.get("y", ""))
            if not xs or len(xs) != len(ys): continue
            sx, sy, tx, ty = xs[0], ys[0], xs[-1], ys[-1]
            u = _find_unit_at(gs, sx, sy)
            side = u.side if u else (int(from_side) if from_side else 0)
            action_dict = {
                "type": "move",
                "side": side,
                "from": [sx, sy], "to": [tx, ty], "path_len": len(xs),
            }
            if u is not None:
                u.x, u.y = tx, ty
                u.moves_left = max(0, u.moves_left - (len(xs) - 1))
                # Village capture: if hex is a village and someone else owned it.
                # Our state-model can't know that without the map. Skipped.
            return action_dict

        if tag == "attack":
            ax = int(sub.attrs.get("weapon", 0) or 0)
            attacker_x = int(sub.attrs.get("x", 0) or 0)
            attacker_y = int(sub.attrs.get("y", 0) or 0)
            defender_x = int(sub.attrs.get("defender_x", 0) or 0)
            defender_y = int(sub.attrs.get("defender_y", 0) or 0)
            u_att = _find_unit_at(gs, attacker_x, attacker_y)
            u_def = _find_unit_at(gs, defender_x, defender_y)
            side = u_att.side if u_att else (int(from_side) if from_side else 0)
            action_dict = {
                "type": "attack",
                "side": side,
                "attacker": [attacker_x, attacker_y],
                "defender": [defender_x, defender_y],
                "weapon": ax,
            }
            # Expected-value damage: each side takes 40% of max_hp.
            # Cheap heuristic; good enough for behavior cloning.
            if u_att is not None:
                u_att.has_attacked = True
                u_att.hp = max(0, u_att.hp - int(0.3 * u_att.max_hp))
            if u_def is not None:
                u_def.hp = max(0, u_def.hp - int(0.5 * u_def.max_hp))
                if u_def.hp == 0:
                    del gs.units[u_def.uid]
            return action_dict

        if tag == "recruit":
            unit_type = sub.attrs.get("type", "")
            tx = int(sub.attrs.get("x", 0) or 0)
            ty = int(sub.attrs.get("y", 0) or 0)
            side = int(from_side) if from_side else gs.current_side
            action_dict = {
                "type": "recruit",
                "side": side,
                "unit_type": unit_type, "to": [tx, ty],
            }
            stats = _unit_stats(unit_type)
            uid = _next_uid(gs)
            gs.units[uid] = Unit(
                uid=uid, unit_type=unit_type, side=side,
                x=tx, y=ty, hp=stats["max_hp"], max_hp=stats["max_hp"],
                moves_left=0,  # recruited units have 0 moves on spawn turn
                max_moves=stats["max_moves"],
            )
            # Deduct gold (real cost requires the units DB; use 15 fallback).
            if side in gs.sides:
                gs.sides[side].gold = max(0, gs.sides[side].gold - 15)
            return action_dict

        if tag == "recall":
            unit_id = sub.attrs.get("value", "")
            tx = int(sub.attrs.get("x", 0) or 0)
            ty = int(sub.attrs.get("y", 0) or 0)
            side = int(from_side) if from_side else gs.current_side
            action_dict = {
                "type": "recall", "side": side,
                "unit_id": unit_id, "to": [tx, ty],
            }
            # Deduct recall cost (20 by default).
            if side in gs.sides:
                gs.sides[side].gold = max(0, gs.sides[side].gold - 20)
            # We don't know the recalled unit's stats; skip creating it.
            return action_dict

    # No recognized action sub-tag — meta command (speak, random_seed,
    # checkup). Caller filters these out by action_dict being None.
    return None


def extract_pairs(path: Path) -> List[dict]:
    """Legacy: emit fully-materialized (state, action) records.

    Bloated output — use extract_replay() for the compact-file format
    used by the training DataLoader. Kept for debugging single files.
    """
    root = parse_replay_file(path)
    gs = build_initial_state(root)

    replays = root.all("replay")
    replay = max(replays, key=lambda r: len(r.all("command")), default=None)
    if replay is None:
        return []
    commands = replay.all("command")
    if not commands:
        return []

    # Meta once per file.
    factions = [gs.sides[s].faction for s in sorted(gs.sides.keys())]
    meta = {
        "game_id": path.stem,
        "scenario_id": gs.scenario_id,
        "factions": factions,
    }

    out: List[dict] = []
    for cmd in commands:
        # Snapshot BEFORE applying the action.
        pre = gs.snapshot()
        action = apply_command(gs, cmd)
        if action is None:
            continue
        # Only keep real player actions.
        if action.get("type") not in ("move", "attack", "recruit", "recall", "end_turn"):
            continue
        record = {**meta, "turn": gs.turn, "state_before": pre, "action": action}
        out.append(record)

    return out


def extract_replay(path: Path) -> Optional[dict]:
    """Parse one replay file into a compact per-game dict.

    Contains everything the DataLoader needs to reconstruct (state, action)
    pairs at training time by replaying the command sequence from an
    initial state. No redundant intermediate states — 100-1000× smaller
    than emitting one record per action.

    Returns None if the file has no player commands (just a save game).
    """
    root = parse_replay_file(path)
    gs = build_initial_state(root)
    # A Wesnoth save typically has TWO [replay] blocks: an empty one at
    # the top carrying just an [upload_log], and the real one further
    # down with the actual command sequence. Pick the non-empty one
    # (largest by command count); first() returns the upload_log shell.
    replays = root.all("replay")
    replay = max(replays, key=lambda r: len(r.all("command")), default=None)
    if replay is None:
        return None
    commands_node = replay.all("command")
    if not commands_node:
        return None

    # Compact command list: each entry is a short tuple encoding the
    # action type + primitive args, small enough to gzip to kilobytes.
    #
    # We also need the [random_seed] following each combat for bit-exact
    # reconstruction. In Wesnoth replays the player issues an action
    # that consumes RNG (recruit with trait roll, attack), then the
    # server emits a separate dependent [command] containing
    # [random_seed] which carries the seed Wesnoth USED for that
    # action. The seed is in the WML stream IMMEDIATELY AFTER the
    # action it describes (verified empirically against
    # replays_raw/.../Freelands_93035.bz2: the [random_seed] block
    # at line 1918 -- new_seed=58dda182, request_id=13 -- is the
    # seed for the FIRST attack at line 1892, not for the recruit
    # at line ~1300).
    #
    # Implementation: track the SINGLE most-recent slot awaiting a
    # seed (whichever was added last -- recruit OR attack). The
    # previous "prefer recruit over attack" logic was buggy: it
    # mis-attributed a post-attack [random_seed] to a recruit
    # earlier in the same turn, leaving the attack with no seed
    # and breaking combat reconstruction from turn 1 onward.
    compact_commands: List[list] = []
    # last_action_slot: index into compact_commands.
    # last_action_kind: "recruit" or "attack" -- which slot in the
    # compact tuple to write to (recruits use slot 4, attacks slot 7).
    last_action_slot: Optional[int] = None
    last_action_kind: Optional[str] = None
    # Most recent attack only -- used for [choose] (advancement
    # picks) which always attach to attacks, never recruits.
    last_attack_slot: Optional[int] = None
    # [choose] commands carry the index the player picked when a unit
    # advances and has multiple advances_to options. They appear after
    # the triggering attack as `dependent` server commands. We collect
    # them in a queue here; the dataset pops from this queue at advance
    # time. Order: attacker-first, defender-second per Wesnoth's
    # attack_unit_and_advance.
    choose_queue: List[int] = []
    # Look ahead at the next command from inside the move handler so
    # we can attach mp_checkup blocks (which Wesnoth emits as a
    # SEPARATE [command dependent=yes] block right after the move,
    # not as a child of the move's [command]). The look-ahead window
    # is 3 commands; mp_checkup typically appears immediately after
    # but server may interleave a [random_seed] block.
    commands_list = list(commands_node)
    for cmd_idx, cmd in enumerate(commands_list):
        # Server-emitted [random_seed] commands attach to the most
        # recent player action that consumed RNG (recruit with trait
        # roll, or attack). The seed in the WML block is the seed
        # Wesnoth USED for that action. Pure musthave-only recruits
        # (undead/mechanical/elemental) don't trigger a [random_seed]
        # because they need no random call -- so a [random_seed]
        # following such a recruit will instead attach to the next
        # action that consumed RNG (which, in our walker, means it
        # gets back-filled into whichever slot is still awaiting one).
        for sub in cmd.children:
            if sub.tag == "random_seed":
                seed_hex = sub.attrs.get("new_seed", "")
                if not seed_hex:
                    continue
                if last_action_slot is None:
                    continue
                if last_action_kind == "recruit":
                    compact_commands[last_action_slot][4] = seed_hex
                else:  # "attack"
                    compact_commands[last_action_slot][7] = seed_hex
                last_action_slot = None
                last_action_kind = None
            elif sub.tag == "choose":
                # Advancement choice picked by the player — append to
                # the most recent attack's advancement-choice list.
                try:
                    val = int(sub.attrs.get("value", 0))
                except ValueError:
                    val = 0
                if last_attack_slot is not None:
                    if len(compact_commands[last_attack_slot]) <= 8:
                        compact_commands[last_attack_slot].append([])
                    compact_commands[last_attack_slot][8].append(val)
                else:
                    # Stray [choose] — stash globally so the next
                    # advance picks it up.
                    choose_queue.append(val)
        for sub in cmd.children:
            t = sub.tag
            if t in ("init_side",):
                side = int(sub.attrs.get("side_number", 0) or 0)
                if side:
                    compact_commands.append(["init_side", side])
                break
            if t == "end_turn":
                compact_commands.append(["end_turn"])
                break
            if t == "move":
                # `from_side` attr on the parent [command] tells us
                # which player issued this move (so which side's unit
                # we should move). Important when our reconstruction
                # has a stale unit at the source hex.
                from_side = int(cmd.attrs.get("from_side", 0) or 0)
                xs = _parse_int_list(sub.attrs.get("x", ""))
                ys = _parse_int_list(sub.attrs.get("y", ""))
                # Wesnoth replays carry the FULL intended path. If the
                # mover sighted an enemy (skip_sighted="only_ally") or
                # ZOC-stopped, the actual final hex is shorter than
                # the path. The engine records that final hex in the
                # following `[checkup]` or `[mp_checkup]` block --
                # which name varies by replay (singleplayer / older
                # replays use `[checkup]`; recent multiplayer replays
                # use `[mp_checkup]`). Both have `[result]` children
                # carrying `final_hex_x/y`.
                final_x: Optional[int] = None
                final_y: Optional[int] = None
                # Wesnoth records the move's actual final hex in a
                # `[checkup]` or `[mp_checkup]` block. The block can
                # be either a CHILD of the move's [command] (older /
                # singleplayer replays), OR a SEPARATE follow-up
                # [command] right after (multiplayer replays:
                # `[command] dependent="yes" [mp_checkup] ... [/mp_checkup]
                # [/command]`). Search both locations.
                def _read_final(node):
                    if node is None:
                        return None, None
                    for r in node.all("result"):
                        if "final_hex_x" in r.attrs:
                            try:
                                return (int(r.attrs.get("final_hex_x", 0)),
                                        int(r.attrs.get("final_hex_y", 0)))
                            except ValueError:
                                pass
                    if "final_hex_x" in node.attrs:
                        try:
                            return (int(node.attrs.get("final_hex_x", 0)),
                                    int(node.attrs.get("final_hex_y", 0)))
                        except ValueError:
                            pass
                    return None, None

                checkup = cmd.first("checkup") or cmd.first("mp_checkup")
                final_x, final_y = _read_final(checkup)
                # If not found as child, look at the next [command] in
                # the stream -- it often carries the mp_checkup block.
                # Window of 3 commands to skip past intervening
                # [random_seed] / [choose] blocks.
                lookahead_idx = cmd_idx + 1
                while (final_x is None
                       and lookahead_idx < len(commands_list)
                       and lookahead_idx <= cmd_idx + 3):
                    nxt = commands_list[lookahead_idx]
                    nxt_chk = nxt.first("checkup") or nxt.first("mp_checkup")
                    if nxt_chk is not None:
                        final_x, final_y = _read_final(nxt_chk)
                        break
                    # Skip if next command is a player action
                    # (move/recruit/attack/end_turn/init_side); the
                    # checkup must precede those.
                    has_action = any(
                        c.tag in ("move", "recruit", "attack",
                                  "end_turn", "init_side", "recall")
                        for c in nxt.children
                    )
                    if has_action:
                        break
                    lookahead_idx += 1
                if xs and len(xs) == len(ys):
                    # If we got an explicit final_hex from [checkup] that
                    # disagrees with the path's last cell, truncate the
                    # path at the recorded stopping point.
                    if (final_x is not None and final_y is not None
                            and (xs[-1], ys[-1]) != (final_x, final_y)):
                        try:
                            stop = next(
                                i for i, (x, y) in enumerate(zip(xs, ys))
                                if x == final_x and y == final_y
                            )
                            xs = xs[:stop + 1]
                            ys = ys[:stop + 1]
                        except StopIteration:
                            # final_hex isn't on the recorded path —
                            # fall back to using it as a single-step
                            # destination. Rare but seen for ambush.
                            xs = [xs[0], final_x]
                            ys = [ys[0], final_y]
                    # Convert WML 1-indexed → Python 0-indexed.
                    xs = [max(0, x - 1) for x in xs]
                    ys = [max(0, y - 1) for y in ys]
                    # 4th slot reserved for from_side (matched against
                    # the moving unit's side at apply time).
                    compact_commands.append(["move", xs, ys, from_side])
                break
            if t == "attack":
                # In 1.18 replays the [attack] command's positions live
                # in nested [source] (attacker) and [destination]
                # (defender) child blocks, not on top-level attrs.
                src = sub.first("source")
                dst = sub.first("destination")
                if src is None or dst is None:
                    break
                # Convert WML 1-indexed → Python 0-indexed.
                attacker_x = max(0, int(src.attrs.get("x", 0) or 0) - 1)
                attacker_y = max(0, int(src.attrs.get("y", 0) or 0) - 1)
                defender_x = max(0, int(dst.attrs.get("x", 0) or 0) - 1)
                defender_y = max(0, int(dst.attrs.get("y", 0) or 0) - 1)
                weapon    = int(sub.attrs.get("weapon", 0) or 0)
                # Defender's chosen weapon (their counter). Stored on
                # the [attack] tag as `defender_weapon` (-1 if none).
                d_weapon  = int(sub.attrs.get("defender_weapon", -1) or -1)
                # Seed is back-filled by the next [random_seed] command.
                compact_commands.append([
                    "attack", attacker_x, attacker_y,
                    defender_x, defender_y, weapon, d_weapon, "",
                ])
                slot = len(compact_commands) - 1
                last_action_slot = slot
                last_action_kind = "attack"
                last_attack_slot = slot   # tracked separately for [choose]
                break
            if t == "recruit":
                unit_type = sub.attrs.get("type", "")
                # Convert WML 1-indexed → Python 0-indexed.
                tx = max(0, int(sub.attrs.get("x", 0) or 0) - 1)
                ty = max(0, int(sub.attrs.get("y", 0) or 0) - 1)
                # 5th slot reserved for the per-recruit trait seed,
                # back-filled by the next [random_seed] command. May
                # remain "" for undead/mechanical/elemental races
                # whose musthave-only trait pool needs no random call
                # (Wesnoth doesn't emit [random_seed] for those, so
                # the slot stays "" and `last_action_slot` advances
                # to the next RNG-consuming action below).
                compact_commands.append(["recruit", unit_type, tx, ty, ""])
                last_action_slot = len(compact_commands) - 1
                last_action_kind = "recruit"
                break
            if t == "recall":
                unit_id = sub.attrs.get("value", "")
                tx = max(0, int(sub.attrs.get("x", 0) or 0) - 1)
                ty = max(0, int(sub.attrs.get("y", 0) or 0) - 1)
                compact_commands.append(["recall", unit_id, tx, ty])
                break

    if not compact_commands:
        return None

    # Initial state — no hex map, loader re-parses map_data from here.
    starting_sides = [
        {
            "side": s.side_num, "faction": s.faction, "gold": s.gold,
            "base_income": s.base_income,
            "village_income": s.village_income,
            "recruit": list(s.recruit_list),
            "leader_type": s.leader_type,
            "color": s.color,
        }
        for s in sorted(gs.sides.values(), key=lambda s: s.side_num)
    ]
    starting_units = [
        {
            "uid": u.uid, "type": u.unit_type, "side": u.side,
            "x": u.x, "y": u.y, "hp": u.hp, "max_hp": u.max_hp,
            "max_moves": u.max_moves, "is_leader": u.is_leader,
            **({"petrified": True} if u.petrified else {}),
        }
        for u in gs.units.values()
    ]

    # Pull map_data from the snapshot (same block we parsed initial
    # state from). Downstream DataLoader will split it into a hex grid.
    snap = root.first("replay_start") or root.first("snapshot") or root.first("scenario")
    map_data = snap.attrs.get("map_data", "") if snap else ""

    # ToD start cycle index. Wesnoth's tod_manager stores `current_time`
    # in [scenario] / [replay_start] / [snapshot] once resolved. For
    # `random_start_time=no` (the default 2p case) it's index 0 (dawn).
    # For `random_start_time=yes`, the engine resolves it at game-start
    # via `tod_manager::resolve_random` using a synced RNG draw — we
    # don't replay that draw, so we drop replays where it's neither
    # disabled nor pre-resolved (caller filters on the returned None).
    tod_start_index = 0
    if snap is not None:
        ct_raw = snap.attrs.get("current_time", "").strip()
        if ct_raw:
            try:
                tod_start_index = int(ct_raw)
            except ValueError:
                pass
        else:
            rst = (snap.attrs.get("random_start_time", "no") or "no").strip().lower()
            if rst in ("yes", "true", "1"):
                log.debug(
                    f"{path.name}: random_start_time=yes with no resolved "
                    f"current_time; dropping for fidelity"
                )
                return None

    # `experience_modifier` lives in the top-level metadata or in
    # [multiplayer]; it scales every unit's max_xp via Wesnoth's
    # `unit_type::experience_needed`:
    #   exp = max(1, (base * experience_modifier + 50) // 100)
    # Defaults to 100 (no modification). Replays use 30..100% commonly;
    # the Aethermaw replay we're auditing uses 50%.
    exp_mod = 100
    mp = root.first("multiplayer")
    if mp is not None:
        try:
            exp_mod = int(mp.attrs.get("experience_modifier", 100) or 100)
        except ValueError:
            pass
    if exp_mod == 100 and root.attrs.get("experience_modifier"):
        try:
            exp_mod = int(root.attrs.get("experience_modifier") or 100)
        except ValueError:
            pass

    # Pre-owned villages from [side] / [village] children. Stored
    # as a list of {x, y, side} so the JSON-loader can apply them
    # to GameState before turn-1 init. (Older extracts didn't carry
    # this field; reconstructor treats absent as empty.)
    starting_villages = [
        {"x": x, "y": y, "side": side}
        for (x, y), side in sorted(gs.villages_owned.items())
    ]

    return {
        "game_id": path.stem,
        "scenario_id": gs.scenario_id,
        "factions": [s["faction"] for s in starting_sides],
        "map_size": list(gs.map_size),
        "map_data": map_data,
        "experience_modifier": exp_mod,
        # 0..5 cycle index for the default 6-step ToD. 0 means turn-1
        # is dawn (the no-randomization default); 1 means turn-1 is
        # morning, etc. Reconstructor offsets `_tod_for_turn` by this.
        "tod_start_index": tod_start_index,
        "starting_sides": starting_sides,
        "starting_units": starting_units,
        "starting_villages": starting_villages,
        "commands": compact_commands,
    }


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def main(argv: List[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(argv) == 2:
        path = Path(argv[1])
        recs = extract_pairs(path)
        for r in recs:
            print(json.dumps(r, separators=(",", ":")))
        log.info(f"# {path.name}: {len(recs)} records")
        return 0

    if len(argv) == 3:
        in_dir = Path(argv[1])
        out_dir = Path(argv[2])
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(in_dir.glob("**/*.bz2"))
        log.info(f"Walking {len(files)} files under {in_dir}")
        stats = Counter()
        index_path = out_dir / "index.jsonl"
        with index_path.open("w", encoding="utf-8") as idx:
            for i, p in enumerate(files, 1):
                # Fast pre-filter: default era only. Avoid parsing the
                # full WML for files we won't keep.
                try:
                    with bz2.open(p, "rb") as fz:
                        head = fz.read(512 * 1024)
                except Exception:
                    stats["files_err"] += 1; continue
                # Check era_id in header.
                head_text = head.decode("utf-8", errors="replace")[:20000]
                if 'era_id="default"' not in head_text \
                        and 'era_id="era_default"' not in head_text:
                    stats["files_skipped_era"] += 1; continue
                # Drop campaign-style "multiplayer" replays. WC_II_2p
                # and similar carry `era_id="era_default"` but layer
                # custom map / heroes / mod resources on top, with
                # 40+ recall-list units at WML (0, 0) that our
                # extractor can't disambiguate from real placements.
                # Test the top-level `campaign=` attr -- empty for
                # true PvP, non-empty for campaign-style mp_campaigns.
                m_camp = re.search(r'^campaign="([^"]+)"', head_text, re.MULTILINE)
                if m_camp and m_camp.group(1).strip():
                    stats["files_skipped_campaign"] += 1; continue
                # Drop replays whose scenario or any [side] enables
                # shroud — encoder fidelity assumes side-only fog and
                # we don't model permanent shroud-clearing. PvP default
                # scenarios don't use shroud, so this is a pure safety
                # net and shouldn't fire. Match `^shroud=` only (not
                # `auto_shroud=` or `mp_shroud=`).
                if re.search(r'(?:^|\s)shroud\s*=\s*"?(yes|true|1)"?',
                             head_text, re.M):
                    stats["files_skipped_shroud"] += 1; continue

                # Full parse.
                try:
                    rep = extract_replay(p)
                except Exception as e:
                    stats["files_err"] += 1
                    log.debug(f"  err {p.name}: {e}")
                    continue
                if rep is None:
                    stats["files_no_commands"] += 1; continue

                # Write compact gzipped JSON. Hash-named so duplicate
                # game_ids from concurrent reruns don't clobber.
                gid = hashlib.sha1(rep["game_id"].encode()).hexdigest()[:12]
                out_path = out_dir / f"{gid}.json.gz"
                with gzip.open(out_path, "wt", encoding="utf-8",
                               compresslevel=6) as fw:
                    json.dump(rep, fw, separators=(",", ":"))

                idx.write(json.dumps({
                    "file": out_path.name,
                    "game_id": rep["game_id"],
                    "scenario_id": rep["scenario_id"],
                    "factions": rep["factions"],
                    "n_commands": len(rep["commands"]),
                }) + "\n")
                stats["files_ok"] += 1
                stats["commands"] += len(rep["commands"])

                if i % 500 == 0:
                    log.info(
                        f"  [{i}/{len(files)}] kept={stats['files_ok']} "
                        f"skipped_era={stats['files_skipped_era']} "
                        f"no_cmds={stats['files_no_commands']} "
                        f"err={stats['files_err']} "
                        f"commands={stats['commands']}",
                    )
        log.info(f"\nDone. {dict(stats)}")
        return 0

    print("usage: replay_extract.py REPLAY.bz2", file=sys.stderr)
    print("       replay_extract.py REPLAYS_DIR OUT_DIR", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
