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
# `[+tag]` is Wesnoth's "merge with last preceding tag of same name" form.
# We capture it as a regular tag (drop the `+` prefix) but mark
# `_is_plus_form=True` so a post-parse pass can apply context-dependent
# semantics: APPEND for sequence parents (e.g. [time] inside [time_area]
# extends the cycle on Tombs of Kesorak / Elensefar Courtyard), MERGE
# for singleton parents (e.g. [+unit] inside [side] adds descriptions
# to the petrified statues in Caves of the Basilisk / Sullas Ruins
# without spawning phantom units).
TAG_OPEN_PLUS_RE = re.compile(r'^\s*\[\+([a-zA-Z_][a-zA-Z0-9_]*)\]\s*$')
TAG_CLOSE_RE = re.compile(r'^\s*\[/([a-zA-Z_][a-zA-Z0-9_]*)\]\s*$')

# Tags that, when prefixed with `[+]`, should APPEND a new child rather
# than merge into the preceding sibling. Wesnoth's actual `[+element]`
# behavior in `wesnoth_src/src/serialization/parser.cpp:217-242` is
# always MERGE: it finds the LAST [element] of same name in the parent
# and reopens it for field/child insertion. We keep this set empty so
# all `[+tag]` uses go through the MERGE branch by default; if a
# specific tag in a future scenario truly needs APPEND semantics, add
# it here explicitly. Audit 2026-05-04: the Tombs of Kesorak [+time]
# blocks merge into the preceding [time], producing a 6-entry cycle
# where each "bright X" entry overrides only the image (and re-asserts
# the same lawful_bonus the macro already set). Stage 17 originally had
# {"time"} which gave the wrong cycle and broke Mage damage on Tombs
# illuminated hexes at turn 7 (off-by-1 dmg per strike).
_PLUS_APPEND_TAGS: set = set()
# Values may be quoted "..." or bare 123 / true / false. We handle both.
KEY_RE = re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*?)\s*$')
# Multi-line string values in WML are quoted and may span lines, closing
# with a standalone `"`. We detect the open-quote here, then glue lines
# until we see the closing quote.


class WMLNode:
    """A WML tag instance: attributes + ordered children (also WMLNodes)."""
    __slots__ = ("tag", "attrs", "children", "_is_plus_form")

    def __init__(self, tag: str, is_plus_form: bool = False):
        self.tag: str = tag
        self.attrs: Dict[str, str] = {}
        self.children: List["WMLNode"] = []
        # True if this node was parsed from `[+tag]` syntax. The
        # post-parse merge pass uses this to decide whether to merge
        # this node into its preceding sibling (Wesnoth's `[+]` rule).
        self._is_plus_form: bool = is_plus_form

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

        # `[+tag]` form — captured separately so a post-parse pass can
        # merge into the preceding sibling (or append, depending on
        # context).
        m = TAG_OPEN_PLUS_RE.match(line)
        if m:
            node = WMLNode(m.group(1), is_plus_form=True)
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

    _resolve_plus_forms(root)
    return root


def _resolve_plus_forms(node: WMLNode) -> None:
    """Apply Wesnoth's `[+tag]` semantics across `node`'s descendants.

    For each child marked `_is_plus_form=True`:
      - If `tag` is in `_PLUS_APPEND_TAGS` (e.g. `time`): keep as a
        new sibling (append semantics — extends the cycle in
        [time_area]).
      - Otherwise: find the most recent preceding sibling with the
        same tag and MERGE this node's attrs+children into it
        (Wesnoth's standard `[+]` rule). If no preceding sibling
        of that tag exists, leave the node in place (treat as a
        fresh tag).

    Concrete cases this handles:
      - Tombs of Kesorak / Elensefar Courtyard `[time_area]`:
        each `[+time]` extends the per-hex ToD cycle from the 6
        macro-expanded entries to 10, so turn 6 on illuminated hexes
        keeps `lawful_bonus=25` (DAY) instead of falling through to
        `secondwatch` (lawful_bonus=-25).
      - Caves of the Basilisk / Sullas Ruins petrified statues:
        `{UNIT_PETRIFY ...} [+unit] description=... [/unit]` adds
        the description text to the just-placed [unit] instead of
        spawning a phantom (description-only) unit alongside it.
    """
    # Recurse first so child-of-child plus forms resolve before we
    # touch this node's children list.
    for c in node.children:
        _resolve_plus_forms(c)
    new_children: List[WMLNode] = []
    for c in node.children:
        if not c._is_plus_form:
            new_children.append(c)
            continue
        if c.tag in _PLUS_APPEND_TAGS:
            # Append semantics: keep as a fresh sibling. Strip the
            # marker so downstream code treats it as a regular tag.
            c._is_plus_form = False
            new_children.append(c)
            continue
        # Merge semantics: find the latest preceding sibling with
        # same tag and fold attrs+children into it.
        target: Optional[WMLNode] = None
        for prev in reversed(new_children):
            if prev.tag == c.tag:
                target = prev
                break
        if target is None:
            # No preceding sibling -- treat as fresh tag.
            c._is_plus_form = False
            new_children.append(c)
            continue
        target.attrs.update(c.attrs)
        target.children.extend(c.children)
        # `c` is dropped (not appended).
    node.children = new_children


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
    # Free upkeep per controlled village (Wesnoth WML `village_support`).
    # Default 1 in vanilla. Reduces effective upkeep paid: net upkeep
    # is `max(0, total_unit_levels - villages * village_support)`.
    village_support: int = 1
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
# Set of unit types whose recruit consumes NO RNG (musthave-only trait
# pool, e.g. all undead / mechanical / elemental). Wesnoth never emits
# `[random_seed]` for these recruits, so an empty seed slot in our
# compact stream does NOT mean the recruit was unfinished.
_RECRUIT_NO_RNG_TYPES: set = set()


def _unit_stats(unit_type: str) -> dict:
    """Return {max_hp, max_moves, is_leader} for a unit type. Loads from
    unit_stats.json (scraped from Wesnoth source) on first call; falls
    back to _DEFAULT_UNIT for unknown types (e.g. Ladder Era variants)."""
    global _UNIT_DB_CACHE, _RECRUIT_NO_RNG_TYPES
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
                ti = v.get("traits") or {}
                # No-RNG recruit: empty random pool, OR num_traits
                # equals len(musthave). Wesnoth's recruit code only
                # rolls when there are random slots to fill.
                pool = list(ti.get("pool") or [])
                musthave = list(ti.get("musthave") or [])
                num_traits = int(ti.get("num_traits") or 0)
                if not pool or num_traits <= len(musthave):
                    _RECRUIT_NO_RNG_TYPES.add(k)
        except FileNotFoundError:
            pass  # silently fall back
    return _UNIT_DB_CACHE.get(unit_type, dict(_DEFAULT_UNIT))


def _recruit_consumes_rng(unit_type: str) -> bool:
    """True iff Wesnoth emits a `[random_seed]` command after recruiting
    this unit type. False for undead/mechanical/elemental (musthave-only
    trait pool)."""
    _unit_stats(unit_type)  # populate cache
    return unit_type not in _RECRUIT_NO_RNG_TYPES


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
            village_support=_safe_int(
                side_node.attrs.get("village_support", 1), 1),
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

    # Concatenate all [replay] blocks; see extract_replay for rationale.
    replays = root.all("replay")
    if not replays:
        return []
    commands: List = []
    for r in replays:
        commands.extend(r.all("command"))
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


def _first_player_action_sig(cmd_nodes) -> Optional[Tuple]:
    """Return a content-signature for the first player action across
    a list of [command] nodes. Used by the trailer-drop heuristic
    to detect duplicates between block i's last action and block
    i+1's first action."""
    for cmd in cmd_nodes:
        for sub in cmd.children:
            t = sub.tag
            if t == "init_side":
                return ("init_side", sub.attrs.get("side_number", ""))
            if t == "end_turn":
                return ("end_turn",)
            if t == "move":
                return ("move", sub.attrs.get("x", ""), sub.attrs.get("y", ""))
            if t == "attack":
                src = sub.first("source")
                dst = sub.first("destination")
                if src is None or dst is None:
                    return None
                return ("attack",
                        int(src.attrs.get("x", 0) or 0) - 1,
                        int(src.attrs.get("y", 0) or 0) - 1,
                        int(dst.attrs.get("x", 0) or 0) - 1,
                        int(dst.attrs.get("y", 0) or 0) - 1,
                        int(sub.attrs.get("weapon", 0) or 0))
            if t == "recruit":
                return ("recruit",
                        sub.attrs.get("type", ""),
                        int(sub.attrs.get("x", 0) or 0) - 1,
                        int(sub.attrs.get("y", 0) or 0) - 1)
            if t == "recall":
                return ("recall",
                        sub.attrs.get("value", ""),
                        int(sub.attrs.get("x", 0) or 0) - 1,
                        int(sub.attrs.get("y", 0) or 0) - 1)
    return None


def _compact_action_sig(compact_entry) -> Optional[Tuple]:
    """Mirror of `_first_player_action_sig` for our compact-format
    entries; used at the boundary to compare a dropped trailer
    against the next block's redo."""
    if not compact_entry:
        return None
    kind = compact_entry[0]
    if kind == "attack":
        # ['attack', ax, ay, dx, dy, a_weapon, d_weapon, seed, ...]
        return ("attack", compact_entry[1], compact_entry[2],
                compact_entry[3], compact_entry[4], compact_entry[5])
    if kind == "recruit":
        # ['recruit', unit_type, tx, ty, seed]
        return ("recruit", compact_entry[1], compact_entry[2], compact_entry[3])
    return None


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
    # A Wesnoth save can have MULTIPLE [replay] blocks at the top
    # level. Two patterns we've observed:
    #
    #   1. Pure replay (or from-scratch save): one empty [replay]
    #      shell carrying just [upload_log], plus one real block
    #      with all commands from turn 1 onwards. Sizes look like
    #      (0, N).
    #   2. Continuation save: TWO non-empty blocks. The first holds
    #      commands from turn 1 to the save point (init_side/start
    #      at cmd[0]); the second holds commands from the save
    #      point onwards (a fresh action like [move] at cmd[0],
    #      no leading init_side because the save was taken
    #      mid-turn). Sizes like (62, 67) on a Turn-5 save.
    #
    # The OLD `max(...)` heuristic picked only the largest block.
    # For continuation saves that meant we'd take the post-save
    # block (which expects the snapshot state, not [replay_start])
    # and feed it into a state initialized from [replay_start] —
    # producing src_missing on the very first move. Concrete case:
    # 2p__Caves_of_the_Basilisk_Turn_5_(120166).bz2; cmd[0] is a
    # move from (23,7) but [replay_start] only has 2 leaders +
    # 15 petrified statues at the canonical positions.
    #
    # Fix: concatenate ALL [replay] blocks in document order. The
    # result is the full turn-1-onwards trajectory matching
    # [replay_start]. The empty-shell case still works because an
    # empty block contributes 0 commands. Tested 2026-05-03
    # against 23-map ladder corpus.
    replays = root.all("replay")
    if not replays:
        return None
    commands_node: List = []
    # Track the cmd_idx at the END of each non-last block, so the
    # main loop can detect block boundaries and drop unfinished
    # trailing actions. See "save-mid-action duplicate" handling
    # below.
    block_boundary_indices: List[int] = []
    # For each boundary, also record the FIRST player action of
    # the next block (as a content-signature). The dropper uses
    # this to distinguish "save-mid-recruit" (next-block first
    # action == this trailer == duplicate, drop) from "completed
    # musthave-only recruit, then save" (next-block first action
    # is something else, keep).
    boundary_next_first_action: dict = {}
    # Also collect ALL recruit hexes used in block i+1 (0-indexed).
    # The save-mid-recruit redo is not always the FIRST action of
    # block i+1 (the player can move first, then re-recruit). If
    # block i+1 contains a recruit at the same hex as the trailer,
    # that's the redo — drop the trailer regardless of whether the
    # redo is action #0. Concrete:
    # 2p__Weldyn_Channel_Turn_14_(157907).bz2: block[0] tail is a
    # recruit Dwarvish Thunderer at (16,3) with no seed; block[1]
    # cmd[0] is a move, cmd[1] is the redo recruit Dwarvish Fighter
    # at (16,3). The first-action sig pointed at the move, so the
    # trailer wasn't dropped.
    boundary_next_recruit_hexes: dict = {}
    for ridx, r in enumerate(replays):
        block_cmds = r.all("command")
        commands_node.extend(block_cmds)
        if ridx < len(replays) - 1 and block_cmds:
            boundary_idx = len(commands_node) - 1
            block_boundary_indices.append(boundary_idx)
            # Find first player-action sig in block ridx+1.
            next_block_cmds = (replays[ridx + 1].all("command")
                               if ridx + 1 < len(replays) else [])
            sig = _first_player_action_sig(next_block_cmds)
            boundary_next_first_action[boundary_idx] = sig
            # Index recruit hexes in block ridx+1 used BEFORE the
            # first `end_turn` / `init_side` -- those are recruits
            # the SAME player issues during the resumed turn. A
            # save-mid-recruit redo happens here. Recruits after
            # end_turn are by a different (or later) turn -- the
            # original trailer (if it was a completed musthave-only
            # recruit) lives there until the unit moves/dies and
            # someone re-recruits at the same hex; mustn't confuse
            # THAT for a redo.
            # False-positive without this filter (Stage 12 audit):
            # 2p__Aethermaw_Turn_18_(100985).bz2 -- block[0] tail is
            # a Ghoul (musthave-only, no seed) at (27,16). Block[1]
            # starts with end_turn (the player ended their turn on
            # load), then on later turns recruits Troll Whelp /
            # Ghoul / Ghost at (27,16) after the original Ghoul
            # moves. Without the end_turn cutoff, condition (d) fired
            # and dropped a legitimately-completed recruit.
            recruit_hexes: set = set()
            for cmd in next_block_cmds:
                stop = False
                for sub in cmd.children:
                    if sub.tag in ("end_turn", "init_side"):
                        stop = True
                        break
                    if sub.tag == "recruit":
                        rx = int(sub.attrs.get("x", 0) or 0) - 1
                        ry = int(sub.attrs.get("y", 0) or 0) - 1
                        recruit_hexes.add((rx, ry))
                if stop:
                    break
            boundary_next_recruit_hexes[boundary_idx] = recruit_hexes
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
    # Track the most-recent compact slot for moves (separate from
    # `last_action_slot` because moves don't consume RNG and so don't
    # need seed back-fill, but we still want to detect save-mid-move
    # duplicates at block boundaries — see Stage 18 / boundary check
    # below).
    last_move_slot: Optional[int] = None
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
    # Save-mid-action duplicate handling. When Wesnoth saves DURING
    # a player's turn (after the player issued an RNG-consuming
    # action like recruit-with-traits or attack, but before the
    # synced engine processed the [random_seed] follow-up), the
    # save flushes the unfinished action to the end of [replay][i]
    # WITHOUT its seed. On load, Wesnoth re-emits the same action
    # (or a replacement, if the player undid + re-issued) at the
    # start of [replay][i+1] WITH proper seeds.
    #
    # Concrete cases:
    #   - 2p__Caves_of_the_Basilisk_Turn_16_(7251).bz2:
    #     [replay][0] last cmd: [recruit] Fencer (18,5), no seed
    #     [replay][1] cmd[0]:   [recruit] Fencer (18,5)
    #     [replay][1] cmd[1]:   [random_seed] for the redo
    #   - 2p__Weldyn_Channel_Turn_14_(157907).bz2:
    #     [replay][0] last cmd: [recruit] Thunderer (17,4), no seed
    #     [replay][1] cmd[1]:   [recruit] Fighter (17,4) -- player
    #                           undid + re-recruited a different unit
    #
    # If we naively concat all blocks, the unfinished trailing
    # action from block i ends up duplicated (same hex, sometimes
    # same unit type, sometimes a stale type that conflicts with
    # the redo). Either way it produces recruit:target_occupied
    # at extract time.
    #
    # Fix: at each block-boundary cmd_idx, after processing the
    # last command of block i, if `last_action_slot` is still set
    # (meaning the trailing recruit/attack didn't get its seed
    # within block i), DROP that compact entry -- it represents
    # an unfinished action that block i+1 will redo properly.
    boundary_set = set(block_boundary_indices)
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
                    new_move = ["move", xs, ys, from_side]
                    # In-block consecutive-duplicate dedup: if the
                    # most recent compact entry is an EXACT duplicate
                    # of this move (same path, same side), drop this
                    # one. Wesnoth emits a duplicate move when a player
                    # surrender (or other server-injected event)
                    # interrupts the action stream and the same move
                    # gets re-recorded on the resume side. Concrete:
                    # 2p__Caves_of_the_Basilisk_Turn_22_(227794).bz2
                    # cmd[336] move (12,13,14)(19,20,19) is followed
                    # by [surrender][/surrender] + [speak]"ok" +
                    # [rename] (async), then the SAME move is emitted
                    # again at cmd[337]. Wesnoth replays the second as
                    # a no-op (the unit is already at the destination)
                    # but our extractor was treating it as a new move,
                    # producing src_missing on a vanished source hex.
                    # Restrict to in-block dedup (don't conflict with
                    # cross-block trailer-drop logic, which uses
                    # `last_move_slot` separately): just compare to
                    # the previous compact entry directly.
                    if (compact_commands
                            and compact_commands[-1] == new_move):
                        # Skip the duplicate, keep last_move_slot
                        # pointing at the original.
                        break
                    compact_commands.append(new_move)
                    last_move_slot = len(compact_commands) - 1
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
            if t == "surrender":
                # When a player surrenders mid-turn AFTER being taken
                # over by another player or AI (the takeover speak
                # appears BEFORE the in-flight move), Wesnoth's replay
                # engine effectively undoes that last move: the move's
                # [checkup] block is recorded as if it executed, but
                # the post-surrender state — driven by the AI/takeover
                # player who continues with skip_sighted="all" — assumes
                # the move did NOT happen.
                #
                # Distinguishing fingerprint: the very next [move] AFTER
                # `[surrender]` uses `skip_sighted="all"`. Normal moves
                # use `skip_sighted="only_ally"` or omit the attr; only
                # AI-driven / engine-replayed moves set "all".
                #
                # Without this drop, the surrendered side ends up with
                # the unit at the moved-to hex in our sim while Wesnoth
                # has it back at the source. Cascades into either
                # final_occupied (something else tries to enter the
                # moved-to hex) or src_missing (something tries to
                # move from the source hex which our sim has empty).
                #
                # Verified 2026-05-08 against:
                #   - 2p__Weldyn_Channel_Turn_30_(205235): cmd[488]
                #     side-1 Gryphon Rider (23,1)→(23,6); surrender;
                #     cmd[489] side-1 move skip_sighted="all". cmd[502]
                #     side-2 Dark Adept move to (23,6) failed
                #     (final_occupied) until cmd[488] dropped.
                #   - 2p__Thousand_Stings_Garrison_Turn_54_(148817):
                #     cmd[693] side-1 (12,14)→(10,15); surrender;
                #     cmd[694] side-1 skip_sighted="all" (12,14)→(18,11)
                #     failed (src_missing) until cmd[693] dropped.
                # NEGATIVE control: 2p__Hornshark_Island_Turn_14_
                # (109842) has [surrender] right after a normal move,
                # but the takeover speak comes AFTER the move (player
                # said "gg wp" then surrendered without leaving an
                # in-flight action). The post-surrender [move] does
                # NOT carry skip_sighted="all", so the move was real
                # and must be kept. The skip_sighted check correctly
                # discriminates this case.
                drop_pre = False
                la_idx = cmd_idx + 1
                while la_idx < len(commands_list):
                    nxt = commands_list[la_idx]
                    saw_action = False
                    for nsub in nxt.children:
                        if nsub.tag == "move":
                            saw_action = True
                            if nsub.attrs.get("skip_sighted", "") == "all":
                                drop_pre = True
                            break
                        if nsub.tag in ("attack", "recruit", "recall",
                                        "init_side", "end_turn"):
                            saw_action = True
                            break
                    if saw_action:
                        break
                    la_idx += 1
                if (drop_pre
                        and last_move_slot is not None
                        and last_move_slot == len(compact_commands) - 1
                        and compact_commands[last_move_slot][0] == "move"):
                    compact_commands.pop(last_move_slot)
                    # Adjust slot trackers.
                    if (last_action_slot is not None
                            and last_action_slot >= last_move_slot):
                        last_action_slot = (None if last_action_slot
                                            == last_move_slot
                                            else last_action_slot - 1)
                    if (last_attack_slot is not None
                            and last_attack_slot >= last_move_slot):
                        last_attack_slot = (None if last_attack_slot
                                            == last_move_slot
                                            else last_attack_slot - 1)
                    last_move_slot = None
                break

        # Block-boundary check: if this cmd_idx is the last command
        # of a non-final [replay] block AND we have an unfinished
        # RNG-consuming action awaiting its seed, decide whether to
        # drop. See "Save-mid-action duplicate handling" above.
        #
        # TIGHTENED HEURISTIC (2026-05-03 audit):
        #   - ATTACKS always emit a [random_seed] when complete (per
        #     strike). So an attack at block-i's tail with no seed
        #     is ALWAYS unfinished -- safe to drop unconditionally.
        #   - RECRUITS may legitimately complete WITHOUT a seed when
        #     the unit type's trait pool is empty (musthave-only
        #     races: undead/mechanical/elemental). For those, "no
        #     seed" doesn't mean unfinished -- it means the engine
        #     didn't need to roll. To avoid dropping legitimately-
        #     completed recruits, only drop a recruit-trailer if its
        #     content matches the FIRST player action of block i+1
        #     (= save-mid-recruit redo). If next-block first action
        #     differs, keep the recruit.
        #
        # Verification: tools/verify_trailer_drop.py audited 500
        # raw replays; all 3 dropper-fires were attacks with
        # legitimate undo/redo on load (block-1 first action was a
        # different action). No false-drop musthave-recruit cases
        # surfaced in that sample, but the risk is real -- 75
        # default-era unit types (Ghoul, Skeleton, Walking Corpse,
        # etc.) are musthave-only.
        if cmd_idx in boundary_set and last_action_slot is not None:
            # The unfinished trailer is at compact index
            # `last_action_slot`. It might NOT be the literal last
            # entry of compact_commands -- player actions like
            # `end_turn` and `init_side` (which are not RNG-consuming
            # so don't update last_action_slot) can be appended after
            # an undone recruit. Concrete: a player issues a recruit,
            # undoes it, issues end_turn -- the replay records the
            # undone recruit followed by end_turn, init_side. Pre-fix
            # we required the trailer to be the very last entry,
            # missing the undone-recruit-then-end-turn case.
            # 2p__Silverhead_Crossing_Turn_31_(127303).bz2 cmd[26]
            # is the canonical example.
            trailer = compact_commands[last_action_slot]
            next_first = boundary_next_first_action.get(cmd_idx)
            trailer_kind = trailer[0]
            should_drop = False
            if trailer_kind == "attack":
                # Attacks ALWAYS need RNG; no-seed = unfinished.
                should_drop = True
            elif trailer_kind == "recruit":
                # Drop only if (a) next block redoes the same recruit
                # at the same hex, OR (b) next block has a recruit at
                # the SAME hex with a DIFFERENT type (undo + replace),
                # OR (c) the trailer recruit is followed in the SAME
                # block by `end_turn` / `init_side` -- meaning the
                # recruit was issued then undone before end-of-turn,
                # so it never completed in Wesnoth's reality. The
                # third condition catches the Silverhead case where
                # block 1 doesn't redo the recruit at all (its first
                # action is unrelated).
                trailer_sig = _compact_action_sig(trailer)
                if (next_first is not None
                        and next_first[0] == "recruit"
                        and trailer_sig is not None
                        and trailer_sig[1:] == next_first[1:]):
                    should_drop = True
                elif (next_first is not None
                        and next_first[0] == "recruit"
                        and trailer_sig is not None
                        and trailer_sig[2:] == next_first[2:]):
                    # Same (x, y), different type => undo+replace.
                    should_drop = True
                else:
                    # Check (c): is the recruit followed by an
                    # end_turn / init_side WITHIN this same block?
                    # If so, it was undone (Wesnoth processes the
                    # recruit, applies undo, the recruit cmd remains
                    # in the [replay] but no [random_seed] follow-up
                    # was emitted because the engine never finalized
                    # the trait roll).
                    # GUARD: skip this check for musthave-only races
                    # (Undead/Mechanical/Elemental) -- those NEVER
                    # emit a [random_seed] regardless of completion,
                    # so "no seed" doesn't imply unfinished. Without
                    # this guard, a legitimately-completed Skeleton
                    # Archer / Ghoul / Walking Corpse recruit at the
                    # end of side 2's turn 1 (followed by end_turn)
                    # gets dropped on every Caves of the Basilisk
                    # replay where side 2 plays Undead.
                    # Concrete: 2p__Caves_of_the_Basilisk_Turn_17_(10379).bz2
                    # block[0] cmd[23] = recruit Skeleton Archer (24,20),
                    # legitimately completed; gets dropped, then
                    # cmd[27]'s move from (23,19) src_missing.
                    trailer_type = trailer[1] if len(trailer) > 1 else ""
                    if _recruit_consumes_rng(trailer_type):
                        for follower in compact_commands[last_action_slot + 1:]:
                            if follower and follower[0] in ("end_turn", "init_side"):
                                should_drop = True
                                break
                    # Check (d): does block i+1 contain ANY recruit
                    # at the same hex (x, y) — possibly after one or
                    # more moves? The redo isn't required to be the
                    # first action of i+1. Same-hex recruit in i+1 =
                    # save-mid-recruit redo, drop the trailer.
                    if (not should_drop and trailer_sig is not None):
                        trailer_hex = (trailer_sig[2], trailer_sig[3])
                        next_hexes = boundary_next_recruit_hexes.get(
                            cmd_idx, set())
                        if trailer_hex in next_hexes:
                            should_drop = True
            if should_drop:
                # Remove the trailer at last_action_slot (NOT
                # necessarily the tail; subsequent end_turn /
                # init_side stay).
                compact_commands.pop(last_action_slot)
                # last_attack_slot may need adjustment.
                if last_attack_slot is not None:
                    if last_attack_slot == last_action_slot:
                        last_attack_slot = None
                    elif last_attack_slot > last_action_slot:
                        last_attack_slot -= 1
                # last_move_slot may need similar adjustment.
                if last_move_slot is not None:
                    if last_move_slot == last_action_slot:
                        last_move_slot = None
                    elif last_move_slot > last_action_slot:
                        last_move_slot -= 1
            last_action_slot = None
            last_action_kind = None

        # Stage 18: save-mid-move duplicate. When Wesnoth saves
        # right after a move completes, the move's [checkup] result
        # IS recorded in block[i] (move was committed). On load,
        # the engine re-emits the same move at block[i+1]'s start
        # WITH another completed [checkup]. Concatenating both
        # makes our sim try to move the same unit twice -- the
        # second attempt fails because the unit is already at the
        # destination (the source hex is now empty).
        # Concrete: 2p__The_Freelands_Turn_18_(170491).bz2: block[0]
        # tail = move (12,15)->(14,18), checkup with result;
        # block[1] cmd[0] = same move, checkup with result; our
        # extractor emits both, second move src_missing at (12,15).
        # Fix: at the boundary, if last_move_slot points to a move
        # whose source/path matches block[i+1]'s first action's
        # move, drop the trailer move.
        if (cmd_idx in boundary_set
                and last_move_slot is not None
                and last_move_slot < len(compact_commands)):
            trailer_move = compact_commands[last_move_slot]
            next_first = boundary_next_first_action.get(cmd_idx)
            if (trailer_move and trailer_move[0] == "move"
                    and next_first is not None
                    and next_first[0] == "move"):
                # `_first_player_action_sig` for moves returns
                # ("move", x_attr, y_attr) where x_attr/y_attr are
                # the FULL WML strings (1-indexed, comma-joined).
                # Convert the trailer's compact (0-indexed lists)
                # back to WML strings to compare.
                xs_wml = ",".join(str(x + 1) for x in trailer_move[1])
                ys_wml = ",".join(str(y + 1) for y in trailer_move[2])
                if (next_first[1] == xs_wml
                        and next_first[2] == ys_wml):
                    compact_commands.pop(last_move_slot)
                    if (last_attack_slot is not None
                            and last_attack_slot > last_move_slot):
                        last_attack_slot -= 1
            last_move_slot = None

    if not compact_commands:
        return None

    # Initial state — no hex map, loader re-parses map_data from here.
    starting_sides = [
        {
            "side": s.side_num, "faction": s.faction, "gold": s.gold,
            "base_income": s.base_income,
            "village_income": s.village_income,
            "village_support": s.village_support,
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
