"""The one reader of the WML that describes a game's starting state.

Two pipelines build an initial `GameState`: reconstruction parses a
replay's starting snapshot (`tools/replay_extract.py`), and generation
parses a multiplayer scenario `.cfg` (`tools/scenario_pool.py`). They
end at the same builder, `replay_dataset._build_initial_gamestate`,
but each used to parse the WML itself, and a save and a `.cfg` spell
these parameters identically: the same `[side]` tag with the same
`side`, `gold`, `income`, `village_gold`, `village_support`, `fog`,
`shroud`, `recruit`, `type` and `color` attributes, the same
`[side][village]` children with 1-indexed `x` / `y`, the same
`[side][unit]` children with `[status] petrified`.

Two parsers of one grammar is not a style problem here, it is a
correctness one: the bit-exact replay sweep certifies the
reconstruction parser over 17,039 games and says nothing about the
generation parser, so a divergence between them is invisible to the
project's strongest test. Both had already drifted -- generation
hardcoded the village economy and the experience modifier and never
read `[side] fog=`, reconstruction never read the scenario-level
`mp_village_gold` spelling -- and `tools/sim_to_replay.py` records a
third divergence in its own comments.

Nodes are duck-typed (`.attrs`, `.first(tag)`, `.all(tag)`), which is
what `replay_extract.WMLNode` provides, so this module imports nothing
from the project and both pipelines can depend on it.

Coordinates: WML is 1-indexed, everything here returns 0-indexed
(CLAUDE.md, Coordinates).
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("wml_state")

# Wesnoth's `game_config::base_income`: a `[side] income=` is an offset
# on it, not a replacement (team.hpp 1.18.4, `base_income() { return
# info_.income + game_config::base_income; }`).
ENGINE_BASE_INCOME = 2
# What a 1v1 game is created with when nothing declares otherwise.
# Measured, not assumed: over the corpus's 17,019 raw replay headers
# the village gold is 2 in 16,712 and the experience modifier 70 in
# 16,671 (tools/analysis/corpus_census.py).
MP_VILLAGE_GOLD = 2
MP_VILLAGE_SUPPORT = 1
MP_EXPERIENCE_MODIFIER = 70

_LEADING_INT = re.compile(r"-?\d+")
_TRUE = ("yes", "true", "1")
_FALSE = ("no", "false", "0")


def wml_int(value, default: Optional[int] = None) -> Optional[int]:
    """A WML integer attribute.

    Tolerates the three malformed spellings both pipelines have met:
    surrounding quotes; the percent form the add-on scenarios write
    (`experience_modifier="70%"`); and attributes that got concatenated
    in the wild (`village_gold="1 controller=human"`, seen in a handful
    of 2p Evil Factory saves), from which the leading integer is
    salvaged rather than dropping the whole replay on a parse error.
    Returns `default` when the value is absent, empty or has no
    leading integer.
    """
    if value is None:
        return default
    text = str(value).strip().strip('"').strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    if not text:
        return default
    try:
        return int(text)
    except (ValueError, TypeError):
        pass
    match = _LEADING_INT.match(text)
    if match:
        try:
            return int(match.group(0))
        except (ValueError, TypeError):
            pass
    return default


def wml_bool_or_none(value) -> Optional[bool]:
    """WML yes/no/true/false/1/0, or None when the value is absent or
    is something else entirely.

    The distinction matters where a key has a third form. Wesnoth's
    `random_start_time=` accepts a VALUE LIST (`"2,4"`, draw one of
    these slots), and a reader that folded that onto False treated it
    as "no random start" and silently began at dawn."""
    if value is None:
        return None
    text = str(value).strip().strip('"').lower()
    if not text:
        return None
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def wml_bool(value, default: bool) -> bool:
    """WML yes/no/true/false/1/0; `default` when absent or malformed."""
    parsed = wml_bool_or_none(value)
    return default if parsed is None else parsed


def wml_list(value) -> List[str]:
    """A comma-separated WML list (`recruit=`), empties dropped."""
    return [item.strip() for item in str(value or "").strip().strip('"').split(",")
            if item.strip()]


def scenario_economy(node) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """(village_gold, village_support, experience_modifier) as this
    scenario or snapshot declares them, each None when it does not.

    The village economy has two spellings and both are read. At
    runtime it is a `[side]` attribute (docs/wesnoth_rules.md
    "`village_gold` / `village_support` are PER-SIDE attributes, set by
    host"), which is what a save carries and what the Mini Maps
    Collection writes; a mainline `.cfg` instead declares the
    game-creation setting `mp_village_gold` on the scenario, which
    multiplayer setup copies onto every side. The per-side form wins
    where both appear, because that is the one the engine reads.
    """
    village_gold = wml_int(node.attrs.get("mp_village_gold"))
    village_support = wml_int(node.attrs.get("mp_village_support"))
    for want in (1, 2):
        side = next((s for s in node.all("side")
                     if wml_int(s.attrs.get("side")) == want), None)
        if side is None:
            continue
        # `is not None`, not truthiness: `village_gold=0` is a real
        # setting, which the scenery sides of the mini maps use.
        per_side_gold = wml_int(side.attrs.get("village_gold"))
        per_side_support = wml_int(side.attrs.get("village_support"))
        if per_side_gold is not None:
            village_gold = per_side_gold
        if per_side_support is not None:
            village_support = per_side_support
        break
    return (village_gold, village_support,
            wml_int(node.attrs.get("experience_modifier")))


def read_villages(side_node, side_num: int) -> List[Dict]:
    """The `[side][village]` children a side owns before play, as
    0-indexed records. Wesnoth's `team::team(const config&)` reads
    them into `villages_` (src/team.cpp:208-217), so they pay turn-1
    income; a `.cfg` may also write the combined `x,y=7,41` form, which
    the scenario loader normalizes before this sees it."""
    out: List[Dict] = []
    for node in side_node.all("village"):
        x = wml_int(node.attrs.get("x"), 0)
        y = wml_int(node.attrs.get("y"), 0)
        if x is None or y is None or x <= 0 or y <= 0:
            continue
        out.append({"x": x - 1, "y": y - 1, "side": side_num})
    return out


def read_unit(unit_node, side_num: int, *, uid: int,
              stats: Optional[Dict] = None) -> Optional[Dict]:
    """One `[side][unit]` as a `starting_units` record, or None when it
    is not on the board.

    Campaign saves park recall-list units at `x="recall" y="recall"`;
    those have no grid position, and letting the coordinate parse raise
    would drop the whole replay. `stats` supplies the unit type's
    defaults for the health and movement attributes a `.cfg` omits.
    """
    x = wml_int(unit_node.attrs.get("x", "0"))
    y = wml_int(unit_node.attrs.get("y", "0"))
    if x is None or y is None:
        return None
    status = unit_node.first("status")
    record: Dict = {
        "uid": uid,
        "type": unit_node.attrs.get("type", "").strip().strip('"'),
        "side": side_num,
        "x": max(0, x - 1),
        "y": max(0, y - 1),
        "is_leader": wml_bool(unit_node.attrs.get("canrecruit"), False),
    }
    # Thousand Stings Garrison embeds petrified Giant Scorpions that
    # block movement but cannot fight; without the flag they would
    # counter-attack at full stats.
    if status is not None and wml_bool(status.attrs.get("petrified"), False):
        record["petrified"] = True
    if stats is not None:
        record["hp"] = wml_int(unit_node.attrs.get("hitpoints"), stats["max_hp"])
        record["max_hp"] = wml_int(unit_node.attrs.get("max_hitpoints"),
                                   stats["max_hp"])
        record["max_moves"] = wml_int(unit_node.attrs.get("max_moves"),
                                      stats["max_moves"])
        record["moves"] = wml_int(unit_node.attrs.get("moves"), stats["max_moves"])
    return record


def read_side(side_node, *, defaults: Optional[Dict] = None) -> Optional[Dict]:
    """One `[side]` as a `starting_sides` record, or None when the tag
    carries no usable side number.

    `defaults` supplies what the block does not declare, which is how
    a `.cfg` (no faction, no recruit list, and often no economy) and a
    save (all of them present) reach the same record shape. Keys it
    honours: `gold`, `village_income`, `village_support`, `faction`,
    `recruit`, `fog`, `shroud`.
    """
    side_num = wml_int(side_node.attrs.get("side"), 0)
    if not side_num:
        return None
    d = defaults or {}
    attrs = side_node.attrs
    return {
        "side": side_num,
        "faction": (attrs.get("faction", "").strip().strip('"')
                    or d.get("faction", "")),
        "gold": wml_int(attrs.get("gold"), d.get("gold", 100)),
        # `income=` is an offset on the engine's base, not a replacement.
        "base_income": (wml_int(attrs.get("income"), 0) or 0) + ENGINE_BASE_INCOME,
        "village_income": wml_int(attrs.get("village_gold"),
                                  d.get("village_income", MP_VILLAGE_GOLD)),
        "village_support": wml_int(attrs.get("village_support"),
                                   d.get("village_support", MP_VILLAGE_SUPPORT)),
        "fog": wml_bool(attrs.get("fog"), d.get("fog", True)),
        "shroud": wml_bool(attrs.get("shroud"), d.get("shroud", False)),
        "recruit": wml_list(attrs.get("recruit")) or list(d.get("recruit", ())),
        "leader_type": attrs.get("type", "").strip().strip('"'),
        "color": attrs.get("color", "").strip().strip('"'),
        "controller": attrs.get("controller", "").strip().strip('"'),
    }


# Wesnoth's default six-slot day: dawn, morning, afternoon, dusk,
# first watch, second watch. The sim hardcodes it in BOTH engines --
# `combat.TOD_DEFAULT_CYCLE` and `rust/wesnoth_core` `DEFAULT_CYCLE` --
# so a scenario that declares a different one would be played at the
# wrong time of day by a sim that never looked.
DEFAULT_BOARD_CYCLE: Tuple[int, ...] = (0, 25, 25, 0, -25, -25)


def board_cycle(node) -> List[int]:
    """The lawful bonus of each of the board's top-level `[time]`
    slots, in order. `[]` when the scenario declares no schedule, which
    means the engine's default applies.

    Not `[time_area]`: those are a zone's cycle and are read per hex.
    `WMLNode.all` does not descend, so they are excluded by construction.
    """
    return [wml_int(t.attrs.get("lawful_bonus"), 0) or 0
            for t in node.all("time")]


def board_cycle_is_default(node) -> bool:
    """Whether this scenario's board schedule is the one both engines
    assume. A declared schedule that equals the default is fine; the
    check is on the VALUES, not on whether it was written down."""
    declared = board_cycle(node)
    return not declared or tuple(declared) == DEFAULT_BOARD_CYCLE


class UnsupportedSchedule(RuntimeError):
    """A scenario declares a board schedule neither engine can play."""


class UnmodelledGate(RuntimeError):
    """A scenario turns a rule on or off through a switch we do not
    read, so we would apply the rule the scenario disables."""


_SCHEDULE_WARNED: set = set()


def check_board_cycle(node, scenario_id: str = "") -> bool:
    """Warn once when a scenario's board schedule is not the default,
    and raise under `WESNOTH_STRICT_WML`.

    The cycle is hardcoded in both engines, so a scenario with, say,
    a 24-slot hourly schedule would be played on a six-slot day
    silently, and every combat in it would use the wrong lawful bonus.
    No scenario in the pool and no game in the corpus declares one
    (measured 2026-09-22, docs/scenario_build_plan_20260922.md), which
    is why the constant stands; this is what keeps that true.

    Making it a READ is the fix, and it has to land in both engines at
    once or they diverge on combat -- the Rust half cannot be built or
    certified on the laptop, so it is box work.
    """
    if board_cycle_is_default(node):
        return True
    declared = board_cycle(node)
    if os.environ.get("WESNOTH_STRICT_WML"):
        raise UnsupportedSchedule(
            f"{scenario_id or 'scenario'} declares a board schedule of "
            f"{len(declared)} slots ({declared}); both engines hardcode "
            f"{list(DEFAULT_BOARD_CYCLE)}")
    key = (scenario_id, tuple(declared))
    if key not in _SCHEDULE_WARNED:
        _SCHEDULE_WARNED.add(key)
        log.warning(
            "%s declares a non-default board schedule %s; the sim will "
            "play it on the default six-slot day, so every lawful bonus "
            "in it is wrong. See wml_state.check_board_cycle.",
            scenario_id or "scenario", declared)
    return False


# The era's quick-leader rule has two gates, and we apply it
# unconditionally. Both are read here so an unconditional rule is a
# CHECKED assumption rather than a lucky one.
QUICK_LEADER_GATES = ("dont_make_me_quick", "make_4mp_leaders_quick")


def quick_leader_gates(node) -> List[str]:
    """Where this scenario touches either gate of `quick_4mp_leaders`.

    `wesnoth_src/data/multiplayer/eras.lua:5-22`: the era gives every
    `max_moves=4` leader the quick trait at prestart, UNLESS the WML
    variable `make_4mp_leaders_quick` is false, and skipping any unit
    whose own `dont_make_me_quick` variable is set. `tools/traits.py`
    applies the rule with neither gate, which is exact only while no
    scenario sets either. Dark Forecast and Isle of Mists DO set
    `dont_make_me_quick` on units
    (`data/multiplayer/scenarios/2p_Dark_Forecast.cfg:68`), so the
    gates are not hypothetical -- those maps are simply not ours.
    """
    out: List[str] = []

    def walk(n, path):
        for key, value in n.attrs.items():
            if key in QUICK_LEADER_GATES:
                out.append(f"{path}.{key}")
            elif (key == "name"
                  and str(value).strip().strip(chr(34)) in QUICK_LEADER_GATES):
                out.append(f"{path}.name={value}")
        for child in n.children:
            walk(child, f"{path}/{child.tag}")

    walk(node, "scenario")
    return out


def check_quick_leader_gates(node, scenario_id: str = "") -> bool:
    """Warn once when a scenario touches a gate we do not model, and
    raise under `WESNOTH_STRICT_WML`."""
    hits = quick_leader_gates(node)
    if not hits:
        return True
    if os.environ.get("WESNOTH_STRICT_WML"):
        raise UnmodelledGate(
            f"{scenario_id or 'scenario'} sets {hits}; tools/traits.py "
            f"applies the era quick-leader rule unconditionally")
    key = (scenario_id, tuple(hits))
    if key not in _SCHEDULE_WARNED:
        _SCHEDULE_WARNED.add(key)
        log.warning(
            "%s touches the quick-leader gates %s, which tools/traits.py "
            "does not model: it gives every 4-MP leader the quick trait "
            "unconditionally. See wml_state.check_quick_leader_gates.",
            scenario_id or "scenario", hits)
    return False


def read_tod(node, *, default_slots: int = 6) -> Tuple[Optional[int], bool, int]:
    """(current_time, random_start_time, number of schedule slots).

    Only the reading is shared: reconstruction has to recover the slot
    the server already drew, while generation draws a fresh one, so the
    two resolution policies stay where they are.
    """
    current = wml_int(node.attrs.get("current_time"))
    random_start = wml_bool(node.attrs.get("random_start_time"), False)
    slots = len(node.all("time")) or default_slots
    return current, random_start, slots


def side_numbers(nodes: Sequence) -> List[int]:
    """The side numbers of a sequence of `[side]` nodes, in order."""
    return [n for n in (wml_int(s.attrs.get("side"), 0) for s in nodes) if n]


# ---------------------------------------------------------------------
# The map grid. A replay inlines it as `map_data=` and a scenario
# points at a `.map` file, but from the raw text on both are the same
# format and there is one reader.
# ---------------------------------------------------------------------

_MAP_INCLUDE = re.compile(r"\s*\{\s*~?([^}]+?)\s*\}\s*")


def resolve_map_file(project_root: Path, *, map_file: str = "",
                     map_data: str = "") -> Optional[Path]:
    """The `.map` a scenario points at, or None when it points at
    neither.

    A scenario writes the reference two ways and both are read: the
    mainline form `map_file=multiplayer/maps/2p_Name.map`, resolved
    under `wesnoth_src/data/`, and the add-on form
    `map_data="{~add-ons/<pkg>/maps/<name>.map}"`, which is Wesnoth's
    preprocessor file inclusion. `~add-ons/` lives under
    `wesnoth_src/data/add-ons/` for the vendored add-ons and under the
    project root for this project's own, so both roots are tried.

    Three partial copies of this lived in the exporter, the replay
    builder and the template builder, all of them handling only the
    mainline form, so each raised on a mini map.
    """
    map_file = (map_file or "").strip().strip('"').lstrip("/")
    map_data = (map_data or "").strip().strip('"')
    if map_file:
        return project_root / "wesnoth_src" / "data" / map_file
    match = _MAP_INCLUDE.match(map_data) if map_data else None
    if match is None:
        return None
    relative = match.group(1).lstrip("/")
    vendored = project_root / "wesnoth_src" / "data" / relative
    return vendored if vendored.is_file() else project_root / relative


def split_map_grid(map_data: str) -> Tuple[List[str], int]:
    """Canonical map_data normalizer: (terrain rows, border size).

    Wesnoth .map / map_data may start with HEADER lines
    (`border_size=1`, `usage=map`) before the terrain grid -- mainline
    maps omit them, add-on maps (the whole Mini Maps collection) carry
    them. Every parser that counts rows MUST strip headers first or
    its entire coordinate frame shifts against Wesnoth's: counting
    them as terrain rows displaced every start position by +2 in y,
    and 29 of 34 sampled 2p_mini_edited replays forked from turn 1
    (2026-08-06 fidelity sweep; the export path had hit the same bug
    on 2026-07-06). The border defaults to Wesnoth's 1
    (wesnoth_src/src/map/map.hpp `border_size`).
    """
    border = 1
    rows: List[str] = []
    in_grid = False
    for line in map_data.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not in_grid and "=" in stripped and "," not in stripped:
            key, _, val = stripped.partition("=")
            if key.strip() == "border_size":
                parsed = wml_int(val)
                if parsed is not None:
                    border = parsed
            continue                      # a header line (usage=, ...)
        in_grid = True
        rows.append(line)
    return rows, border


def map_starting_positions(map_data: str) -> Dict[int, Tuple[int, int]]:
    """{side: (x, y)} for the cells carrying a starting-position label
    (`"1 Kh"` marks where side 1's leader spawns), 0-indexed and
    border-stripped like `parse_map_data`.

    A cell may carry several labels, and a label may be a NAME rather
    than a side (`"lake Gs^Vc"`); those are special locations no side
    starts on and `start_position_side` drops them (map.cpp:324-327).
    Border cells are skipped: a marker there would otherwise come back
    with a negative coordinate.
    """
    from tools.terrain_resolver import split_start_position, start_position_side

    out: Dict[int, Tuple[int, int]] = {}
    rows, border = split_map_grid(map_data)
    if not rows:
        return out
    for y, row in enumerate(rows):
        if y < border or y >= len(rows) - border:
            continue
        cells = [c.strip() for c in row.split(",")]
        for x, cell in enumerate(cells):
            if x < border or x >= len(cells) - border or not cell:
                continue
            label, _code = split_start_position(cell)
            for name in label.split():
                side = start_position_side(name)
                if side is not None:
                    out[side] = (x - border, y - border)
    return out
