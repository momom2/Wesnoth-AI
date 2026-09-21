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

import re
from typing import Dict, List, Optional, Sequence, Tuple

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


def wml_bool(value, default: bool) -> bool:
    """WML yes/no/true/false/1/0; `default` when absent or malformed."""
    if value is None:
        return default
    text = str(value).strip().strip('"').lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return default


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
