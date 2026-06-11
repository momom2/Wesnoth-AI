"""Eval scenario generation: faction defs, map list, and `[test]` builders.

Generates per-matchup `[test] id=eval_*` scenarios under
`add-ons/wesnoth_ai/scenarios/eval/`. Each scenario:
  - Has one side using our custom Lua AI stage (the same one the
    training scenario uses), driven by the Python transformer policy.
  - Has the other side using Wesnoth's default RCA AI (no
    `{~add-ons/wesnoth_ai/ai_config.cfg}` include for that side).

The full set of generated scenarios is auto-included by `_main.cfg`'s
glob `{~add-ons/wesnoth_ai/scenarios/eval/}`. Wesnoth picks them up at
launch; we run a particular matchup with `wesnoth --test eval_<id>`.

Faction data (leader + recruits) is lifted from
`wesnoth_src/data/multiplayer/factions/*-default.cfg` for the 1.18.4
default era. We pick the FIRST listed leader for determinism — random
leader selection would inflate eval variance for no scientific gain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


log = logging.getLogger("eval_scenarios")


# ---------------------------------------------------------------------
# Faction data (Wesnoth 1.18.4 default era)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Faction:
    name:    str          # Wesnoth user-facing name (e.g., "Drakes")
    leader:  str          # leader unit type (first in default era list)
    recruit: List[str]    # recruit list, comma-joined into the [side] block


FACTIONS: List[Faction] = [
    Faction("Drakes",
            "Drake Flare",
            ["Drake Burner", "Drake Clasher", "Drake Glider",
             "Drake Fighter", "Saurian Skirmisher", "Saurian Augur"]),
    Faction("Knalgan Alliance",
            "Dwarvish Steelclad",
            ["Dwarvish Guardsman", "Dwarvish Fighter", "Dwarvish Ulfserker",
             "Dwarvish Thunderer", "Thief", "Poacher", "Footpad",
             "Gryphon Rider"]),
    Faction("Loyalists",
            "Lieutenant",
            ["Cavalryman", "Horseman", "Spearman", "Fencer",
             "Heavy Infantryman", "Bowman", "Mage", "Merman Fighter"]),
    Faction("Northerners",
            "Orcish Warrior",
            ["Orcish Grunt", "Troll Whelp", "Wolf Rider", "Orcish Archer",
             "Orcish Assassin", "Naga Fighter", "Goblin Spearman"]),
    Faction("Rebels",
            "Elvish Captain",
            ["Elvish Fighter", "Elvish Archer", "Mage", "Elvish Shaman",
             "Elvish Scout", "Wose", "Merman Hunter"]),
    Faction("Undead",
            "Dark Sorcerer",
            ["Skeleton", "Skeleton Archer", "Walking Corpse", "Ghost",
             "Vampire Bat", "Dark Adept", "Ghoul"]),
]

FACTION_BY_NAME = {f.name: f for f in FACTIONS}


# ---------------------------------------------------------------------
# Map data (subset of competitive 2p, picked for variety)
# ---------------------------------------------------------------------
#
# Picked for diversity: open / closed / split / asymmetric layouts.
# All ship with stock Wesnoth 1.18; map_file paths verified against
# wesnoth_src/data/multiplayer/scenarios/2p_*.cfg. Excluded the maps
# with extra scenario-event mechanics (Aethermaw, Howling Ghost,
# Thousand Stings) -- those work but introduce variance not tied to
# faction skill, so save them for a later evaluation pass.

@dataclass(frozen=True)
class MapDef:
    short:      str    # short identifier used in scenario ids ("caves", "den")
    map_file:   str    # WML-side map_file= path
    name:       str    # human-readable name


MAPS: List[MapDef] = [
    MapDef("caves",       "multiplayer/maps/2p_Caves_of_the_Basilisk.map",
           "Caves of the Basilisk"),
    MapDef("den",         "multiplayer/maps/2p_Den_of_Onis.map",
           "Den of Onis"),
    MapDef("sablestone",  "multiplayer/maps/2p_Sablestone_Delta.map",
           "Sablestone Delta"),
    MapDef("hornshark",   "multiplayer/maps/2p_Hornshark_Island.map",
           "Hornshark Island"),
    MapDef("hamlets",     "multiplayer/maps/2p_Hamlets.map",
           "Hamlets"),
    MapDef("freelands",   "multiplayer/maps/2p_The_Freelands.map",
           "The Freelands"),
]

MAP_BY_SHORT = {m.short: m for m in MAPS}


# ---------------------------------------------------------------------
# Scenario template + generator
# ---------------------------------------------------------------------

# `our_side` selects which side (1 or 2) is driven by our Python AI.
# The other side gets default Wesnoth RCA (no ai_config.cfg include).
def scenario_id(map_short: str, our_faction: str, opp_faction: str,
                our_side: int) -> str:
    """Stable id used by `wesnoth --test <id>`. ASCII-only, no spaces."""
    def _slug(s: str) -> str:
        return s.lower().replace(" ", "_").replace("'", "")
    return (
        f"eval_{map_short}"
        f"_{_slug(our_faction)}_vs_{_slug(opp_faction)}_s{our_side}"
    )


# WML side block — `{ai_include}` is filled with either the include
# line (for our AI) or the empty string (for default Wesnoth RCA).
_SIDE_TEMPLATE = """\
    [side]
        side={side}
        controller=ai
        save_id={save_id}
        team_name=team_{side}
        user_team_name="{faction_name}"
        type={leader}
        id={save_id}_leader
        name="{faction_name} leader"
        canrecruit=yes
        recruit={recruits}
        gold=100
        village_gold=2
        income=0
{ai_include}
    [/side]
"""

_SCENARIO_TEMPLATE = """\
# AUTO-GENERATED by tools/eval_scenarios.py -- do not edit by hand.
# Regenerate via tools/eval_vs_builtin.py (the eval entry point invokes
# the generator).
#
# Matchup: {our_faction} (side {our_side}, our AI) vs {opp_faction} (side {opp_side}, default RCA)
# Map: {map_name}

[test]
    id={scenario_id}
    name="EVAL: {our_faction} (ours) vs {opp_faction}, {map_name}, s{our_side}"

    map_file={map_file}

    {{DEFAULT_SCHEDULE}}
    turns=-1
    victory_when_enemies_defeated=yes

{side1}
{side2}

    # Per-process random game_id. Without this every eval Wesnoth
    # process falls back to the Lua-side default ("game_0") and 4
    # parallel games stomp on the same add-ons/wesnoth_ai/games/game_0/
    # action.lua + share each other's state-frame std_prints. This
    # mirrors training_scenario.cfg's preload event verbatim -- if you
    # touch it there, mirror the change here too (or extract a macro).
    [event]
        name=preload
        first_time_only=yes

        [lua]
            code=<<
                math.randomseed(os.time() + math.floor(os.clock() * 1000000))
                local game_id = string.format("g%09d", math.random(1, 999999999))
                wml.variables.game_id = game_id
                std_print("=== Eval scenario starting: {scenario_id} ===")
                std_print("Game ID: " .. game_id)
            >>
        [/lua]
    [/event]

    # Turn-boundary marker so log tails are easy to read.
    [event]
        name=turn refresh
        first_time_only=no

        [lua]
            code=<<
                std_print(string.format("=== Turn %d, Side %d ===",
                    wesnoth.current.turn, wesnoth.current.side))
            >>
        [/lua]
    [/event]

    # Terminal-state notification. Critical when the OPPOSING side
    # (default RCA) lands the killing blow on our leader: in that case
    # our turn_stage never gets to run again, so without this hook
    # Python would only learn the game ended via the 90s state-read
    # timeout.
    #
    # We hook `[event] name=die [filter] canrecruit=yes [/filter]`
    # rather than `[event] name=victory` because for all-AI [test]
    # scenarios the engine doesn't reliably fire the victory event.
    # The die event always fires when any leader unit dies and is
    # what training_scenario.cfg uses for its end-game detection.
    # A leader died -> emit a final state frame so Python doesn't have
    # to wait out the 90s state-read timeout. On a 2p map, the LOSER
    # side is the one whose leader died and the WINNER is the other.
    #
    # Why two handlers instead of `[filter] canrecruit=yes` plus a
    # dynamic loser-side lookup: in Wesnoth 1.18 the Lua-side
    # `wesnoth.current.event_context` is built from a config that has
    # NO `unit` field (only x1/y1/unit_x/unit_y/data; see
    # src/scripting/game_lua_kernel.cpp around line 1716). So we can't
    # ask "who died?" from Lua at die-event time. The cleanest answer
    # is to use Wesnoth's own [filter] side= matching: each handler
    # only fires for one side, so the side is known statically.

    [event]
        name=die
        first_time_only=no

        [filter]
            canrecruit=yes
            side=1
        [/filter]

        [lua]
            code=<<
                std_print("Leader of side 1 has died!")
                local m = wesnoth.require("~add-ons/wesnoth_ai/lua/turn_stage.lua")
                m.emit_terminal_state(1)
            >>
        [/lua]
    [/event]

    [event]
        name=die
        first_time_only=no

        [filter]
            canrecruit=yes
            side=2
        [/filter]

        [lua]
            code=<<
                std_print("Leader of side 2 has died!")
                local m = wesnoth.require("~add-ons/wesnoth_ai/lua/turn_stage.lua")
                m.emit_terminal_state(2)
            >>
        [/lua]
    [/event]
[/test]
"""


def _build_side_block(side: int, faction: Faction, our_side: int) -> str:
    is_ours = side == our_side
    ai_include = (
        "        {~add-ons/wesnoth_ai/ai_config.cfg}" if is_ours else ""
    )
    return _SIDE_TEMPLATE.format(
        side=side,
        save_id=("ours" if is_ours else "rca") + f"_s{side}",
        faction_name=faction.name,
        leader=faction.leader,
        recruits=", ".join(faction.recruit),
        ai_include=ai_include,
    )


def build_scenario_cfg(
    map_def: MapDef,
    side1_faction: Faction,
    side2_faction: Faction,
    our_side: int,
) -> Tuple[str, str]:
    """Return (scenario_id, full WML text) for one eval matchup."""
    if our_side not in (1, 2):
        raise ValueError(f"our_side must be 1 or 2, got {our_side}")
    our_faction = side1_faction if our_side == 1 else side2_faction
    opp_faction = side2_faction if our_side == 1 else side1_faction
    sid = scenario_id(map_def.short, our_faction.name,
                      opp_faction.name, our_side)
    text = _SCENARIO_TEMPLATE.format(
        scenario_id=sid,
        our_faction=our_faction.name,
        opp_faction=opp_faction.name,
        our_side=our_side,
        opp_side=3 - our_side,
        map_name=map_def.name,
        map_file=map_def.map_file,
        side1=_build_side_block(1, side1_faction, our_side),
        side2=_build_side_block(2, side2_faction, our_side),
    )
    return sid, text


def generate_eval_scenarios(
    matchups:  List[Tuple[MapDef, Faction, Faction, int]],
    out_dir:   Path,
    main_cfg:  Path,
) -> List[str]:
    """Write per-matchup .cfg files into `out_dir` and ensure `main_cfg`
    glob-includes them. Returns the list of scenario ids written.

    Idempotent: re-running with the same matchups overwrites the same
    files; running with a NEW set wipes any stale eval_*.cfg first so
    Wesnoth doesn't trip on duplicate scenario ids from a prior run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Wipe stale generated files (anything matching our naming pattern).
    for stale in out_dir.glob("eval_*.cfg"):
        stale.unlink()

    written: List[str] = []
    for map_def, fa, fb, our_side in matchups:
        sid, text = build_scenario_cfg(map_def, fa, fb, our_side)
        (out_dir / f"{sid}.cfg").write_text(text, encoding="utf-8")
        written.append(sid)

    _ensure_main_cfg_includes(main_cfg, out_dir)
    log.info(f"generated {len(written)} eval scenarios under {out_dir}")
    return written


_GLOB_INCLUDE_LINE = "{~add-ons/wesnoth_ai/scenarios/eval/}"


def _ensure_main_cfg_includes(main_cfg: Path, eval_dir: Path) -> None:
    """Append a glob include of `eval_dir` to `_main.cfg` if missing.

    Wesnoth's WML preprocessor evaluates `{X/}` (trailing slash) as
    "include every .cfg under X". Once added, every subsequent
    Wesnoth launch sees all files in eval_dir/ automatically -- we
    don't have to touch _main.cfg again on regeneration.
    """
    text = main_cfg.read_text(encoding="utf-8")
    if _GLOB_INCLUDE_LINE in text:
        return
    if not text.endswith("\n"):
        text += "\n"
    text += (
        "\n# Auto-generated eval scenarios. Pre-loaded so they're\n"
        "# available via `wesnoth --test eval_<id>`. Empty until\n"
        "# tools/eval_vs_builtin.py runs.\n"
        f"{_GLOB_INCLUDE_LINE}\n"
    )
    main_cfg.write_text(text, encoding="utf-8")
    log.info(f"added eval-glob include to {main_cfg}")


# ---------------------------------------------------------------------
# Matchup builders -- common configurations
# ---------------------------------------------------------------------

def all_pairs() -> List[Tuple[Faction, Faction]]:
    """All UNIQUE faction pairs incl. mirrors -- 6 mirrors + 15 distinct = 21."""
    out: List[Tuple[Faction, Faction]] = []
    for i, fa in enumerate(FACTIONS):
        for j, fb in enumerate(FACTIONS):
            if i <= j:
                out.append((fa, fb))
    return out


def cross_pairs() -> List[Tuple[Faction, Faction]]:
    """Distinct-faction pairs only -- 15. Skips mirrors."""
    return [(a, b) for a, b in all_pairs() if a != b]


def build_matchup_grid(
    maps:        List[MapDef],
    pairs:       List[Tuple[Faction, Faction]],
    swap_sides:  bool = True,
) -> List[Tuple[MapDef, Faction, Faction, int]]:
    """Each (map, factionA, factionB) generates one or two matchups.

    With swap_sides=True (default), each matchup is played twice:
    once with our AI as side 1, once as side 2. Cancels first-mover
    advantage from the eval estimate.
    """
    out: List[Tuple[MapDef, Faction, Faction, int]] = []
    for m in maps:
        for fa, fb in pairs:
            out.append((m, fa, fb, 1))
            if swap_sides:
                out.append((m, fa, fb, 2))
    return out
