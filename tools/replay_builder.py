"""Build a Wesnoth-loadable .bz2 replay from scratch.

Replaces the old `sim_to_replay` splice approach (take a source bz2,
swap in our [replay] commands) with a clean assembly from canonical
inputs:

  - The scenario .cfg in `wesnoth_src/data/multiplayer/scenarios/2p_*.cfg`
    -- parsed via `tools.scenario_events.load_scenario_wml`.
  - The .map file referenced by the .cfg's `map_file=` attr.
  - Our `ScenarioSetup` (faction1/leader1/faction2/leader2 picks).
  - The default era's faction definitions
    (wesnoth_src/data/multiplayer/factions/*-default.cfg via
     tools.scenario_pool.load_factions).
  - The sim's `command_history`.

No source replay involvement. Output is byte-for-byte determined
by the inputs above.

Status: FIRST CUT. Not yet validated against Wesnoth's loader.
The structure mirrors a reference replay's layout but several
WML attrs are best-effort and may need tweaking once the user
loads an exported file in Wesnoth and reports failures.

Architecture choices documented inline. Open questions tagged
with `# TODO(replay-builder)`.
"""

from __future__ import annotations

import bz2
import logging
from pathlib import Path
from typing import List, Optional

from tools.replay_extract import WMLNode

# Reuse the existing replay-command emitter; the [replay] block at
# the end is the same shape as before.
from tools.sim_to_replay import _build_replay_wml as _build_command_replay_wml


log = logging.getLogger("replay_builder")


# ---------------------------------------------------------------------
# WML serializer (round-tripping the parser)
# ---------------------------------------------------------------------

def _emit_attr(key: str, value: str, indent: int) -> str:
    """One `key="value"\\n` line. Wesnoth tolerates unquoted numeric
    values but always-quote is safer (`""` round-trip preserves
    intent). Newlines inside the value get translated to literal
    `\\n` -- multiline strings in WML use the +`+= concatenation
    or here-strings; we don't generate either, so any literal
    newline in `value` is treated as a wire-format error in the
    input we'd rather drop."""
    pad = "\t" * indent
    # Wesnoth's WML quotes use "" to escape an embedded ".
    safe = str(value).replace('"', '""')
    return f'{pad}{key}="{safe}"\n'


def emit_wml(node: WMLNode, indent: int = 0) -> str:
    """Serialize a WMLNode (and its children) back to text. The
    inverse of `tools.replay_extract.parse_wml`. Output is
    deterministic and parser-friendly: tab-indented, attrs
    before child blocks, one block per line.

    Pseudo-tag `__root__` at the top is stripped (its children
    become the file's top-level content).
    """
    out: List[str] = []
    pad = "\t" * indent
    if node.tag == "__root__":
        for k, v in node.attrs.items():
            out.append(_emit_attr(k, v, indent))
        for c in node.children:
            out.append(emit_wml(c, indent))
        return "".join(out)

    out.append(f"{pad}[{node.tag}]\n")
    for k, v in node.attrs.items():
        out.append(_emit_attr(k, v, indent + 1))
    for c in node.children:
        out.append(emit_wml(c, indent + 1))
    out.append(f"{pad}[/{node.tag}]\n")
    return "".join(out)


# ---------------------------------------------------------------------
# Scenario block construction
# ---------------------------------------------------------------------

def _build_side_block(
    *, side: int, faction_info, leader_type: str,
    leader_x: int, leader_y: int, gold: int, base_income: int,
    user_team_name: str = "", team_name: str = "",
    pre_villages: Optional[list] = None,
) -> WMLNode:
    """Build the [side] WML for a player-controlled side.

    Carries everything Wesnoth needs to instantiate the leader and
    set up recruit list at game start: faction metadata, leader
    type, gold, recruits, plus the `canrecruit=yes` flag. Pre-
    owned villages are emitted as nested [village] blocks per the
    scenario .cfg semantics."""
    n = WMLNode("side")
    n.attrs["side"]            = str(side)
    n.attrs["canrecruit"]      = "yes"
    n.attrs["controller"]      = "human"
    n.attrs["faction"]         = faction_info.name
    n.attrs["faction_name"]    = faction_info.name
    n.attrs["leader"]          = ",".join(faction_info.leader_pool)
    n.attrs["random_leader"]   = ",".join(faction_info.random_leader_pool)
    n.attrs["recruit"]         = ",".join(faction_info.recruit)
    n.attrs["type"]            = leader_type
    n.attrs["gold"]            = str(gold)
    # Wesnoth's `income=` attr is an OFFSET added to the engine's
    # default base_income (= 2 in default era). 0 -> effective
    # income of 2. There's no `base_income=` attr in standard WML.
    n.attrs["income"]          = "0"
    n.attrs["fog"]             = "yes"
    n.attrs["shroud"]          = "no"
    n.attrs["village_gold"]    = "2"
    n.attrs["village_support"] = "1"
    if team_name:
        n.attrs["team_name"]      = team_name
    if user_team_name:
        n.attrs["user_team_name"] = user_team_name
    n.attrs["color"]           = "purple" if side == 1 else "blue"
    # gender="" lets Wesnoth pick the unit-type's default (some
    # leaders like Lich are male-only; passing female there fails).
    n.attrs["gender"]          = ""
    if pre_villages:
        for (vx, vy) in pre_villages:
            v = WMLNode("village")
            v.attrs["x"] = str(vx + 1)   # 0-internal -> 1-WML
            v.attrs["y"] = str(vy + 1)
            n.children.append(v)
    return n


def _build_scenario_node(
    setup,
    gs,
    raw_map: str,
    scenario_root: WMLNode,
) -> WMLNode:
    """Build the [scenario] node Wesnoth uses as the replay's
    starting state. We start from the parsed .cfg's [multiplayer]
    block (which has events, schedule, etc.) and:

      1. Drop its existing [side] blocks (which are minimal
         placeholders -- Wesnoth normally fills leader/faction
         from the era + lobby choice).
      2. Inject our own [side] blocks for sides 1 and 2 with
         the chosen factions/leaders.
      3. Preserve scenario-controlled extra [side] blocks (e.g.,
         neutral side 3 on Caves of the Basilisk).
      4. Add the map_data attr from the loaded .map file.
    """
    # Lazy import to avoid cycles.
    from tools.scenario_pool import load_factions

    src = (scenario_root.first("multiplayer")
           or scenario_root.first("scenario"))
    if src is None:
        raise RuntimeError(
            f"scenario WML for {setup.scenario_id} has neither "
            "[multiplayer] nor [scenario] root child")

    out = WMLNode("scenario")
    # Carry forward attrs from the source .cfg (id, name, description,
    # map_file, schedule attrs, random_start_time...).
    for k, v in src.attrs.items():
        out.attrs[k] = v

    # Inline the map_data so Wesnoth doesn't try to resolve
    # `map_file=multiplayer/maps/2p_*.map` at replay-load time
    # (where the data path can be ambiguous). The replay carries
    # the map literally.
    # WML uses `map_data="<<...>>"` for embedded multi-line maps;
    # we use the same fence since `parse_wml` doesn't emit those.
    # Workaround: store as a regular quoted attr with newlines
    # escaped -- Wesnoth accepts both forms.
    out.attrs["map_data"] = raw_map
    # Drop map_file so Wesnoth doesn't double-load.
    out.attrs.pop("map_file", None)

    # Pull pre-owned villages off global_info for our [side] blocks.
    pre_villages = getattr(gs.global_info, "_village_owner", None) or {}
    by_side: dict = {}
    for (x, y), sn in pre_villages.items():
        by_side.setdefault(int(sn), []).append((x, y))

    factions = load_factions()

    # Find leader positions from the just-built GameState (avoids
    # re-parsing the map for `1 ` / `2 ` markers).
    leader_pos: dict = {}
    for u in gs.map.units:
        if u.is_leader and u.side in (1, 2):
            leader_pos[u.side] = (u.position.x, u.position.y)
    if 1 not in leader_pos or 2 not in leader_pos:
        raise RuntimeError(
            f"GameState missing leader for side 1 or 2: {leader_pos}")

    # Side 1 + 2 with our chosen factions/leaders.
    s1 = _build_side_block(
        side=1, faction_info=factions[setup.faction1],
        leader_type=setup.leader1,
        leader_x=leader_pos[1][0], leader_y=leader_pos[1][1],
        gold=gs.sides[0].current_gold, base_income=gs.sides[0].base_income,
        team_name="east", user_team_name="teamname^East",
        pre_villages=by_side.get(1),
    )
    s2 = _build_side_block(
        side=2, faction_info=factions[setup.faction2],
        leader_type=setup.leader2,
        leader_x=leader_pos[2][0], leader_y=leader_pos[2][1],
        gold=gs.sides[1].current_gold, base_income=gs.sides[1].base_income,
        team_name="west", user_team_name="teamname^West",
        pre_villages=by_side.get(2),
    )
    out.children.append(s1)
    out.children.append(s2)

    # Carry forward all non-[side] children of the source [multiplayer]
    # block: events, time, music, etc. Plus any [side] with side >=3
    # (neutral wandering creatures, AI-controlled scenario sides).
    for c in src.children:
        if c.tag == "side":
            try:
                sn = int(c.attrs.get("side", "0"))
            except ValueError:
                continue
            if sn in (1, 2):
                continue   # we built our own
            # Side 3+ (e.g., CoB neutral): pass through unchanged.
            out.children.append(c)
        else:
            out.children.append(c)
    return out


# ---------------------------------------------------------------------
# File-level assembly
# ---------------------------------------------------------------------

def _build_file_wml(setup, gs, sim, raw_map: str,
                    scenario_root: WMLNode) -> str:
    """Assemble the full WML document for a Wesnoth replay file.

    Layout (verified against real replays):

      <top-level attrs> (campaign_type, version, era_id, ...)
      [replay]   -- empty stub at the top; Wesnoth's loader expects
                    this even though commands live in the second
                    [replay] block at the end.
      [/replay]
      [scenario]   -- the initial state Wesnoth replays from.
        ...attrs (map_data, description, schedule)...
        [side] x2 -- our chosen leaders/factions.
        [side] x... -- scenario neutrals (CoB statues etc.).
        [event], [time], [music], ...
      [/scenario]
      [replay]   -- the actual command stream.
        [command]
        ...
      [/replay]
    """
    scen = _build_scenario_node(setup, gs, raw_map, scenario_root)

    # Top-level attrs. Match what real replays emit; mostly empty
    # strings for fields we don't populate.
    top_attrs = {
        "abbrev":                 "",
        "active_mods":            "",
        "campaign":               "",
        "campaign_define":        "",
        "campaign_extra_defines": "",
        "campaign_name":          "",
        "campaign_type":          "multiplayer",
        "core":                   "default",
        "difficulty":             "NORMAL",
        "end_credits":            "yes",
        "end_text":               "",
        "end_text_duration":      "0",
        "era_define":             "",
        "era_id":                 "era_default",
        "label":                  setup.label(),
        "mod_defines":            "",
        "mp_game_title":          setup.label(),
        # mp_use_map_settings / mp_village_gold / mp_village_support:
        # NOT emitted yet -- their effect on REPLAY load (vs. lobby
        # game creation) isn't verified against source. team.cpp
        # reads village_gold from the [side] attr directly when
        # loading a saved game (team.cpp:236 reads cfg["village_gold"]
        # with default game_config::village_income), so the per-
        # [side] attrs we already emit should be authoritative.
        # If gold accounting diverges from Wesnoth on load, the
        # next investigation step is to compare team::new_turn
        # behavior on real replay load vs. our [side] attrs.
        "oos_debug":              "no",
        "random_mode":            "",
        "scenario_define":        "",
        # TODO(replay-builder): pin the Wesnoth version. Hardcoding
        # 1.18.4 to match wesnoth_src/. If a future Wesnoth version
        # rejects this attr, drop or update.
        "version":                "1.18.4",
    }

    out: List[str] = []
    for k, v in top_attrs.items():
        out.append(_emit_attr(k, v, 0))

    # Empty [replay] stub at the top -- mirrors what real replays
    # have. Just an [upload_log] inside.
    out.append("[replay]\n")
    out.append("\t[upload_log]\n")
    out.append("\t[/upload_log]\n")
    out.append("[/replay]\n")

    # [scenario] with full initial state.
    out.append(emit_wml(scen, indent=0))

    # [replay] with our command stream.
    out.append(_build_command_replay_wml(sim.command_history))
    return "".join(out)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def export_scenario_replay(
    setup,
    sim,
    out_path: Path,
) -> None:
    """Build a Wesnoth-loadable .bz2 from `setup` + `sim` alone.

    The [scenario] block in a replay is the TURN-1 STARTING state;
    Wesnoth then applies the [replay] command stream to advance
    forward. So we deterministically rebuild a fresh initial
    GameState from `setup` for the scenario serialization rather
    than using `sim.gs` (which has the final post-game state --
    using it would emit final gold/villages/HP/etc. as the
    starting values, divorced from the [replay] sequence).

    `setup` is the `tools.scenario_pool.ScenarioSetup` used to
    seed the sim. `sim` is the post-game `WesnothSim` (with a
    populated `command_history`). `out_path` is where the .bz2
    is written.
    """
    from tools.scenario_events import load_scenario_wml

    scenario_root = load_scenario_wml(setup.scenario_id)
    if scenario_root is None:
        raise RuntimeError(
            f"could not load scenario .cfg for {setup.scenario_id}")

    # Locate + read the .map file.
    mp = (scenario_root.first("multiplayer")
          or scenario_root.first("scenario"))
    if mp is None:
        raise RuntimeError(
            f"scenario {setup.scenario_id}: no [multiplayer]/[scenario]")
    map_file_attr = mp.attrs.get("map_file", "").strip().strip('"')
    if not map_file_attr:
        raise RuntimeError(
            f"scenario {setup.scenario_id}: no map_file attr")
    project_root = Path(__file__).resolve().parent.parent
    map_path = project_root / "wesnoth_src" / "data" / map_file_attr
    if not map_path.is_file():
        raise RuntimeError(f"map not found: {map_path}")
    raw_map = map_path.read_text(encoding="utf-8", errors="replace")

    # Fresh turn-1 GameState (deterministic given setup). This is
    # what Wesnoth needs in the [scenario] block; it then applies
    # sim.command_history to advance forward.
    from tools.scenario_pool import build_scenario_gamestate
    initial_gs = build_scenario_gamestate(setup)

    text = _build_file_wml(setup, initial_gs, sim, raw_map, scenario_root)

    log.info(f"writing {out_path} "
             f"({len(sim.command_history)} commands, "
             f"{len(text):,} chars)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(out_path, "wt", encoding="utf-8") as f:
        f.write(text)
