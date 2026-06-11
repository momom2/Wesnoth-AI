"""Dump a reconstructed GameState back into a Wesnoth-loadable save.

Pairs with tools/replay_dataset.py for end-to-end validation: take any
mid-replay GameState our pipeline reconstructs, write it as a .gz save
file, and let the user open it in Wesnoth. If the rendering matches
what Wesnoth would have shown at that moment, our reconstruction is
faithful; if anything is off (HP mismatch, wrong terrain, missing unit,
gold drift), the discrepancy is visually obvious.

Wesnoth save format (1.18.x): a gzip-compressed WML text file. The
shell of a multiplayer-replay save looks like:

    abbrev=""
    ...top-level metadata key=value pairs...
    version="1.18.4"

    [replay]              # empty wrapper expected; we leave it bare
        [upload_log]
        [/upload_log]
    [/replay]

    [scenario]            # the actual game state at this moment
        map_data="..."
        turn_at="..."
        [side]
            ...attrs...
            [unit]
                ...attrs...
            [/unit]
            ...
        [/side]
        ...
        [time]            # ToD cycle entries
            ...
        [/time]
        ...
    [/scenario]

    [carryover_sides_start]
        next_scenario="..."
    [/carryover_sides_start]

    [multiplayer]
        ...
    [/multiplayer]

We aim for "minimum viable save Wesnoth will load and display", not
byte-for-byte fidelity — the goal is that opening the file shows the
units we reconstructed at the right hexes with the right HP.

CLI:
    python tools/dump_savestate.py REPLAY.json.gz [--turn N] [--out save.gz]

Dependencies: classes, combat, tools.replay_dataset
Dependents: (manual user validation step)
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Make `import classes` etc. work whether we're invoked from the repo
# root or the tools/ subdir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classes import (
    GameState, Hex, Position, SideInfo, Terrain, TerrainModifiers, Unit,
)
import combat as cb
from tools.replay_dataset import (
    iter_replay_pairs_with_state, _stats_for, _terrain_at,
)


# ----------------------------------------------------------------------
# WML emission
# ----------------------------------------------------------------------

def _wml_value(v) -> str:
    """Render one WML attribute value. Wesnoth's own saves quote
    EVERYTHING (including ints), so we follow suit — leniency on the
    save loader's part shouldn't be relied upon. Inner quotes get
    doubled (WML convention)."""
    if isinstance(v, bool):
        return f'"{"yes" if v else "no"}"'
    s = str(v).replace('"', '""')
    return f'"{s}"'


def _emit_attrs(attrs: Dict[str, object], indent: str) -> List[str]:
    return [f'{indent}{k}={_wml_value(v)}' for k, v in attrs.items()]


def _emit_block(tag: str, attrs: Dict[str, object],
                children: Iterable, indent: str = "") -> List[str]:
    """children: iterable of (tag, attrs_dict, children_iter) triples
    OR raw strings already-formatted lines (for opaque pass-through)."""
    out = [f'{indent}[{tag}]']
    for line in _emit_attrs(attrs, indent + "\t"):
        out.append(line)
    for ch in children:
        if isinstance(ch, str):
            out.append(ch)
        else:
            ctag, cattrs, cchildren = ch
            out.extend(_emit_block(ctag, cattrs, cchildren, indent + "\t"))
    out.append(f'{indent}[/{tag}]')
    return out


# ----------------------------------------------------------------------
# Hex grid → map_data string
# ----------------------------------------------------------------------

# Reverse of replay_dataset._TERRAIN_BASE — pick a canonical code per
# Terrain enum. This won't perfectly round-trip stylistic codes (Gg vs
# Gs vs Gd) but Wesnoth treats them as the same base, so loading still
# renders correctly.
_BASE_CODE = {
    Terrain.FROZEN: "Aa", Terrain.FLAT: "Gg",
    Terrain.HILLS: "Hh", Terrain.MOUNTAINS: "Mm",
    Terrain.SHALLOWWATER: "Ww", Terrain.DEEPWATER: "Wo",
    Terrain.SWAMP: "Ss", Terrain.SAND: "Ds",
    Terrain.CAVE: "Ql", Terrain.IMPASSABLE: "Xu",
    Terrain.UNWALKABLE: "Uu", Terrain.CASTLE: "Ch",  # human castle
    Terrain.VILLAGE: "Gg",   # base under a village is grassland by default
    Terrain.FOREST:  "Gg",   # forest overlay
}


def _hex_code(hex_obj: Hex, leader_hex_for_side: Dict[int, Position]) -> str:
    """Render one Hex back to a Wesnoth map code (e.g. 'Gg^Vh')."""
    tts = hex_obj.terrain_types
    base_terrain = next(
        (t for t in (Terrain.CASTLE, Terrain.CAVE, Terrain.MOUNTAINS,
                     Terrain.HILLS, Terrain.SHALLOWWATER, Terrain.DEEPWATER,
                     Terrain.SWAMP, Terrain.SAND, Terrain.FROZEN,
                     Terrain.IMPASSABLE, Terrain.UNWALKABLE, Terrain.FLAT)
         if t in tts),
        Terrain.FLAT,
    )
    base = _BASE_CODE.get(base_terrain, "Gg")
    overlay = ""
    if Terrain.VILLAGE in tts:
        overlay = "Vh"   # human village overlay
    elif Terrain.FOREST in tts:
        overlay = "Fp"   # pine forest
    if TerrainModifiers.KEEP in hex_obj.modifiers:
        base = "Kh"      # keep
    elif TerrainModifiers.CASTLE in hex_obj.modifiers and Terrain.CASTLE in tts:
        base = "Ch"

    # Starting-position prefix: if this hex matches a side's leader,
    # emit "1 Gg^Vh" form. Wesnoth needs the prefix or it complains.
    prefix = ""
    for side, pos in leader_hex_for_side.items():
        if pos.x == hex_obj.position.x and pos.y == hex_obj.position.y:
            prefix = f"{side} "
            break

    code = base + (f"^{overlay}" if overlay else "")
    return prefix + code


def _build_map_data(gs: GameState,
                    leader_pos_by_side: Dict[int, Position]) -> str:
    """Re-emit the hex grid as the row-major map_data string Wesnoth
    consumes. Cells are comma-separated; rows newline-separated."""
    sx, sy = gs.map.size_x, gs.map.size_y
    grid: List[List[str]] = [
        ["Gg" for _ in range(sx)] for _ in range(sy)
    ]
    pos_to_hex = {(h.position.x, h.position.y): h for h in gs.map.hexes}
    for y in range(sy):
        for x in range(sx):
            h = pos_to_hex.get((x, y))
            if h is None:
                grid[y][x] = "_off^_usr"
            else:
                grid[y][x] = _hex_code(h, leader_pos_by_side)
    rows = [", ".join(row) for row in grid]
    return "\n".join(rows)


# ----------------------------------------------------------------------
# Unit emission
# ----------------------------------------------------------------------

def _unit_block(u: Unit) -> List[str]:
    """Emit a single [unit] block matching Wesnoth's save format.

    Three non-obvious requirements for Wesnoth to load the unit
    correctly:
      - `random_traits="no"` — tells the engine not to roll new traits.
      - `[modifications]` block listing the actual traits — without it
        Wesnoth treats the unit as fresh and re-rolls.
      - All attribute values quoted (even numbers) — matches what
        real saves produce.

    Trait IDs come from `u.traits` (populated by tools.traits during
    recruit/leader-spawn).
    """
    stats = _stats_for(u.name)
    traits = sorted(u.traits) if u.traits else []
    # Match the real-save attribute list as closely as we can. Most
    # of these get auto-filled from the unit-type registry on load,
    # but emitting them explicitly avoids relying on default-fill
    # behavior that varies by Wesnoth version.
    attrs = {
        "attacks_left":   1,
        "canrecruit":     "yes" if u.is_leader else "no",
        "cost":           int(stats.get("cost", u.cost or 14)),
        "experience":     int(u.current_exp),
        "extra_recruit":  "",
        "facing":         "se",
        "flag_rgb":       "magenta",
        "gender":         "male",
        "generate_name":  "yes",
        "halo":           "",
        "hitpoints":      int(u.current_hp),
        "id":             u.id,
        "language_name":  u.name,
        "max_attacks":    int(stats.get("max_attacks", 1)),
        "max_experience": int(u.max_exp),
        "max_hitpoints":  int(u.max_hp),
        "max_moves":      int(u.max_moves),
        "moves":          int(u.current_moves),
        "name":           "",
        "overlays":       "",
        "race":           stats.get("race", ""),
        "random_traits":  "no",
        "recall_cost":    -1,
        "resting":        "no",
        "role":           "",
        "side":           u.side,
        "type":           u.name,
        "undead_variation": "",
        "underlying_id":  int(u.id[1:]) if u.id[1:].isdigit() else 0,
        "unrenamable":    "no",
        "upkeep":         "full",
        "usage":          stats.get("usage", "fighter"),
        "variation":      "",
        "vision":         -1,
        # x/y always emitted: Wesnoth respects them when present and
        # places the unit at the listed hex. (Real saves omit them
        # only for leaders that haven't moved from the keep, which is
        # less reliable for mid-game state reconstruction.)
        "x":              u.position.x + 1,   # Wesnoth is 1-indexed
        "y":              u.position.y + 1,
    }
    out = ['\t[unit]']
    for line in _emit_attrs(attrs, "\t\t"):
        out.append(line)
    out.append("\t\t[variables]")
    out.append("\t\t[/variables]")
    out.append("\t\t[filter_recall]")
    out.append("\t\t[/filter_recall]")
    out.append("\t\t[status]")
    for status in sorted(u.statuses):
        out.append(f'\t\t\t{status}="yes"')
    out.append("\t\t[/status]")
    out.append("\t\t[modifications]")
    for trait_id in traits:
        out.extend(_trait_block_wml(trait_id))
    out.append("\t\t[/modifications]")
    out.append('\t[/unit]')
    return out


# Canonical trait WML (id + male_name + effect children) so Wesnoth's
# loader actually applies the trait when reading the save. Direct port
# of the trait macros in `wesnoth_src/data/core/macros/traits.cfg`.
# We only emit the gameplay-relevant fields (id + effects); strings
# like `male_name` are localized and Wesnoth fills them from the
# trait registry by id at display time.
_TRAIT_EFFECTS: Dict[str, List[Tuple[str, str, str]]] = {
    "strong": [
        ("attack", "range", "melee"), ("attack", "increase_damage", "1"),
        ("hitpoints", "increase_total", "1"),
    ],
    "dextrous": [
        ("attack", "range", "ranged"), ("attack", "increase_damage", "1"),
    ],
    "quick": [
        ("movement", "increase", "1"),
        ("hitpoints", "increase_total", "-5%"),
    ],
    "intelligent": [
        ("max_experience", "increase", "-20%"),
    ],
    "resilient": [
        ("hitpoints", "increase_total", "4"),
        ("hitpoints_per_level", "increase_total", "1"),  # times=per level
    ],
    "healthy": [
        ("hitpoints", "increase_total", "1"),
        ("hitpoints_per_level", "increase_total", "1"),
        ("healthy_marker", "set", "yes"),
    ],
    "weak": [
        ("attack", "range", "melee"), ("attack", "increase_damage", "-1"),
        ("hitpoints", "increase_total", "-1"),
    ],
    "slow": [
        ("movement", "increase", "-1"),
        ("hitpoints", "increase_total", "5%"),
    ],
    "dim": [
        ("max_experience", "increase", "20%"),
    ],
    "aged": [
        ("movement", "increase", "-1"),
        ("hitpoints", "increase_total", "-8"),
    ],
    "fearless": [
        ("fearless_marker", "set", "yes"),
    ],
    "feral": [
        ("defense_village_cap", "set", "50"),
    ],
    # Musthaves with status-only effects.
    "undead": [
        ("status", "add", "unpoisonable"),
        ("status", "add", "undrainable"),
        ("status", "add", "unplagueable"),
    ],
    "mechanical": [
        ("status", "add", "unpoisonable"),
        ("status", "add", "undrainable"),
        ("status", "add", "unplagueable"),
    ],
    "elemental": [
        ("status", "add", "unpoisonable"),
        ("status", "add", "undrainable"),
        ("status", "add", "unplagueable"),
    ],
    "loyal": [
        ("loyal_marker", "set", "yes"),
    ],
}


def _trait_block_wml(trait_id: str) -> List[str]:
    """Emit one [trait] block matching what Wesnoth's save would
    include — id + [effect] children matching `traits.cfg`. Without the
    [effect] children, Wesnoth's UI shows the trait but doesn't apply
    its in-game effect on load. The id-only emission is the smaller
    failure mode we want to avoid."""
    out = ["\t\t\t[trait]", f'\t\t\t\tid="{trait_id}"']
    for apply_to, key, value in _TRAIT_EFFECTS.get(trait_id, []):
        if apply_to == "hitpoints_per_level":
            out.append("\t\t\t\t[effect]")
            out.append('\t\t\t\t\tapply_to="hitpoints"')
            out.append(f'\t\t\t\t\ttimes="per level"')
            out.append(f'\t\t\t\t\t{key}={_wml_value(value)}')
            out.append("\t\t\t\t[/effect]")
        elif apply_to == "attack":
            # Two-line per-attack effect — pair adjacent (range, ...) entries.
            # We emit them as one [effect] block; each subsequent
            # `(attack, k, v)` tuple is a key-value pair within.
            # In our table, attack-effects come as TWO consecutive tuples:
            # (attack, range, X) then (attack, increase_damage, Y).
            # Detect by lookahead omitted here — easier to special-case.
            pass
        elif apply_to in ("status", "fearless_marker", "healthy_marker",
                          "loyal_marker", "defense_village_cap"):
            # Map our pseudo-keys back to Wesnoth's apply_to.
            real_apply = {
                "status": "status",
                "fearless_marker": "fearless",
                "healthy_marker": "healthy",
                "loyal_marker": "loyal",
                "defense_village_cap": "defense",
            }[apply_to]
            out.append("\t\t\t\t[effect]")
            out.append(f'\t\t\t\t\tapply_to="{real_apply}"')
            if key == "add":
                out.append(f'\t\t\t\t\tadd="{value}"')
            elif key == "set":
                out.append(f'\t\t\t\t\tset={_wml_value(value)}')
            out.append("\t\t\t\t[/effect]")
        else:
            out.append("\t\t\t\t[effect]")
            out.append(f'\t\t\t\t\tapply_to="{apply_to}"')
            out.append(f'\t\t\t\t\t{key}={_wml_value(value)}')
            out.append("\t\t\t\t[/effect]")
    # Re-emit attack effects properly as paired blocks.
    attack_tuples = [t for t in _TRAIT_EFFECTS.get(trait_id, [])
                     if t[0] == "attack"]
    if attack_tuples:
        # Pairs are (attack, "range", R) followed by (attack, "increase_damage", D).
        for i in range(0, len(attack_tuples), 2):
            range_val = attack_tuples[i][2]
            dmg_val = attack_tuples[i + 1][2]
            out.append("\t\t\t\t[effect]")
            out.append('\t\t\t\t\tapply_to="attack"')
            out.append(f'\t\t\t\t\trange="{range_val}"')
            out.append(f'\t\t\t\t\tincrease_damage={_wml_value(dmg_val)}')
            out.append("\t\t\t\t[/effect]")
    out.append("\t\t\t[/trait]")
    return out


# ----------------------------------------------------------------------
# [side] emission
# ----------------------------------------------------------------------

def _side_block(side_idx: int, side: SideInfo, units: List[Unit],
                raw_side_meta: Optional[dict] = None,
                villages_owned: Optional[List[Tuple[int, int]]] = None,
                ) -> List[str]:
    """One [side] block with all its [unit] children. `raw_side_meta`
    is the matching dict from the original replay's `starting_sides`
    (carries the canonical `leader_type` and `color`).

    Wesnoth's load behavior: if a side has both `type="X"` and a nested
    [unit canrecruit=yes], the engine spawns BOTH a fresh leader from
    `type=` AND keeps the [unit] block — leading to two leaders. Real
    saves work around this by setting `type="null"` when the leader is
    materialized as a [unit] child. We follow suit.
    """
    leader = next((u for u in units if u.is_leader), None)
    raw = raw_side_meta or {}
    color = raw.get("color") or ["red", "blue", "green", "purple"][(side_idx - 1) % 4]
    # When we have a materialized leader [unit] block, omit `type=`
    # from [side] entirely. Wesnoth's loader ALSO spawns from `type=`
    # when present, even if a [unit canrecruit=yes] is already there
    # — that produced the double-leader bug. Real saves rely on the
    # [unit] block alone for the leader's identity.
    attrs = {
        "side": side_idx,
        "controller": "human",
        "team_name": f"Team {side_idx}",
        "user_team_name": f"Team {side_idx}",
        "color": color,
        "gold": int(side.current_gold),
        "income": int(side.base_income),
        "village_gold": 1,
        "village_support": 1,
        "faction": side.faction or "Custom",
        "recruit": ",".join(side.recruits),
        "canrecruit": "yes",
        "fog": "no",
        "shroud": "no",
    }
    if leader is None:
        # No materialized leader — fall back to spawning via [side] type=.
        attrs["type"] = (
            raw.get("leader_type")
            or (side.recruits[0] if side.recruits else "Spearman")
        )
    children = []
    # Village ownership — Wesnoth WML uses [village]x=N y=M[/village]
    # blocks inside [side]. Without these, all hexes that LOOK like
    # villages on the map render as unowned in the UI even when our
    # reconstruction has tracked them as captured.
    for vx, vy in (villages_owned or []):
        children.extend([
            "\t[village]",
            f'\t\tx={vx + 1}',  # 1-indexed
            f'\t\ty={vy + 1}',
            "\t[/village]",
        ])
    for u in units:
        # Embed each unit block as raw lines (already-formatted strings).
        children.extend(_unit_block(u))
    return _emit_block("side", attrs, children)


# ----------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------

def dump_savestate(gs: GameState, scenario_id: str = "multiplayer_test",
                   wesnoth_version: str = "1.18.4") -> str:
    """Render the GameState to one big WML text blob ready to gzip.

    If the GameState was constructed via replay_dataset (which stashes
    `_raw_map_data` and `_raw_starting_sides` on `global_info`), we use
    the verbatim original strings rather than re-rendering — that keeps
    Wesnoth's hundreds of distinct terrain codes (Gg vs Gs vs Gd, Aa vs
    Aab, Hh vs Ha, etc.) intact. Falling back to the lossy hex-grid
    render when those raw strings aren't available.
    """
    leader_pos_by_side: Dict[int, Position] = {}
    units_by_side: Dict[int, List[Unit]] = {}
    for u in sorted(gs.map.units, key=lambda u: (u.side, u.position.y, u.position.x)):
        units_by_side.setdefault(u.side, []).append(u)
        if u.is_leader:
            leader_pos_by_side.setdefault(u.side, u.position)

    raw_map_data = getattr(gs.global_info, "_raw_map_data", "")
    if raw_map_data:
        map_data = raw_map_data
    else:
        map_data = _build_map_data(gs, leader_pos_by_side)
    raw_sides = getattr(gs.global_info, "_raw_starting_sides", []) or []
    raw_side_by_num = {int(s.get("side", 0)): s for s in raw_sides}

    lines: List[str] = []
    # Top-level metadata
    lines.extend(_emit_attrs({
        "abbrev": "",
        "campaign_type": "multiplayer",
        "core": "default",
        "difficulty": "NORMAL",
        "label": scenario_id,
        "version": wesnoth_version,
    }, ""))
    lines.append("")

    # Empty [replay] wrapper — Wesnoth tolerates this for non-replay saves.
    lines.extend(_emit_block("replay", {}, [
        ("upload_log", {}, []),
    ]))
    lines.append("")

    # [scenario]
    scenario_attrs = {
        "id": scenario_id,
        "map_data": map_data,
        "turn_at": int(gs.global_info.turn_number),
        "current_tod": gs.global_info.time_of_day or "morning",
        "name": scenario_id,
    }
    # Village-ownership map (per-replay state stashed by replay_dataset).
    village_owner: Dict[Tuple[int, int], int] = (
        getattr(gs.global_info, "_village_owner", {}) or {}
    )
    villages_by_side: Dict[int, List[Tuple[int, int]]] = {}
    for (vx, vy), owner in village_owner.items():
        villages_by_side.setdefault(owner, []).append((vx, vy))

    scenario_children: List = []
    for side_idx in sorted(units_by_side.keys() | {1, 2}):
        if 1 <= side_idx <= len(gs.sides):
            side = gs.sides[side_idx - 1]
            scenario_children.append(
                _SideRaw(_side_block(
                    side_idx, side, units_by_side.get(side_idx, []),
                    raw_side_meta=raw_side_by_num.get(side_idx),
                    villages_owned=villages_by_side.get(side_idx, []),
                ))
            )

    # Default ToD cycle (vanilla Wesnoth):
    for tod_id, lawful in cb.TOD_DEFAULT_CYCLE:
        scenario_children.append(_TimeRaw(_emit_block("time", {
            "id": tod_id,
            "lawful_bonus": lawful,
        }, [])))

    # Wesnoth save-file structure (per src/savegame.cpp):
    #   [snapshot]   — materialized current state (what we dump here)
    #   [replay_start] — original Turn-1 state (we leave empty)
    #   [replay]     — command history (we leave empty)
    #
    # Wesnoth's loader auto-spawns leaders from each [side]'s `type=`
    # attribute when loading from [scenario]. We emit our state as
    # [snapshot] instead so the loader uses the materialized [unit]
    # blocks directly without duplicating leaders. (Real in-game saves
    # do exactly this.)
    scen_lines = ["[snapshot]"]
    for k, v in scenario_attrs.items():
        scen_lines.append(f"\t{k}={_wml_value(v)}")
    for ch in scenario_children:
        if isinstance(ch, _SideRaw):
            for line in ch.lines:
                scen_lines.append("\t" + line)
        elif isinstance(ch, _TimeRaw):
            for line in ch.lines:
                scen_lines.append("\t" + line)
    scen_lines.append("[/snapshot]")
    lines.extend(scen_lines)
    lines.append("")
    # Empty [replay_start] and [replay] so Wesnoth recognises the file
    # as a save (with snapshot) rather than a scenario template.
    lines.extend(_emit_block("replay_start", {}, []))
    lines.append("")

    # Carryover stub
    lines.extend(_emit_block("carryover_sides_start", {
        "next_scenario": scenario_id,
    }, []))
    lines.append("")

    # Multiplayer settings stub
    lines.extend(_emit_block("multiplayer", {
        "active_mods": "",
        "mp_era": "default",
        "mp_era_name": "Default",
        "mp_scenario": scenario_id,
        "mp_scenario_name": scenario_id,
        "mp_use_map_settings": "yes",
        "mp_village_gold": 1,
        "mp_village_support": 1,
        "scenario": scenario_id,
        "savegame": "no",
    }, []))

    return "\n".join(lines) + "\n"


# Marker classes to hand pre-rendered line lists into the assembly above
# without re-flattening. (Lightweight — avoids the indentation gymnastics
# of a recursive emitter.)
class _SideRaw:
    def __init__(self, lines: List[str]): self.lines = lines
class _TimeRaw:
    def __init__(self, lines: List[str]): self.lines = lines


# ----------------------------------------------------------------------
# CLI driver
# ----------------------------------------------------------------------

def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("replay", type=Path,
                   help="Compact replay (.json.gz) from replay_extract.py")
    p.add_argument("--at-pair", type=int, default=-1,
                   help="Yield save at this (state, action) pair index. "
                        "-1 means final state.")
    p.add_argument("--out", type=Path, default=Path("/tmp/recon_save.gz"),
                   help="Where to write the gzipped save (default /tmp).")
    args = p.parse_args(argv)

    target_idx = args.at_pair
    state: Optional[GameState] = None
    last_state: Optional[GameState] = None
    for i, (gs, _) in enumerate(iter_replay_pairs_with_state(args.replay)):
        last_state = gs
        if target_idx >= 0 and i == target_idx:
            state = gs
            break
    if state is None:
        state = last_state
    if state is None:
        print("No states reconstructed — replay was empty?", file=sys.stderr)
        return 1

    blob = dump_savestate(state, scenario_id=str(args.replay.stem))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        f.write(blob)
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes).")
    print(f"Reconstructed at pair index {target_idx if target_idx >= 0 else 'final'}; "
          f"turn={state.global_info.turn_number} units={len(state.map.units)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
