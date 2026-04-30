"""Scenario-based seed pool for self-play.

Replaces the previous "use replay starting states as seeds" approach.
For each self-play game we:

  1. Pick a random scenario from the 21 Ladder Era maps.
  2. Pick a random faction + leader for each side from the default era.
  3. Build a fresh GameState from the scenario's WML (.cfg + .map),
     placing the chosen leaders at the map's player-start positions
     and giving each side the chosen faction's recruit list.
  4. Hand off to `WesnothSim(gs, scenario_id=...)` which then fires
     the scenario's prestart events (CoB petrified neutrals,
     Aethermaw morph setup, etc.).

Why this beats replay-as-seed: replays carry idiosyncratic starting
states (specific leader picks, sometimes weird recruit overrides);
self-play wants a clean canonical start with random matchups so
the policy doesn't overfit to whatever pairings happened to be
popular in the corpus.

Default era is the source for factions / leaders / recruits.
Source files:
  wesnoth_src/data/multiplayer/factions/*-default.cfg
  wesnoth_src/data/multiplayer/scenarios/2p_*.cfg
  wesnoth_src/data/multiplayer/maps/2p_*.map
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from classes import GameState, Position
from tools.replay_dataset import (
    _build_initial_gamestate, _setup_scenario_events,
)
from tools.scenario_events import load_scenario_wml


log = logging.getLogger("scenario_pool")


# ---------------------------------------------------------------------
# The Ladder Era's 21 PvP maps (Competitive + Classic + Adventurous).
# Source: ~/Documents/My Games/Wesnoth1.18/data/add-ons/Ladder_Era/
# map_picker/{Competitive,Classic,Adventurous}_Maps.cfg
# Verified 2026-04-30.
# ---------------------------------------------------------------------

LADDER_SCENARIO_IDS: List[str] = [
    "multiplayer_Aethermaw",
    "multiplayer_Arcanclave_Citadel",
    "multiplayer_Basilisk",                   # Caves of the Basilisk
    "multiplayer_Clearing_Gushes",
    "multiplayer_Den_of_Onis",
    "multiplayer_Elensefar_Courtyard",        # case-corrected for .cfg lookup
    "multiplayer_Fallenstar_Lake",
    "multiplayer_Hamlets",
    "multiplayer_Hellhole",
    "multiplayer_Howling_Ghost_Badlands",
    "multiplayer_Ruined_Passage",
    "multiplayer_Ruphus_Isle",
    "multiplayer_Sablestone_Delta",
    "multiplayer_Silverhead_Crossing",
    "multiplayer_Sullas_Ruins",
    "multiplayer_Swamp_of_Dread",
    "multiplayer_The_Freelands",
    "multiplayer_The_Walls_of_Pyrennis",
    "multiplayer_Thousand_Stings_Garrison",   # case-corrected
    "multiplayer_Tombs_of_Kesorak",
    "multiplayer_Weldyn_Channel",
]


# ---------------------------------------------------------------------
# Default era factions
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class FactionInfo:
    name: str
    leader_pool: List[str]          # all allowed leader types
    random_leader_pool: List[str]   # subset used for type=random
    recruit: List[str]              # base recruit list


_FACTIONS_CACHE: Optional[Dict[str, FactionInfo]] = None


def _strip_comments(text: str) -> str:
    """Drop full-line `#` comments and trailing-`#` comments.
    The faction .cfg files are simple but `#textdomain` lines and
    `# wmllint:` hints would otherwise confuse the simple parser."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        # Trim trailing comment (very crude; fine for our cfg shape).
        if "#" in line:
            line = line.split("#", 1)[0]
        out.append(line)
    return "\n".join(out)


def _parse_kv_list(value: str) -> List[str]:
    """`leader=Lieutenant,Swordsman,Pikeman` → ['Lieutenant', ...]."""
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_faction_cfg(path: Path) -> Optional[FactionInfo]:
    """Faction .cfg files are one [multiplayer_side] block with
    flat key=value attrs. Hand-roll the parse instead of dragging
    in the WML parser -- these files are simple enough."""
    text = _strip_comments(path.read_text(encoding="utf-8", errors="replace"))
    name: Optional[str] = None
    leader_pool: List[str] = []
    random_leader_pool: List[str] = []
    recruit: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("[") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        # Strip quotes / leading underscore (translation marker).
        val = val.strip()
        if val.startswith("_"):
            val = val[1:].strip()
        val = val.strip('"')
        if key == "id":
            name = val
        elif key == "leader":
            leader_pool = _parse_kv_list(val)
        elif key == "random_leader":
            random_leader_pool = _parse_kv_list(val)
        elif key == "recruit":
            recruit = _parse_kv_list(val)
    if name is None or not leader_pool:
        return None
    if not random_leader_pool:
        # Some factions may omit `random_leader=` and reuse `leader=`.
        random_leader_pool = list(leader_pool)
    return FactionInfo(
        name=name,
        leader_pool=leader_pool,
        random_leader_pool=random_leader_pool,
        recruit=recruit,
    )


def load_factions(faction_dir: Optional[Path] = None) -> Dict[str, FactionInfo]:
    """Parse `wesnoth_src/data/multiplayer/factions/*-default.cfg`
    once and cache. Returns dict[faction_name -> FactionInfo].

    Default era factions: Drakes, Knalgan Alliance, Loyalists,
    Northerners, Rebels, Undead. Dunefolk has a -default.cfg too
    but isn't in the *standard* default era's faction list -- skip
    by name.
    """
    global _FACTIONS_CACHE
    if _FACTIONS_CACHE is not None:
        return _FACTIONS_CACHE
    if faction_dir is None:
        # Default location relative to project root.
        root = Path(__file__).resolve().parent.parent
        faction_dir = root / "wesnoth_src" / "data" / "multiplayer" / "factions"
    if not faction_dir.is_dir():
        raise RuntimeError(
            f"faction dir not found: {faction_dir} -- "
            f"check wesnoth_src/ submodule")
    out: Dict[str, FactionInfo] = {}
    # The mainline default era's 6 factions. Anything else (Dunefolk
    # is not in default era) is skipped.
    DEFAULT_ERA_FACTIONS = {
        "Drakes", "Knalgan Alliance", "Loyalists",
        "Northerners", "Rebels", "Undead",
    }
    for cfg in sorted(faction_dir.glob("*-default.cfg")):
        info = _parse_faction_cfg(cfg)
        if info is None:
            log.warning(f"could not parse {cfg.name}")
            continue
        if info.name not in DEFAULT_ERA_FACTIONS:
            continue
        out[info.name] = info
    if len(out) != len(DEFAULT_ERA_FACTIONS):
        missing = DEFAULT_ERA_FACTIONS - set(out)
        log.warning(f"missing factions: {missing}")
    _FACTIONS_CACHE = out
    return out


# ---------------------------------------------------------------------
# Map starting positions
# ---------------------------------------------------------------------

def extract_player_starts(raw_map: str) -> Dict[int, Position]:
    """Walk the .map data for cells like `"1 Kh"` / `"2 Kh"` --
    Wesnoth's marker for player N's starting hex. Returns
    {N: Position(x, y)} in 0-indexed border-stripped coords (same
    convention as parse_map_data / parse_terrain_codes).

    The leading digit + space prefix marks the keep where side N's
    leader spawns. Multiple maps occasionally have additional
    starts (3, 4, ...) for FFA but ladder maps only have 1 and 2.
    """
    out: Dict[int, Position] = {}
    rows = [r for r in raw_map.splitlines() if r.strip()]
    if not rows:
        return out
    border = 1
    for y_b, row in enumerate(rows):
        if y_b < border or y_b >= len(rows) - border:
            continue
        cells = [c.strip() for c in row.split(",")]
        for x_b, cell in enumerate(cells):
            if x_b < border or x_b >= len(cells) - border:
                continue
            if not cell:
                continue
            # Markers look like "1 Kh" or "2 Kh^Vhh" -- digit, space,
            # terrain code. Anything else: skip.
            if (len(cell) >= 2 and cell[0].isdigit() and cell[1] == " "):
                player = int(cell[0])
                out[player] = Position(x=x_b - border, y=y_b - border)
    return out


# ---------------------------------------------------------------------
# Setup + state assembly
# ---------------------------------------------------------------------

@dataclass
class ScenarioSetup:
    scenario_id: str
    faction1: str
    leader1: str
    faction2: str
    leader2: str

    def label(self) -> str:
        """Short human-readable label for logging."""
        sid = self.scenario_id.replace("multiplayer_", "")
        return f"{sid}:{self.faction1}({self.leader1})_vs_{self.faction2}({self.leader2})"


# Faction that MUST appear on at least one side every game.
# Set to None to fall back to fully-uniform faction sampling.
# Currently locked to Knalgan Alliance per user request 2026-04-30:
# every self-play game has at least one side playing Knalgan, so the
# policy gets concentrated training as / against that faction. The
# OTHER side samples uniformly from all 6 factions including
# Knalgan, so Knalgan-vs-Knalgan mirror matches still happen
# (~16.7% of games); cross-faction Knalgan matches are ~83.3%.
FORCED_FACTION: Optional[str] = "Knalgan Alliance"


def random_setup(
    rng: random.Random,
    *,
    forced_faction: Optional[str] = ...,
) -> ScenarioSetup:
    """Pick a random scenario + 2 (faction, leader) pairs.

    `forced_faction`: if set to a faction name, one side is
    randomly chosen to play it; the other samples uniformly from
    all 6 default-era factions (mirrors still possible on that
    side). Pass None for fully-uniform per-side sampling. The
    sentinel `...` means "use the module-level FORCED_FACTION
    default" -- used so callers that don't override don't have
    to reach into the module to learn the default.

    Leaders are sampled from each faction's `random_leader=` pool,
    matching Wesnoth's `type=random` behavior.
    """
    if forced_faction is ...:
        forced_faction = FORCED_FACTION
    factions = load_factions()
    if not factions:
        raise RuntimeError("no factions loaded")
    scenario_id = rng.choice(LADDER_SCENARIO_IDS)

    if forced_faction is not None and forced_faction in factions:
        # Pick which side gets the forced faction (50/50). The other
        # side samples uniformly from ALL factions, so mirrors still
        # occur ~1/6 of the time on that side.
        forced_side = rng.choice((1, 2))
        other_faction = rng.choice(list(factions.keys()))
        if forced_side == 1:
            f1, f2 = forced_faction, other_faction
        else:
            f1, f2 = other_faction, forced_faction
    else:
        f1 = rng.choice(list(factions.keys()))
        f2 = rng.choice(list(factions.keys()))

    l1 = rng.choice(factions[f1].random_leader_pool)
    l2 = rng.choice(factions[f2].random_leader_pool)
    return ScenarioSetup(
        scenario_id=scenario_id,
        faction1=f1, leader1=l1,
        faction2=f2, leader2=l2,
    )


def build_scenario_gamestate(
    setup: ScenarioSetup,
    *,
    starting_gold: Optional[int] = None,
    base_income: int = 2,
    village_gold: int = 2,
    village_upkeep: int = 1,
    experience_modifier: int = 70,
) -> GameState:
    """Assemble a fresh GameState from scenario WML + faction picks.

    Reuses `_build_initial_gamestate` by constructing the dict that
    function consumes (mirrors the shape of a replay's extracted
    json.gz). Scenario events are NOT fired here -- they fire in
    `WesnothSim.__init__` via `_setup_scenario_events`. So the
    caller wraps the returned state in `WesnothSim(gs, scenario_id=...)`
    before stepping.

    `starting_gold=None` (default): read each side's gold from the
    scenario's [side] `gold=` attr (Arcanclave specifies 175;
    Hamlets has none, falls back to 100). Pass an int to override.

    `experience_modifier=70` matches standard PvP defaults (each
    advance needs 70% of base XP). Other args mirror what
    `apply_pvp_defaults` would inject post-build, but we set them
    inline so the build is one-shot.
    """
    factions = load_factions()
    if setup.faction1 not in factions:
        raise ValueError(f"unknown faction1: {setup.faction1}")
    if setup.faction2 not in factions:
        raise ValueError(f"unknown faction2: {setup.faction2}")

    # Parse the scenario .cfg for the map_file reference.
    root = load_scenario_wml(setup.scenario_id)
    if root is None:
        raise RuntimeError(
            f"scenario .cfg for {setup.scenario_id} not found")
    mp = root.first("multiplayer") or root.first("scenario")
    if mp is None:
        raise RuntimeError(
            f"scenario {setup.scenario_id} has no [multiplayer] / "
            f"[scenario] block")
    map_file_attr = mp.attrs.get("map_file", "").strip().strip('"')
    if not map_file_attr:
        raise RuntimeError(
            f"scenario {setup.scenario_id} has no map_file attr")

    # Resolve map path.
    project_root = Path(__file__).resolve().parent.parent
    map_path = (project_root / "wesnoth_src" / "data" / map_file_attr)
    if not map_path.is_file():
        raise RuntimeError(f"map file not found: {map_path}")
    raw_map = map_path.read_text(encoding="utf-8", errors="replace")

    starts = extract_player_starts(raw_map)
    if 1 not in starts or 2 not in starts:
        raise RuntimeError(
            f"map for {setup.scenario_id} is missing player 1 or 2 "
            f"start markers: {starts}")

    # Per-side gold from the scenario's [side] blocks. Arcanclave
    # specifies gold=175; most maps don't, falling back to the
    # `starting_gold` arg or 100.
    side_gold: Dict[int, int] = {}
    side_pre_villages: Dict[int, List[Position]] = {}
    for s in mp.all("side"):
        try:
            sn = int(s.attrs.get("side", "0"))
        except ValueError:
            continue
        if sn not in (1, 2):
            continue
        if "gold" in s.attrs:
            try:
                side_gold[sn] = int(s.attrs["gold"])
            except ValueError:
                pass
        # Pre-owned villages from [village] subblocks. Wesnoth
        # auto-captures these to the side at scenario start, which
        # affects income from turn 1 onward.
        for v in s.all("village"):
            try:
                vx = int(v.attrs.get("x", "0")) - 1
                vy = int(v.attrs.get("y", "0")) - 1
            except ValueError:
                continue
            side_pre_villages.setdefault(sn, []).append(
                Position(x=vx, y=vy))

    def _gold_for(sn: int) -> int:
        if starting_gold is not None:
            return starting_gold
        return side_gold.get(sn, 100)

    # Build the dict that _build_initial_gamestate consumes. The keys
    # mirror what tools/replay_extract.py emits per replay.
    leader1_pos = starts[1]
    leader2_pos = starts[2]
    starting_units = [
        {
            "uid": 1,
            "type": setup.leader1,
            "side": 1,
            "x": leader1_pos.x,
            "y": leader1_pos.y,
            "is_leader": True,
        },
        {
            "uid": 2,
            "type": setup.leader2,
            "side": 2,
            "x": leader2_pos.x,
            "y": leader2_pos.y,
            "is_leader": True,
        },
    ]
    starting_sides = [
        {
            "side": 1,
            "faction": setup.faction1,
            "gold": _gold_for(1),
            "recruit": list(factions[setup.faction1].recruit),
            "base_income": base_income,
            "nb_villages_controlled": len(side_pre_villages.get(1, [])),
        },
        {
            "side": 2,
            "faction": setup.faction2,
            "gold": _gold_for(2),
            "recruit": list(factions[setup.faction2].recruit),
            "base_income": base_income,
            "nb_villages_controlled": len(side_pre_villages.get(2, [])),
        },
    ]
    data = {
        "map_data":            raw_map,
        "game_id":             f"sim_{setup.label()}",
        "scenario_id":         setup.scenario_id,
        "starting_units":      starting_units,
        "starting_sides":      starting_sides,
        "experience_modifier": experience_modifier,
        "tod_start_index":     0,
    }

    gs = _build_initial_gamestate(data)
    # Override the global_info defaults written by
    # _build_initial_gamestate to match the user's pvp settings.
    gs.global_info.village_gold = village_gold
    gs.global_info.village_upkeep = village_upkeep
    gs.global_info.base_income = base_income

    # `_build_initial_gamestate` hardcodes nb_villages_controlled=0,
    # so pre-owned villages need a post-build patch. Critical for
    # turn-1 income on scenarios like Arcanclave where side 2
    # auto-owns 2 villages (`[side]/[village]` blocks).
    for sn, positions in side_pre_villages.items():
        if 1 <= sn <= len(gs.sides):
            from dataclasses import replace as _replace
            old = gs.sides[sn - 1]
            gs.sides[sn - 1] = _replace(
                old, nb_villages_controlled=len(positions))

    # Stash pre-owned villages on global_info. The encoder doesn't
    # currently use this, but the village ownership tracker
    # (`_village_owner` map in global_info) is consulted by
    # _capture_village to detect "revisit" vs "capture" when a unit
    # walks onto a village. Marking these as owned at start avoids
    # spurious "captured a village!" rewards on turn 1 if a unit
    # walks onto a pre-owned village.
    if side_pre_villages:
        owner_map = {}
        for sn, positions in side_pre_villages.items():
            for p in positions:
                owner_map[(p.x, p.y)] = sn
        setattr(gs.global_info, "_village_owner", owner_map)
    return gs
