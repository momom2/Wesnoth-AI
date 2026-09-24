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
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from wesnoth_ai.classes import GameState, Position
from tools.replay_dataset import (
    _build_initial_gamestate,
)
from tools.scenario_events import load_scenario_wml
from tools.wml_state import (MP_EXPERIENCE_MODIFIER, MP_VILLAGE_GOLD,
                             MP_VILLAGE_SUPPORT, check_board_cycle,
                             check_quick_leader_gates, map_starting_positions,
                             read_side, read_tod, read_villages,
                             resolve_map_file, wml_int)
from tools.wml_state import scenario_economy as _read_scenario_economy


log = logging.getLogger("scenario_pool")


# ---------------------------------------------------------------------
# The Ladder Era's 21 PvP maps (Competitive + Classic + Adventurous).
# Source: ~/Documents/My Games/Wesnoth1.18/data/add-ons/Ladder_Era/
# map_picker/{Competitive,Classic,Adventurous}_Maps.cfg
# Verified 2026-04-30.
# ---------------------------------------------------------------------

# Authoritative ladder map list -- the union of every official map
# pool (Ladder Era's Competitive + Classic + Adventurous packs).
# IDs are case-EXACT against wesnoth_src/data/multiplayer/scenarios/
# 2p_*.cfg's `[multiplayer] id=` (verified 2026-05-11): two scenarios
# use lowercase tokens (Elensefar Courtyard, Thousand Stings Garrison),
# the rest are CamelCase. Other modules that filter against this set
# (sim_self_play._is_ladder_map, etc.) consume this constant rather
# than re-defining their own copies -- single source of truth.
LADDER_SCENARIO_IDS: List[str] = [
    "multiplayer_Aethermaw",
    "multiplayer_Arcanclave_Citadel",
    "multiplayer_Basilisk",                   # Caves of the Basilisk
    "multiplayer_Clearing_Gushes",
    "multiplayer_Den_of_Onis",
    "multiplayer_elensefar_courtyard",
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
    "multiplayer_thousand_stings_garrison",
    "multiplayer_Tombs_of_Kesorak",
    "multiplayer_Weldyn_Channel",
]


# Engagement-curriculum subset: 1v1 scenarios from the Mini Maps
# Collection add-on (vendored under wesnoth_src/data/add-ons/
# Mini_Maps_Collection/). These maps are TINY (35-273 cells vs the
# smallest Ladder map's 690), so leaders start ~5-15 hexes apart
# and engagement happens by turn 3-5. Standard mainline 1v1 economy
# constraints (each side gets the gold the scenario specifies, no
# income); fast skirmishes.
#
# Use case: 2026-05-16 diagnosis showed the policy plateaus at ~1%
# attack rate on the Ladder pool regardless of reward magnitude.
# Hypothesis: the long-march exploration problem dominates on
# larger maps; constraining to mini maps removes that barrier
# almost entirely. Turn on via the `--mini-maps` flag
# (sim_self_play.py) or MINI_MAPS=1 (cluster sbatch).
#
# Selected (1v1 scenarios only, 2 player sides + optional scenery):
#   2p_mini                          8x6 = 48 cells
#   2p_mini_edited                   8x8 = 64 cells
#   enclave_micro_isar               9x7 = 63 cells
#   Modified_Tiny_Close_Relation     7x5 = 35 cells   (smallest!)
#   around_mini                      11x13 = 143 cells
#   enclave_mini_fallenstar_1v1      13x13 = 169 cells
#   enclave_small_fallenstar_1v1     21x13 = 273 cells
#   Benji_Autumn_Siege_small         18x12 = 216 cells
#
# Excluded: 2v2 / 3v3 / FFA scenarios (>2 player sides — our self-
# play infrastructure is 1v1 only), and `oasis_mini` (5-side FFA).
MINI_MAP_SCENARIO_IDS: List[str] = [
    "2p_mini",
    "2p_mini_edited",
    # "around_mini" removed 2026-07-14 (user): its side-2 starting
    # Tentacle (water-locked player unit) is out until further notice.
    "enclave_micro_isar",
    "enclave_mini_fallenstar_1v1",
    "enclave_small_fallenstar_1v1",
    "Modified_Tiny_Close_Relation",
    "Benji_Autumn_Siege_small",
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
    """{N: Position} for the cells marking where side N's leader
    spawns. The parsing is `wml_state.map_starting_positions`, shared
    with the replay reader, which used to carry its own copy; this
    wraps the coordinates in `Position` for the pool's callers.
    """
    return {side: Position(x=x, y=y)
            for side, (x, y) in map_starting_positions(raw_map).items()}


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
    # Play this game with fog of war disabled (units always visible;
    # hide-cover abilities still conceal). Set by `random_setup` on a
    # `fogless_ratio` fraction of LADDER-pool games; applied by
    # `build_scenario_gamestate` as `global_info._fog = False`.
    fogless: bool = False
    # Pool category ("ladder"/"fogless"/"mini"); stashed on
    # `global_info._scenario_category` so category-scoped prior
    # biases (action_sampler.prior_bias_end_turn) can detect it from
    # the state alone -- trainer re-forwards included.
    category: str = ""

    # Turn-1 ToD slot override (0=dawn .. 5=second_watch on the
    # default schedule). None -> the scenario's own set-time default
    # (`current_time` from the expanded template, e.g. Fallenstar
    # Lake starts at 5). `random_setup` fills this via
    # `sample_tod_start` so random_start_time=yes scenarios (the
    # Mini_Maps pool) draw a fresh slot per game, mirroring the
    # engine; it doubles as the config hook to force any fixed
    # start time on any map.
    tod_start: Optional[int] = None

    def label(self) -> str:
        """Short human-readable label. Filesystem-safe: no colons,
        slashes, asterisks, etc. Wesnoth's save dialog rejects
        filenames with ` " * / : < > ? \\ | ~` and uses the
        `label=` attr as the default save name on replay end, so
        a label with colons triggers a "save names may not contain
        ..." warning even when the user didn't try to save."""
        sid = self.scenario_id.replace("multiplayer_", "")
        return (f"{sid} - {self.faction1} ({self.leader1}) "
                f"vs {self.faction2} ({self.leader2})")


# Faction that MUST appear on at least one side every game.
# Set to None to fall back to fully-uniform faction sampling.
# Currently locked to Knalgan Alliance per user request 2026-04-30:
# every self-play game has at least one side playing Knalgan, so the
# policy gets concentrated training as / against that faction. The
# OTHER side samples uniformly from all 6 factions including
# Knalgan, so Knalgan-vs-Knalgan mirror matches still happen
# (~16.7% of games); cross-faction Knalgan matches are ~83.3%.
FORCED_FACTION: Optional[str] = "Knalgan Alliance"


def classify_scenario(scenario_id: str) -> str:
    """Map-class of a scenario id: "ladder" (the 21-map competitive
    pool), "mini" (anything else we set up — the Mini Maps engagement
    curriculum), "" for an unknown/empty id. Exists to SPLIT outcome statistics per class: the
    aggregate decisive rate over a mixed curriculum proved misleading
    (2026-07-03: ~50% aggregate while ladder maps were 0/8 decisive)."""
    if not scenario_id:
        return ""
    if scenario_id in LADDER_SCENARIO_IDS:
        return "ladder"
    return "mini"


# Training-mix categories (2026-07-20 user redesign): the game mix
# is specified as ABSOLUTE proportions of all games, one categorical
# roll per game -- no cascading "fraction of the remainder"
# multiplication. "midgame" is rolled here but sampled by the caller
# (it needs the human value corpus, not a scenario pool);
# "fogless" is a LADDER-pool game with fog off.
MIX_CATEGORIES = ("midgame", "mini", "fogless", "ladder")


def validate_mix(*, midgame: float = 0.0, mini: float = 0.0,
                 fogless: float = 0.0,
                 ladder: float = 1.0) -> None:
    """Raise ValueError unless every ratio is in [0, 1] and the four
    sum to 1 (the user must account for the full distribution
    explicitly -- no silent remainders). The sum check stays even as
    categories come and go: it is what turned a silent mis-mix into a
    loud failure (2026-08-04 drill-era incident; user ruling
    2026-08-10 keeps it)."""
    vals = {"midgame": midgame, "mini": mini,
            "fogless": fogless, "ladder": ladder}
    for name, v in vals.items():
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"scenario-mix ratio {name}={v} outside [0, 1]")
    total = sum(vals.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"scenario-mix ratios must sum to 1, got {total:.6f}: "
            + ", ".join(f"{k}={v}" for k, v in vals.items()))


def roll_mix(rng: random.Random, *, midgame: float = 0.0,
             mini: float = 0.0,
             fogless: float = 0.0, ladder: float = 1.0) -> str:
    """One categorical roll over MIX_CATEGORIES with absolute
    weights. Validates the mix on every call (cheap; one call per
    game)."""
    validate_mix(midgame=midgame, mini=mini,
                 fogless=fogless, ladder=ladder)
    r = rng.random()
    for name, w in (("midgame", midgame), ("mini", mini),
                    ("fogless", fogless)):
        if r < w:
            return name
        r -= w
    return "ladder"


def random_setup(
    rng: random.Random,
    *,
    forced_faction: Optional[str] = ...,
    mini_maps: bool = False,
    category: str = "ladder",
) -> ScenarioSetup:
    """Pick a random scenario + 2 (faction, leader) pairs.

    `forced_faction`: if set to a faction name, one side is
    randomly chosen to play it; the other samples uniformly from
    all 6 default-era factions (mirrors still possible on that
    side). Pass None for fully-uniform per-side sampling. The
    sentinel `...` means "use the module-level FORCED_FACTION
    default" -- used so callers that don't override don't have
    to reach into the module to learn the default.

    `mini_maps`: when True, sample only from the 5-map subset in
    MINI_MAP_SCENARIO_IDS (smallest Ladder maps, ~700-870 cells).
    Legacy "100% mini" toggle; overrides `category`.

    `category`: which pool THIS game comes from -- "ladder" (fogged
    21-map pool), "fogless" (ladder pool, fog of war off), or "mini"
    (MINI_MAP_SCENARIO_IDS).
    Mixing is the CALLER's job: roll once per game with `roll_mix`
    (absolute proportions summing to 1) and pass the result here.
    "midgame" is not accepted -- midgame starts are sampled from
    the human value corpus by the training loop, not from a
    scenario pool.

    Leaders are sampled from each faction's `random_leader=` pool,
    matching Wesnoth's `type=random` behavior.
    """
    if forced_faction is ...:
        forced_faction = FORCED_FACTION
    factions = load_factions()
    if not factions:
        raise RuntimeError("no factions loaded")
    if mini_maps:
        category = "mini"
    pools = {"ladder": LADDER_SCENARIO_IDS,
             "fogless": LADDER_SCENARIO_IDS,
             "mini": MINI_MAP_SCENARIO_IDS}
    if category not in pools:
        raise ValueError(f"unknown scenario category {category!r} "
                         f"(expected one of {sorted(pools)})")
    scenario_id = rng.choice(pools[category])
    # Only ladder-pool games play fogless (mini scenarios are the
    # engagement curriculum where fog barely matters and the pools
    # should stay comparable across runs).
    fogless = (category == "fogless")

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
        fogless=fogless,
        category=category,
        tod_start=sample_tod_start(scenario_id, rng),
    )


def _scenario_tod_info(scenario_id: str) -> tuple:
    """(current_time, random_start, n_slots) for a scenario we are
    about to BUILD, read from the scenario's own expansion.

    Schedule macros resolve to a `current_time` attr -- e.g.
    {DEFAULT_SCHEDULE_SECOND_WATCH} (Fallenstar Lake, Ruined Passage)
    emits current_time=5, so real Wesnoth starts those maps at second
    watch, not dawn. `current_time` is None when the scenario declares
    none; `n_slots` counts the schedule's top-level [time] blocks,
    falling back to 6.

    `read_tod` reads the TOP-LEVEL attributes and the top-level [time]
    children, which is what the engine's tod_manager ctor reads: an
    area-local `current_time` is that area's phase, not the game's
    start slot, and `WMLNode.all` does not descend into [time_area].

    ONE SOURCE. Until 2026-09-22 this read the committed template with
    three private regexes while every other part of generation read
    `load_scenario_wml`, so one scenario was built from two renderings
    of itself. They agree on this triple for all 28 pool scenarios,
    measured, and `tests/test_expansion_diff.py` is what keeps them
    agreeing -- the templates are the EXPORT path's input now.
    """
    root = load_scenario_wml(scenario_id)
    block = None if root is None else (
        root.first("multiplayer") or root.first("scenario"))
    if block is None:
        return None, False, 6
    return read_tod(block)


def _scenario_tod_start(scenario_id: str) -> int:
    """Deterministic turn-1 ToD slot: the scenario's `current_time`
    (0 when absent). This is what `build_scenario_gamestate` uses
    when the caller didn't pick a slot -- the set-time-faithful
    default. Before 2026-07-15 this was hardcoded 0 and fresh
    self-play on the second-watch maps ran a ToD cycle shifted 5
    slots from real Wesnoth.
    """
    current_time, _, _ = _scenario_tod_info(scenario_id)
    return current_time if current_time is not None else 0


def sample_tod_start(scenario_id: str, rng: random.Random) -> int:
    """Turn-1 ToD slot the ENGINE would give a fresh game here.

    Mirrors 1.18.4 tod_manager.cpp:51-55 + resolve_random(): a
    `current_time` attr wins outright; otherwise
    `random_start_time=yes` draws a uniform slot, like the engine's
    synced-RNG draw at scenario init; otherwise slot 0. NB the
    Mini_Maps pool is NOT uniformly random-start: 2p_mini,
    2p_mini_edited and Modified_Tiny_Close_Relation lack
    `random_start_time` and are deterministic slot-0 -- exactly the
    3 maps carrying the passivity side-asymmetry (workflow verdict
    2026-08-04; the stale "whole pool" claim here hid that split). Called by `random_setup` so training
    minis start at a random ToD while set-time ladder maps keep
    their scenario value (user policy 2026-07-15). The replay
    exporter pins whatever slot the sim used, so Wesnoth playback
    always matches.
    """
    current_time, random_start, n_slots = _scenario_tod_info(scenario_id)
    if current_time is not None:
        return current_time
    if random_start:
        return rng.randrange(n_slots)
    # De-confound lever (BACKLOG 3c, pre-register before trusting):
    # WESNOTH_MINI_RANDOM_TOD=1 forces a random start slot on the
    # fixed-ToD mini templates. Env-inherited so spool workers pick
    # it up with no CLI forwarding (three forwarding bugs taught
    # that lesson). The replay exporter pins the sampled slot as
    # always, so Wesnoth playback stays faithful either way.
    if (os.environ.get("WESNOTH_MINI_RANDOM_TOD")
            and scenario_id in MINI_MAP_SCENARIO_IDS):
        return rng.randrange(n_slots)
    return 0


def scenario_economy(scenario_id: str) -> Tuple[Optional[int], Optional[int],
                                                Optional[int]]:
    """(village_gold, village_support, experience_modifier) as the
    scenario declares them, each None when it does not.

    One line of parsing, in `tools/wml_state`, shared with the replay
    reader: a save and a .cfg spell these the same way, and two
    parsers of one grammar is how the generation path came to hardcode
    the village economy while the bit-exact sweep certified only the
    other parser.
    """
    root = load_scenario_wml(scenario_id)
    if root is None:
        return None, None, None
    mp = root.first("multiplayer") or root.first("scenario")
    if mp is None:
        return None, None, None
    return _read_scenario_economy(mp)


def build_scenario_gamestate(
    setup: ScenarioSetup,
    *,
    starting_gold: Optional[int] = None,
    base_income: int = 2,
    village_gold: Optional[int] = None,
    village_upkeep: Optional[int] = None,
    experience_modifier: Optional[int] = None,
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

    `village_gold`, `village_upkeep` and `experience_modifier` read
    the same way: None (the default) takes the scenario's value and
    falls back to the multiplayer default, an int overrides. They
    travel to `_build_initial_gamestate` in the same dict fields a
    replay record carries (`starting_sides[*].village_income` /
    `.village_support`, top-level `experience_modifier`), so a game
    built from a scenario and a game rebuilt from a replay go through
    one code path rather than two.

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
    check_board_cycle(mp, setup.scenario_id)
    check_quick_leader_gates(mp, setup.scenario_id)
    project_root = Path(__file__).resolve().parent.parent
    map_path = resolve_map_file(project_root,
                                map_file=mp.attrs.get("map_file", ""),
                                map_data=mp.attrs.get("map_data", ""))
    if map_path is None:
        raise RuntimeError(
            f"scenario {setup.scenario_id} has no map_file or "
            f"resolvable map_data attr")
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
    side_income: Dict[int, int] = {}   # [side] income= OFFSET (engine
    #                                    adds it to the global base)
    side_pre_villages: Dict[int, List[Position]] = {}
    # Sides the ENGINE never gives a turn: controller=null parses to
    # an empty team, and playsingle_controller::skip_empty_sides
    # (playsingle_controller.cpp:198-210) advances the turn loop past
    # every `team::is_empty()` side -- no [init_side], no actions,
    # ever. Our armed-neutral machinery must honor this: Silverhead
    # Crossing's side-3 "Shapeshifter" (an armed Tentacle of the
    # Deep, controller=null) got a full sim turn per round, whose
    # exported [init_side] side_number=3 commands drifted playback's
    # turn counter and with it the ToD phase -- the 2026-07-19
    # damage-desync (19-vs-12 on the Wose strike = afternoon vs
    # night). Minis' tentacle sides declare controller=ai and keep
    # their turn.
    null_controller_sides: set = set()
    # The converse census: extra sides that DO act. The engine's
    # only turn-loop skip is team::is_empty() (= controller=null);
    # any other declared side keeps its turn every round even after
    # its last unit dies (2026-07-21 mini OOS: dropping side 3 at
    # tentacle extinction desynced every exported tentacle game --
    # playback errored "Expacted was a [command] from side 3").
    neutral_actor_sides: set = set()
    side_vision: Dict[int, Dict[str, bool]] = {}
    for s in mp.all("side"):
        sn = wml_int(s.attrs.get("side"), 0)
        if not sn:
            continue
        if s.attrs.get("controller", "").strip() == "null":
            null_controller_sides.add(sn)
        elif sn not in (1, 2):
            neutral_actor_sides.add(sn)
        if sn not in (1, 2):
            continue
        # Fog and shroud as the side declares them (the lobby's defaults
        # when it does not): the scenario supplies the default and
        # `setup.fogless` overrides it below. Three minis declare fog=no.
        declared = read_side(s)
        side_vision[sn] = {"fog": declared["fog"], "shroud": declared["shroud"]}
        gold = wml_int(s.attrs.get("gold"))
        if gold is not None:
            side_gold[sn] = gold
        # [side] income= is an OFFSET on the global base income, not
        # a replacement: team.hpp (1.18.4) `base_income() { return
        # info_.income + game_config::base_income; }`. Thousand
        # Stings Garrison sets income=-2 (net 0 on the no-villages
        # garrison map); ignoring it overpaid both sides 2/turn
        # (found 2026-07-21 with the starting-gold override bug).
        income = wml_int(s.attrs.get("income"))
        if income is not None:
            side_income[sn] = income
        # Pre-owned villages from [village] subblocks. Wesnoth
        # auto-captures these to the side at scenario start, which
        # affects income from turn 1 onward. Read by the same function
        # the replay path uses, which also drops the 0-coordinate
        # placeholders a hand-edited scenario can carry.
        for v in read_villages(s, sn):
            side_pre_villages.setdefault(sn, []).append(
                Position(x=v["x"], y=v["y"]))

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

    # Pre-placed units defined directly in the scenario .cfg's
    # [side] blocks -- two distinct users:
    #
    # (a) PLAYER sides 1/2: scenarios with fixed pre-placed armies
    #     (see DRILL_SCENARIO_IDS). Ordinary controllable units.
    #
    # (b) Neutral / scenery sides >= 3. The mainline 2p maps that use this:
    #
    #   - Thousand Stings Garrison: side=3 controller=null, ~50 petrified
    #     Giant Scorpion statues placed via UNIT_PETRIFY macros.
    #   - Caves of the Basilisk: side 2..N as petrified former heroes
    #     (also UNIT_PETRIFY style, named individuals).
    #   - Aethermaw etc: similar scenery sides.
    #
    # Wesnoth spawns these from the .cfg at scenario start. Without
    # them in our sim's GameState, the recruit-hex legality mask
    # treats their hexes as empty -- the policy can pick e.g. (29, 12)
    # on TSG, which is occupied by a statue, and Wesnoth then
    # rejects the [recruit] command at replay-load with "found
    # [recruit] command expecting user choice".
    #
    # We don't add a SideInfo for these sides: the sim rotates
    # 1<->2 based on `len(gs.sides)`, and the statues are inert
    # (petrified -> 0 moves, no attacks, has_attacked=True). Keeping
    # them out of `starting_sides` avoids accidental side-3 turns;
    # leaving them in `starting_units` (with side=3) is enough for
    # `gs.map.units` to list them and the legality mask to see the
    # hexes as occupied.
    # Pre-placed PLAYER units (sides 1/2) come first: the capability
    # Some scenarios field fixed armies as [unit] blocks under [side]. They
    # are ordinary controllable units (trait-less -- such cfgs
    # carry random_traits=no so Wesnoth playback instantiates them
    # trait-less too, matching this model).
    next_uid = 1000   # leave room above the leader uids; placed
                      # units don't need contiguous numbering.
    for s in mp.all("side"):
        try:
            sn = int(s.attrs.get("side", "0"))
        except ValueError:
            continue
        for u in s.all("unit"):
            try:
                ux_wml = int(u.attrs.get("x", "0"))
                uy_wml = int(u.attrs.get("y", "0"))
            except ValueError:
                continue
            if ux_wml <= 0 or uy_wml <= 0:
                continue
            utype = u.attrs.get("type", "").strip().strip('"')
            if not utype:
                continue
            # [status] petrified=yes -> render as a petrified statue
            # (no moves, no attacks, can't act). Scenery-side only
            # in practice (TSG / CoB).
            petrified = False
            status_node = u.first("status")
            if status_node is not None:
                pet_attr = status_node.attrs.get("petrified", "").strip().lower()
                if pet_attr in ("yes", "true", "1"):
                    petrified = True
            starting_units.append({
                "uid": next_uid,
                "type": utype,
                "side": sn,
                "x": ux_wml - 1,        # 1-WML -> 0-internal
                "y": uy_wml - 1,
                "is_leader": False,
                "petrified": petrified,
            })
            next_uid += 1
    # The scenario's own economy, unless the caller forced one.
    scn_village_gold, scn_village_support, scn_exp = scenario_economy(
        setup.scenario_id)
    if village_gold is None:
        village_gold = (scn_village_gold if scn_village_gold is not None
                        else MP_VILLAGE_GOLD)
    if village_upkeep is None:
        village_upkeep = (scn_village_support if scn_village_support is not None
                          else MP_VILLAGE_SUPPORT)
    if experience_modifier is None:
        experience_modifier = (scn_exp if scn_exp is not None
                               else MP_EXPERIENCE_MODIFIER)
    starting_sides = [
        {
            "side": 1,
            "faction": setup.faction1,
            "gold": _gold_for(1),
            "recruit": list(factions[setup.faction1].recruit),
            "base_income": base_income + side_income.get(1, 0),
            "nb_villages_controlled": len(side_pre_villages.get(1, [])),
            "village_income": village_gold,
            "village_support": village_upkeep,
            **side_vision.get(1, {}),
        },
        {
            "side": 2,
            "faction": setup.faction2,
            "gold": _gold_for(2),
            "recruit": list(factions[setup.faction2].recruit),
            "base_income": base_income + side_income.get(2, 0),
            "nb_villages_controlled": len(side_pre_villages.get(2, [])),
            "village_income": village_gold,
            "village_support": village_upkeep,
            **side_vision.get(2, {}),
        },
    ]
    data = {
        "map_data":            raw_map,
        "game_id":             f"sim_{setup.label()}",
        "scenario_id":         setup.scenario_id,
        "starting_units":      starting_units,
        "starting_sides":      starting_sides,
        "experience_modifier": experience_modifier,
        "tod_start_index":     (setup.tod_start
                                if setup.tod_start is not None
                                else _scenario_tod_start(setup.scenario_id)),
    }

    gs = _build_initial_gamestate(data)
    # The village economy and the experience modifier came through the
    # dict above, as they do for a replay record. `base_income` has no
    # dict field -- `_build_initial_gamestate` writes the engine's 2
    # and every side carries its own `base_income` (the per-[side]
    # `income=` offset is already folded in) -- so the global copy,
    # which only feeds `state_key`, is set here.
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

    # Stash pre-owned villages on global_info. The `_village_owner`
    # map is consulted by _capture_village (to detect "revisit" vs
    # "capture") AND, since 2026-07-11, by the encoder's per-village
    # ownership flags. Marking these as owned at start avoids
    # spurious "captured a village!" rewards on turn 1 if a unit
    # walks onto a pre-owned village.
    if side_pre_villages:
        owner_map = {}
        for sn, positions in side_pre_villages.items():
            for p in positions:
                owner_map[(p.x, p.y)] = sn
        setattr(gs.global_info, "_village_owner", owner_map)

    # Fogless game: underscore attr so GlobalInfo.__deepcopy__
    # carries it through MCTS state copies. Consumed by
    # visibility.units_visible_to (skips the sight-disc gate) and
    # the encoder's village-ownership fog rule.
    if setup.fogless:
        setattr(gs.global_info, "_fog", False)
    # Pool category stash: lets category-scoped prior biases
    # (action_sampler.prior_bias_end_turn) detect mini games from
    # the state alone, symmetrically at rollout and trainer
    # re-forward. Str: aliased safely by GlobalInfo.__deepcopy__.
    if getattr(setup, "category", ""):
        setattr(gs.global_info, "_scenario_category", setup.category)
    # Engine-skipped sides (controller=null): the sim's neutral-turn
    # gate consults this so a null side never acts (see the census
    # comment at the [side] parse above).
    if null_controller_sides:
        setattr(gs.global_info, "_null_controller_sides",
                frozenset(null_controller_sides))
    # Acting extra sides (controller != null, side > 2): the sim's
    # neutral-turn gate keys on THIS static census, never on living
    # units -- see the census comment at the [side] parse above.
    if neutral_actor_sides:
        setattr(gs.global_info, "_neutral_actor_sides",
                frozenset(neutral_actor_sides))
    return gs
