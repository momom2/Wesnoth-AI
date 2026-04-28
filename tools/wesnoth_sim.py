"""Pure-Python headless Wesnoth simulator for self-play training.

Why this exists: running Wesnoth proper (`--test`, `--multiplayer`,
plugin-driven, or wesnothd-relayed) is slow on Windows (SDL frame loop)
and unsolved on the cluster (Linux Wesnoth install is non-trivial,
prior headless attempts didn't pan out -- see
memory/reference_wesnoth_headless_attempt.md). A from-scratch Python
simulator that's bit-exact with Wesnoth's game logic has TWO huge
advantages:

  - 100x+ faster than `--test`: no SDL, no rendering, no IPC.
  - Runs trivially on the cluster: pure Python + numpy + torch.

We already have most of what's needed. The replay-reconstruction
pipeline in tools/replay_dataset.py is bit-exact against Wesnoth (we
verified across 50+ replays during the corpus build) -- it reads the
WML command stream and applies it to a `GameState`. The simulator
reuses that machinery and just swaps the data source: instead of
reading commands from a replay, it queries a Python policy.

What's faithful to Wesnoth (because it shares the replay-recon code):
  - Unit stats / attacks / resistances / abilities (via unit_stats.json).
  - Combat math (combat.py).
  - Trait engine (traits.py): capped defenses, undead/mechanical/elemental
    statuses, defense_overrides.
  - Plague reanimation (Walking Corpse / Soulless variants).
  - Per-turn healing / poison / cures (mirrors heal.cpp::calculate_healing).
  - Time-of-day cycle (lawful_bonus per turn).
  - Scenario events (time_area, store_locations, Aethermaw morph, etc).
  - Village capture on entry.
  - Slow-status drop at end_turn.

What's NOT yet covered (deferred -- check before claiming
self-play parity):
  - Default RCA AI as the opponent. For self-play we don't need this
    (both sides driven by our policy); for eval against built-in AI
    we'd need to reimplement RCA, which is a separate large project.
  - Unit advancement on level-up (replays handle this implicitly via
    the next [unit] command's level info; the simulator would need
    to apply it directly when XP threshold hits).
  - Some scenario events on maps we don't currently train on.
  - Random number generation for trait rolls and combat. The
    replay-recon code uses seeded RNG matching Wesnoth's; for
    self-play we just use Python's random module (different RNG
    stream than Wesnoth, but statistically equivalent).

Usage:
    sim = WesnothSim.from_replay("replays_dataset/<id>.json.gz")
    while not sim.done:
        action = policy.select_action(sim.state, game_label="sim")
        sim.step(action)
    print(sim.winner, sim.turn_count)

Or for AI-vs-AI:
    result = sim.run_game(policy_a, policy_b, max_turns=100)
"""

from __future__ import annotations

import gzip
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make project root importable when run as a script.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from classes import GameState, Position, SideInfo
from replay_dataset import (
    _apply_command,
    _build_initial_gamestate,
    _setup_scenario_events,
)


log = logging.getLogger("wesnoth_sim")


# ---------------------------------------------------------------------
# "Use map settings" -- standard PvP defaults for self-play
# ---------------------------------------------------------------------
# Source replays in our corpus come from games with arbitrary host
# settings: custom-era 800g start, experience_modifier=30, village_gold=6,
# whatever. For self-play training we want consistent ladder-PvP
# behavior across the corpus -- otherwise the policy would learn to
# exploit per-host quirks rather than play standard 2p.
#
# This dataclass + helper override the source replay's settings with
# vanilla 2p multiplayer defaults. Only used by self-play paths
# (`tools/sim_self_play.py`); supervised training on real replays
# MUST keep source-replay settings intact for bit-exact reconstruction.

@dataclass
class PvPDefaults:
    """Standard 2p ladder settings ('use map settings' equivalent
    when self-playing). Defaults match Wesnoth vanilla:
      - 100 gold start
      - +2 base income (game_config::base_income, side income offset 0)
      - +2 gold per village
      - -1 upkeep per village (village_support)
      - 70% experience modifier
    Construct a different instance to deviate (e.g. 75 gold for tournament
    settings, 50% xp for fast-train experiments)."""
    starting_gold:        int = 100
    base_income:          int = 2
    village_gold:         int = 2
    village_support:      int = 1
    experience_modifier:  int = 70


def apply_pvp_defaults(gs: GameState, defaults: PvPDefaults) -> None:
    """Override the GameState's settings with `defaults` in place.
    Touches every side's gold + base_income, plus the global village
    economy and experience modifier. Doesn't touch unit positions,
    factions, recruit lists, or terrain -- only the economy/xp knobs."""
    gs.sides = [
        SideInfo(
            player=s.player,
            recruits=s.recruits,
            current_gold=defaults.starting_gold,
            base_income=defaults.base_income,
            nb_villages_controlled=s.nb_villages_controlled,
            faction=s.faction,
        )
        for s in gs.sides
    ]
    gs.global_info.village_gold = defaults.village_gold
    gs.global_info.village_upkeep = defaults.village_support
    setattr(gs.global_info, "_experience_modifier",
            defaults.experience_modifier)


# ---------------------------------------------------------------------
# Terrain-aware movement-cost lookup
# ---------------------------------------------------------------------
# Wesnoth deducts terrain-dependent MP per hex entered. _apply_command
# in replay_dataset.py uses a flat 1-MP-per-step deduction (good enough
# for replay reconstruction; replays already encode the realized
# trajectory). For self-play simulation we need real costs because the
# sim's move-validity check, AND any exported replay's playback,
# depend on it. Wesnoth's playback runs `plot_turn` and rejects the
# whole move with "corrupt movement" if the very first hex costs more
# than the unit has remaining (see wesnoth_src/src/actions/move.cpp).

_MOVETYPE_COSTS_CACHE: Dict[str, dict] = {}


def _movetype_costs(unit_type: str) -> dict:
    """Look up `{terrain_key: int_cost}` for a unit type. Returns an
    empty dict if not found (caller falls back to cost=1)."""
    if unit_type in _MOVETYPE_COSTS_CACHE:
        return _MOVETYPE_COSTS_CACHE[unit_type]
    try:
        # Lazy import: replay_dataset's module-level _UNIT_DB load
        # already happens during sim init, so this is essentially
        # free.
        import json
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "unit_stats.json"
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        units = data.get("units", {})
        movetypes = data.get("movement_types", {})
        u = units.get(unit_type, {})
        mt = u.get("movement_type")
        costs = movetypes.get(mt, {}).get("movement_costs", {}) if mt else {}
        _MOVETYPE_COSTS_CACHE[unit_type] = costs
        return costs
    except Exception as e:
        log.debug(f"sim: movetype lookup failed for {unit_type!r}: {e}")
        _MOVETYPE_COSTS_CACHE[unit_type] = {}
        return {}


def _move_cost(unit, terrain_key: str) -> int:
    """How many MP does `unit` need to enter a hex of terrain
    `terrain_key`? Falls back to 1 if data is missing (matches the
    sim's pre-existing flat-cost behavior)."""
    costs = _movetype_costs(unit.name)
    cost = costs.get(terrain_key, 1)
    # Wesnoth uses 99 as the impassable sentinel; anything >= 99 is
    # effectively unenterable.
    return int(cost) if cost is not None else 1


# Overlays that affect MOVEMENT cost. There are two patterns in
# wesnoth_src/data/core/terrain.cfg, with different semantics:
#
#   1. `mvt_alias=-,_bas,X` (MINUS marker + _bas): Wesnoth resolves
#      as MAX(base_cost, X_cost). Forest, mushroom-grove, swamp
#      overlays. Listed in `_OVERLAY_MOVEMENT_MAX`.
#
#   2. `mvt_alias=X` (no _bas) OR `aliasof=X` (no _bas): the overlay
#      OVERRIDES the base for movement. Impassable walls (^Xo, ^Pr|,
#      ^Pw|), bridges (^Br|), unwalkable embellishments (^Eqf, ^Qov).
#      A bridge over water makes the hex walkable as flat -- the
#      water cost is irrelevant -- so we can't just append "flat"
#      and MAX, we must REPLACE. Listed in
#      `_OVERLAY_MOVEMENT_OVERRIDE`.
#
# Without these, e.g. `Chw^Xo` (sunken-ruin castle with impassable
# overlay) is read as castle+shallow_water (cost 1-3) instead of
# impassable (99); the sim emits moves into walls that Wesnoth
# rejects as "corrupt movement". Both tables mined from terrain.cfg
# with the helper scripts in tools/wesnoth_sim.py's docstrings.

_OVERLAY_MOVEMENT_MAX: Dict[str, List[str]] = {
    # Forest variants (F*) all alias to "forest" for movement.
    "Fp": ["forest"], "Fpa": ["forest"], "Ft": ["forest"],
    "Ftr": ["forest"], "Ftd": ["forest"], "Ftp": ["forest"],
    "Fts": ["forest"], "Fda": ["forest"], "Fdf": ["forest"],
    "Fds": ["forest"], "Fdw": ["forest"], "Fet": ["forest"],
    "Feta": ["forest"], "Fetd": ["forest"], "Feth": ["forest"],
    "Fma": ["forest"], "Fmf": ["forest"], "Fms": ["forest"],
    "Fmw": ["forest"],
    # Mushroom grove variants → fungus for movement.
    "Tf": ["fungus"], "Tfi": ["fungus"],
    # Underground forest variants (Qhhf / Qhuf) → forest.
    "Qhhf": ["forest"], "Qhuf": ["forest"],
    # Desert / dust overlays.
    "Dc": ["sand"], "Dr": ["hills"],
    # Wreckage on water → swamp movement.
    "Wkf": ["swamp_water"],
    # Impassable mine/cave overlays (Xm/Xv) -- the impassable check
    # in `_move_cost_at_hex` collapses these to 99.
    "Xm": ["impassable"], "Xv": ["impassable"],
}

_OVERLAY_MOVEMENT_OVERRIDE: Dict[str, List[str]] = {
    # Impassable wall overlays. ^Xo is "Impassable Overlay" (=Xt);
    # ^Pr*, ^Pw* are gate / portal walls (rusty gate, wooden door);
    # all override the base entirely. e.g. `Chw^Xo` (castle on
    # water + wall) is impassable, NOT castle+water.
    "Xo": ["impassable"],
    "Pr|": ["impassable"], "Pr/": ["impassable"], "Pr\\": ["impassable"],
    "Pw|": ["impassable"], "Pw/": ["impassable"], "Pw\\": ["impassable"],
    # Unwalkable embellishments (chasm, gorge details). aliasof=Qt.
    "Eqf": ["unwalkable"], "Eqp": ["unwalkable"],
    "Qhux": ["unwalkable"], "Qov": ["unwalkable"],
    # Bridges -- mvt_alias=Gt,Rt: walkable as flat regardless of
    # what's underneath. A bridge over deep water is cost 1, not 99.
    "Br|": ["flat"], "Br/": ["flat"], "Br\\": ["flat"],
    # Underground fungus overlays that REPLACE base.
    "Uf": ["fungus"], "Ufi": ["fungus"],
}


def _move_cost_at_hex(unit, gs, x: int, y: int) -> int:
    """Resolve the movement cost for `unit` entering hex (x, y) per
    Wesnoth's actual rules. Hexes with overlays (e.g. `Gs^Fp` =
    grass+forest, `Mm^Xm` = mountain+impassable, `Re^Tf` =
    road+fungus) have multiple underlying terrain keys, and the cost
    depends on the overlay's `mvt_alias` MINUS marker (prefer-high).
    For impassable / mine / unwalkable overlays Wesnoth picks the MAX
    cost; for forest / fungus / similar movement-affecting overlays
    likewise.

    Without this, our move-validator returns the cheapest underlier
    and the sim emits moves Wesnoth rejects on playback as 'corrupt
    movement' -- the unit's MP differs by the overlay-induced delta."""
    # Pull the raw terrain code so we can apply movement-specific
    # overlay rules (the existing defense-keys table doesn't include
    # movement-only overlays like Tf=fungus).
    codes = getattr(gs.global_info, "_terrain_codes", {}) or {}
    code = codes.get((x, y))
    keys: List[str]
    if not code:
        # Fall back to the defense-keys path so we still get something
        # sensible for synthetic scenarios where _terrain_codes wasn't
        # populated.
        from replay_dataset import _terrain_keys_at
        keys = _terrain_keys_at(gs, x, y) or ["flat"]
    else:
        from replay_dataset import _defense_keys_for_code
        # Strip "1 ", "2 " starting-position markers like
        # _parse_hex_code does, so codes like "2 Ke" resolve to "Ke".
        c = code
        if c[:1].isdigit() and c[1:2] == " ":
            c = c[2:]
        if "^" in c:
            overlay = c.split("^", 1)[1]
            if overlay in _OVERLAY_MOVEMENT_OVERRIDE:
                # OVERRIDE-style overlay: bridge / wall / impassable
                # gate. The base terrain is irrelevant for movement
                # -- a bridge over deep water is walkable, an
                # impassable wall on grass is impassable. Use the
                # overlay's keys exclusively.
                keys = list(_OVERLAY_MOVEMENT_OVERRIDE[overlay])
            else:
                # MAX-style overlay (forest/fungus/etc.) OR pure
                # cosmetic overlay (Em/Es/Bs*): start from the base
                # defense keys and append any extra MAX keys.
                keys = list(_defense_keys_for_code(c))
                for k in _OVERLAY_MOVEMENT_MAX.get(overlay, []):
                    if k not in keys:
                        keys.append(k)
        else:
            keys = list(_defense_keys_for_code(c))
        if not keys:
            keys = ["flat"]
    if any(k in ("impassable", "unwalkable") for k in keys):
        return 99
    costs = _movetype_costs(unit.name)
    return max(int(costs.get(k, 1) or 1) for k in keys)


# ---------------------------------------------------------------------
# Per-action seed generation (shared with sim_to_replay)
# ---------------------------------------------------------------------
# The sim must use the SAME seed for trait rolls / damage rolls that
# the exported `.bz2` replay will hand to Wesnoth via [random_seed].
# Otherwise the sim's recruited Cavalryman might roll the `quick`
# trait (+1 MP), allowing 3 forest moves; Wesnoth, getting a different
# seed (or rolling differently from a different seed), gives the unit
# `intelligent` instead -- 8 MP, only 2 forest moves -- and rejects
# the third move as "corrupt movement" on playback.
#
# Solution: deterministic per-request seeds derived from a counter.
# Both the sim and sim_to_replay call this with the same request_id,
# get the same hex string, and so trait/damage rolls match.

def request_seed(request_id: int) -> str:
    """Deterministic 8-hex-char seed for the n-th random-rolling
    command in a sim run. Re-implemented (not imported) to keep the
    dependency one-way: sim_to_replay imports from wesnoth_sim, not
    the reverse."""
    import hashlib
    h = hashlib.sha256(f"sim_to_replay:{request_id}".encode()).hexdigest()
    return h[:8]


# ---------------------------------------------------------------------
# Recruit cost lookup
# ---------------------------------------------------------------------
# Wesnoth's recruit handler refuses a recruit if the side can't afford
# the unit's cost (see wesnoth_src/src/synced_commands.cpp recruit
# handler around the `u_type->cost() > beginning_gold` check). Our
# `_apply_command` is permissive (clamps gold to 0), so the sim's
# move-validation layer needs the cost to gate recruits before they
# reach _apply_command.

_RECRUIT_COSTS_CACHE: Dict[str, int] = {}


def _recruit_cost_for(unit_type: str) -> int:
    """Look up the recruit cost (gold) for a unit type from
    `unit_stats.json`. Returns 14 (the smallfoot/orcishfoot Footpad
    baseline) if not found, so an unknown type fails the gold check
    safely on a small budget rather than slipping through with cost=0."""
    if unit_type in _RECRUIT_COSTS_CACHE:
        return _RECRUIT_COSTS_CACHE[unit_type]
    try:
        import json
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "unit_stats.json"
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        cost = int(data.get("units", {}).get(unit_type, {}).get("cost", 14))
        _RECRUIT_COSTS_CACHE[unit_type] = cost
        return cost
    except Exception as e:
        log.debug(f"sim: recruit cost lookup failed for {unit_type!r}: {e}")
        _RECRUIT_COSTS_CACHE[unit_type] = 14
        return 14


@dataclass
class SimResult:
    """Outcome of a simulated game."""
    winner:        int          # 0 = draw/timeout, 1 or 2 = side index
    turns:         int          # final turn number reached
    side1_actions: int          # how many actions side 1's policy emitted
    side2_actions: int
    ended_by:      str          # 'leader_killed' | 'max_turns' | 'max_actions' | 'no_legal'
    trajectory:    List["SimStep"] = field(default_factory=list)


@dataclass
class SimStep:
    """One (state, action, side) record. Reward is computed downstream
    by the trainer's reward function -- the simulator stays generic."""
    state:  GameState     # game state BEFORE the action
    action: dict          # the policy's action dict
    side:   int           # which side's turn it was


@dataclass
class RecordedCommand:
    """One entry in the simulator's command history -- enough to
    reconstruct a Wesnoth-loadable replay.

    `kind` is one of "init_side" | "move" | "attack" | "recruit" |
    "recall" | "end_turn". `cmd` is the raw command list as consumed
    by `_apply_command`. `side` is the side that acted (0 for system
    events that don't have a clear actor; in practice init_side and
    end_turn carry the relevant side index).

    `extras` stores side-channel data the WML format needs but
    `_apply_command` doesn't: today only `leader_pos` for recruit
    (the [from] coordinates), captured at recruit time so the
    exporter doesn't have to reconstruct game state.
    """
    kind:   str
    side:   int
    cmd:    list
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------

class WesnothSim:
    """One self-play game running fully in-process. Bit-exact game
    logic via the existing replay-reconstruction code."""

    # Hard caps to prevent infinite games. The values are tuned for
    # competitive 2p maps where a typical game ends in 20-40 turns and
    # ~150-300 player actions per side.
    DEFAULT_MAX_TURNS   = 80
    DEFAULT_MAX_ACTIONS_PER_SIDE = 500

    def __init__(
        self,
        initial_state: GameState,
        scenario_id:   str,
        max_turns:     int = DEFAULT_MAX_TURNS,
        max_actions_per_side: int = DEFAULT_MAX_ACTIONS_PER_SIDE,
    ):
        self.gs = initial_state
        self.scenario_id = scenario_id
        self.max_turns = max_turns
        self.max_actions_per_side = max_actions_per_side

        # Wire scenario-specific events (time_area, store_locations,
        # Aethermaw morph, etc.) -- mirrors what replay_dataset does
        # at the top of iter_replay_pairs.
        _setup_scenario_events(self.gs, scenario_id)

        self.done:      bool = False
        self.winner:    int  = 0
        self.ended_by:  str  = ""
        self._actions_by_side: Dict[int, int] = {1: 0, 2: 0}

        # Command history -- ordered list of every command applied to
        # the game state. Used by sim_to_replay.py to export a
        # Wesnoth-loadable .bz2 replay so the user can inspect a
        # simulated game in the Wesnoth GUI.
        self.command_history: List[RecordedCommand] = []

        # Per-action RNG-request counter. Increments on every command
        # that consumes a Wesnoth synced [random_seed] (recruit,
        # attack). Both the sim's _build_recruit_unit / combat code
        # and sim_to_replay's WML emitter derive seeds from this
        # counter so the trait rolls / damage rolls agree bit-exact
        # between simulator and Wesnoth playback.
        self._rng_requests: int = 0

        # Turn 0 is pre-game. The first init_side(1) bumps to turn 1
        # AND fires turn-1 events / healing. Mirror that here.
        self._begin_side_turn(1)

    # ----- factory ---------------------------------------------------

    @classmethod
    def from_replay(
        cls,
        gz_path: Path,
        *,
        pvp_defaults: Optional[PvPDefaults] = None,
        **kwargs,
    ) -> "WesnothSim":
        """Build an initial state from any replay's `starting_units` +
        scenario_id. The replay is used as a source of map / faction /
        starting-unit configuration only -- we discard the command
        stream and let the policy decide what happens from turn 1.

        `pvp_defaults`: when provided, overrides the source replay's
        economy / experience settings with standard PvP values. Use
        for self-play; leave None for replay reconstruction in
        supervised training."""
        with gzip.open(Path(gz_path), "rt", encoding="utf-8") as f:
            data = json.load(f)
        gs = _build_initial_gamestate(data)
        scenario_id = data.get("scenario_id", "")
        if pvp_defaults is not None:
            apply_pvp_defaults(gs, pvp_defaults)
        return cls(gs, scenario_id=scenario_id, **kwargs)

    # ----- public stepping API ---------------------------------------

    @property
    def state(self) -> GameState:
        return self.gs

    @property
    def current_side(self) -> int:
        return self.gs.global_info.current_side

    def fork(self) -> "WesnothSim":
        """Cheap clone for MCTS-style branching. Deepcopies the
        game state (via Map.__deepcopy__'s fast-path) and the small
        scalar fields. **Drops command_history** -- forks aren't
        meant to be exported to bz2; if you mutate the parent's
        command_history aliased to the fork, you'd corrupt the
        export pipeline. Forks always start with an empty history.

        Mirrors `Trainer.step` and `WesnothSim.step` semantics:
        anything mutable downstream is properly isolated, immutable
        config (max_turns, max_actions_per_side, scenario_id) is
        aliased."""
        import copy as _copy
        out = WesnothSim.__new__(WesnothSim)
        out.gs = _copy.deepcopy(self.gs)
        out.scenario_id = self.scenario_id
        out.max_turns = self.max_turns
        out.max_actions_per_side = self.max_actions_per_side
        out.done     = self.done
        out.winner   = self.winner
        out.ended_by = self.ended_by
        out._actions_by_side = dict(self._actions_by_side)
        out._rng_requests    = self._rng_requests
        out.command_history  = []   # forks don't track history
        return out

    def step(self, action: dict) -> bool:
        """Apply one action. Returns True if the game is over after
        this step. The action dict is the same shape the policy
        produces (see action_sampler.SampledAction.action)."""
        if self.done:
            return True

        side_now = self.gs.global_info.current_side

        # Capture leader position BEFORE applying the action -- recruit
        # exports need it for the WML [from] block, and even though
        # _apply_command's recruit handler doesn't move the leader,
        # taking the snapshot here keeps `_action_to_command` pure.
        leader_pos: Optional[Tuple[int, int]] = None
        if action.get("type") == "recruit":
            for u in self.gs.map.units:
                if u.side == side_now and u.is_leader:
                    leader_pos = (u.position.x, u.position.y)
                    break

        cmd = self._action_to_command(action)
        if cmd is None:
            # Action was illegal-shaped or referred to a missing unit.
            # Treat as a wasted turn -- end the side's turn.
            log.debug(f"sim: untranslatable action {action!r}; ending turn")
            cmd = ["end_turn"]

        if cmd[0] == "end_turn":
            _apply_command(self.gs, ["end_turn"])
            self.command_history.append(RecordedCommand(
                kind="end_turn", side=side_now, cmd=["end_turn"]))
            # Advance to the next side. 2p only for now.
            n_sides = max(2, len(self.gs.sides))
            next_side = (side_now % n_sides) + 1
            self._begin_side_turn(next_side)
        else:
            # Snapshot the village owner BEFORE the move applies so
            # we can tell capture (prior owner != side) from revisit
            # (prior owner == side, no MP zeroing).
            prev_village_owner: Optional[int] = None
            if cmd[0] == "move":
                tx, ty = cmd[1][-1], cmd[2][-1]
                owner_map = getattr(
                    self.gs.global_info, "_village_owner", None) or {}
                prev_village_owner = int(owner_map.get((tx, ty), 0))

            # Snapshot attacker/defender (id, name, side) BEFORE attack so
            # we can detect mid-attack advancement post-apply. Wesnoth's
            # `attack_unit_and_advance` (attack.cpp:1556-1573) advances
            # attacker first, then defender, each producing one
            # `[command] dependent="yes" [choose] value=K [/choose]` block
            # in the replay. If we don't emit those, playback OOSes the
            # first time a model lands a level-up kill.
            pre_att = pre_dfd = None
            if cmd[0] == "attack":
                ax, ay = cmd[1], cmd[2]
                dx, dy = cmd[3], cmd[4]
                for u in self.gs.map.units:
                    if u.position.x == ax and u.position.y == ay:
                        pre_att = (u.id, u.name, u.side)
                    elif u.position.x == dx and u.position.y == dy:
                        pre_dfd = (u.id, u.name, u.side)

            _apply_command(self.gs, cmd)
            # Post-apply terrain MP correction. _apply_command
            # already flat-deducted 1 MP; we owe an extra
            # (cost - 1) so the unit's MP reflects Wesnoth's
            # terrain-based cost.
            terrain_cost = action.get("_terrain_cost") if isinstance(action, dict) else None
            if cmd[0] == "move" and terrain_cost is not None and terrain_cost > 1:
                tx, ty = cmd[1][-1], cmd[2][-1]
                self._deduct_extra_mp(tx, ty, terrain_cost - 1)
            # Post-apply ZoC / ambush / village-capture MP zeroing.
            # Wesnoth's `unit_mover::try_actual_movement`
            # (move.cpp:1042-1054) calls `set_movement(0, true)` when
            # the unit lands in a ZoC'd hex, gets ambushed by a
            # hidden enemy, or captures a non-friendly village. Our
            # sim was missing all three, so the unit could keep
            # moving past the stop point -- subsequent moves would be
            # valid in our view but Wesnoth's playback (with MP=0)
            # would reject them as "corrupt movement".
            if cmd[0] == "move":
                tx, ty = cmd[1][-1], cmd[2][-1]
                self._apply_post_move_stops(
                    tx, ty, side_now, prev_village_owner)
            extras: dict = {}
            if cmd[0] == "recruit" and leader_pos is not None:
                extras["leader_pos"] = leader_pos
            # Post-attack advancement detection: an attacker/defender
            # whose name changed (or vanished from the unit set replaced
            # by a same-id but different-name advanced version) crossed
            # the XP threshold and was advanced. We pick `targets[0]` in
            # `_maybe_advance_unit`, so the choice index is always 0.
            # Order MUST match Wesnoth's attack_unit_and_advance:
            # attacker first, defender second.
            if cmd[0] == "attack" and (pre_att or pre_dfd):
                # Build id -> Unit map post-apply.
                by_id = {u.id: u for u in self.gs.map.units}
                advance_choices: List[Tuple[int, int]] = []  # (side, idx)
                for pre in (pre_att, pre_dfd):
                    if pre is None:
                        continue
                    pre_id, pre_name, pre_side = pre
                    post = by_id.get(pre_id)
                    if post is not None and post.name != pre_name:
                        advance_choices.append((pre_side, 0))
                if advance_choices:
                    extras["advance_choices"] = advance_choices
            self.command_history.append(RecordedCommand(
                kind=cmd[0], side=side_now, cmd=list(cmd), extras=extras))

        self._actions_by_side[side_now] = self._actions_by_side.get(side_now, 0) + 1
        # Post-step invariants. Run only under __debug__ so production
        # runs (`python -O`) skip the check; this catches divergences
        # at the source command rather than during Wesnoth playback or
        # in a corrupt training step several minutes later.
        if __debug__:
            self._assert_invariants(after_cmd=cmd[0])
        self._check_game_over()
        return self.done

    def run_game(
        self,
        policy_side1,
        policy_side2,
        game_label: str = "sim",
        record_trajectory: bool = False,
    ) -> SimResult:
        """Drive the game to completion with two policies. Useful for
        smoke tests; the trainer's outer loop typically calls .step
        directly so it can observe / shape rewards per step."""
        traj: List[SimStep] = []
        while not self.done:
            side = self.current_side
            policy = policy_side1 if side == 1 else policy_side2
            action = policy.select_action(self.gs, game_label=game_label)
            if record_trajectory:
                traj.append(SimStep(state=self.gs, action=action, side=side))
            self.step(action)
        return SimResult(
            winner=self.winner,
            turns=self.gs.global_info.turn_number,
            side1_actions=self._actions_by_side.get(1, 0),
            side2_actions=self._actions_by_side.get(2, 0),
            ended_by=self.ended_by,
            trajectory=traj,
        )

    # ----- internals -------------------------------------------------

    # Cover abilities -- units with one of these CAN be hidden in
    # the matching terrain/ToD, but only if not already revealed.
    # `_hide_cover_active` and the per-turn `_uncovered_units` set
    # together gate when ambush actually fires.
    _AMBUSH_ABILITIES = frozenset({
        "ambush", "nightstalk", "concealment", "submerge",
    })

    def _hide_cover_active(self, unit) -> bool:
        """True if `unit` has a hide ability AND its current hex
        terrain (or the ToD, for nightstalk) satisfies the ability's
        cover condition. Verified abilities and their covers from
        wesnoth_src/data/core/abilities.cfg:

          - ambush:      forest terrain
          - concealment: village terrain
          - submerge:    deep_water terrain
          - nightstalk:  current ToD has lawful_bonus < 0 (night /
                         second_watch)
        """
        from replay_dataset import _terrain_keys_at, _lawful_bonus_at
        abilities = unit.abilities or set()
        if not (abilities & self._AMBUSH_ABILITIES):
            return False
        keys = _terrain_keys_at(self.gs, unit.position.x, unit.position.y)
        if "ambush" in abilities and "forest" in keys:
            return True
        if "concealment" in abilities and "village" in keys:
            return True
        if "submerge" in abilities and "deep_water" in keys:
            return True
        if "nightstalk" in abilities:
            bonus = _lawful_bonus_at(
                self.gs, unit.position.x, unit.position.y,
                self.gs.global_info.turn_number,
            )
            if bonus < 0:
                return True
        return False

    def _refresh_uncovered_state(self, current_side: int) -> None:
        """Called at each side's init_side. Implements Wesnoth's
        `unit::new_turn` reset of STATE_UNCOVERED for the side's own
        units, plus the start-of-turn re-evaluation: a hidden unit
        that's already adjacent to ANY enemy at turn start is
        considered exposed for this turn -- even if the enemy moves
        away later, the unit doesn't re-hide until its OWN side's
        next init_side.

        This handles the user-flagged edge case: a hidden unit that
        starts adjacent to an enemy is revealed; if the enemy then
        moves away, the unit remains revealed.
        """
        from tools.abilities import hex_neighbors

        uncovered: set = getattr(
            self.gs.global_info, "_uncovered_units", None) or set()

        # Step 1: side-current units re-hide (STATE_UNCOVERED reset).
        for u in list(self.gs.map.units):
            if u.side == current_side and u.id in uncovered:
                uncovered.discard(u.id)

        # Step 2: for each hidden-ability unit on OTHER sides, check
        # if it's adjacent to any unit of `current_side`. If so, it's
        # already exposed (visible to current_side from the start).
        side_unit_positions = {
            (u.position.x, u.position.y)
            for u in self.gs.map.units
            if u.side == current_side
        }
        for u in self.gs.map.units:
            if u.side == current_side:
                continue
            if not self._hide_cover_active(u):
                continue
            for nx, ny in hex_neighbors(u.position.x, u.position.y):
                if (nx, ny) in side_unit_positions:
                    uncovered.add(u.id)
                    break

        setattr(self.gs.global_info, "_uncovered_units", uncovered)

    def _apply_post_move_stops(
        self,
        x: int,
        y: int,
        side: int,
        prev_village_owner: Optional[int] = None,
    ) -> None:
        """If the unit at (x, y) of `side` just moved into a hex that
        Wesnoth would zero its MP on, set current_moves = 0. Three
        independent triggers (move.cpp:1042-1054), each with the
        precise condition Wesnoth checks:

          - Ambush: an adjacent enemy has an ACTIVE hide ability
            (cover terrain/ToD matches the ability) AND that enemy
            has not yet been uncovered this turn. Skirmisher does
            NOT bypass ambush. After firing, the enemy becomes
            uncovered (won't ambush again until its own side's next
            init_side reset).
          - ZoC: an adjacent enemy of level >= 1 emits ZoC over (x,y)
            and the mover lacks `skirmisher`.
          - Village capture: entering a village whose previous owner
            was NOT this side. Friendly revisit doesn't zero MP.
            Pre-move owner is snapshotted in `step()` before
            `_apply_command` updates the village owner map.
        """
        from tools.abilities import hex_neighbors
        from replay_dataset import _stats_for
        from classes import TerrainModifiers

        unit = next(
            (u for u in self.gs.map.units
             if u.position.x == x and u.position.y == y and u.side == side),
            None,
        )
        if unit is None or unit.current_moves <= 0:
            return

        zero_mp = False

        # Ambush + ZoC: walk adjacent hexes once.
        uncovered: set = getattr(
            self.gs.global_info, "_uncovered_units", None) or set()
        is_skirmisher = "skirmisher" in (unit.abilities or set())
        for nx, ny in hex_neighbors(x, y):
            enemy = next(
                (u for u in self.gs.map.units
                 if u.position.x == nx and u.position.y == ny
                 and u.side != side),
                None,
            )
            if enemy is None:
                continue
            # Ambush: hide ability with matching cover, not already
            # uncovered. Bypasses skirmisher.
            if (enemy.id not in uncovered
                    and self._hide_cover_active(enemy)):
                zero_mp = True
                # Once an enemy ambushes, it's revealed for the rest
                # of the turn (won't ambush twice from the same
                # cover this turn).
                uncovered.add(enemy.id)
                setattr(self.gs.global_info, "_uncovered_units",
                        uncovered)
                break
            # ZoC: only level >= 1 enemies emit, only stops
            # non-skirmishers.
            if not is_skirmisher:
                level = int(_stats_for(enemy.name).get("level", 1))
                if level >= 1:
                    zero_mp = True
                    break

        # Village capture: only zero MP if the prior owner was NOT
        # this side. Friendly revisits leave MP alone (Wesnoth's
        # `if (orig_village_owner != current_side_)` gate).
        if not zero_mp and prev_village_owner is not None and prev_village_owner != side:
            hex_obj = next(
                (h for h in self.gs.map.hexes
                 if h.position.x == x and h.position.y == y),
                None,
            )
            if hex_obj is not None and TerrainModifiers.VILLAGE in hex_obj.modifiers:
                zero_mp = True

        if zero_mp:
            self._set_unit_mp(x, y, side, 0)

    def _set_unit_mp(self, x: int, y: int, side: int, new_mp: int) -> None:
        """Replace the unit at (x, y) of `side` with a copy whose
        `current_moves` is set to `new_mp`. Mirrors `_deduct_extra_mp`
        but with an absolute target value rather than a delta."""
        from classes import Unit
        new_units = set()
        for u in self.gs.map.units:
            if u.position.x == x and u.position.y == y and u.side == side:
                base = {k: v for k, v in u.__dict__.items()
                        if not k.startswith("_")}
                base["current_moves"] = max(0, new_mp)
                replacement = Unit(**base)
                for k, v in u.__dict__.items():
                    if k.startswith("_"):
                        setattr(replacement, k, v)
                new_units.add(replacement)
            else:
                new_units.add(u)
        self.gs.map.units = new_units

    def _deduct_extra_mp(self, x: int, y: int, extra: int) -> None:
        """Subtract `extra` more MP from the unit now at (x, y),
        clamped to >= 0. Used to top up _apply_command's flat-1
        deduction with the actual terrain cost - 1."""
        from classes import Unit
        new_units = set()
        for u in self.gs.map.units:
            if u.position.x == x and u.position.y == y:
                base = {k: v for k, v in u.__dict__.items()
                        if not k.startswith("_")}
                base["current_moves"] = max(0, u.current_moves - extra)
                replacement = Unit(**base)
                for k, v in u.__dict__.items():
                    if k.startswith("_"):
                        setattr(replacement, k, v)
                new_units.add(replacement)
            else:
                new_units.add(u)
        self.gs.map.units = new_units

    def _begin_side_turn(self, side: int) -> None:
        """Fire init_side(side). Replay-recon's _apply_command for
        init_side handles: setting current_side, incrementing turn
        (when side == 1), updating time-of-day, firing scenario
        turn-start events, and computing healing / poison / curing
        for `side`'s units. Game-over can also fire here (e.g. poison
        kills a leader)."""
        _apply_command(self.gs, ["init_side", side])
        self.command_history.append(RecordedCommand(
            kind="init_side", side=side, cmd=["init_side", side]))
        # Refresh hidden/uncovered tracking: own-side units re-hide,
        # other-side hidden units adjacent to our units become exposed.
        # Must run AFTER _apply_command (turn number / ToD updated) so
        # nightstalk's lawful_bonus check sees the right ToD.
        self._refresh_uncovered_state(side)
        self._check_game_over()

    def _assert_invariants(self, *, after_cmd: str) -> None:
        """Cheap structural sanity check on the unit set. Catches
        common bugs early -- rather than seeing them later as Wesnoth
        OOSes or corrupt rewards. Each check below has actually
        triggered during development of the sim.

        Invariants:

          (a) `current_hp` in [0, max_hp]. HP > max_hp means an effect
              (drain, healing) overshot; HP < 0 means we forgot to
              clamp a damage roll.

          (b) `current_moves` in [0, max_moves]. MP > max_moves means
              `_deduct_extra_mp` over-credited or `_begin_side_turn`
              re-applied the reset twice; MP < 0 means we deducted
              past zero somewhere.

          (c) No two units on the same hex (Wesnoth never allows
              stacking, and `_apply_command` for "move" doesn't check
              -- the gate is in `_action_to_command`).

          (d) Per side, AT MOST one leader. Two leaders on one side is
              recoverable via `_check_game_over`'s heuristic but
              indicates a recruit/recall logic bug.

        Raises AssertionError with enough context to debug.
        """
        seen_hexes: Dict[Tuple[int, int], str] = {}
        leaders_per_side: Dict[int, List[str]] = {}
        for u in self.gs.map.units:
            # (a) HP bounds.
            if u.current_hp < 0 or u.current_hp > u.max_hp:
                raise AssertionError(
                    f"sim invariant: unit {u.id} ({u.name!r}) HP out of "
                    f"range: current_hp={u.current_hp}, max_hp={u.max_hp} "
                    f"(after cmd={after_cmd!r}, turn={self.gs.global_info.turn_number})")
            # (b) MP bounds.
            if u.current_moves < 0 or u.current_moves > u.max_moves:
                raise AssertionError(
                    f"sim invariant: unit {u.id} ({u.name!r}) MP out of "
                    f"range: current_moves={u.current_moves}, "
                    f"max_moves={u.max_moves} "
                    f"(after cmd={after_cmd!r}, turn={self.gs.global_info.turn_number})")
            # (c) No hex stacking.
            key = (u.position.x, u.position.y)
            if key in seen_hexes:
                raise AssertionError(
                    f"sim invariant: hex {key} occupied by both "
                    f"{seen_hexes[key]!r} and {u.id!r} "
                    f"(after cmd={after_cmd!r}, turn={self.gs.global_info.turn_number})")
            seen_hexes[key] = u.id
            # (d) Leader count per side.
            if u.is_leader:
                leaders_per_side.setdefault(u.side, []).append(u.id)
        for side, leaders in leaders_per_side.items():
            if len(leaders) > 1:
                raise AssertionError(
                    f"sim invariant: side {side} has {len(leaders)} "
                    f"leaders ({leaders!r}); at most one allowed "
                    f"(after cmd={after_cmd!r}, turn={self.gs.global_info.turn_number})")

    def _check_game_over(self) -> None:
        if self.done:
            return
        # Leader-alive heuristic: a side is alive iff it has at least
        # one canrecruit (leader) unit on the map.
        sides_alive = {u.side for u in self.gs.map.units if u.is_leader}
        if 1 in sides_alive and 2 in sides_alive:
            # Both leaders alive -- check turn / action limits.
            if self.gs.global_info.turn_number > self.max_turns:
                self.done = True
                self.winner = 0
                self.ended_by = "max_turns"
                return
            if any(c >= self.max_actions_per_side
                   for c in self._actions_by_side.values()):
                self.done = True
                self.winner = 0
                self.ended_by = "max_actions"
                return
            return
        # At least one leader missing -- game over.
        self.done = True
        self.ended_by = "leader_killed"
        if 1 in sides_alive:
            self.winner = 1
        elif 2 in sides_alive:
            self.winner = 2
        else:
            self.winner = 0   # mutual elimination

    def _action_to_command(self, action: dict) -> Optional[list]:
        """Translate the policy's action dict into the command list
        format _apply_command consumes. Returns None for malformed
        actions; the caller treats that as an end-turn fallback."""
        # Lazy import to avoid a circular dep at module-load time.
        from tools.abilities import hex_neighbors

        atype = action.get("type")
        if atype == "end_turn":
            return ["end_turn"]
        if atype == "move":
            start: Position = action["start_hex"]
            target: Position = action["target_hex"]
            # Validate the move is legal: target must be a hex
            # neighbor of start, the source must hold one of OUR
            # units, and the unit must have enough MP to ENTER the
            # target's terrain. Wesnoth's replay engine re-checks
            # this on playback (plot_turn in move.cpp) and rejects
            # any violation as 'corrupt movement', so we have to
            # enforce it here too -- the underlying _apply_command
            # is permissive (flat 1 MP per step, no terrain check)
            # and would otherwise accept teleports / 0-MP moves /
            # over-cost moves.
            if (target.x, target.y) not in hex_neighbors(start.x, start.y):
                log.debug(f"sim: rejecting non-adjacent move "
                          f"{(start.x, start.y)}->{(target.x, target.y)}")
                return None
            mover = next(
                (u for u in self.gs.map.units
                 if u.position.x == start.x and u.position.y == start.y
                 and u.side == self.current_side),
                None,
            )
            if mover is None:
                return None
            # Target hex must be unoccupied (Wesnoth never allows
            # stacking; the playback engine would assert).
            if any(u.position.x == target.x and u.position.y == target.y
                   for u in self.gs.map.units):
                return None
            cost = _move_cost_at_hex(mover, self.gs, target.x, target.y)
            if cost >= 99 or mover.current_moves < cost:
                return None
            # Stash for step() to do the post-apply MP correction
            # (since _apply_command flat-deducts 1, we owe an extra
            # cost-1 on top).
            action["_terrain_cost"] = cost
            # _apply_command's move handler reads xs[0], ys[0] (start)
            # and xs[-1], ys[-1] (target). The intermediate path is
            # only used to fire enter_hex events, which our scenarios
            # don't depend on. Pass [start, target] -- minimal valid
            # path -- so the unit lands at the target hex.
            return ["move",
                    [start.x, target.x],
                    [start.y, target.y],
                    self.current_side]   # from_side filter
        if atype == "attack":
            start: Position = action["start_hex"]
            target: Position = action["target_hex"]
            weapon = int(action.get("attack_index", 0))
            # Allocate a synced-RNG seed so combat damage rolls match
            # what Wesnoth replays back from the [random_seed]
            # follow-up command sim_to_replay emits. cmd[7] is the
            # seed slot consumed by replay_dataset._apply_command.
            self._rng_requests += 1
            seed = request_seed(self._rng_requests)
            # d_weapon=-1 means "let combat.py pick the defender's
            # best weapon", same as Wesnoth's auto-selection.
            return ["attack",
                    start.x, start.y,
                    target.x, target.y,
                    weapon, -1, seed]
        if atype == "recruit":
            unit_type = action["unit_type"]
            target: Position = action["target_hex"]
            # Validate the recruit BEFORE letting it through. Wesnoth
            # playback re-checks each [recruit] command against the
            # current side's gold and the keep-castle network; if the
            # side can't afford the unit or the target hex isn't part
            # of the leader's castle, playback errors with "cannot
            # recruit unit: ...". Our `_apply_command` is permissive --
            # it deducts cost and clamps gold to 0, then accepts the
            # recruit -- so without this gate the sim emits illegal
            # recruits that Wesnoth rejects.
            cost = _recruit_cost_for(unit_type)
            side_idx = self.current_side - 1
            if 0 <= side_idx < len(self.gs.sides):
                gold = int(self.gs.sides[side_idx].current_gold)
                if gold < cost:
                    log.debug(
                        f"sim: refusing recruit {unit_type!r} (cost={cost} "
                        f"> gold={gold})")
                    return None
            # Allocate a synced-RNG seed for the trait roll. Without
            # this both sides diverge: the sim might give the recruit
            # `quick` (+1 MP) while Wesnoth's playback rolls a
            # different trait, leading to "corrupt movement" the
            # first time the unit's MP differs between the two views.
            # cmd[4] is the trait_seed slot consumed by
            # _build_recruit_unit's MTRng path.
            self._rng_requests += 1
            seed = request_seed(self._rng_requests)
            return ["recruit", unit_type, target.x, target.y, seed]
        if atype == "recall":
            # Recall is NOT supported end-to-end. The sim has no
            # recall list (gs.global_info doesn't track per-side
            # surviving units), `_apply_command` for "recall" is a
            # no-op (replay_dataset.py:1341-1343), and
            # sim_to_replay._wml_for_command WOULD emit a [recall]
            # block citing a unit_id that doesn't exist on Wesnoth's
            # recall list -- playback then errors with "no such unit
            # on recall list".
            #
            # action_sampler today never emits recall actions (no
            # recall slots in the actor head), so this branch is
            # dormant. Returning None here makes step() fall back to
            # end_turn -- equivalent to the old behavior in terms of
            # game progress, but with the broken [recall] WML
            # suppressed if a future caller does produce one. The
            # alternative (proper recall support) needs:
            #   1. recall list in gs.global_info, populated on
            #      level-up of dying-side units;
            #   2. recall slots in encoder.recruit_tokens;
            #   3. the action_sampler, action_executor, and exporter
            #      all wired to the new slots.
            log.warning(
                f"sim: recall not supported end-to-end; treating as "
                f"end_turn (action={action!r})")
            return None
        log.debug(f"sim: unknown action type {atype!r}")
        return None


# ---------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------

def _smoke(argv: List[str]) -> int:
    """`python tools/wesnoth_sim.py REPLAY.json.gz`
    runs one game with the dummy policy on both sides and prints
    the outcome. Useful for sanity-checking the simulator without
    a trained model."""
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("replay", type=Path,
                    help=".json.gz replay to take initial state from")
    ap.add_argument("--max-turns", type=int, default=40)
    args = ap.parse_args(argv[1:])

    from dummy_policy import DummyPolicy
    sim = WesnothSim.from_replay(args.replay, max_turns=args.max_turns)
    print(f"scenario:   {sim.scenario_id}")
    print(f"factions:   "
          f"side1={sim.gs.sides[0].faction if sim.gs.sides else '?'} "
          f"side2={sim.gs.sides[1].faction if len(sim.gs.sides) > 1 else '?'}")
    print(f"map size:   {sim.gs.map.size_x}x{sim.gs.map.size_y}")
    print(f"start units: {len(sim.gs.map.units)}")
    print()
    pa, pb = DummyPolicy(), DummyPolicy()
    res = sim.run_game(pa, pb)
    print(f"winner:     {res.winner}  (0=draw)")
    print(f"ended by:   {res.ended_by}")
    print(f"turns:      {res.turns}")
    print(f"actions:    s1={res.side1_actions}  s2={res.side2_actions}")
    return 0


if __name__ == "__main__":
    sys.exit(_smoke(sys.argv))
