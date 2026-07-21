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

    # Rescale the ALREADY-CONSTRUCTED starting units' xp-to-advance to
    # the target modifier. _build_initial_gamestate baked each starting
    # unit's max_exp using the SOURCE replay's experience_modifier (often
    # a custom host value), but recruits made later use this PvP default
    # -- so without this rescale a leader/pre-placed unit would advance
    # on a different xp threshold than its own recruits in the same game.
    # Idempotent: recomputing at the same modifier yields the same value.
    import dataclasses
    from replay_dataset import _stats_for, _scaled_max_exp
    target_mod = int(defaults.experience_modifier)
    rescaled = set()
    for u in gs.map.units:
        base_exp = int(_stats_for(u.name).get("experience", 50))
        rescaled.add(dataclasses.replace(
            u, max_exp=_scaled_max_exp(base_exp, target_mod)))
    gs.map.units = rescaled


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

# Memoized terrain->MP resolution. `terrain_resolver.mvt_cost` walks the
# terrain alias graph recursively from scratch on every call (its own
# lru_cache import was never wired up), and `_move_cost_at_hex` is called
# per frontier hex inside `_find_attack_hex`'s Dijkstra -- millions of
# times over a training run. The result is a pure function of
# (stripped_terrain_code, unit_type, slowed), so cache it. Keyed on
# unit.name (not movetype) to match `_MOVETYPE_COSTS_CACHE`'s granularity.
_MVT_RESOLVE_CACHE: Dict[Tuple[str, str, bool], int] = {}


_UNIT_STATS_DATA: Optional[dict] = None


def _unit_stats_data() -> dict:
    """Parse `unit_stats.json` ONCE per process and cache it. Both
    `_movetype_costs` and `_recruit_cost_for` index into this instead of
    re-reading + re-parsing the ~400KB file on every cold cache key
    (their per-key caches still avoid re-indexing once warm)."""
    global _UNIT_STATS_DATA
    if _UNIT_STATS_DATA is None:
        import json
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "unit_stats.json"
        with path.open(encoding="utf-8") as f:
            _UNIT_STATS_DATA = json.load(f)
    return _UNIT_STATS_DATA


def _movetype_costs(unit_type: str, slowed: bool = False) -> dict:
    """Look up `{terrain_key: int_cost}` for a unit type. Returns an
    empty dict if not found (caller falls back to cost=1).

    Prefers the unit-type's per-unit `movement_costs` (which the
    scraper now emits as a fully merged table layering any
    `[movement_costs]` overrides on top of the movetype's defaults).
    Falls back to the raw movement_type table for older
    `unit_stats.json` files predating that scraper change.

    When `slowed=True`, every cost is doubled, EXCEPT UNREACHABLE
    sentinels (>=99 stay >=99). Mirrors movetype.hpp:69-72:
        return slowed && result != UNREACHABLE ? 2 * result : result;
    """
    cache_key = (unit_type, slowed)
    if cache_key in _MOVETYPE_COSTS_CACHE:
        return _MOVETYPE_COSTS_CACHE[cache_key]
    try:
        data = _unit_stats_data()
        units = data.get("units", {})
        movetypes = data.get("movement_types", {})
        u = units.get(unit_type, {})
        # Per-unit merged costs take precedence; fall back to the
        # movement_type's table for backward compat.
        costs = u.get("movement_costs")
        if not costs:
            mt = u.get("movement_type")
            costs = (movetypes.get(mt, {}).get("movement_costs", {})
                     if mt else {})
        if slowed:
            costs = {k: (v if v >= 99 else 2 * v) for k, v in costs.items()}
        _MOVETYPE_COSTS_CACHE[cache_key] = costs
        return costs
    except Exception as e:
        log.debug(f"sim: movetype lookup failed for {unit_type!r}: {e}")
        _MOVETYPE_COSTS_CACHE[cache_key] = {}
        return {}


def _move_cost(unit, terrain_key: str) -> int:
    """How many MP does `unit` need to enter a hex of terrain
    `terrain_key`? Falls back to 1 if data is missing (matches the
    sim's pre-existing flat-cost behavior). Honors the `slowed`
    status (doubles every cost except UNREACHABLE)."""
    slowed = "slowed" in (getattr(unit, "statuses", set()) or set())
    costs = _movetype_costs(unit.name, slowed=slowed)
    cost = costs.get(terrain_key, 1)
    # Wesnoth uses 99 as the impassable sentinel; anything >= 99 is
    # effectively unenterable.
    return int(cost) if cost is not None else 1


def _describe_action(action: dict) -> str:
    """Compact human-readable form of a policy action dict for the
    replay-comment instrumentation (attempted-action [speak] lines)."""
    t = action.get("type", "?")
    if t == "move":
        s, d = action.get("start_hex"), action.get("target_hex")
        return (f"move ({getattr(s,'x','?')},{getattr(s,'y','?')})->"
                f"({getattr(d,'x','?')},{getattr(d,'y','?')})")
    if t == "attack":
        s, d = action.get("start_hex"), action.get("target_hex")
        return (f"attack ({getattr(s,'x','?')},{getattr(s,'y','?')})->"
                f"({getattr(d,'x','?')},{getattr(d,'y','?')}) "
                f"w={action.get('attack_index', 0)}")
    if t == "recruit":
        d = action.get("target_hex")
        return (f"recruit {action.get('unit_type','?')}@"
                f"({getattr(d,'x','?')},{getattr(d,'y','?')})")
    return str(t)


def _move_cost_at_hex(unit, gs, x: int, y: int) -> int:
    """Resolve the movement cost for `unit` entering hex (x, y).

    Delegates to `tools.terrain_resolver.mvt_cost`, which scrapes
    `wesnoth_src/data/core/terrain.cfg` into `terrain_db.json` and
    walks the alias graph exactly as Wesnoth does (see
    `wesnoth_src/src/movetype.cpp:276-369` for `calc_value` and
    `wesnoth_src/src/terrain/terrain.cpp:208-244` + 334-377 for
    composite construction). Replaces the prior hand-rolled
    `_OVERLAY_MOVEMENT_OVERRIDE` / `_OVERLAY_MOVEMENT_MAX` tables,
    which covered ~30 overlays out of 137+ and silently mispriced
    every uncovered case.

    See `docs/wesnoth_rules.md` ("Movement / mvt_alias resolution")
    for the rule statement and source quotes.
    """
    from tools.terrain_resolver import mvt_cost as _resolve_mvt
    codes = getattr(gs.global_info, "_terrain_codes", {}) or {}
    code = codes.get((x, y))
    # Honor "slowed" status: doubles each terrain cost (except
    # UNREACHABLE). movetype.hpp:69-72.
    slowed = "slowed" in (getattr(unit, "statuses", set()) or set())
    if not code:
        # No raw terrain code recorded for this hex (synthetic
        # tests, partial state). Fall through to the defense-keys
        # path which gives us a semantically-decent flat fallback.
        from replay_dataset import _terrain_keys_at
        keys = _terrain_keys_at(gs, x, y) or ["flat"]
        costs = _movetype_costs(unit.name, slowed=slowed)
        per_key = [int(costs.get(k, 1) or 1) for k in keys]
        return min(per_key) if per_key else 99
    # Strip "1 ", "2 " starting-position markers ("2 Ke" -> "Ke")
    # before resolving; the marker is a placement hint, not part of
    # the terrain code.
    c = code
    if c[:1].isdigit() and c[1:2] == " ":
        c = c[2:]
    cache_key = (c, unit.name, slowed)
    cached = _MVT_RESOLVE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    val = _resolve_mvt(c, _movetype_costs(unit.name, slowed=slowed))
    _MVT_RESOLVE_CACHE[cache_key] = val
    return val


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
        data = _unit_stats_data()
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
    ended_by:      str          # 'leader_killed' | 'max_turns' | 'max_actions' | 'no_progress' | 'no_legal'
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
        apply_scenario_events: bool = True,
        begin_side: int = 1,
        no_progress_turns: int = 0,
    ):
        self.gs = initial_state
        self.scenario_id = scenario_id
        self.max_turns = max_turns
        self.max_actions_per_side = max_actions_per_side
        # No-progress stalemate rule (2026-07-21, chess 50-move-rule
        # analog): a "progress event" is any objective state change --
        # unit count changed (kill/recruit), net HP dropped (combat/
        # poison damage), or village ownership changed. After
        # `no_progress_turns` consecutive FULL turns without one, the
        # game ends as a draw (ended_by='no_progress'). 0 = rule OFF;
        # the TRACKER always runs (cheap fingerprint per step) so
        # observe-mode data collects would-fire statistics.
        self.no_progress_turns = int(no_progress_turns)
        self._last_progress_turn = initial_state.global_info.turn_number
        self._max_quiet = 0
        self._quiet_resumed: list = []   # lengths of quiet streaks >=3
        #                                  that ended with real progress

        # Wire scenario-specific events (time_area, store_locations,
        # Aethermaw morph, etc.) -- mirrors what replay_dataset does
        # at the top of iter_replay_pairs. Mid-game starts pass
        # False: reconstruction already fired them, and prestart
        # unit placement (CoB statues) must not double-apply.
        if apply_scenario_events:
            _setup_scenario_events(self.gs, scenario_id)

        self.done:      bool = False
        self.winner:    int  = 0
        self.ended_by:  str  = ""
        self._actions_by_side: Dict[int, int] = {1: 0, 2: 0}

        # Starting leaders, snapshotted BEFORE any play: replay export
        # renders [side] blocks from the STARTING setup, and reading
        # leaders from the final state breaks on every decisive game
        # (the loser's leader is dead — observed 2026-07-03, the first
        # time a decisive game was exported) and mis-types leaders
        # that advanced mid-game. {side: (unit_name, Position)}.
        self.initial_leaders: Dict[int, tuple] = {
            u.side: (u.name, u.position)
            for u in self.gs.map.units if u.is_leader
        }

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

        # Search-only seed salt. Empty on every LIVE game sim --
        # seeds then come from request_seed(counter), the bit-exact
        # replay-export contract. MCTS chance-node sampling sets a
        # fresh salt on each search fork before stepping a stochastic
        # action, so repeated forks of the same parent sample
        # DIFFERENT combat outcomes / trait rolls instead of
        # replaying one predetermined seed. Never set this on a sim
        # whose command_history will be exported.
        self._seed_salt: str = ""

        # last_step_rejected: did the most recent .step() call refuse
        # to apply the action (rather than apply or fall back to
        # end_turn)? Currently only set on recruit-rejection (target
        # hex was god-view-occupied; the harness should re-decide
        # rather than waste the turn). Callers that loop on
        # rejection check this flag; existing callers that ignore
        # it see a `step()` that's a silent no-op for the rejected
        # decision -- they'd typically just call step() again,
        # which is exactly the right behavior.
        self.last_step_rejected: bool = False
        # Consecutive rejected-step counter for the mask-less-caller
        # loop guard (see step()). Reset whenever a command applies.
        self._consecutive_rejects: int = 0

        # Turn 0 is pre-game. The first init_side(1) bumps to turn 1
        # AND fires turn-1 events / healing. Mirror that here.
        # Mid-game starts resume at the side whose init_side the cut
        # landed on (adversarial review 2026-07-12 C1: hardcoding side
        # 1 skipped side 2's turn and double-turned side 1 -- free
        # income+healing tempo bias in every continuation).
        self._begin_side_turn(begin_side)

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
        out.no_progress_turns = self.no_progress_turns
        out._last_progress_turn = self._last_progress_turn
        out._max_quiet = 0
        out._quiet_resumed = []
        out.done     = self.done
        out.winner   = self.winner
        out.ended_by = self.ended_by
        out._actions_by_side = dict(self._actions_by_side)
        out._rng_requests    = self._rng_requests
        out._seed_salt       = self._seed_salt
        out.command_history  = []   # forks don't track history
        return out

    def enable_engagement_stats(self):
        """Attach a per-game EngagementStats accumulator to THIS sim.
        fork() never carries it, so MCTS search forks stay
        instrumentation-free (zero cost in rollouts)."""
        from tools.engagement_stats import EngagementStats
        self._engagement = EngagementStats()
        return self._engagement

    def _apply_with_stats(self, cmd) -> None:
        """_apply_command with the thread-local engagement event sink
        installed for the duration (combat + heal events from
        replay_dataset). Plain _apply_command when stats are off."""
        es = getattr(self, "_engagement", None)
        if es is None:
            _apply_command(self.gs, cmd)
            return
        from tools.engagement_stats import (clear_event_sink,
                                            set_event_sink)
        set_event_sink(es.on_event)
        try:
            _apply_command(self.gs, cmd)
        finally:
            clear_event_sink()

    def apply_neutral_attack(self, action: dict) -> bool:
        """Execute one pre-validated NEUTRAL-side attack (side >= 3
        RCA turn, tools/neutral_ai.py). Mirrors step()'s attack-apply
        bookkeeping (pre snapshots, advancement choices, checkup
        strikes, history) WITHOUT step()'s end-turn fallback: a
        neutral attack must never rotate the player turn order.
        Returns False if the action didn't translate to an attack
        command (caller aborts its loop)."""
        cmd, _cost = self._action_to_command(action)
        if cmd is None or cmd[0] != "attack":
            log.warning(f"neutral attack failed to translate: {action!r}")
            return False
        side_now = self.gs.global_info.current_side
        self._apply_with_stats(cmd)
        extras: dict = {}
        # Advancement [choose] events straight from the applier's
        # side-channel (one per advancement step, AMLA and chain
        # links included; attacker-first order preserved). The old
        # name-change diff missed AMLAs and double-advances
        # (validation pipeline catch, 2026-07-15).
        advance_choices = list(getattr(
            self.gs.global_info, "_last_advance_events", []) or [])
        if advance_choices:
            setattr(self.gs.global_info, "_last_advance_events", [])
            extras["advance_choices"] = advance_choices
            es = getattr(self, "_engagement", None)
            if es is not None:
                for _adv_side, _ in advance_choices:
                    if _adv_side in (1, 2):
                        es.advancements[_adv_side] += 1
        strikes = getattr(self.gs.global_info,
                          "_last_checkup_strikes", None)
        if strikes:
            extras["checkup_strikes"] = strikes
            setattr(self.gs.global_info, "_last_checkup_strikes", None)
        self.command_history.append(RecordedCommand(
            kind="attack", side=side_now, cmd=list(cmd),
            extras=extras))
        if __debug__:
            self._assert_invariants(after_cmd="attack")
        self._check_game_over()
        return True

    def _next_seed(self) -> str:
        """Allocate the next synced-RNG seed. Live sims (no salt)
        derive it purely from the request counter -- the bit-exact
        contract sim_to_replay's WML emitter relies on. Salted sims
        (MCTS search forks) mix the salt in so identical counters
        yield independent rolls across forks."""
        self._rng_requests += 1
        if not self._seed_salt:
            return request_seed(self._rng_requests)
        import hashlib
        h = hashlib.sha256(
            f"{self._seed_salt}:{self._rng_requests}".encode()).hexdigest()
        return h[:8]

    def _find_attack_hex(self, attacker, target) -> Optional[Position]:
        """Pick a hex the attacker can move to and attack `target` from.

        Returns the chosen attack hex (a neighbor of `target` the
        attacker can LAND on this turn, per the shared
        observable-state planner), or None if no such hex exists.

        Why this helper: the policy emits attack actions with
        `start_hex = unit's current position`. When `start_hex` isn't
        adjacent to `target`, Wesnoth's replay engine rejects the
        bare `[attack]` command (battle_context disables out-of-range
        weapons; the attack handler then returns without consuming the
        [random_seed] follow-up, and the next outer-loop iteration
        errors with "found dependent command in replay while
        is_synced=false"). So before recording an [attack], we have to
        emit an explicit [move] putting the attacker on a neighbor of
        the target hex -- mirroring what Wesnoth's UI does when the
        player clicks an enemy from a non-adjacent unit.

        Knowledge level: the plan runs on the ACTING SIDE'S
        OBSERVABLE state (tools/pathfind_sim.ReachContext), exactly
        like a player's own attack order -- hidden units neither
        block nor ZoC the approach; they resolve during the nested
        move's execution walk (blocked/ambush truncation aborts the
        attack, as in Wesnoth).

        Choice among candidate hexes: minimal Wesnoth float cost
        (MP + defense/ally subcosts, pathfind.cpp:815-820), i.e. the
        cheapest approach preferring defensible terrain -- the same
        preference order Wesnoth's own pathfinder applies.
        """
        from tools.abilities import hex_neighbors
        from tools.pathfind_sim import ReachContext, unit_reach

        target_neighbors = set(hex_neighbors(target.x, target.y))
        ctx = ReachContext.for_side(
            self.gs, attacker.side, exclude_unit=attacker)
        reach = unit_reach(attacker, self.gs, ctx)
        # The attacker's own hex is a free "landing" if already
        # adjacent (defensive: callers normally handle that case).
        if reach.start in target_neighbors:
            return Position(x=reach.start[0], y=reach.start[1])
        candidates = [
            pos for pos in reach.landable if pos in target_neighbors
        ]
        if not candidates:
            return None
        best = min(candidates, key=lambda pos: reach.cost[pos])
        return Position(x=best[0], y=best[1])


    def step(self, action: dict) -> bool:
        """Apply one action. Returns True if the game is over after
        this step. Wraps `_step_inner` with the no-progress tracker:
        a cheap state fingerprint (unit count, total HP, village
        ownership) taken before/after detects objective progress;
        full quiet turns are counted at each turn increment and --
        when `no_progress_turns` > 0 -- end the game as a stalemate
        draw. Nested step() calls (move-to-attack pre-moves) run the
        wrapper too; that is harmless (each level sees its own
        delta)."""
        if self.done:
            return True
        gi = self.gs.global_info
        t0 = gi.turn_number
        n0 = len(self.gs.map.units)
        hp0 = sum(u.current_hp for u in self.gs.map.units)
        vo0 = dict(getattr(gi, "_village_owner", None) or {})
        over = self._step_inner(action)
        gi = self.gs.global_info
        progressed = (
            len(self.gs.map.units) != n0
            or sum(u.current_hp for u in self.gs.map.units) < hp0
            or dict(getattr(gi, "_village_owner", None) or {}) != vo0)
        if progressed:
            quiet = max(0, gi.turn_number - self._last_progress_turn - 1)
            if quiet >= 3:
                self._quiet_resumed.append(quiet)
            self._last_progress_turn = gi.turn_number
        elif gi.turn_number > t0:
            # A turn boundary passed without progress this step.
            quiet = max(0, gi.turn_number - self._last_progress_turn - 1)
            self._max_quiet = max(self._max_quiet, quiet)
            if (self.no_progress_turns > 0 and not self.done
                    and quiet >= self.no_progress_turns):
                self.done = True
                self.winner = 0
                self.ended_by = "no_progress"
                return True
        return over

    def noprogress_summary(self) -> dict:
        """Per-game tracker readout for telemetry / offline
        evaluation of candidate K values. `tail_quiet` is the quiet
        streak the game ENDED on (never resumed)."""
        tail = max(0, self.gs.global_info.turn_number
                   - self._last_progress_turn - 1)
        return {
            "max_quiet": max(self._max_quiet, tail),
            "tail_quiet": tail,
            "resumed_streaks": list(self._quiet_resumed),
        }

    def _step_inner(self, action: dict) -> bool:
        if self.done:
            return True

        _es = getattr(self, "_engagement", None)
        if _es is not None and action.get("type") == "attack":
            # Count the POLICY-chosen attack (before move-to-attack
            # normalization; nested pre-move step()s are moves and
            # can't double count).
            _es.note_attack_attempt(self.gs, action)

        # Implicit move-to-attack: if the policy picked an attack on a
        # target that the actor isn't adjacent to, plan a pre-move to a
        # reachable attack hex and dispatch it via a nested step()
        # call. Then continue with the (now-adjacent) attack. The
        # policy's mask permits attacks within `current_moves + 1`
        # hexes of an enemy (the +1 lets the actor cover one MP
        # then strike), so this branch fires only on actions the mask
        # already approved.
        if action.get("type") == "attack":
            from tools.abilities import hex_neighbors
            start = action.get("start_hex")
            target = action.get("target_hex")
            if (start is not None and target is not None
                and (target.x, target.y) not in hex_neighbors(start.x, start.y)):
                # Locate the attacker and target units. If either is
                # missing, the attack is illegal anyway -- fall through
                # to _action_to_command to handle the rejection.
                attacker = next(
                    (u for u in self.gs.map.units
                     if u.position.x == start.x and u.position.y == start.y
                     and u.side == self.current_side), None)
                defender = next(
                    (u for u in self.gs.map.units
                     if u.position.x == target.x and u.position.y == target.y),
                    None)
                if attacker is None or defender is None:
                    # Stale action -- units moved / died. Drop it; the
                    # outer loop will pick a fresh action next call.
                    self._forced_end_turn_note = (
                        f"attempted {_describe_action(action)}; "
                        f"stale (unit moved/died) -> end_turn")
                    action = {"type": "end_turn"}
                else:
                    attack_hex = self._find_attack_hex(attacker, target)
                    if attack_hex is None:
                        # No landable attack hex. The mask computes
                        # attackability from the SAME observable-state
                        # reach, so this is a mask/sim disagreement --
                        # loud, and re-decide rather than burn the
                        # turn.
                        log.warning(
                            f"sim: attack {start.x},{start.y}->"
                            f"{target.x},{target.y} has no landable "
                            f"attack hex; mask/sim reachability "
                            f"disagreement -- re-deciding")
                        self.last_step_rejected = True
                        return self.done
                    else:
                        # Dispatch the move first; if it produces a
                        # game-over (e.g. capture-the-flag scenario),
                        # propagate.
                        if self.step({
                            "type": "move",
                            "start_hex": start,
                            "target_hex": attack_hex,
                        }):
                            return True
                        # Verify the move actually landed: ZoC / ambush
                        # / village-capture stops can zero MP without
                        # changing position, but the position update
                        # always succeeds when _apply_command accepts
                        # the move. If for any reason the attacker
                        # didn't land on attack_hex, abort: the next
                        # action_sampler call will pick a fresh action.
                        moved = next(
                            (u for u in self.gs.map.units
                             if u.id == attacker.id
                             and u.position.x == attack_hex.x
                             and u.position.y == attack_hex.y), None)
                        if moved is None:
                            return self.done
                        # Update the action to attack from the new hex.
                        action = {
                            **action,
                            "start_hex": attack_hex,
                        }

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

        # Reset the per-step rejection signal. The harness consults
        # this AFTER step() to decide whether to re-decide rather than
        # advance.
        self.last_step_rejected = False

        cmd, terrain_cost = self._action_to_command(action)
        if cmd is not None and cmd[0] in ("__retry_recruit__",
                                          "__reject_action__"):
            # Recruit hex god-view-occupied, or a move target outside
            # the planner's landable set. Normally: bail out without
            # applying anything and let the caller re-decide
            # (`last_step_rejected`). BUT a policy that does NOT
            # consult the legality mask (scripted DummyPolicy, buggy
            # caller) can deterministically re-emit the same doomed
            # action forever -- caught 2026-07-17 as an infinite
            # no-op loop in the validation suite. Bound it: after
            # _MAX_CONSECUTIVE_REJECTS rejected steps with no applied
            # command in between, degrade to end_turn (the pre-2026-07
            # behavior) with a loud warning. Mask-consulting policies
            # never accumulate rejects (the mask and the planner
            # agree), so the bound only fires for mask-less callers.
            self._consecutive_rejects = getattr(
                self, "_consecutive_rejects", 0) + 1
            if self._consecutive_rejects < self._MAX_CONSECUTIVE_REJECTS:
                self.last_step_rejected = True
                return self.done
            log.warning(
                f"sim: {self._consecutive_rejects} consecutive "
                f"rejected steps (last: {action!r}); caller is not "
                f"re-deciding from the mask -- ending turn to "
                f"guarantee progress")
            self._forced_end_turn_note = (
                f"attempted {_describe_action(action)}; "
                f"{self._consecutive_rejects} consecutive rejects "
                f"-> forced end_turn (loop guard)")
            cmd = ["end_turn"]
            terrain_cost = None
        self._consecutive_rejects = 0
        if cmd is None:
            # Action was illegal-shaped or referred to a missing unit.
            # Treat as a wasted turn -- end the side's turn.
            log.debug(f"sim: untranslatable action {action!r}; ending turn")
            self._forced_end_turn_note = (
                f"attempted {_describe_action(action)}; "
                f"untranslatable -> end_turn")
            cmd = ["end_turn"]
            terrain_cost = None

        if cmd[0] == "end_turn":
            if _es is not None:
                # "Right before end of turn": hexes revealed during
                # the turn are still visible here and may re-hide
                # afterwards (user spec 2026-07-12).
                _es.note_end_turn(self.gs, side_now)
            _apply_command(self.gs, ["end_turn"])
            _et_extras: dict = {}
            _note = getattr(self, "_forced_end_turn_note", None)
            if _note is not None and action.get("type") != "end_turn":
                # SIM-FORCED end_turn: the policy attempted something
                # else. Policy-chosen end_turns carry no note --
                # absence of the annotation IS the provenance signal.
                _et_extras["attempted"] = _note
            self._forced_end_turn_note = None
            self.command_history.append(RecordedCommand(
                kind="end_turn", side=side_now, cmd=["end_turn"],
                extras=_et_extras))
            # Advance to the next side. 2p only for now.
            n_sides = max(2, len(self.gs.sides))
            next_side = (side_now % n_sides) + 1
            # NB reward attribution (review 2026-07-14 M4): the
            # neutral turn's effects land inside side 2's end_turn
            # step, so REINFORCE-path compute_delta credits side 2
            # for tentacle damage to side 1. Production training is
            # MCTS (per-step shaping is a no-op there); fix the
            # delta split before any REINFORCE run on tentacle maps.
            # Neutral side-3 turn (Mini_Maps tentacles, 2026-07-14):
            # Wesnoth's side order is 1, 2, 3 within a turn, so the
            # RCA combat turn for armed side-3 units runs after side
            # 2 ends, before init_side(1) increments the turn. Runs
            # in MCTS forks too -- search must anticipate tentacle
            # retaliation.
            if side_now == 2 and not self.done:
                from visibility import is_scenery_unit
                # controller=null sides NEVER act: the engine's turn
                # loop skips empty teams entirely
                # (playsingle_controller.cpp:198-210
                # skip_empty_sides), so emitting a sim turn (and its
                # exported [init_side]) for one desynchronizes
                # playback's turn counter and drifts ToD -- the
                # 2026-07-19 Silverhead damage desync ("Nani the
                # Shapeshifter" is an ARMED null-side tentacle).
                # Minis' tentacle sides are controller=ai and keep
                # their turn.
                _null_sides = getattr(
                    self.gs.global_info, "_null_controller_sides",
                    None) or frozenset()
                if any(u.side not in (1, 2)
                       and u.side not in _null_sides
                       and not is_scenery_unit(u)
                       for u in self.gs.map.units):
                    from tools.neutral_ai import run_neutral_side_turn
                    run_neutral_side_turn(self, side=3)
                    if self.done and self.gs.global_info.current_side \
                            not in (1, 2):
                        # A tentacle killed a leader: the game is
                        # over and the WINNER is the player side
                        # whose leader survives (_check_game_over's
                        # leader-alive rule). Terminal states must
                        # not report current_side=3 -- every
                        # downstream consumer (telemetry, terminal
                        # observation, GameOutcome) indexes by
                        # player side. Point it at the survivor.
                        if self.winner in (1, 2):
                            self.gs.global_info.current_side = \
                                self.winner
            if not self.done:
                self._begin_side_turn(next_side)
        else:
            self._apply_with_stats(cmd)
            # Move MP, truncation (blocked/ambush), reveals, and the
            # ZoC / village-capture MP zeroing are all resolved
            # INSIDE _apply_command's move handler via
            # `pathfind_sim.walk_move_path` -- one truncation
            # semantics shared with replay reconstruction (the old
            # post-apply `_deduct_extra_mp` / `_apply_post_move_stops`
            # fix-ups are gone with the flat-1-MP deduction they
            # corrected).
            extras: dict = {}
            if cmd[0] == "move":
                # Truncated walk (blocked/ambush): annotate the
                # ATTEMPTED destination so the exported replay shows
                # what the policy wanted (comment instrumentation,
                # 2026-07-19).
                walk = getattr(self.gs.global_info,
                               "_last_move_walk", None) or {}
                if walk and walk.get("landed") != walk.get("ordered"):
                    extras["attempted"] = (
                        f"move ordered to {walk['ordered']}, stopped "
                        f"at {walk['landed']} ({walk['stop_reason']})")
            if cmd[0] == "recruit" and leader_pos is not None:
                extras["leader_pos"] = leader_pos
            # Advancement [choose] events from the applier's
            # side-channel (one per advancement step, AMLA and
            # multi-advance chain links included; attacker-first
            # order preserved -- matches attack_unit_and_advance).
            if cmd[0] == "attack":
                advance_choices = list(getattr(
                    self.gs.global_info, "_last_advance_events", [])
                    or [])
                if advance_choices:
                    setattr(self.gs.global_info,
                            "_last_advance_events", [])
                    extras["advance_choices"] = advance_choices
                    if _es is not None:
                        for _adv_side, _ in advance_choices:
                            if _adv_side in (1, 2):
                                _es.advancements[_adv_side] += 1
            if cmd[0] == "attack":
                # Per-strike checkup payloads recorded by
                # resolve_attack (stashed by the shared attack
                # handler). Exported as [checkup][result] children;
                # Wesnoth playback compares each strike's
                # chance/hits/damage/dies and OOS-errors on
                # divergence -- the export-side verification
                # contract (see test_rng_accounting.py).
                strikes = getattr(self.gs.global_info,
                                  "_last_checkup_strikes", None)
                if strikes:
                    extras["checkup_strikes"] = strikes
                    setattr(self.gs.global_info,
                            "_last_checkup_strikes", None)
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
                # Deepcopy the PRE-action state: self.step() below mutates
                # self.gs in place, so storing the live reference would make
                # every SimStep.state alias the single terminal state (its
                # docstring promises "the game state BEFORE the action").
                # Matches the per-decision snapshot the trainer/MCTS rely on.
                import copy as _copy
                traj.append(SimStep(state=_copy.deepcopy(self.gs),
                                    action=action, side=side))
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
    # Loop guard: max consecutive planner-rejected steps before the
    # sim forces end_turn (mask-less caller protection; see step()).
    _MAX_CONSECUTIVE_REJECTS = 8

    _AMBUSH_ABILITIES = frozenset({
        "ambush", "nightstalk", "concealment", "submerge",
    })

    def _refresh_uncovered_state(self, current_side: int) -> None:
        """Called at each side's init_side. Implements Wesnoth's
        `unit::new_turn` reset of STATE_UNCOVERED for the side's own
        units (unit.cpp:1277): a hider that was revealed (ambush
        trigger, blocked-move reveal, or its own attack) re-hides at
        ITS side's turn start.

        Adjacency-based discovery is deliberately NOT persisted
        here: `would_be_discovered` is a LIVE predicate (a hider
        adjacent to an enemy is visible only while the enemy stays
        adjacent -- display_context.cpp:29-49), modelled by
        `visibility._discovered_by_adjacency` at observation time.
        (An earlier revision persisted turn-start adjacency reveals
        for the whole turn; source check 2026-07-17 showed the
        engine has no such rule.)
        """
        uncovered: set = getattr(
            self.gs.global_info, "_uncovered_units", None) or set()
        for u in list(self.gs.map.units):
            if u.side == current_side and u.id in uncovered:
                uncovered.discard(u.id)
        setattr(self.gs.global_info, "_uncovered_units", uncovered)

    def _begin_side_turn(self, side: int) -> None:
        """Fire init_side(side). Replay-recon's _apply_command for
        init_side handles: setting current_side, incrementing turn
        (when side == 1), updating time-of-day, firing scenario
        turn-start events, and computing healing / poison / curing
        for `side`'s units. Game-over can also fire here (turn-limit
        checks; NB poison cannot kill -- healing clamps at 1 HP)."""
        self._apply_with_stats(["init_side", side])
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
              clamp a damage roll. Verified against 1.18 source: drain
              caps at `max_hp - hp` (attack.cpp:1037), healing caps in
              calculate_healing (heal.cpp), WML `[effect]
              apply_to=hitpoints` clamps unless `violate_maximum=yes`
              (unit.cpp:2167-2170). The only mainline-supported paths
              to over-cap are Lua `unit.hitpoints = N` (lua_unit.cpp:
              437 -> unit.hpp:519, no clamp) and `set_max_hitpoints(N)`
              with N below current hp (unit.hpp:515, no hp clamp) --
              neither happens in 2p ladder PvP. If this fires on a 2p
              replay, it's a real sim bug; if it fires on a campaign /
              custom-era replay, the upstream filter should be
              tightened to exclude it.

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

    def _action_to_command(
        self, action: dict,
    ) -> Tuple[Optional[list], Optional[int]]:
        """Translate the policy's action dict into the command list
        format _apply_command consumes.

        Returns ``(cmd, terrain_cost)``:
          - ``cmd`` is the list `_apply_command` consumes, or None if
            the action was malformed / illegal (caller treats None as
            an end-turn fallback);
          - ``terrain_cost`` is the move's terrain MP cost when
            ``cmd[0] == "move"``, else None. The caller uses this to
            fix up MP after `_apply_command`'s flat 1-MP deduction.

        Pure: never mutates the caller's `action` dict. The previous
        implementation stashed `action["_terrain_cost"] = cost` which
        was side-effecting -- a caller hashing or reusing the dict
        would see the stashed key, and any bypass path that
        constructed a cmd directly would silently underdeduct MP.
        """
        # Lazy import to avoid a circular dep at module-load time.
        from tools.abilities import hex_neighbors

        atype = action.get("type")
        if atype == "end_turn":
            return ["end_turn"], None
        if atype == "move":
            start: Position = action["start_hex"]
            target: Position = action["target_hex"]
            # Move-action invariant: actions only carry (start, target)
            # hex pairs -- single-step moves. The policy's mask
            # restricts moves to adjacent hexes, and Wesnoth replays
            # we EXTRACT can carry multi-hex paths (replay_extract
            # preserves them), but the SIM doesn't model multi-hex
            # action dicts. If a future change exposes multi-step
            # moves to the policy:
            #   1. The action dict gains a `path: List[Position]`
            #      field (or similar) -- assert below catches the
            #      transition.
            #   2. `_action_to_command` would need to validate each
            #      step's adjacency + MP cost and emit the full
            #      path in the WML, including firing enter_hex
            #      events for scenarios that depend on them.
            #   3. sim_to_replay's `_wml_for_command` would need to
            #      emit the full path verbatim.
            if "path" in action:
                raise NotImplementedError(
                    "explicit `path` field in move action dict not "
                    "supported; the sim plans the route itself "
                    "(Wesnoth-default cost model, see "
                    "tools/pathfind_sim.py). Pass (start_hex, "
                    "target_hex)."
                )
            mover = next(
                (u for u in self.gs.map.units
                 if u.position.x == start.x and u.position.y == start.y
                 and u.side == self.current_side),
                None,
            )
            if mover is None:
                return None, None
            # Plan the route from the ACTING SIDE'S OBSERVABLE state
            # -- the same knowledge the legality mask used to offer
            # this target, and the same knowledge Wesnoth's client
            # uses for a player's move order (mouse_events.cpp
            # get_route). Hidden units neither block nor ZoC here;
            # they resolve at execution (walk_move_path: blocked /
            # ambush truncation).
            from tools.pathfind_sim import (
                ReachContext, unit_reach, route_to)
            ctx = ReachContext.for_side(
                self.gs, self.current_side, exclude_unit=mover)
            reach = unit_reach(mover, self.gs, ctx)
            tpos = (target.x, target.y)
            if tpos not in reach.landable:
                # The mask uses the same planner on the same
                # observable state, so this means mask and sim
                # disagree -- a contract bug, not a policy mistake.
                # Loud (WARNING, not debug) + re-decide instead of
                # burning the turn.
                log.warning(
                    f"sim: move target {tpos} not landable for "
                    f"{mover.id}@{(start.x, start.y)} "
                    f"(mp={mover.current_moves}); mask/sim "
                    f"reachability disagreement -- re-deciding")
                return ["__reject_action__"], None
            path = route_to(reach, tpos)
            cmd_move = ["move",
                        [p[0] for p in path],
                        [p[1] for p in path],
                        self.current_side]   # from_side filter
            return cmd_move, None
        if atype == "attack":
            start: Position = action["start_hex"]
            target: Position = action["target_hex"]
            weapon = int(action.get("attack_index", 0))
            # Defensive adjacency check. step() normally normalizes
            # non-adjacent attack actions into a move + adjacent
            # attack pair before reaching here, but if a future caller
            # bypasses step() or my move-planning logic misses a case,
            # emitting an [attack] from a non-adjacent source produces
            # an invalid Wesnoth replay (battle_context disables
            # out-of-range weapons; the [random_seed] follow-up
            # orphans, errors with "found dependent command in replay
            # while is_synced=false"). Reject rather than emit garbage.
            if (target.x, target.y) not in hex_neighbors(start.x, start.y):
                log.debug(
                    f"sim: rejecting non-adjacent attack "
                    f"{(start.x, start.y)}->{(target.x, target.y)}")
                return None, None
            # Allocate a synced-RNG seed so combat damage rolls match
            # what Wesnoth replays back from the [random_seed]
            # follow-up command sim_to_replay emits. cmd[7] is the
            # seed slot consumed by replay_dataset._apply_command.
            seed = self._next_seed()
            # Resolve the defender's counter-weapon NOW (exact
            # engine-rating port, tools/combat_outcomes): the
            # command must carry a concrete index so the sim applies
            # retaliation and Wesnoth playback uses the same counter
            # we resolved. Lazy import: combat_outcomes pulls in
            # replay_dataset, which this module must not import at
            # module level.
            from tools.combat_outcomes import choose_counter_weapon
            att_u = next(
                (u for u in self.gs.map.units
                 if u.position.x == start.x and u.position.y == start.y),
                None)
            dfd_u = next(
                (u for u in self.gs.map.units
                 if u.position.x == target.x and u.position.y == target.y),
                None)
            from visibility import is_scenery_unit
            if dfd_u is not None and is_scenery_unit(dfd_u):
                # Wesnoth-as-played refuses attacks on incapacitated
                # or scenery units (UI gate, mouse_events.cpp:753 --
                # see docs/wesnoth_rules.md). The legality mask should
                # never send one; refuse rather than resolve combat
                # against a statue (defense in depth after the
                # 2026-07-11 scenery-encoding bug, where exactly that
                # happened for months).
                es = getattr(self, "_engagement", None)
                if es is not None and self.current_side in (1, 2):
                    es.attacks_rejected_sim[self.current_side] += 1
                log.debug(
                    f"sim: rejecting attack on scenery/petrified "
                    f"target at {(target.x, target.y)}")
                return None, None
            d_weapon = (choose_counter_weapon(self.gs, att_u, dfd_u, weapon)
                        if att_u is not None and dfd_u is not None
                        else -1)
            return ["attack",
                    start.x, start.y,
                    target.x, target.y,
                    weapon, d_weapon, seed], None
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
            # Leader-on-keep + castle-network connectivity, via the
            # SHARED helper the legality mask consumes
            # (visibility.leader_castle_network) -- audit 2026-07-17
            # found the sim skipped connectivity entirely, so a
            # mask-less caller could emit recruits Wesnoth playback
            # rejects ("cannot recruit unit: ..."). Violation = the
            # caller ignored the mask -> loud reject + re-decide
            # (bounded by the consecutive-reject guard).
            from visibility import leader_castle_network
            _leader = next(
                (u for u in self.gs.map.units
                 if u.side == self.current_side and u.is_leader), None)
            if _leader is None:
                return None, None
            _on_keep, _network = leader_castle_network(self.gs, _leader)
            if not _on_keep or (target.x, target.y) not in _network:
                log.warning(
                    f"sim: recruit {unit_type!r} on "
                    f"({target.x},{target.y}) rejected: "
                    f"{'leader off keep' if not _on_keep else 'hex outside leader castle network'}"
                    f" -- re-deciding")
                return ["__reject_action__"], None
            # Type must be on the side's recruit list (mask offers
            # only own_recruit_types; engine playback rejects
            # off-list recruits).
            side_idx = self.current_side - 1
            if 0 <= side_idx < len(self.gs.sides):
                _rlist = self.gs.sides[side_idx].recruits or ()
                if _rlist and unit_type not in _rlist:
                    log.warning(
                        f"sim: recruit {unit_type!r} not on side "
                        f"{self.current_side}'s recruit list -- "
                        f"re-deciding")
                    return ["__reject_action__"], None
            cost = _recruit_cost_for(unit_type)
            if 0 <= side_idx < len(self.gs.sides):
                gold = int(self.gs.sides[side_idx].current_gold)
                if gold < cost:
                    log.debug(
                        f"sim: refusing recruit {unit_type!r} (cost={cost} "
                        f"> gold={gold})")
                    return None, None
            # God-view occupancy check. The sampler's mask only sees
            # what the model can see (visible units); the sim has
            # ground truth and knows about fog-hidden enemies on
            # castle hexes. If we'd be recruiting on top of an
            # actually-occupied hex, signal "rejected for retry"
            # rather than "rejected for end_turn fallback":
            #   - Add hex to gs.global_info._recruit_rejected_hexes.
            #   - Return ("__retry_recruit__", None) -- a sentinel
            #     step() recognizes and turns into a no-op (no apply,
            #     no end_turn, no history append). The harness sees
            #     `last_step_rejected=True` and re-decides with the
            #     new rejection state.
            for u in self.gs.map.units:
                if u.position.x == target.x and u.position.y == target.y:
                    rejected = (
                        getattr(self.gs.global_info,
                                "_recruit_rejected_hexes", None) or set()
                    )
                    rejected.add((target.x, target.y))
                    setattr(self.gs.global_info,
                            "_recruit_rejected_hexes", rejected)
                    log.debug(
                        f"sim: recruit on ({target.x},{target.y}) "
                        f"rejected (occupied by {u.id!r}, side {u.side}); "
                        f"adding to rejection set, harness should retry"
                    )
                    return ["__retry_recruit__"], None
            # Allocate a synced-RNG seed for the trait roll. Without
            # this both sides diverge: the sim might give the recruit
            # `quick` (+1 MP) while Wesnoth's playback rolls a
            # different trait, leading to "corrupt movement" the
            # first time the unit's MP differs between the two views.
            # cmd[4] is the trait_seed slot consumed by
            # _build_recruit_unit's MTRng path.
            seed = self._next_seed()
            return ["recruit", unit_type, target.x, target.y, seed], None
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
            return None, None
        log.debug(f"sim: unknown action type {atype!r}")
        return None, None


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
