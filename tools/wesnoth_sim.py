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

from classes import GameState, Position
from replay_dataset import (
    _apply_command,
    _build_initial_gamestate,
    _setup_scenario_events,
)


log = logging.getLogger("wesnoth_sim")


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


# Overlays that affect MOVEMENT cost but NOT defense. Mined from
# wesnoth_src/data/core/terrain.cfg by scanning every [terrain_type]
# whose `mvt_alias` follows the pattern `-,_bas,<X>` (MINUS marker:
# Wesnoth resolves these as MAX(base_cost, X_cost)). Defense for the
# same overlays is empty / aliasof=_bas, which is why the existing
# `_OVERLAY_DEFENSE_KEYS` doesn't cover them.
#
# Without these, e.g. `Re^Tf` (road with mushroom-grove overlay) is
# read as flat (cost 1 for woodland) when Wesnoth actually treats it
# as MAX(flat=1, fungus=2)=2 -- the sim then emits a move into Re
# next that Wesnoth rejects as 'corrupt movement' because the unit
# now has 1 fewer MP than the sim believed.
_OVERLAY_MOVEMENT_KEYS: Dict[str, List[str]] = {
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
    # Forest+frozen overlays (Fda / Fma / Fpa) — already handled
    # above as "forest"; the additional `At` (frozen) doesn't usually
    # matter because forest cost typically dominates, but include
    # frozen as a safety key for movetypes where frozen > forest.
    # (We already added these to the base map; an entry here would
    # be an extra `frozen` key; not adding to keep the table tight.)
    # Cosmetic / structural overlays Wesnoth lists with `aliasof=_bas`
    # only (Em, Es, Edp, Bs*, etc.) -- no movement impact, so they
    # are intentionally absent here.
    # Impassable overlays mirror what _OVERLAY_DEFENSE_KEYS already
    # provides for movement (the 99-collapse path catches them).
    "Xm": ["impassable"], "Xv": ["impassable"],
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
        keys = list(_defense_keys_for_code(c))
        if "^" in c:
            overlay = c.split("^", 1)[1]
            for k in _OVERLAY_MOVEMENT_KEYS.get(overlay, []):
                if k not in keys:
                    keys.append(k)
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
    def from_replay(cls, gz_path: Path, **kwargs) -> "WesnothSim":
        """Build an initial state from any replay's `starting_units` +
        scenario_id. The replay is used as a source of map / faction /
        starting-unit configuration only -- we discard the command
        stream and let the policy decide what happens from turn 1."""
        with gzip.open(Path(gz_path), "rt", encoding="utf-8") as f:
            data = json.load(f)
        gs = _build_initial_gamestate(data)
        scenario_id = data.get("scenario_id", "")
        return cls(gs, scenario_id=scenario_id, **kwargs)

    # ----- public stepping API ---------------------------------------

    @property
    def state(self) -> GameState:
        return self.gs

    @property
    def current_side(self) -> int:
        return self.gs.global_info.current_side

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
            _apply_command(self.gs, cmd)
            # Post-apply terrain MP correction. _apply_command
            # already flat-deducted 1 MP; we owe an extra
            # (cost - 1) so the unit's MP reflects Wesnoth's
            # terrain-based cost.
            terrain_cost = action.get("_terrain_cost") if isinstance(action, dict) else None
            if cmd[0] == "move" and terrain_cost is not None and terrain_cost > 1:
                tx, ty = cmd[1][-1], cmd[2][-1]
                self._deduct_extra_mp(tx, ty, terrain_cost - 1)
            extras: dict = {}
            if cmd[0] == "recruit" and leader_pos is not None:
                extras["leader_pos"] = leader_pos
            self.command_history.append(RecordedCommand(
                kind=cmd[0], side=side_now, cmd=list(cmd), extras=extras))

        self._actions_by_side[side_now] = self._actions_by_side.get(side_now, 0) + 1
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
        self._check_game_over()

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
            unit_id = action["unit_id"]
            target: Position = action["target_hex"]
            # Recall is a no-op in replay-recon (no recall list
            # tracked). For self-play we similarly skip; the action
            # gets logged as a wasted turn.
            return ["recall", unit_id, target.x, target.y]
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
