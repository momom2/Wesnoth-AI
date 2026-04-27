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
        cmd = self._action_to_command(action)
        if cmd is None:
            # Action was illegal-shaped or referred to a missing unit.
            # Treat as a wasted turn -- end the side's turn.
            log.debug(f"sim: untranslatable action {action!r}; ending turn")
            cmd = ["end_turn"]

        if cmd[0] == "end_turn":
            _apply_command(self.gs, ["end_turn"])
            # Advance to the next side. 2p only for now.
            n_sides = max(2, len(self.gs.sides))
            next_side = (side_now % n_sides) + 1
            self._begin_side_turn(next_side)
        else:
            _apply_command(self.gs, cmd)

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

    def _begin_side_turn(self, side: int) -> None:
        """Fire init_side(side). Replay-recon's _apply_command for
        init_side handles: setting current_side, incrementing turn
        (when side == 1), updating time-of-day, firing scenario
        turn-start events, and computing healing / poison / curing
        for `side`'s units. Game-over can also fire here (e.g. poison
        kills a leader)."""
        _apply_command(self.gs, ["init_side", side])
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
        atype = action.get("type")
        if atype == "end_turn":
            return ["end_turn"]
        if atype == "move":
            start: Position = action["start_hex"]
            target: Position = action["target_hex"]
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
            # d_weapon=-1 means "let combat.py pick the defender's
            # best weapon", same as Wesnoth's auto-selection.
            return ["attack",
                    start.x, start.y,
                    target.x, target.y,
                    weapon, -1, ""]
        if atype == "recruit":
            unit_type = action["unit_type"]
            target: Position = action["target_hex"]
            # trait_seed empty -> _build_recruit_unit picks default
            # traits via Python's random; not bit-exact with Wesnoth's
            # synced RNG but distribution is the same (and self-play
            # doesn't need exact replay-ability across runs).
            return ["recruit", unit_type, target.x, target.y, ""]
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
